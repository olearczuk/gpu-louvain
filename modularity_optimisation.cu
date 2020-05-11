#include "modularity_optimisation.cuh"
#include <climits>
#include <cstdio>
#include <thrust/partition.h>
#include <cmath>

/**
 * Computes hashing (using double hashing) for open-addressing purposes of arrays in prepareHashArrays function.
 * @param val   value we want to insert
 * @param index current position
 * @param prime size of hash array
 * @return hash
 */
__device__ int getHash(int val, int index, int prime) {
	int h1 = val % HASHING;
	int h2 = 1 + (val % (HASHING - 1));
	return (h1 + index * h2) % prime;
}

__device__ float atomicMaxFloat(float *addr, float value) {
	float old;
	old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
		  __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

	return old;
}

__global__ void printEdgesSum(int *V, float *vertexEdgesSum) {
	for (int v = 0; v < *V; v++) {
		printf("vertexEdgesSum %d %f\n", v+1, vertexEdgesSum[v]);
	}
}

/**
 * Computes sum of weights of edges adjacent to vertices (results are stored in vertexEdgesSum).
 * @param V              number of vertices
 * @param vertexEdgesSum vertex -> sum of edges adjacent to vertex
 * @param edgesIndex     vertex -> begin index of edges (in weights array)
 * @param weights        array of weights of edges
 */
__global__ void computeEdgesSum(int *V, float *vertexEdgesSum, int *edgesIndex, float *weights) {
	int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (vertex < *V) {
		vertexEdgesSum[vertex] = 0;
		for (int i = edgesIndex[vertex]; i < edgesIndex[vertex + 1]; i++)
			vertexEdgesSum[vertex] += weights[i];
	}
}

/**
 * Reset weights of communities.
 * @param communities     number of communities
 * @param communityWeight community -> weight (sum of edges adjacent to vertices of community)
 */
__global__ void resetCommunityWeight(int communities, float *communityWeight) {
	int community = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (community < communities)
		communityWeight[community] = 0;
}

/**
 * Computes sum of weights of edges adjacent to vertices (results are stored in vertexEdgesSum).
 * @param V               number of vertices
 * @param communityWeight community -> weight (sum of edges adjacent to vertices of community)
 * @param vertexCommunity vertex -> community assignment
 * @param vertexEdgesSum  vertex -> sum of edges adjacent to vertex
 */
__global__ void computeCommunityWeight(int V, float *communityWeight, int *vertexCommunity,
									   float *vertexEdgesSum) {
	int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (vertex < V) {
		int community = vertexCommunity[vertex];
		atomicAdd(&communityWeight[community], vertexEdgesSum[vertex]);
	}
}

/**
 * Fills content of hashCommunity and hash_weights arrays that are later used in computeGain function.
 * @param community        neighbour's community
 * @param prime            prime number used for hashing
 * @param weight		   neighbour's weight
 * @param hashWeight	   table of sum of weights between vertices and communities
 * @param hashCommunity	   table informing which community's info is stored in given index
 * @param hashTablesOffset offset of the vertex in hash arrays (single hash array may contain multiple vertices)
 */
__device__ void prepareHashArrays(int community, int prime, float weight, float *hashWeight, int *hashCommunity,
								  int hashTablesOffset) {
	bool foundPosition = false;
	int it = 0;
	while (!foundPosition) {
		int curPos = hashTablesOffset + getHash(community, it++, prime);
		if (hashCommunity[curPos] == community)
			atomicAdd(&hashWeight[curPos], weight);
			// TODO - uses inelegant solution with -1
		else if (hashCommunity[curPos] == -1) {
			if (atomicCAS(&hashCommunity[curPos], -1, community) == -1)
				atomicAdd(&hashWeight[curPos], weight);
			else if (hashCommunity[curPos] == community)
				atomicAdd(&hashWeight[curPos], weight);
		}
		foundPosition = hashCommunity[curPos] == community;
	}
}

/**
 * Computes gain that would be obtained if we would move vertex to community.
 * @param vertex      	   vertex number
 * @prime                  prime number used for hashing (and size of vertex's area in hash arrays)
 * @param community 	   neighbour's community
 * @param currentCommunity current community of vertex
 * @param communityWeight  community -> weight (sum of edges adjacent to vertices of community)
 * @param vertexEdgesSum   vertex -> sum of edges adjacent to vertex
 * @param hashCommunity    table informing which community's info is stored in given index
 * @param hashWeight       table of sum of weights between vertices and communities
 * @param hashTablesOffset offset of the vertex in hash arrays (single hash array may contain multiple vertices
 * @return gain that would be obtained by moving vertex to community
 */
__device__ float computeGain(int vertex, int prime, int community, int currentCommunity, float *communityWeight,
							 float *vertexEdgesSum, int *hashCommunity, float *hashWeight, int hashTablesOffset, int neighbour) {
	float communitySum = communityWeight[community];
	float currentCommunitySum = communityWeight[currentCommunity] - vertexEdgesSum[vertex];
	float vertexToCommunity = 0, vertexToCurrentCommunity = 0;
	for (int i = 0; i < prime; i++) {
		int index = hashTablesOffset + i;
		if (hashCommunity[index] == community)
			vertexToCommunity = hashWeight[index];
		else if (hashCommunity[index] == currentCommunity)
			vertexToCurrentCommunity = hashWeight[index];
	}
	// TODO - gain does not have to be depending on vertexToCurrentCommunity
	float gain = (vertexToCommunity - vertexToCurrentCommunity) / M +
				 vertexEdgesSum[vertex] * (currentCommunitySum - communitySum) / (2 * M * M);
	if (vertex == -1) {
		printf("VERT: %d, EDGE: %d, MOD_GAIN: %f, e_i: %f, k_i: %f, a_C(j): %f, a_C(i)-i: %f, m: %f, e_C(i)-i: %f\n",
				vertex, neighbour, gain, vertexToCommunity, vertexEdgesSum[vertex], communitySum, currentCommunitySum, M, vertexToCurrentCommunity);
	}
	return gain;
}

/**
 * Finds new vertex -> community assignment (stored in newVertexCommunity) that maximise gains for each vertex.
 * @param V                  number of vertices
 * @param vertices		     vertices
 * @param prime              prime number used for hashing
 * @param vertexCommunity    vertex -> community assignment
 * @param edgesIndex         vertex -> begin index of edges (in edges and weights arrays)
 * @param edges              array of neighbours
 * @param weights            array of weights of edges
 * @param communityWeight    community -> weight (sum of edges adjacent to vertices of community)
 * @param vertexEdgesSum     vertex -> sum of edges adjacent to vertex
 * @param newVertexCommunity new vertex -> community assignment
 */
__global__ void computeMove(int V, int *vertices, int prime, int *vertexCommunity, int *edgesIndex, int *edges,
							float *weights, float *communityWeight, float *vertexEdgesSum, int *newVertexCommunity,
							int *communitySize) {
	int verticesPerBlock = blockDim.y;
	int concurrentNeighbours = blockDim.x;
	int hashTablesOffset = threadIdx.y * prime;
	int bestGainsIndex = threadIdx.y;
	int vertexIndex = blockIdx.x * verticesPerBlock + threadIdx.y;
	extern __shared__ int s[];
	int *hashCommunity = s;
	auto *hashWeight = (float*)&hashCommunity[verticesPerBlock * prime];
	auto *bestGains = (float*)&hashCommunity[2 * verticesPerBlock * prime];
//	int *neighbourChosen = &hashCommunity[2 * verticesPerBlock * prime + verticesPerBlock];
	bestGains[bestGainsIndex] = 0;
//	neighbourChosen[bestGainsIndex] = V;
	// TODO - maybe there is more elegant solution
	for (int i = 0; i < prime; i++) {
		hashWeight[hashTablesOffset + i] = 0;
		hashCommunity[hashTablesOffset + i] = -1;
	}

	if (vertexIndex < V) {
		int vertex = vertices[vertexIndex];
		int currentCommunity = vertexCommunity[vertex];
		int bestCommunity = currentCommunity;
		int bestNeighbour = vertex;
		float bestGain = 0;
		// putting data in hash table
		int neighbourIndex = threadIdx.x + edgesIndex[vertex];

		while (neighbourIndex < edgesIndex[vertex + 1]) {
			int neighbour = edges[neighbourIndex];
			int community = vertexCommunity[neighbour];
			float weight = weights[neighbourIndex];
			// this lets us achieve ei -> C(i)\{i} instead of ei -> C(i)
			if (neighbour != vertex)
				prepareHashArrays(community, prime, weight, hashWeight, hashCommunity, hashTablesOffset);
			neighbourIndex += concurrentNeighbours;
		}
		// TODO - this should be only for `big case` - 6th bucket (similarly for 7th bucket)
		// TODO - gain in computeGain does not have to be depending on vertexToCurrentCommunity, so this is not needed
		if (concurrentNeighbours > WARP_SIZE)
			__syncthreads();
		// choosing community
		neighbourIndex = threadIdx.x + edgesIndex[vertex];
		while (neighbourIndex < edgesIndex[vertex + 1]) {
			int neighbour = edges[neighbourIndex];
			int community = vertexCommunity[neighbour];
			if (vertex == -1) {
				printf("vertex: %d, neighbour: %d, curcom: %d, curcomsize: %d, community: %d, community size: %d\n", vertex, neighbour, currentCommunity,
						communitySize[currentCommunity], community, communitySize[community]);
			}
			// TODO - this should only be checked when both communities have size = 1
			if ((community < currentCommunity || communitySize[community] > 1 || communitySize[currentCommunity] > 1) &&
			community != currentCommunity) {
				float gain = computeGain(vertex, prime, community, currentCommunity, communityWeight, vertexEdgesSum,
										 hashCommunity, hashWeight, hashTablesOffset, neighbour);
				if (vertex == -1)
					printf("vertex: %d, neighbour: %d, gain %f\n", vertex, neighbour, gain);
				if (gain > bestGain || (gain == bestGain && community < bestCommunity)) {
					bestGain = gain;
					bestCommunity = community;
					bestNeighbour = neighbour;
//					if (vertex == 5)
//						printf("vertex %d, new bestGain %f, new bestCommunity %d\n", vertex, gain, bestCommunity);
				}
			}
			neighbourIndex += concurrentNeighbours;
		}
		// TODO - this can be done more efficiently with `tree` method
		if (bestGain > 0) {
			newVertexCommunity[vertex] = INT_MAX;
			atomicMaxFloat(&bestGains[bestGainsIndex], bestGain);
			// TODO - this should be only for `big case` - 6th bucket (similarly for 7th bucket)
			if (concurrentNeighbours > WARP_SIZE)
				__syncthreads();
			if (bestGains[bestGainsIndex] == bestGain) {
				if (vertex == -1)
					printf("trying gain %f bestComunity %d, %d\n", bestGain, bestCommunity, newVertexCommunity[vertex]);
				atomicMin(&newVertexCommunity[vertex], bestCommunity);
				if (vertex == -1) {
					printf("vertex %d, best community %d gain %f\n", vertex, newVertexCommunity[vertex], bestGains[bestGainsIndex]);
				}
				// TODO - this should be only for `big case` - 6th bucket (similarly for 7th bucket)
				if (concurrentNeighbours > WARP_SIZE)
					__syncthreads();
//				if (newVertexCommunity[vertex] == bestCommunity)
//					atomicMin(&neighbourChosen[bestGainsIndex], bestNeighbour);
//				 TODO - this should be only for `big case` - 6th bucket (similarly for 7th bucket)
//				if (concurrentNeighbours > WARP_SIZE)
//					__syncthreads();
//				if (neighbourChosen[bestGainsIndex] == bestNeighbour)
//					atomicAdd(totalGain, bestGain);
			}
		}
	}
}

/**
 * Updates vertexCommunity   content based on newVertexCommunity content
 * @param V                  number of vertices
 * @param vertices           vertices
 * @param newVertexCommunity vertex -> community assignment (that gets updated)
 * @param vertexCommunity    new vertex -> community assignment
 */
__global__ void updateVertexCommunity(int V, int *vertices, int *newVertexCommunity, int *vertexCommunity, int *communitySize) {
	int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (index < V) {
		int vertex = vertices[index];
		int oldCommunity = vertexCommunity[vertex];
		int newCommunity = newVertexCommunity[vertex];
		if (oldCommunity != newCommunity) {
			vertexCommunity[vertex] = newVertexCommunity[vertex];
			atomicSub(&communitySize[oldCommunity], 1);
			atomicAdd(&communitySize[newCommunity], 1);
		}
	}
}

__global__ void updateOriginalToCommunity(device_structures deviceStructures) {
	int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (vertex < *deviceStructures.originalV) {
		int community = deviceStructures.originalToCommunity[vertex];
		deviceStructures.originalToCommunity[vertex] = deviceStructures.vertexCommunity[community];
	}
}

void printVertexAssignments(host_structures& structures) {
//	for (int v = 0; v < structures.V; v++)
//		printf("%d ", structures.vertexCommunity[v]);
//	printf("\n");
	for (int c = 0; c < structures.V; c++) {
		printf("%d", c + 1);
		for (int v = 0; v < structures.V; v++)
			if (c == structures.vertexCommunity[v])
				printf(" %d", v + 1);
		printf("\n");
	}
}

struct isInBucket
{
	isInBucket(int llowerBound, int uupperBound, int *eedgesIndex) {
		lowerBound = llowerBound;
		upperBound = uupperBound;
		edgesIndex = eedgesIndex;
	}

	int lowerBound, upperBound;
	int *edgesIndex;
	__host__ __device__
	bool operator()(const int &v) const
	{
		int edgesNumber = edgesIndex[v + 1] - edgesIndex[v];
		return edgesNumber > lowerBound && edgesNumber <= upperBound;
	}
};



bool optimiseModularity(float minGain, device_structures& deviceStructures, host_structures& hostStructures) {
	int blocks = (hostStructures.V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	float totalGain = minGain;
	// TODO - this should be done once
	HANDLE_ERROR(cudaMemcpyToSymbol(M, &hostStructures.M, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(&hostStructures.V, deviceStructures.V, sizeof(float), cudaMemcpyDeviceToHost));
	// TODO - move to structure (additionally no reason to compute it again in 2nd phase)
	int *communitySize;
	HANDLE_ERROR(cudaMalloc((void**)&communitySize, hostStructures.V * sizeof(int)));
	thrust::fill(thrust::device, communitySize, communitySize + hostStructures.V, 1);
	// computing new vertices weights
	computeEdgesSum<<<blocks, THREADS_PER_BLOCK>>>(deviceStructures.V, deviceStructures.vertexEdgesSum,
												   deviceStructures.edgesIndex, deviceStructures.weights);
//	printEdgesSum<<<1, 1>>>(deviceStructures.V, deviceStructures.vertexEdgesSum);

	HANDLE_ERROR(cudaMemcpy(hostStructures.edgesIndex, deviceStructures.edgesIndex,
			(hostStructures.V + 1) * (sizeof(int)), cudaMemcpyDeviceToHost));
	bool wasAnythingChanged = false;
	int bucketsSize = 8;
	int buckets[] = {0, 4, 8, 16, 32, 84, 319, INT_MAX};
	int primes[] = {7, 13, 29, 53, 127, 479};
	// x - number of neighbours processed concurrently, y - vertices per block
	dim3 dims[] {
			{4, 32},
			{8, 16},
			{16, 8},
			{32, 4},
			{32, 4},
			{128, 1},
			{128, 1},
	};
	int vertices[hostStructures.V];
	for (int i = 0; i < hostStructures.V; i++)
		vertices[i] = i;
	int *deviceVertices;
	HANDLE_ERROR(cudaMalloc((void**)&deviceVertices, hostStructures.V * sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(deviceVertices, vertices, hostStructures.V * sizeof(int), cudaMemcpyHostToDevice));

	// TODO - remove this and add this to community_aggregation
	resetCommunityWeight<<<blocks, THREADS_PER_BLOCK>>>(hostStructures.V, deviceStructures.communityWeight);
	computeCommunityWeight<<<blocks, THREADS_PER_BLOCK>>>(hostStructures.V,
			deviceStructures.communityWeight, deviceStructures.vertexCommunity, deviceStructures.vertexEdgesSum);
	int counter = 0;
	while (totalGain >= minGain) {
//		if (counter == 2)
//			break;
		counter++;
		float modularityBefore = calculateModularity(hostStructures.V, deviceStructures);
		// TODO - separate case for last bucket
		for(int bucketNum= 0; bucketNum < bucketsSize - 2; bucketNum++) {
			dim3 blockDimension = dims[bucketNum];
			int vertexDegree = buckets[bucketNum + 1];
			int prime = primes[bucketNum];
			auto predicate = isInBucket(buckets[bucketNum], buckets[bucketNum + 1], hostStructures.edgesIndex);
			// TODO thrust::device or thrust::host
			int *deviceVerticesEnd = thrust::partition(thrust::device, deviceVertices, deviceVertices + hostStructures.V, predicate);
			int V = thrust::distance(deviceVertices, deviceVerticesEnd);
			if (V > 0) {
				int sharedMemSize = blockDimension.y * prime * (sizeof(float) + sizeof(int)) +
									blockDimension.y * sizeof(float);
//								 blockDimension.y * (sizeof(float) + sizeof(int));
				int blocksDegrees = (V * vertexDegree + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
				computeMove<<<blocksDegrees, blockDimension, sharedMemSize>>>(V, deviceVertices, prime,
						deviceStructures.vertexCommunity, deviceStructures.edgesIndex, deviceStructures.edges,
						deviceStructures.weights, deviceStructures.communityWeight, deviceStructures.vertexEdgesSum,
						deviceStructures.newVertexCommunity, communitySize);
			}
			// updating vertex -> community assignment
			updateVertexCommunity<<<blocks, THREADS_PER_BLOCK>>>(V, deviceVertices,
					deviceStructures.newVertexCommunity, deviceStructures.vertexCommunity, communitySize);
			// updating community weight
			resetCommunityWeight<<<blocks, THREADS_PER_BLOCK>>>(hostStructures.V, deviceStructures.communityWeight);
			computeCommunityWeight<<<blocks, THREADS_PER_BLOCK>>>(hostStructures.V,
																  deviceStructures.communityWeight, deviceStructures.vertexCommunity, deviceStructures.vertexEdgesSum);
		}

//		dim3 blockDimension = {8, 16};
//		int V = hostStructures.V;
//		int vertexDegree = 8;
//		int prime = 13;
//		int sharedMemSize = blockDimension.y * prime * (sizeof(float) + sizeof(int)) +
//				blockDimension.y * (sizeof(float) + sizeof(int));
//		int blocksDegrees = (V * vertexDegree + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//		computeMove<<<blocksDegrees, blockDimension, sharedMemSize>>>(V, deviceVertices, prime,
//																	  deviceStructures.vertexCommunity, deviceStructures.edgesIndex, deviceStructures.edges,
//																	  deviceStructures.weights, deviceStructures.communityWeight, deviceStructures.vertexEdgesSum,
//																	  deviceStructures.newVertexCommunity, deviceStructures.originalToCommunity);
//
//		updateVertexCommunity<<<blocks, THREADS_PER_BLOCK>>>(V, deviceVertices,
//															 deviceStructures.newVertexCommunity, deviceStructures.vertexCommunity);
		// updating community weight
//		resetCommunityWeight<<<blocks, THREADS_PER_BLOCK>>>(hostStructures.V, deviceStructures.communityWeight);
//		computeCommunityWeight<<<blocks, THREADS_PER_BLOCK>>>(hostStructures.V,
//															  deviceStructures.communityWeight, deviceStructures.vertexCommunity, deviceStructures.vertexEdgesSum);

//		HANDLE_ERROR(cudaMemcpy(&totalGain, deviceStructures.totalGain, sizeof(float), cudaMemcpyDeviceToHost));
		float modularityAfter = calculateModularity(hostStructures.V, deviceStructures);
		totalGain = modularityAfter - modularityBefore;
		printf("before: %.10f, after: %.10f\n", modularityBefore, modularityAfter);
		wasAnythingChanged = wasAnythingChanged | (totalGain > 0);
	}
	HANDLE_ERROR(cudaMemcpy(hostStructures.vertexCommunity, deviceStructures.vertexCommunity,
							hostStructures.V * sizeof(float), cudaMemcpyDeviceToHost));
	printVertexAssignments(hostStructures);
	updateOriginalToCommunity<<<(hostStructures.originalV + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(deviceStructures);
	HANDLE_ERROR(cudaFree(deviceVertices));
	return wasAnythingChanged;
}

__global__ void calculateModularityPerVertex(device_structures deviceStructures) {
	int community = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (community < *deviceStructures.V) {
		float e_i_to_ci = 0;
		for (int vertex = 0; vertex < *deviceStructures.V; vertex++) {
			if (deviceStructures.vertexCommunity[vertex] == community) {
				for (int i = deviceStructures.edgesIndex[vertex]; i < deviceStructures.edgesIndex[vertex+1]; i++) {
					int neighbour = deviceStructures.edges[i];
					int neighbourCommunity = deviceStructures.vertexCommunity[neighbour];
					if (neighbourCommunity == community)
						e_i_to_ci += deviceStructures.weights[i];
				}
			}
		}
		float M = *deviceStructures.M;
		atomicAdd(deviceStructures.modularity, e_i_to_ci / (2 * M));
		float communityWeight = deviceStructures.communityWeight[community];
		atomicAdd(deviceStructures.modularity, -1 * communityWeight * communityWeight / (4 * M * M));
	}
}

float calculateModularity(int V, device_structures deviceStructures) {
	float modularity = 0;
	thrust::fill(thrust::device, deviceStructures.modularity, deviceStructures.modularity + 1, (float) 0);
	int blocksNumber = (V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	calculateModularityPerVertex<<<blocksNumber, THREADS_PER_BLOCK>>>(deviceStructures);
	HANDLE_ERROR(cudaMemcpy(&modularity, deviceStructures.modularity, sizeof(float), cudaMemcpyDeviceToHost));
	return modularity;
}
