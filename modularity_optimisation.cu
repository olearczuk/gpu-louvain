#include "modularity_optimisation.cuh"
#include <climits>
#include <cstdio>
#include <thrust/partition.h>

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

/**
 * Computes sum of weights of edges adjacent to vertices (results are stored in vertexEdgesSum).
 * @param V              number of vertices
 * @param vertexEdgesSum vertex -> sum of edges adjacent to vertex
 * @param edgesIndex     vertex -> begin index of edges (in weights array)
 * @param weights        array of weights of edges
 */
__global__ void computeEdgesSum(int *V, float *vertexEdgesSum, int *edgesIndex, float *weights) {
	int vertexIndex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (vertexIndex < *V) {
		vertexEdgesSum[vertexIndex] = 0;
		for (int i = edgesIndex[vertexIndex]; i < edgesIndex[vertexIndex + 1]; i++)
			vertexEdgesSum[vertexIndex] += weights[i];
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
 * @param vertexIndexes   vertex indexes
 * @param communityWeight community -> weight (sum of edges adjacent to vertices of community)
 * @param vertexCommunity vertex -> community assignment
 * @param vertexEdgesSum  vertex -> sum of edges adjacent to vertex
 */
__global__ void computeCommunityWeight(int V, int *vertexIndexes, float *communityWeight, int *vertexCommunity,
									   float *vertexEdgesSum) {
	int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (index < V) {
		int vertexIndex = vertexIndexes[index];
		int community = vertexCommunity[vertexIndex];
		atomicAdd(&communityWeight[community], vertexEdgesSum[vertexIndex]);
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
 * @param vertexIndex      vertex index
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
__device__ float computeGain(int vertexIndex, int prime, int community, int currentCommunity, float *communityWeight,
							 float *vertexEdgesSum, int *hashCommunity, float *hashWeight, int hashTablesOffset) {
	float communitySum = communityWeight[community];
	float currentCommunitySum = communityWeight[currentCommunity] - vertexEdgesSum[vertexIndex];
	float vertexToCommunity = 0, vertexToCurrentCommunity = 0;
	for (int i = 0; i < prime; i++) {
		int index = hashTablesOffset + i;
		if (hashCommunity[index] == community)
			vertexToCommunity = hashWeight[index];
		else if (hashCommunity[index] == currentCommunity)
			vertexToCurrentCommunity = hashWeight[index];
	}
	float gain = (vertexToCommunity - vertexToCurrentCommunity) / M +
				 vertexEdgesSum[vertexIndex] * (currentCommunitySum - communitySum) / (2 * M * M);
	return gain;
}

/**
 * Finds new vertex -> community assignment (stored in newVertexCommunity) that maximise gains for each vertex.
 * @param V                  number of vertices
 * @param vertices		     vertices
 * @param vertexIndexes      vertex indexes
 * @param prime              prime number used for hashing
 * @param vertexCommunity    vertex -> community assignment
 * @param edgesIndex         vertex -> begin index of edges (in edges and weights arrays)
 * @param edges              array of neighbours
 * @param weights            array of weights of edges
 * @param communityWeight    community -> weight (sum of edges adjacent to vertices of community)
 * @param vertexEdgesSum     vertex -> sum of edges adjacent to vertex
 * @param totalGain          variable that stores sum of partial gains
 * @param newVertexCommunity new vertex -> community assignment
 */
__global__ void computeMove(int V, int *vertices, int *vertexIndexes, int prime, int *vertexCommunity, int *edgesIndex, int *edges,
							float *weights, float *communityWeight, float *vertexEdgesSum, float *totalGain, int
							*newVertexCommunity) {
	int verticesPerBlock = blockDim.y;
	int concurrentNeighbours = blockDim.x;
	int hashTablesOffset = threadIdx.y * prime;
	int bestGainsIndex = threadIdx.y;
	int index = blockIdx.x * verticesPerBlock + threadIdx.y;
	extern __shared__ int s[];
	int *hashCommunity = s;
	auto *hashWeight = (float*)&hashCommunity[verticesPerBlock * prime];
	auto *bestGains = (float*)&hashCommunity[2 * verticesPerBlock * prime];
	int *neighbourChosen = &hashCommunity[2 * verticesPerBlock * prime + verticesPerBlock];
	bestGains[bestGainsIndex] = 0;
	neighbourChosen[bestGainsIndex] = V;
	// TODO - maybe there is more elegant solution
	for (int i = 0; i < prime; i++) {
		hashWeight[hashTablesOffset + i] = 0;
		hashCommunity[hashTablesOffset + i] = -1;
	}

	if (index < V) {
		int vertexIndex = vertexIndexes[index];
		int vertex = vertices[vertexIndex];;
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
		if (concurrentNeighbours > WARP_SIZE)
			__syncthreads();
		// choosing community
		neighbourIndex = threadIdx.x + edgesIndex[vertex];
		while (neighbourIndex < edgesIndex[vertex + 1]) {
			int neighbour = edges[neighbourIndex];
			int community = vertexCommunity[neighbour];
			if (community < currentCommunity) {
				float gain = computeGain(vertex, prime, community, currentCommunity, communityWeight, vertexEdgesSum,
										 hashCommunity, hashWeight, hashTablesOffset);
				if (gain > bestGain || (gain == bestGain && community < bestCommunity)) {
					bestGain = gain;
					bestCommunity = community;
					bestNeighbour = neighbour;
				}
			}
			neighbourIndex += concurrentNeighbours;
		}
		// TODO - this can be done more efficiently with `tree` method
		if (bestGain > 0) {
			atomicMaxFloat(&bestGains[bestGainsIndex], bestGain);
			// TODO - this should be only for `big case` - 6th bucket (similarly for 7th bucket)
			if (concurrentNeighbours > WARP_SIZE)
				__syncthreads();
			if (bestGains[bestGainsIndex] == bestGain) {
				atomicMin(&newVertexCommunity[vertex], bestCommunity);
				// TODO - this should be only for `big case` - 6th bucket (similarly for 7th bucket)
				if (concurrentNeighbours > WARP_SIZE)
					__syncthreads();
				if (newVertexCommunity[vertex] == bestCommunity)
					atomicMin(&neighbourChosen[bestGainsIndex], bestNeighbour);
				// TODO - this should be only for `big case` - 6th bucket (similarly for 7th bucket)
				if (concurrentNeighbours > WARP_SIZE)
					__syncthreads();
				if (neighbourChosen[bestGainsIndex] == bestNeighbour)
					atomicAdd(totalGain, bestGain);
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
__global__ void updateVertexCommunity(int V, int *vertices, int *newVertexCommunity, int *vertexCommunity) {
	int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (index < V) {
		int vertex = vertices[index];
		vertexCommunity[vertex] = newVertexCommunity[vertex];
	}
}

void printVertexAssignments(host_structures& structures) {
	for (int c = 0; c < structures.V; c++) {
		printf("%d", c + 1);
		for (int v = 0; v < structures.originalV; v++)
			if (c == structures.vertexCommunity[v])
				printf(" %d", v + 1);
		if (c < structures.V - 1)
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
	// computing new vertices weights
	computeEdgesSum<<<blocks, THREADS_PER_BLOCK>>>(deviceStructures.V, deviceStructures.vertexEdgesSum,
												   deviceStructures.edgesIndex, deviceStructures.weights);

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
	int verticesIndexes[hostStructures.V];
	for (int i = 0; i < hostStructures.V; i++)
		verticesIndexes[i] = i;
	int *deviceVerticesIndexes;
	HANDLE_ERROR(cudaMalloc((void**)&deviceVerticesIndexes, hostStructures.V * sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(deviceVerticesIndexes, verticesIndexes, hostStructures.V * sizeof(int), cudaMemcpyHostToDevice));
	while (totalGain >= minGain) {
		totalGain = 0;
		HANDLE_ERROR(cudaMemcpy(deviceStructures.totalGain, &totalGain, sizeof(float), cudaMemcpyHostToDevice));

		// TODO - separate case for last bucket
		for(int bucketNum= 0; bucketNum < bucketsSize - 2; bucketNum++) {
			dim3 blockDimension = dims[bucketNum];
			int vertexDegree = buckets[bucketNum + 1];
			int prime = primes[bucketNum];
			auto predicate = isInBucket(buckets[bucketNum], buckets[bucketNum + 1], hostStructures.edgesIndex);
			// TODO thrust::device or thrust::host
			int *deviceVerticesEnd = thrust::partition(thrust::device, deviceVerticesIndexes, deviceVerticesIndexes + hostStructures.V, predicate);
			int V = thrust::distance(deviceVerticesIndexes, deviceVerticesEnd);
			if (V > 0) {
				int sharedMemSize = blockDimension.y * prime * (sizeof(float) + sizeof(int)) +
								 blockDimension.y * (sizeof(float) + sizeof(int));
				int blocksDegrees = (V * vertexDegree + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
				computeMove<<<blocksDegrees, blockDimension, sharedMemSize>>>(V, deviceStructures.vertices, deviceVerticesIndexes, prime,
						deviceStructures.vertexCommunity, deviceStructures.edgesIndex, deviceStructures.edges,
						deviceStructures.weights, deviceStructures.communityWeight, deviceStructures.vertexEdgesSum,
						deviceStructures.totalGain, deviceStructures.newVertexCommunity);
			}
			// updating vertex -> community assignment
			updateVertexCommunity<<<blocks, THREADS_PER_BLOCK>>>(V, deviceVerticesIndexes,
																 deviceStructures.newVertexCommunity, deviceStructures.vertexCommunity);
			// updating community weight
			resetCommunityWeight<<<blocks, THREADS_PER_BLOCK>>>(hostStructures.V, deviceStructures.communityWeight);
			computeCommunityWeight<<<blocks, THREADS_PER_BLOCK>>>(hostStructures.V, deviceVerticesIndexes,
																  deviceStructures.communityWeight, deviceStructures.vertexCommunity, deviceStructures.vertexEdgesSum);
		}
//		dim3 blockDimension = {8, 16};
//		int V = hostStructures.V;
//		int vertexDegree = 8;
//		int prime = 13;
//		int sharedMemSize = blockDimension.y * prime * (sizeof(float) + sizeof(int)) +
//				blockDimension.y * (sizeof(float) + sizeof(int));
//		int blocksDegrees = (V * vertexDegree + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//		computeMove<<<blocksDegrees, blockDimension, sharedMemSize>>>(V, deviceVerticesIndexes, prime,
//				deviceStructures.vertexCommunity, deviceStructures.edgesIndex, deviceStructures.edges,
//				deviceStructures.weights, deviceStructures.communityWeight, deviceStructures.vertexEdgesSum,
//				deviceStructures.totalGain, deviceStructures.newVertexCommunity);
//
//		updateVertexCommunity<<<blocks, THREADS_PER_BLOCK>>>(V, deviceVerticesIndexes,
//				deviceStructures.newVertexCommunity, deviceStructures.vertexCommunity);
//		// updating community weight
//		resetCommunityWeight<<<blocks, THREADS_PER_BLOCK>>>(hostStructures.V, deviceStructures.communityWeight);
//		computeCommunityWeight<<<blocks, THREADS_PER_BLOCK>>>(hostStructures.V, deviceVerticesIndexes,
//				deviceStructures.communityWeight, deviceStructures.vertexCommunity, deviceStructures.vertexEdgesSum);

		HANDLE_ERROR(cudaMemcpy(&totalGain, deviceStructures.totalGain, sizeof(float), cudaMemcpyDeviceToHost));
		printf("%f\n", totalGain);
		wasAnythingChanged = wasAnythingChanged | (totalGain > 0);
	}
	HANDLE_ERROR(cudaMemcpy(hostStructures.vertexCommunity, deviceStructures.vertexCommunity,
			hostStructures.V * sizeof(float), cudaMemcpyDeviceToHost));
	printVertexAssignments(hostStructures);
	HANDLE_ERROR(cudaFree(deviceVerticesIndexes));
	return wasAnythingChanged;
}