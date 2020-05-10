#include "community_aggregation.cuh"
#include <thrust/scan.h>

/**
 * Computes hash (using double hashing) for open-addressing purposes of arrays in prepareHashArrays function.
 * @param val   value we want to insert
 * @param index current position
 * @param prime size of hash array
 * @return hash
 */
__device__ int getHashAggregation(int val, int index, int prime) {
	int h1 = val % HASHING;
	int h2 = 1 + (val % (HASHING - 1));
	return (h1 + index * h2) % prime;
}

/**
 * Fills content of hashCommunity and hashWeights arrays that are later used in mergeCommunity function.
 * @param community        neighbour's community
 * @param prime            prime number used for hashing
 * @param weight		   neighbour's weight
 * @param hashWeight	   table of sum of weights between vertices and communities
 * @param hashCommunity	   table informing which community's info is stored in given index
 * @param hashTablesOffset offset of the vertex in hash arrays (single hash array may contain multiple vertices)
 * @return curPos, if this was first addition, -1 otherwise
 */
__device__ int prepareHashArraysAggregation(int community, int prime, float weight, float *hashWeight, int *hashCommunity,
								  int hashTablesOffset) {
	int it = 0;
	while (true) {
		int curPos = hashTablesOffset + getHashAggregation(community, it++, prime);
		if (hashCommunity[curPos] == community) {
			atomicAdd(&hashWeight[curPos], weight);
			return -1;
		}
			// TODO - uses inelegant solution with -1
		else if (hashCommunity[curPos] == -1) {
			if (atomicCAS(&hashCommunity[curPos], -1, community) == -1) {
				float weightBefore = atomicAdd(&hashWeight[curPos], weight);
				if (weightBefore == 0)
					return curPos;
				return -1;
			}
			else if (hashCommunity[curPos] == community) {
				float weightBefore = atomicAdd(&hashWeight[curPos], weight);
				if (weightBefore == 0)
					return curPos;
				return -1;
			}
		}
	}
}

__global__ void initEdgeIndexToCurPos(int E, int *initEdgeIndexToCurPos) {
	int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (index < E) {
		initEdgeIndexToCurPos[index] = -1;
	}
}

__global__ void initArrays(int V, int *communitySize, int *communityDegree, int *newID) {
	int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (index < V) {
		communitySize[index] = 0;
		communityDegree[index] = 0;
		newID[index] = 0;
	}
}

__global__ void fillArrays(int V, int *communitySize, int *communityDegree, int *newID, int *vertexCommunity, int *edgesIndex) {
	int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (vertex < V) {
		int community = vertexCommunity[vertex];
		atomicAdd(&communitySize[community], 1);
		int vertexDegree = edgesIndex[vertex + 1] - edgesIndex[vertex];
		atomicAdd(&communityDegree[community], vertexDegree);
		newID[community] = 1;
	}
}

/**
 * orderVerices is responsible for generating ordered (meaning vertices in the same community are placed
 * next to each other) vertices.
 * @param V               - number of vertices
 * @param orderedVertices - ordered vertices
 * @param vertexStart     - community -> begin index in orderedVertices array
 *                          NOTE: atomicAdd changes values in this array, that's why it has to be reset afterwards
 * @param vertexCommunity - vertex -> community
 */
__global__ void orderVertices(int V, int *orderedVertices, int *vertexStart, int *vertexCommunity) {
	int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (vertex < V) {
		int community = vertexCommunity[vertex];
		int index = atomicAdd(&vertexStart[community], 1);
		orderedVertices[index] = vertex;
	}
}

// TODO - take into account scenarios when concurrentEdges > WARP_SIZE (some __syncthreads())
__global__ void mergeCommunity(int V, int *communities, device_structures deviceStructures, int prime, int *edgePos,
		int *communityDegree, int *orderedVertices, int *vertexStart, int *communitySize, int *edgeIndexToCurPos, int *newEdges,
		float *newWeights) {
	int communitiesOwned = 0;
	int communitiesPerBlock = blockDim.y;
	int concurrentThreads = blockDim.x;
	int hashTablesOffset = threadIdx.y * prime;
	extern __shared__ int s[];
	int *hashCommunity = s;
	auto *hashWeight = (float*)&hashCommunity[communitiesPerBlock * prime];
	auto *communitiesOwnedPrefixSum = (int*)&hashWeight[communitiesPerBlock * prime];
	int communityIndex = blockIdx.x * communitiesPerBlock + threadIdx.y;

	if (communityIndex < V) {
		int community = communities[communityIndex];
		if (communitySize[community] > 0) {
				if (threadIdx.x == 0) {
					// updating number of vertices in new graph
					atomicAdd(deviceStructures.V, 1);
					for (int i = 0; i < prime; i++) {
						hashWeight[hashTablesOffset + i] = 0;
						hashCommunity[hashTablesOffset + i] = -1;
					}
				}

				// filling hash tables content for every vertex in community
				for (int vertexIndex = 0; vertexIndex < communitySize[community]; vertexIndex++) {
					int vertex = orderedVertices[vertexStart[community] + vertexIndex];
					int vertexBaseIndex = deviceStructures.edgesIndex[vertex];
					int vertexDegree = deviceStructures.edgesIndex[vertex + 1] - vertexBaseIndex;

					for (int neighbourIndex = threadIdx.x;
						 neighbourIndex < vertexDegree; neighbourIndex += concurrentThreads) {
						int index = vertexBaseIndex + neighbourIndex;
						int neighbour = deviceStructures.edges[index];
						float weight = deviceStructures.weights[index];
						int neighbourCommunity = deviceStructures.vertexCommunity[neighbour];
						// TODO - this step is to make sure M stays the same for all iterations
						if (neighbourCommunity == community && neighbour != vertex)
							weight /= 2;
						int curPos = prepareHashArraysAggregation(neighbourCommunity, prime, weight, hashWeight,
																  hashCommunity, hashTablesOffset);
						if (curPos > -1) {
							communitiesOwned++;
							edgeIndexToCurPos[index] = curPos;
						}
					}
				}
				// updating number of edges in new graph
				atomicAdd(deviceStructures.E, communitiesOwned);

				// TODO - perform concurrent prefix sum
				int baseIndex = threadIdx.y * concurrentThreads;
				int communitiesOwnedIndex = baseIndex + threadIdx.x;
				for (int i = 0; i < concurrentThreads; i++) {
					if (i == threadIdx.x) {
						if (threadIdx.x != concurrentThreads - 1)
							communitiesOwnedPrefixSum[communitiesOwnedIndex + 1] = communitiesOwned;
						if (threadIdx.x == 0)
							communitiesOwnedPrefixSum[communitiesOwnedIndex] = 0;
						else
							communitiesOwnedPrefixSum[communitiesOwnedIndex] += communitiesOwnedPrefixSum[
									communitiesOwnedIndex - 1];
						if (threadIdx.x == concurrentThreads - 1) {
							// setting real community degree
							communityDegree[community] =
									communitiesOwnedPrefixSum[communitiesOwnedIndex] + communitiesOwned;
						}
					}
				}

				int newEdgesIndex = edgePos[community] + communitiesOwnedPrefixSum[communitiesOwnedIndex];
				for (int vertexIndex = 0; vertexIndex < communitySize[community]; vertexIndex++) {
					int vertex = orderedVertices[vertexStart[community] + vertexIndex];
					int vertexBaseIndex = deviceStructures.edgesIndex[vertex];
					int vertexDegree = deviceStructures.edgesIndex[vertex + 1] - vertexBaseIndex;

					for (int neighbourIndex = threadIdx.x;
						 neighbourIndex < vertexDegree; neighbourIndex += concurrentThreads) {
						int index = vertexBaseIndex + neighbourIndex;
						int curPos = edgeIndexToCurPos[index];
						if (curPos > -1) {
							newEdges[newEdgesIndex] = hashCommunity[curPos];
							newWeights[newEdgesIndex] = hashWeight[curPos];
							newEdgesIndex++;
						}
					}
				}
		}
	}
}

// NOTE: vertexStart now contains edgeIndex of compressed array
__global__ void compressEdges(int V, device_structures deviceStructures, int *communityDegree, int *newEdges,
		float *newWeights, int *communitySize, int *newID, int *edgePos, int *vertexStart) {
	int communitiesPerBlock = blockDim.y;
	int concurrentThreads = blockDim.x;
	int community = blockIdx.x * communitiesPerBlock + threadIdx.y;
	if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
		deviceStructures.edgesIndex[*deviceStructures.V] = *deviceStructures.E;
	}
	if (community < V && communitySize[community] > 0) {
		int neighboursBaseIndex = edgePos[community];
		if (threadIdx.x == 0) {
			int communityNewID = newID[community];
			deviceStructures.vertexCommunity[communityNewID] = communityNewID;
			deviceStructures.newVertexCommunity[communityNewID] = communityNewID;
			deviceStructures.edgesIndex[communityNewID] = vertexStart[community];
		}
		for (int neighbourIndex = threadIdx.x; neighbourIndex < communityDegree[community]; neighbourIndex += concurrentThreads) {
			int newIndex = neighbourIndex + neighboursBaseIndex;
			int oldIndex = vertexStart[community] + neighbourIndex;
			deviceStructures.edges[oldIndex] = newID[newEdges[newIndex]];
			deviceStructures.weights[oldIndex] = newWeights[newIndex];
		}
	}
}

__global__ void updateOriginalToCommunity(device_structures deviceStructures, int *newID) {
	int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (vertex < *deviceStructures.originalV) {
		int community = deviceStructures.originalToCommunity[vertex];
		deviceStructures.originalToCommunity[vertex] = newID[community];
	}
}

__global__ void printNewGraph(device_structures deviceStructures) {
	for (int vertex = 0; vertex < *deviceStructures.V; vertex++) {
		int vertexDegree = deviceStructures.edgesIndex[vertex + 1] - deviceStructures.edgesIndex[vertex];
		printf("%d, %d: [", vertex + 1, vertexDegree);
		for (int i = 0; i < vertexDegree; i++) {
			int index = i + deviceStructures.edgesIndex[vertex];
			printf(" (%d, %f)", deviceStructures.edges[index] + 1, deviceStructures.weights[index]);
		}
		printf(" ]\n");
	}
}

struct IsInBucketAggregation
{
	IsInBucketAggregation(int llowerBound, int uupperBound, int *ccomunityDegree) {
		lowerBound = llowerBound;
		upperBound = uupperBound;
		communityDegree = ccomunityDegree;
	}

	int lowerBound, upperBound;
	int *communityDegree;
	__host__ __device__
	bool operator()(const int &v) const
	{
		int edgesNumber = communityDegree[v];
		return edgesNumber > lowerBound && edgesNumber <= upperBound;
	}
};

void aggregateCommunities(device_structures &deviceStructures, host_structures &hostStructures) {
	int blocks = (hostStructures.V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int *communitySize, *communityDegree, *newID, *edgePos;
	int *vertexStart;
	int *orderedVertices;
	int *edgeIndexToCurPos;
	int *newEdges;
	float *newWeights;
	int V = hostStructures.V, E = hostStructures.E;
	HANDLE_ERROR(cudaMalloc((void**)&communitySize, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&communityDegree, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&newID, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&edgePos, V * sizeof(int)));;
	HANDLE_ERROR(cudaMalloc((void**)&vertexStart, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&orderedVertices, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&edgeIndexToCurPos, E * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&newEdges, E * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&newWeights, E * sizeof(float)));

	int vertices[hostStructures.V];
	for (int i = 0; i < hostStructures.V; i++)
		vertices[i] = i;
	int *deviceVertices;
	HANDLE_ERROR(cudaMalloc((void**)&deviceVertices, hostStructures.V * sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(deviceVertices, vertices, hostStructures.V * sizeof(int), cudaMemcpyHostToDevice));

	initArrays<<<blocks, THREADS_PER_BLOCK>>>(V, communitySize, communityDegree, newID);
	fillArrays<<<blocks, THREADS_PER_BLOCK>>>(V, communitySize, communityDegree, newID,
			deviceStructures.vertexCommunity, deviceStructures.edgesIndex);
	thrust::exclusive_scan(thrust::device, newID, newID + V , newID);
	thrust::exclusive_scan(thrust::device, communityDegree, communityDegree + V, edgePos);
	thrust::exclusive_scan(thrust::device, communitySize, communitySize + V, vertexStart);

	orderVertices<<<blocks, THREADS_PER_BLOCK>>>(V, orderedVertices, vertexStart,
			deviceStructures.vertexCommunity);
//	 resetting vertexStart state to one before orderVertices call
	thrust::exclusive_scan(thrust::device, communitySize, communitySize + V, vertexStart);

	initEdgeIndexToCurPos<<<(hostStructures.E + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(E, edgeIndexToCurPos);

	int bucketsSize = 4;
	int buckets[] = {0, 127, 479, INT_MAX};
	int primes[] = {191, 719};
	dim3 dims[] {
			{32, 4},
			{128, 1},
			{128, 1},
	};

	E = 0;
	HANDLE_ERROR(cudaMemcpy(deviceStructures.E, &E, sizeof(int), cudaMemcpyHostToDevice));
	int zeroV = 0;
	HANDLE_ERROR(cudaMemcpy(deviceStructures.V, &zeroV, sizeof(int), cudaMemcpyHostToDevice));

	// TODO - separate case for last bucket
	for (int bucketNum = 0; bucketNum < bucketsSize - 2; bucketNum++) {
			dim3 blockDimension = dims[bucketNum];
			int commDegree = buckets[bucketNum + 1];
			int prime = primes[bucketNum];
			auto predicate = IsInBucketAggregation(buckets[bucketNum], buckets[bucketNum + 1], communityDegree);
			// TODO thrust::device or thrust::host
			int *deviceVerticesEnd = thrust::partition(thrust::device, deviceVertices, deviceVertices + hostStructures.V, predicate);
			int partitionSize = thrust::distance(deviceVertices, deviceVerticesEnd);
			if (partitionSize > 0) {
				unsigned int sharedMemSize = blockDimension.y * prime * (sizeof(float) + sizeof(int)) + blockDimension.y * blockDimension.x * sizeof(int);
				unsigned int blocksDegrees = (partitionSize * commDegree + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
				mergeCommunity<<<blocksDegrees, blockDimension, sharedMemSize>>>(partitionSize, deviceVertices, deviceStructures, prime, edgePos,
						communityDegree, orderedVertices, vertexStart, communitySize, edgeIndexToCurPos, newEdges, newWeights);
			}
	}

	// vertexStart will contain starting indexes in compressed list
	thrust::exclusive_scan(thrust::device, communityDegree, communityDegree + V, vertexStart);
	int blocksDegrees = (V * WARP_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	dim3 blockDimension = {WARP_SIZE, THREADS_PER_BLOCK / WARP_SIZE};
	compressEdges<<<blocksDegrees, blockDimension>>>(V, deviceStructures, communityDegree, newEdges, newWeights,
			communitySize, newID, edgePos, vertexStart);

	updateOriginalToCommunity<<<(hostStructures.originalV + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(deviceStructures, newID);


	HANDLE_ERROR(cudaMemcpy(&hostStructures.E, deviceStructures.E, sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(&hostStructures.V, deviceStructures.V, sizeof(int), cudaMemcpyDeviceToHost));
	printNewGraph<<<1, 1>>>(deviceStructures);

}

void printOriginalToCommunity(device_structures& deviceStructures, host_structures& hostStructures) {
	HANDLE_ERROR(cudaMemcpy(&hostStructures.V, deviceStructures.V, sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(&hostStructures.originalToCommunity, deviceStructures.originalToCommunity,
			hostStructures.originalV * sizeof(int), cudaMemcpyDeviceToHost));
	for (int c = 0; c < hostStructures.V; c++) {
		printf("%d", c+1);
		for (int v = 0; v < hostStructures.originalV; v++)
			if (c == hostStructures.originalToCommunity[v])
				printf(" %d", v + 1);
		printf("\n");
	}
}