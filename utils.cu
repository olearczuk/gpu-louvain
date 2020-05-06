#include "utils.cuh"
#include <vector>
#include <iostream>

host_structures readInputData() {
    int V, E;
    std::cin >> V >> V >> E;
    int v1, v2;
    float w;
    host_structures hostStructures;
	hostStructures.originalV = V;
	hostStructures.V = V;
    HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.vertexCommunity, V * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.communityWeight, V * sizeof(float), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.edgesIndex, (V + 1) * sizeof(int), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.originalToCommunity, V * sizeof(int), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.vertices, V * sizeof(int), cudaHostAllocDefault));

    std::vector<std::pair<int, float>> neighbours[V];
    for (int v = 0; v < V; v++) {
		hostStructures.vertexCommunity[v] = v;
		hostStructures.originalToCommunity[v] = v;
		hostStructures.vertices[v] = v;
    }
    // TODO: here is assumption that graph is undirected
    int aux = E;
    for (int i = 0; i < aux; i++) {
        std::cin >> v1 >> v2 >> w;
        v1--;
        v2--;
		hostStructures.communityWeight[v1] += w;
        neighbours[v1].emplace_back(v2, w);
        if (v1 != v2) {
            E++;
			hostStructures.communityWeight[v2] += w;
            neighbours[v2].emplace_back(v1, w);
        }
		hostStructures.M += w;
    }
    HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.edges, E * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.weights, E * sizeof(float), cudaHostAllocDefault));
	hostStructures.E = E;
    int index = 0;
    for (int v = 0; v < V; v++) {
		hostStructures.edgesIndex[v] = index;
        for (auto & it : neighbours[v]) {
			hostStructures.edges[index] = it.first;
			hostStructures.weights[index] = it.second;
            index++;
        }
    }
	hostStructures.edgesIndex[V] = E;
    return hostStructures;
}

void copyStructures(host_structures& hostStructures, device_structures& deviceStructures) {
	int V = hostStructures.V, E = hostStructures.E;
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.vertexCommunity, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.communityWeight, V * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.edges, E * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.weights, E * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.edgesIndex, (V + 1) * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.originalToCommunity, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.vertexEdgesSum, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.newVertexCommunity, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.vertices, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.V, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.E, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.originalV, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.totalGain, sizeof(float)));

	HANDLE_ERROR(cudaMemcpy(deviceStructures.vertexCommunity, hostStructures.vertexCommunity, V * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.newVertexCommunity, hostStructures.vertexCommunity, V * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.communityWeight, hostStructures.communityWeight, V * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.edges, hostStructures.edges, E * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.weights, hostStructures.weights, E * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.edgesIndex, hostStructures.edgesIndex, (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.originalToCommunity, hostStructures.originalToCommunity, V * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.vertices, hostStructures.vertices, V * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.V, &hostStructures.V, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.E, &hostStructures.E, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.originalV, &hostStructures.originalV, sizeof(int), cudaMemcpyHostToDevice));
}

void deleteStructures(host_structures& hostStructures, device_structures& deviceStructures) {
    HANDLE_ERROR(cudaFreeHost(hostStructures.vertexCommunity));
    HANDLE_ERROR(cudaFreeHost(hostStructures.communityWeight));
    HANDLE_ERROR(cudaFreeHost(hostStructures.edges));
    HANDLE_ERROR(cudaFreeHost(hostStructures.weights));
    HANDLE_ERROR(cudaFreeHost(hostStructures.edgesIndex));
    HANDLE_ERROR(cudaFreeHost(hostStructures.originalToCommunity));

	HANDLE_ERROR(cudaFree(deviceStructures.V));
	HANDLE_ERROR(cudaFree(deviceStructures.originalV));
    HANDLE_ERROR(cudaFree(deviceStructures.vertexCommunity));
	HANDLE_ERROR(cudaFree(deviceStructures.communityWeight));
	HANDLE_ERROR(cudaFree(deviceStructures.edges));
	HANDLE_ERROR(cudaFree(deviceStructures.weights));
	HANDLE_ERROR(cudaFree(deviceStructures.edgesIndex));
	HANDLE_ERROR(cudaFree(deviceStructures.originalToCommunity));
	HANDLE_ERROR(cudaFree(deviceStructures.vertexEdgesSum));
	HANDLE_ERROR(cudaFree(deviceStructures.newVertexCommunity));
	HANDLE_ERROR(cudaFree(deviceStructures.vertices));
	HANDLE_ERROR(cudaFree(deviceStructures.totalGain));
	HANDLE_ERROR(cudaFree(deviceStructures.E));
}
