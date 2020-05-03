#include "utils.cuh"
#include <vector>
#include <iostream>

host_structures read_input_data() {
    int V, E;
    std::cin >> V >> V >> E;
    int v1, v2;
    float w;
    host_structures host_struct;
    host_struct.original_V = V;
    host_struct.V = V;
    HANDLE_ERROR(cudaHostAlloc((void**)&host_struct.vertex_community, V * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_struct.community_weight, V * sizeof(float), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_struct.edges_index, (V+1) * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_struct.original_to_community, V * sizeof(int), cudaHostAllocDefault));

    std::vector<std::pair<int, float>> neighbours[V];
    for (int v = 0; v < V; v++) {
        host_struct.vertex_community[v] = v;
        host_struct.original_to_community[v] = v;
    }
    // TODO: here is assumption that graph is undirected
    int aux = E;
    for (int i = 0; i < aux; i++) {
        std::cin >> v1 >> v2 >> w;
        v1--;
        v2--;
        host_struct.community_weight[v1] += w;
        neighbours[v1].emplace_back(v2, w);
        if (v1 != v2) {
            E++;
            host_struct.community_weight[v2] += w;
            neighbours[v2].emplace_back(v1, w);
//            host_struct.M += w;
        }
        host_struct.M += w;
    }
//    host_struct.M /= 2;
    HANDLE_ERROR(cudaHostAlloc((void**)&host_struct.edges, E * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_struct.weights, E * sizeof(float), cudaHostAllocDefault));
	host_struct.E = E;
    int index = 0;
    for (int v = 0; v < V; v++) {
        host_struct.edges_index[v] = index;
        for (auto & it : neighbours[v]) {
            host_struct.edges[index] = it.first;
            host_struct.weights[index] = it.second;
            index++;
        }
    }
    host_struct.edges_index[V] = E;
    return host_struct;
}

void copy_structures(host_structures& host_struct, device_structures& dev_struct) {
	int V = host_struct.V, E = host_struct.E;
	HANDLE_ERROR(cudaMalloc((void**)&dev_struct.vertex_community, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_struct.community_weight, V * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_struct.edges, E * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_struct.weights, E * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_struct.edges_index, (V+1) * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_struct.original_to_community, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_struct.vertex_edges_sum, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_struct.new_vertex_community, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_struct.V, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_struct.original_V, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_struct.total_gain, sizeof(float)));

	HANDLE_ERROR(cudaMemcpy(dev_struct.vertex_community, host_struct.vertex_community, V * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_struct.new_vertex_community, host_struct.vertex_community, V * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_struct.community_weight, host_struct.community_weight, V * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_struct.edges, host_struct.edges, E * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_struct.weights, host_struct.weights, E * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_struct.edges_index, host_struct.edges_index, (V+1) * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_struct.original_to_community, host_struct.original_to_community, V * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_struct.V, &host_struct.V, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_struct.original_V, &host_struct.original_V, sizeof(int), cudaMemcpyHostToDevice));
}

void delete_structures(host_structures& host_struct, device_structures& dev_struct) {
    HANDLE_ERROR(cudaFreeHost(host_struct.vertex_community));
    HANDLE_ERROR(cudaFreeHost(host_struct.community_weight));
    HANDLE_ERROR(cudaFreeHost(host_struct.edges));
    HANDLE_ERROR(cudaFreeHost(host_struct.weights));
    HANDLE_ERROR(cudaFreeHost(host_struct.edges_index));
    HANDLE_ERROR(cudaFreeHost(host_struct.original_to_community));

    HANDLE_ERROR(cudaFree(dev_struct.vertex_community));
	HANDLE_ERROR(cudaFree(dev_struct.community_weight));
	HANDLE_ERROR(cudaFree(dev_struct.edges));
	HANDLE_ERROR(cudaFree(dev_struct.weights));
	HANDLE_ERROR(cudaFree(dev_struct.edges_index));
	HANDLE_ERROR(cudaFree(dev_struct.original_to_community));
	HANDLE_ERROR(cudaFree(dev_struct.new_vertex_community));
	HANDLE_ERROR(cudaFree(dev_struct.vertex_edges_sum));
	HANDLE_ERROR(cudaFree(dev_struct.V));
	HANDLE_ERROR(cudaFree(dev_struct.original_V));
	HANDLE_ERROR(cudaFree(dev_struct.total_gain));
}
