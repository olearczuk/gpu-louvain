#include "utils.hpp"
#include <vector>
#include <iostream>

data_structures read_input_data() {
    int V, E;
    std::cin >> V >> V >> E;
    int v1, v2;
    float w;
    int cur_weight;
    data_structures structures = {
        .V = V,
        .vertex_community = new int[V],
        .community_weight = new float[V],
        .vertices = new int[V],
    };

    std::vector<std::pair<int, float>> neighbours[V];
    for (int v = 0; v < V; v++) {
        structures.vertex_community[v] = v;
        structures.vertices[v] = v;
    }
    // TODO: here is assumption that graph is undirected
    int aux = E;
    for (int i = 0; i < aux; i++) {
        std::cin >> v1 >> v2 >> w;
        v1--;
        v2--;
        structures.community_weight[v1] += w;
        neighbours[v1].push_back(std::pair<int, float>(v2, w));
        if (v1 != v2) {
            E++;
            structures.community_weight[v2] += w;
            neighbours[v2].push_back(std::pair<int, float>(v1, w));
        }
        structures.M += w;

    }
    structures.edges = new int[E];
    structures.weights = new float[E];
    structures.edges_index = new int[V+1];

    int index = 0;
    for (int v = 0; v < V; v++) {
        structures.edges_index[v] = index;
        for (auto it = neighbours[v].begin(); it != neighbours[v].end(); it++) {
            structures.edges[index] = it->first;
            structures.weights[index] = it->second;
            index++;
        }
    }
    structures.edges_index[V] = E;
    return structures;
}

void delete_structures(data_structures& structures) {
    delete[] structures.vertex_community;
    delete[] structures.community_weight;
    delete[] structures.vertices;
    delete[] structures.edges;
    delete[] structures.weights;
    delete[] structures.edges_index;
}