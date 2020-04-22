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
        .original_V = V,
        .V = V,
        .vertex_community = new int[V],
        .community_weight = new float[V],
        .edges_index = new int[V+1],
        .original_to_community = new int[V],
    };

    std::vector<std::pair<int, float>> neighbours[V];
    for (int v = 0; v < V; v++) {
        structures.vertex_community[v] = v;
        structures.original_to_community[v] = v;
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
//            structures.M += w;
        }
        structures.M += w;

    }
//    structures.M /= 2;
    structures.edges = new int[E];
    structures.weights = new float[E];

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

float get_community_weight(int community, data_structures& structures) {
    float total_weight = 0;
    for (int vertex = 0; vertex < structures.V; vertex++) {
        if (structures.vertex_community[vertex] != community)
            continue;
        for (int j = structures.edges_index[vertex]; j < structures.edges_index[vertex + 1]; j++)
            // TODO: is it okay for sure?  In UNDIRECTED graph edges within community are added 2 times
            total_weight += structures.weights[j];
    }
    return total_weight;
}

void print_vertex_assignments(data_structures& structures) {
    for (int c = 0; c < structures.V; c++) {
        std::cout << c + 1;
        for (int v = 0; v < structures.original_V; v++)
            if (c == structures.original_to_community[v])
                std::cout << " " << v + 1;
        if (c < structures.V - 1)
            std::cout << "\n";
    }
}

void delete_structures(data_structures& structures) {
    delete[] structures.vertex_community;
    delete[] structures.community_weight;
    delete[] structures.edges;
    delete[] structures.weights;
    delete[] structures.edges_index;
    delete[] structures.original_to_community;
}
