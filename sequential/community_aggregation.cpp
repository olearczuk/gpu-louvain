#include "community_aggregation.hpp"
#include <iostream>
#include <utility>

void prefix_sum(int *res, int *arr, int size) {
    res[0] = 0;
    for (int i = 1; i < size + 1; i++)
        res[i] = res[i - 1] + arr[i - 1];
}

// TODO remove
void print_community_assignment(int *community_size, int *new_id, int size) {
    for (int c = 0; c < size; c++)
        if (community_size[c] > 0)
            std::cout << c + 1<< " -> " << new_id[c] + 1 << "\n";
}

void compute_communities_weights(float *communities_weights, int *ordered_vertices,
        data_structures& structures) {
    for (int i = 0; i < structures.V; i++)
        for (int j = 0; j < structures.V; j++)
            communities_weights[i * structures.V + j] = 0;
    for (int i = 0; i < structures.V; i++) {
        int vertex = ordered_vertices[i];
        int cur_comm = structures.vertex_community[vertex];
        for (int j = structures.edges_index[vertex]; j < structures.edges_index[vertex + 1]; j++) {
            int neighbour = structures.edges[j];
            float weight = structures.weights[j];
            // TODO: assumption that there are only positive edges (caused be creating "holes" in aggregation)
            if (weight == 0)
                break;
            int neighbour_comm = structures.vertex_community[neighbour];
            if (cur_comm == neighbour_comm && vertex != neighbour)
                weight /= 2;
            communities_weights[cur_comm * structures.V + neighbour_comm] += weight;
        }
    }
}

void aggregate_communities(data_structures& structures) {
    int community_size[structures.V];
    int community_degree[structures.V];
    for (int i = 0; i < structures.V; i++) {
        community_size[i] = 0;
        community_degree[i] = 0;
    }
    int E = 0;
    for (int i = 0; i < structures.V; i++) {
        int vertex = structures.vertices[i];
        int community = structures.vertex_community[vertex];
        community_size[community]++;
        int vertex_degree = structures.edges_index[vertex + 1] - structures.edges_index[vertex];
        community_degree[community] += vertex_degree;
        E += vertex_degree;
    }

    int new_id_[structures.V];
    int new_id[structures.V + 1];
    for (int c = 0; c < structures.V; c++) {
        if (community_size[c] > 0)
            new_id_[c] = 1;
        else
            new_id_[c] = 0;
    }
    prefix_sum(new_id, new_id_, structures.V);
    // TODO remove
    print_community_assignment(community_size, new_id, structures.V);

    // this will be used in setting new structures.edges_index;
    int edge_pos[structures.V + 1];
    prefix_sum(edge_pos, community_degree, structures.V);

    // this step makes vertices ordered based on their community
    int vertex_start[structures.V + 1];
    prefix_sum(vertex_start, community_size, structures.V);
    vertex_start[structures.V] = structures.V;
    int ordered_vertices[structures.V];
    for (int i = 0; i < structures.V; i++) {
        int vertex = structures.vertices[i];
        int community = structures.vertex_community[vertex];
        ordered_vertices[vertex_start[community]] = vertex;
        vertex_start[community]++;
    }
    // computing distances between communities
    // TODO this should be executed with 'tricky hashing'

//    std::cout << "-----------------------\n";
    float communities_distances[structures.V * structures.V];
    compute_communities_weights(communities_distances, ordered_vertices, structures);

    int edges[E];
    float new_weights[E];
    for (int i = 0; i < E; i++)
        new_weights[i] = 0;
    // TODO: communities should be partitioned using community_degree array
    int communities_number = new_id[structures.V];
    structures.edges_index[communities_number] = E;
    float new_community_weight[communities_number];
    for (int c = 0; c < structures.V; c++) {
        if (community_size[c] == 0)
            continue;
        int id = new_id[c];
        int current_index = edge_pos[c];
        structures.edges_index[id] = current_index;
        for (int c1 = 0; c1 < structures.V; c1++) {
            if (community_size[c1] == 0 || communities_distances[c * structures.V + c1] == 0)
                continue;
            int id1 = new_id[c1];
            edges[current_index] = id1;
            new_weights[current_index] = communities_distances[c * structures.V + c1];
            new_community_weight[id] += new_weights[current_index];
            current_index++;
        }
    }

//    std::cout << "-----------------------\n";

    structures.V = communities_number;
    for (int i = 0; i < E; i++) {
        structures.edges[i] = edges[i];
        structures.weights[i] = new_weights[i];
    }
    for (int c = 0; c < communities_number; c++) {
        structures.vertices[c] = c;
        structures.vertex_community[c] = c;
        structures.community_weight[c] = get_community_weight(c, structures);
    }
    for (int i = 0; i < structures.V; i++) {
        int vertex = structures.vertices[i];
//        std::cout << structures.edges_index[vertex] << " " << structures.edges_index[vertex+1] << "\n";
        std::cout << vertex + 1 << ": [";
        for (int i = structures.edges_index[vertex]; i < structures.edges_index[vertex + 1]; i++) {
            float weight = structures.weights[i];
            // TODO: assumption that there are only positive edges (caused be creating "holes" in aggregation)

            if (weight == 0)
                break;
            std::cout << " (" << structures.edges[i] + 1 << ", " << structures.weights[i] << ")";
        }
        std::cout << " ]\n";
    }
}

