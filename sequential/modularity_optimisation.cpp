#include <iostream>
#include "modularity_optimisation.hpp"

/**
 * Computes sum of edges (vertex, *).
 */
float get_edges_sum(int vertex, data_structures& structures) {
    float sum = 0;
    for (int i = structures.edges_index[vertex]; i < structures.edges_index[vertex + 1]; i++) {
        sum += structures.weights[i];
    }
    return sum;
}

/**
 * Computes sum of edges (vertex, vertex in community).
 * In case of vertex belonging to the community, edges (vertex, vertex) are discarded.
 */
float get_edges_to_community(int vertex, int community, data_structures& structures) {
    int current_community = structures.vertex_community[vertex];
    float sum = 0;
    for (int i = structures.edges_index[vertex]; i < structures.edges_index[vertex + 1]; i++) {
        int neighbour = structures.edges[i];
        int neighbour_community = structures.vertex_community[neighbour];
        if (neighbour_community == community && (community != current_community || neighbour != vertex)) {
            float weight = structures.weights[i];
            sum += weight;
        }
    }
    return sum;
}

/**
 * Computes sum of edges (v, *), where v is a vertex in the given community.
 */
float get_community_weight(int community, data_structures& structures) {
    float total_weight = 0;
    for (int i = 0; i < structures.V; i++) {
        int vertex = structures.vertices[i];
        if (structures.vertex_community[vertex] != community)
            continue;
        for (int j = structures.edges_index[vertex]; j < structures.edges_index[vertex + 1]; j++)
            // TODO: is it okay for sure?  In UNDIRECTED graph edges within community are added 2 times
            total_weight += structures.weights[j];
    }
    return total_weight;
}

/**
 * Computes new community that offers the best gain.
 * In case of few communities offering equal gains, one with the lowest index is chosen.
 * In situation when none of communities offers positive gain, current community is returned.
 * @return (selected community, gain)
 */
std::pair<int, float> find_new_community(int vertex, float vertex_edges_sum, data_structures& structures) {
    // TODO this function should be executed with 'tricky hashing'
    float M = structures.M;
    int current_community = structures.vertex_community[vertex];
    // edges k -> <anything> are discarded
    float current_comm_sum = structures.community_weight[current_community] - vertex_edges_sum;
    float vertex_current_comm = get_edges_to_community(vertex, current_community, structures);
    float act_best_gain = 0;
    int act_best_community = current_community;
    for (int i = structures.edges_index[vertex]; i < structures.edges_index[vertex + 1]; i++) {
        int community = structures.vertex_community[structures.edges[i]];
        if (community >= current_community)
            continue;
        float comm_sum = structures.community_weight[community];
        float vertex_comm = get_edges_to_community(vertex, community, structures);
        float gain = (vertex_comm - vertex_current_comm) / M +
                     vertex_edges_sum * (current_comm_sum - comm_sum) / (2 * M * M);
        if (gain > act_best_gain || (gain == act_best_gain && community < act_best_community)) {
            act_best_gain = gain;
            act_best_community = community;
        }
    }
    return std::pair<int, float>(act_best_community, act_best_gain);
}

/**
 * Iterates over vertices and finds new communities using auxiliary `find_new_community` function.
 * After retrieving new community, it changes relevant fields in the given `structures`.
 * @return total gain
 */
float find_new_communities(data_structures& structures) {
    float total_gain = 0;
    int new_community;
    int new_vertex_community[structures.V];
    float vertex_edges_sum[structures.V];
    for (int i = 0; i < structures.V; i++) {
        int vertex = structures.vertices[i];
        vertex_edges_sum[vertex] = get_edges_sum(vertex, structures);
    }

    // TODO: vertices should be partitioned
    for (int i = 0; i < structures.V; i++) {
        int vertex = structures.vertices[i];
        std::pair<int, float> community_info = find_new_community(vertex, vertex_edges_sum[vertex], structures);
        new_community = community_info.first;
        total_gain += community_info.second;
        new_vertex_community[vertex] = new_community;
    }

    for (int v = 0; v < structures.V; v++) {
        structures.vertex_community[v] = new_vertex_community[v];
    }

    for (int c = 0; c < structures.V; c++) {
        structures.community_weight[c] = get_community_weight(c, structures);
    }
    return total_gain;
}

/**
 * Finds new communities as long as gain is equal or higher than minimal gain.
 * Returns information, whether any changes in vertex -> community assignment were done.
 */
bool optimise_modularity(float min_gain, data_structures& structures) {
    float gain = min_gain;
    bool was_anything_changed = false;
    while (gain >= min_gain) {
        gain = find_new_communities(structures);
        was_anything_changed = was_anything_changed || gain > 0;
        std::cout << gain << "\n";
    }
    return was_anything_changed;
}