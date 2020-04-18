#include <iostream>
#include <map>
#include <vector>

using namespace std;

struct data_structures {
    float M = 0;
    int V;
    // vertex -> community
    int *vertex_community;
    // sum of edges adjacent to community
    float *community_weight;
    int *vertices;
    // array of neighbours
    int *edges;
    // array of weights of edges
    float *weights;
    // sum of edges adjacent to vertex
    float *vertex_edges_sum;
    // starting index of edges for given vertex (compressed neighbours list)
    int *edges_index;
};

void delete_structures(data_structures& structures) {
    delete(structures.vertex_community);
    delete(structures.community_weight);
    delete(structures.vertices);
    delete(structures.edges);
    delete(structures.weights);
    delete(structures.edges_index);
    delete(structures.vertex_edges_sum);
}

/**
 * Computes sum of edges vertex -> *.
 */
float get_edges_sum(int vertex, data_structures& structures) {
    float sum = 0;
    for (int i = structures.edges_index[vertex]; i < structures.edges_index[vertex + 1]; i++) {
        sum += structures.weights[i];
    }
    return sum;
}

/**
 * Computes sum of edges vertex -> community.
 * In case of vertex belonging to the community, edges vertex -> vertex are discarded.
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
 * Computes sum of edges v -> *, where v is a vertex in the given community.
 */
float get_community_weight(int community, data_structures& structures) {
    float total_weight = 0;
    for (int i = 0; i < structures.V; i++) {
        int vertex = structures.vertices[i];
        if (structures.vertex_community[vertex] != community)
            continue;
        for (int j = structures.edges_index[vertex]; j < structures.edges_index[vertex + 1]; j++) {
            int neighbour = structures.edges[j];
            int neighbour_community = structures.vertex_community[neighbour];
            total_weight += structures.weights[j];
        }
    }
    return total_weight;
}

/**
 * Computes new community that offers the best gain.
 * In case of few communities offering the sam gain, the one with lowest index is chosen.
 * In case none of communities offered positive gain, current community is returned.
 * @return (selected community, gain)
 */
pair<int, float> find_new_community(int vertex, data_structures& structures) {
    float M = structures.M;
    float vertex_edges_sum = structures.vertex_edges_sum[vertex];
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
    return pair<int, float>(act_best_community, act_best_gain);
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

    for (int i = 0; i < structures.V; i++) {
        int vertex = structures.vertices[i];
        structures.vertex_edges_sum[vertex] = get_edges_sum(vertex, structures);
    }

    for (int i = 0; i < structures.V; i++) {
        int vertex = structures.vertices[i];
        pair<int, float> community_info = find_new_community(vertex, structures);
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
 * Reads input data and initialises values of global variables.
 */
data_structures read_input_data() {
    int V, E;
    cin >> V >> V >> E;
    int v1, v2;
    float w;
    int cur_weight;
    data_structures structures;
    structures.V = V;
    structures.community_weight = new float[V];
    structures.vertex_community = new int[V];
    structures.vertices = new int[V];
    structures.vertex_edges_sum = new float[V];
    vector<pair<int, float>> neighbours[V];
    for (int v = 0; v < V; v++) {
        structures.vertex_community[v] = v;
        structures.vertices[v] = v;
    }
    // TODO: here is assumption that graph is undirected
    int aux = E;
    for (int i = 0; i < aux; i++) {
        cin >> v1 >> v2 >> w;
        v1--;
        v2--;
        structures.community_weight[v1] += w;
        neighbours[v1].push_back(pair<int, float>(v2, w));
        if (v1 != v2) {
            E++;
            structures.community_weight[v2] += w;
            neighbours[v2].push_back(pair<int, float>(v1, w));
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

/**
 * Finds new communities as long as gain is equal or higher than minimal gain.
 * Returns information, whether any in vertex -> community were done.
 */
bool phase_one(float min_gain, data_structures& structures) {
    float gain = min_gain;
    bool was_anything_changed = false;
    while (gain >= min_gain) {
        gain = find_new_communities(structures);
        was_anything_changed = was_anything_changed || gain > 0;
        cout << gain << "\n";
    }
    return was_anything_changed;
}

int main() {
    float min_gain = 0.001;
    data_structures structures = read_input_data();
    phase_one(min_gain, structures);

    for (int i = 0; i < structures.V; i++) {
        int vertex = structures.vertices[i];
        cout << vertex + 1 << " " << structures.vertex_community[vertex] + 1 << "\n";
    }

    delete_structures(structures);
}