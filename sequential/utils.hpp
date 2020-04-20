#ifndef __UTILS__HPP__
#define __UTILS__HPP__

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
    // starting index of edges for given vertex (compressed neighbours list)
    int *edges_index;
};

/**
 * Reads input data and initialises values of global variables.
 */
data_structures read_input_data();

void delete_structures(data_structures& structures);

/**
 * Computes sum of edges (v, *), where v is a vertex in the given community.
 */
float get_community_weight(int community, data_structures& structures);

#endif /* __UTILS__HPP__ */