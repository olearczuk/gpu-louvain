#ifndef __UTILS__CUH__
#define __UTILS__CUH__

#include <stdio.h>
#include <climits>

const int THREADS_PER_BLOCK = 128;
const int HASHING = 1000000007;

struct host_structures {
    float M = 0;
    // original number of vertices
    int original_V;
    // current number of vertices
    int V, E;
    // vertex -> community
    int *vertex_community;
    // sum of edges adjacent to community
    float *community_weight;
    // array of neighbours
    int *edges;
    // array of weights of edges
    float *weights;
    // starting index of edges for given vertex (compressed neighbours list)
    int *edges_index;
    // represents final assignment of vertex to community
    int *original_to_community;
};

struct device_structures {
	int *V, *original_V;
	// vertex -> community
	int *vertex_community;
	// sum of edges adjacent to community
	float *community_weight;
	// array of neighbours
	int *edges;
	// array of weights of edges
	float *weights;
	// starting index of edges for given vertex (compressed neighbours list)
	int *edges_index;
	// represents final assignment of vertex to community
	int *original_to_community;
	// sums of edges adjacent to vertices
	float *vertex_edges_sum;
	// auxiliary array used for remembering new community
	int *new_vertex_community;
	// total gain stored from every iteration of phase 1
	float *total_gain;
};

/**
 * Reads input data and initialises values of global variables.
 */
host_structures read_input_data();

/**
 * Deletes both host, and device structures.
 * @param host_struct structures stored in host memory
 * @param dev_struct  structures stored in device memory
 */
void  delete_structures(host_structures& host_struct, device_structures& dev_struct);

void copy_structures(host_structures& host_struct, device_structures& dev_struct);

/**
 * Prints assignment of original vertices to final communities.
 * @param host_struct structures stored in host memory
 * @param dev_struct  structures stored in device memory
 */
void print_vertex_assignments(host_structures& host_struct, device_structures& dev_struct);

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString(err),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err) (HandleError( err, __FILE__, __LINE__ ))

#endif /* __UTILS__CUH__ */