#include "modularity_optimisation.cuh"
#include <utility>
#include <climits>
#include <stdio.h>

__device__ int get_hash(int val, int index, int m) {
	int h1 = val % HASHING;
	int h2 = 1 + (val % (HASHING - 1));
	return (h1 + index * h2) % m;
}

__device__ float atomic_max_float(float * addr, float value) {
	float old;
	old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
		  __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

	return old;
}

/**
 * Computes sum of weights of edges adjacent to vertices (results are stored in vertex_edges_sum).
 * @param V                number of vertices
 * @param vertex_edges_sum vertex -> sum of edges adjacent to vertex
 * @param edges_index      vertex -> begin index of edges (in weights array)
 * @param weights          array of weights of edges
 */
__global__ void compute_edges_sum(int *V, float *vertex_edges_sum, int *edges_index, float *weights) {
	int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (vertex < *V) {
		vertex_edges_sum[vertex] = 0;
		for (int i = edges_index[vertex]; i < edges_index[vertex + 1]; i++)
			vertex_edges_sum[vertex] += weights[i];
	}
}

/**
 * Computes sum of weights of edges adjacent to vertices (results are stored in vertex_edges_sum).
 * @param V                number of vertices
 * @param community_weight community -> weight (sum of edges adjacent to vertices of community)
 * @param vertex_community vertex -> community assignment
 * @param vertex_edges_sum vertex -> sum of edges adjacent to vertex
 */
__global__ void compute_community_weight(int *V, float *community_weight, int *vertex_community, float *vertex_edges_sum) {
	int community = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (community < *V) {
		community_weight[community] = 0;
		for (int vertex = 0; vertex < *V; vertex++) {
			if (vertex_community[vertex] == community) {
				community_weight[community] += vertex_edges_sum[vertex];
			}
		}
	}
}

const int VERTEX_DEGREE = 32;
const int VERTEX_DEGREE_PRIME = 127;

/**
 * Fills content of hash_community and hash_weights arrays that are later used in compute_gain function.
 * @param community         neighbour's community
 * @param weight			neighbour's weight
 * @param hash_weight		table of sum of weights between vertices and communities
 * @param hash_community	table informing which community's info is stored in given index
 * @param hash_tables_offset offset of the vertex in hash arrays (single hash array may contain multiple vertices - if
 *							VERTEX_DEGREE < THREADS_PER_BLOCK)
 */
__device__ void prepare_hash_arrays(int community, float weight, float *hash_weight, int *hash_community,
		int hash_tables_offset) {
	bool found_position = false;
	int it = 0;
	while (!found_position) {
		int cur_pos = hash_tables_offset + get_hash(community, it++, VERTEX_DEGREE_PRIME);
		if (hash_community[cur_pos] == community)
			atomicAdd(&hash_weight[cur_pos], weight);
			// TODO - uses inelegant solution with -1
		else if (hash_community[cur_pos] == -1) {
			if (atomicCAS(&hash_community[cur_pos], -1, community) == -1)
				atomicAdd(&hash_weight[cur_pos], weight);
			else if (hash_community[cur_pos] == community)
				atomicAdd(&hash_weight[cur_pos], weight);
		}
		found_position = hash_community[cur_pos] == community;
	}
}

/**
 * Computes gain that would be obtained if we would move vertex to community.
 * @param vertex            vertex
 * @param community 	    neighbour's community
 * @param current_comm 	    current community of vertex
 * @param community_weight  community -> weight (sum of edges adjacent to vertices of community)
 * @param vertex_edges_sum  vertex -> sum of edges adjacent to vertex
 * @param hash_community    table informing which community's info is stored in given index
 * @param hash_weight       table of sum of weights between vertices and communities
 * @param hash_tables_offset offset of the vertex in hash arrays (single hash array may contain multiple vertices - if
 *							VERTEX_DEGREE < THREADS_PER_BLOCK)
 * @return gain that would be obtained by moving vertex to community
 */
__device__ float compute_gain(int vertex, int community, int current_comm, float *community_weight,
		float *vertex_edges_sum, int *hash_community, float *hash_weight, int hash_tables_offset) {
	float comm_sum = community_weight[community];
	float current_comm_sum = community_weight[current_comm] - vertex_edges_sum[vertex];
	float vertex_community = 0, vertex_curr_comm = 0;
	for (int i = 0; i < VERTEX_DEGREE; i++) {
		int index = hash_tables_offset + i;
		if (hash_community[index] == community)
			vertex_community = hash_weight[index];
		else if (hash_community[index] == current_comm)
			vertex_curr_comm = hash_weight[index];
	}
	float gain = (vertex_community - vertex_curr_comm) / M +
				 vertex_edges_sum[vertex] * (current_comm_sum - comm_sum) / (2 * M * M);
	return gain;
}

/**
 * Finds new vertex -> community assignment (stored in new_vertex_community) that maximise gains for each vertex.
 * @param V                    number of vertices
 * @param vertex_community     vertex -> community assignment
 * @param edges_index          vertex -> begin index of edges (in edges and weights arrays)
 * @param edges                array of neighbours
 * @param weights              array of weights of edges
 * @param community_weight     community -> weight (sum of edges adjacent to vertices of community)
 * @param vertex_edges_sum     vertex -> sum of edges adjacent to vertex
 * @param total_gain           variable that stores sum of partial gains
 * @param new_vertex_community new vertex -> community assignment
 */
__global__ void compute_move(int *V, int *vertex_community, int *edges_index, int *edges, float *weights,
		float *community_weight, float *vertex_edges_sum, float *total_gain, int *new_vertex_community) {
	__shared__ float hash_weight[THREADS_PER_BLOCK / VERTEX_DEGREE * VERTEX_DEGREE_PRIME];
	__shared__ int hash_community[THREADS_PER_BLOCK / VERTEX_DEGREE * VERTEX_DEGREE_PRIME];
	__shared__ float best_gains[THREADS_PER_BLOCK / VERTEX_DEGREE];
	__shared__ int neighbour_chosen[THREADS_PER_BLOCK / VERTEX_DEGREE];
	for (int i = 0; i < THREADS_PER_BLOCK / VERTEX_DEGREE; i++) {
		best_gains[i] = 0;
		neighbour_chosen[i] = *V;
	}
	// TODO - maybe there is more elegant solution
	for (int i = 0; i < THREADS_PER_BLOCK / VERTEX_DEGREE * VERTEX_DEGREE_PRIME; i++) {
		hash_weight[i] = 0;
		hash_community[i] = -1;
	}

	int number = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	int hash_tables_offset = threadIdx.x / VERTEX_DEGREE * VERTEX_DEGREE_PRIME;
	int best_gains_index = hash_tables_offset / VERTEX_DEGREE_PRIME;
	int vertex = number / VERTEX_DEGREE;
	int neighbour_index = number % VERTEX_DEGREE;
	if (vertex < *V) {
		neighbour_index += edges_index[vertex];
		int current_comm = vertex_community[vertex];
		// putting data in hash table
		if (neighbour_index < edges_index[vertex + 1]) {
			int neighbour = edges[neighbour_index];
			int community = vertex_community[neighbour];
			float weight = weights[neighbour_index];
			// this lets us achieve ei -> C(i)\{i} instead of ei -> C(i)
			if (neighbour != vertex)
				prepare_hash_arrays(community, weight, hash_weight, hash_community, hash_tables_offset);
			// TODO - probably necessary only for big degree cases
			__syncthreads();
			// TODO - is this really needed?
			if (community < current_comm) {
				float gain = compute_gain(vertex, community, current_comm, community_weight, vertex_edges_sum,
						hash_community, hash_weight, hash_tables_offset);
				// TODO - this can be done more efficiently with `tree` method
				atomic_max_float(&best_gains[best_gains_index], gain);
				if (best_gains[best_gains_index] == gain) {
					atomicMin(&new_vertex_community[vertex], community);
					if (new_vertex_community[vertex] == community)
						atomicMin(&neighbour_chosen[best_gains_index], neighbour);
						if (neighbour_chosen[best_gains_index] == neighbour)
							atomicAdd(total_gain, gain);
				}
			}
		}
	}
}

/**
 * Updates vertex_community content based on new_vertex_community content
 * @param V                    number of vertices
 * @param new_vertex_community vertex -> community assignment (that gets updated)
 * @param vertex_community     new vertex -> community assignment
 */
__global__ void update_vertex_community(int *V, int *new_vertex_community, int *vertex_community) {
	int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (vertex < *V) {
		vertex_community[vertex] = new_vertex_community[vertex];
	}
}

void print_vertex_assignments(host_structures& structures) {
	for (int c = 0; c < structures.V; c++) {
		printf("%d", c + 1);
		for (int v = 0; v < structures.original_V; v++)
			if (c == structures.vertex_community[v])
				printf(" %d", v + 1);
		if (c < structures.V - 1)
			printf("\n");
	}
}

bool optimise_modularity(float min_gain, device_structures& dev_struct, host_structures& host_struct) {
	int blocks = (host_struct.V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int blocks_degrees = (host_struct.V * VERTEX_DEGREE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	float total_gain = min_gain;
	// TODO - this should be done once
	HANDLE_ERROR(cudaMemcpyToSymbol(M, &host_struct.M, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(&host_struct.V, dev_struct.V, sizeof(float), cudaMemcpyDeviceToHost));
	// computing new vertices weights
	compute_edges_sum<<<blocks, THREADS_PER_BLOCK>>>(dev_struct.V, dev_struct.vertex_edges_sum,
													 dev_struct.edges_index, dev_struct.weights);
	bool was_anything_changed = false;
	while (total_gain >= min_gain) {
		total_gain = 0;
		HANDLE_ERROR(cudaMemcpy(dev_struct.total_gain, &total_gain, sizeof(float), cudaMemcpyHostToDevice));
		// TODO partition at this stage
		// finding new communities
		compute_move<<<blocks_degrees, THREADS_PER_BLOCK>>>(dev_struct.V, dev_struct.vertex_community,
															dev_struct.edges_index,
															dev_struct.edges, dev_struct.weights,
															dev_struct.community_weight, dev_struct.vertex_edges_sum,
															dev_struct.total_gain, dev_struct.new_vertex_community);

		HANDLE_ERROR(cudaMemcpy(&total_gain, dev_struct.total_gain, sizeof(float), cudaMemcpyDeviceToHost));
		// updating vertex -> community assignment
		update_vertex_community<<<blocks, THREADS_PER_BLOCK>>>(dev_struct.V, dev_struct.new_vertex_community,
															   dev_struct.vertex_community);
		// updating community weight
		compute_community_weight<<<blocks, THREADS_PER_BLOCK>>>(dev_struct.V, dev_struct.community_weight,
																dev_struct.vertex_community, dev_struct.vertex_edges_sum);
		printf("%f\n", total_gain);
		was_anything_changed = was_anything_changed | (total_gain > 0);
	}
	HANDLE_ERROR(
			cudaMemcpy(host_struct.vertex_community, dev_struct.vertex_community, host_struct.V * sizeof(float),
					   cudaMemcpyDeviceToHost));
	print_vertex_assignments(host_struct);
	return was_anything_changed;
}