#include "modularity_optimisation.cuh"
#include <climits>
#include <cstdio>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>

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
 * @param vertices         vertices
 * @param community_weight community -> weight (sum of edges adjacent to vertices of community)
 * @param vertex_community vertex -> community assignment
 * @param vertex_edges_sum vertex -> sum of edges adjacent to vertex
 */
__global__ void compute_community_weight(int V, int *vertices, float *community_weight,int *vertex_community,
		float *vertex_edges_sum) {
	int community = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (community < V) {
		community_weight[community] = 0;
		for (int index = 0; index < V; index++) {
			int vertex = vertices[index];
			if (vertex_community[vertex] == community) {
				community_weight[community] += vertex_edges_sum[vertex];
			}
		}
	}
}

/**
 * Fills content of hash_community and hash_weights arrays that are later used in compute_gain function.
 * @param community         neighbour's community
 * @param prime             prime number used for hashing
 * @param weight			neighbour's weight
 * @param hash_weight		table of sum of weights between vertices and communities
 * @param hash_community	table informing which community's info is stored in given index
 * @param hash_tables_offset offset of the vertex in hash arrays (single hash array may contain multiple vertices - if
 *							VERTEX_DEGREE < THREADS_PER_BLOCK)
 */
__device__ void prepare_hash_arrays(int community, int prime, float weight, float *hash_weight, int *hash_community,
		int hash_tables_offset) {
	bool found_position = false;
	int it = 0;
	while (!found_position) {
		int cur_pos = hash_tables_offset + get_hash(community, it++, prime);
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
 * @prime                   prime number used for hashing (and size of vertex's area in hash arrays)
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
__device__ float compute_gain(int vertex, int prime, int community, int current_comm, float *community_weight,
		float *vertex_edges_sum, int *hash_community, float *hash_weight, int hash_tables_offset) {
	float comm_sum = community_weight[community];
	float current_comm_sum = community_weight[current_comm] - vertex_edges_sum[vertex];
	float vertex_community = 0, vertex_curr_comm = 0;
	for (int i = 0; i < prime; i++) {
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
 * @param vertices			   vertices
 * @param prime                prime number used for hashing
 * @param vertex_community     vertex -> community assignment
 * @param edges_index          vertex -> begin index of edges (in edges and weights arrays)
 * @param edges                array of neighbours
 * @param weights              array of weights of edges
 * @param community_weight     community -> weight (sum of edges adjacent to vertices of community)
 * @param vertex_edges_sum     vertex -> sum of edges adjacent to vertex
 * @param total_gain           variable that stores sum of partial gains
 * @param new_vertex_community new vertex -> community assignment
 */
__global__ void compute_move(int V, int *vertices, int prime, int *vertex_community, int *edges_index, int *edges,
		float *weights, float *community_weight, float *vertex_edges_sum, float *total_gain, int *new_vertex_community) {
	int vertices_per_block = blockDim.y;
	int neighbours_radius = blockDim.x;
	int hash_tables_offset = threadIdx.y * prime;
	int best_gains_index = threadIdx.y;
	int vertex_index = blockIdx.x * vertices_per_block + threadIdx.y;
	extern __shared__ int s[];
	int *hash_community = s;
	auto *hash_weight = (float*)&hash_community[vertices_per_block * prime];
	auto *best_gains = (float*)&hash_community[2 * vertices_per_block * prime];
	int *neighbour_chosen = &hash_community[2 * vertices_per_block * prime + vertices_per_block];
	best_gains[best_gains_index] = 0;
	neighbour_chosen[best_gains_index] = V;
	// TODO - maybe there is more elegant solution
	for (int i = 0; i < prime; i++) {
		hash_weight[hash_tables_offset + i] = 0;
		hash_community[hash_tables_offset + i] = -1;
	}

	if (vertex_index < V) {
		int vertex = vertices[vertex_index];
		int current_comm = vertex_community[vertex];
		int best_community = current_comm;
		int best_neighbour = vertex;
		float best_gain = 0;
		// putting data in hash table
		int neighbour_index = threadIdx.x + edges_index[vertex];

		while (neighbour_index < edges_index[vertex + 1]) {
			int neighbour = edges[neighbour_index];
			int community = vertex_community[neighbour];
			float weight = weights[neighbour_index];
			// this lets us achieve ei -> C(i)\{i} instead of ei -> C(i)
			if (neighbour != vertex)
				prepare_hash_arrays(community, prime, weight, hash_weight, hash_community, hash_tables_offset);
			neighbour_index += neighbours_radius;
		}
		// TODO - this should be only for `big case` - 6th bucket (similarly for 7th bucket)
		__syncthreads();
		// choosing community
		neighbour_index = threadIdx.x + edges_index[vertex];
		while (neighbour_index < edges_index[vertex + 1]) {
			int neighbour = edges[neighbour_index];
			int community = vertex_community[neighbour];
			if (community < current_comm) {
				float gain = compute_gain(vertex, prime, community, current_comm, community_weight, vertex_edges_sum,
						hash_community, hash_weight, hash_tables_offset);
				if (gain > best_gain || (gain == best_gain && community < best_community)) {
					best_gain = gain;
					best_community = community;
					best_neighbour = neighbour;
				}
			}
			neighbour_index += neighbours_radius;
		}
		// TODO - this can be done more efficiently with `tree` method
		if (best_gain > 0) {
			atomic_max_float(&best_gains[best_gains_index], best_gain);
			// TODO - this should be only for `big case` - 6th bucket (similarly for 7th bucket)
			__syncthreads();
			if (best_gains[best_gains_index] == best_gain) {
				atomicMin(&new_vertex_community[vertex], best_community);
				// TODO - this should be only for `big case` - 6th bucket (similarly for 7th bucket)
				__syncthreads();
				if (new_vertex_community[vertex] == best_community)
					atomicMin(&neighbour_chosen[best_gains_index], best_neighbour);
				// TODO - this should be only for `big case` - 6th bucket (similarly for 7th bucket)
				__syncthreads();
				if (neighbour_chosen[best_gains_index] == best_neighbour)
					atomicAdd(total_gain, best_gain);
			}
		}
	}
}

/**
 * Updates vertex_community content based on new_vertex_community content
 * @param V                    number of vertices
 * @param vertices             vertices
 * @param new_vertex_community vertex -> community assignment (that gets updated)
 * @param vertex_community     new vertex -> community assignment
 */
__global__ void update_vertex_community(int V, int *vertices, int *new_vertex_community, int *vertex_community) {
	int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (index < V) {
		int vertex = vertices[index];
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

struct is_in_bucket
{
	is_in_bucket(int lower_bound_, int upper_bound_, int *edges_index_) {
		lower_bound = lower_bound_;
		upper_bound = upper_bound_;
		edges_index = edges_index_;
	}

	int lower_bound{}, upper_bound{};
	int *edges_index{};
	__host__ __device__
	bool operator()(const int &v) const
	{
		int edges_number = edges_index[v + 1] - edges_index[v];
		return edges_number > lower_bound && edges_number <= upper_bound;
	}
};

bool optimise_modularity(float min_gain, device_structures& dev_struct, host_structures& host_struct) {
	int blocks = (host_struct.V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	float total_gain = min_gain;
	// TODO - this should be done once
	HANDLE_ERROR(cudaMemcpyToSymbol(M, &host_struct.M, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(&host_struct.V, dev_struct.V, sizeof(float), cudaMemcpyDeviceToHost));
	// computing new vertices weights
	compute_edges_sum<<<blocks, THREADS_PER_BLOCK>>>(dev_struct.V, dev_struct.vertex_edges_sum,
													 dev_struct.edges_index, dev_struct.weights);

	HANDLE_ERROR(cudaMemcpy(host_struct.edges_index, dev_struct.edges_index, (host_struct.V+1) * (sizeof(int)), cudaMemcpyDeviceToHost));
	bool was_anything_changed = false;
	int buckets_size = 8;
	int buckets[] = {0, 4, 8, 16, 32, 84, 319, INT_MAX};
	int primes[] = {7, 13, 29, 53, 127, 479};
	// x - radius in which threads look for neighbours, y - vertices per block
	dim3 block_dims[] {
			{4, 32},
			{8, 16},
			{16, 8},
			{32, 4},
			{32, 4},
			{128, 1},
			{128, 1},
	};
	int A[host_struct.V];
	for (int i = 0; i < host_struct.V; i++)
		A[i] = i;
	int *A_device;
	HANDLE_ERROR(cudaMalloc((void**)&A_device, host_struct.V * sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(A_device, A, host_struct.V * sizeof(int), cudaMemcpyHostToDevice));

	while (total_gain >= min_gain) {
		total_gain = 0;
		HANDLE_ERROR(cudaMemcpy(dev_struct.total_gain, &total_gain, sizeof(float), cudaMemcpyHostToDevice));
		// TODO partition at this stage
		// finding new communities

		for(int bucket_num= 0; bucket_num < buckets_size - 2; bucket_num++) {
			dim3 block_dim = block_dims[bucket_num];
			int vertex_degree = buckets[bucket_num + 1];
			int prime = primes[bucket_num];
			auto predicate = is_in_bucket(buckets[bucket_num], buckets[bucket_num+1], host_struct.edges_index);
			int *A_device_end = thrust::partition(thrust::device, A_device, A_device + host_struct.V, predicate);
			int V = thrust::distance(A_device, A_device_end);
			if (V > 0) {
				int shmem_size = block_dim.y * prime * (sizeof(float) + sizeof(int)) +
								 block_dim.y * (sizeof(float) + sizeof(int));
				int blocks_degrees = (V * vertex_degree + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
				compute_move<<<blocks_degrees, block_dim, shmem_size>>>(V, A_device, prime,
						dev_struct.vertex_community, dev_struct.edges_index, dev_struct.edges,
						dev_struct.weights, dev_struct.community_weight, dev_struct.vertex_edges_sum,
						dev_struct.total_gain, dev_struct.new_vertex_community);
			}
			// updating vertex -> community assignment
			update_vertex_community<<<blocks, THREADS_PER_BLOCK>>>(V, A_device, dev_struct.new_vertex_community,
																   dev_struct.vertex_community);
			// updating community weight
			compute_community_weight<<<blocks, THREADS_PER_BLOCK>>>(host_struct.V, A_device, dev_struct.community_weight,
																	dev_struct.vertex_community, dev_struct.vertex_edges_sum);
		}

//		unsigned int vertex_degree = 8;
//		int prime = vertex_degree;
//		unsigned int vertices_per_block = THREADS_PER_BLOCK/vertex_degree;
//		dim3 block_dim = {vertex_degree, vertices_per_block};
////		int prime = 127;
////		dim3 block_dim = {4, 2};
//		int shmem_size = vertices_per_block * prime * (sizeof(float) + sizeof(int)) +
//						 vertices_per_block * (sizeof(float) + sizeof(int));
//		int blocks_degrees = (host_struct.V * vertex_degree + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//		compute_move<<<blocks_degrees, block_dim, shmem_size>>>(host_struct.V, A_device, prime,
//				dev_struct.vertex_community, dev_struct.edges_index, dev_struct.edges, dev_struct.weights,
//				dev_struct.community_weight, dev_struct.vertex_edges_sum, dev_struct.total_gain,
//				dev_struct.new_vertex_community);

		HANDLE_ERROR(cudaMemcpy(&total_gain, dev_struct.total_gain, sizeof(float), cudaMemcpyDeviceToHost));
		printf("%f\n", total_gain);
		was_anything_changed = was_anything_changed | (total_gain > 0);
	}
	HANDLE_ERROR(
			cudaMemcpy(host_struct.vertex_community, dev_struct.vertex_community, host_struct.V * sizeof(float),
					   cudaMemcpyDeviceToHost));
	print_vertex_assignments(host_struct);
	return was_anything_changed;
}