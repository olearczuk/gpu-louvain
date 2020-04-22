#include <iostream>
#include "modularity_optimisation.hpp"
#include "community_aggregation.hpp"

int main() {
    float min_gain = 0.1;
    data_structures structures = read_input_data();
    float total_gain = 0;
    for (;;) {
        if (!optimise_modularity(min_gain, structures))
            break;
        aggregate_communities(structures);
    }
    std::cout << compute_modularity(structures) << "\n";
    print_vertex_assignments(structures);
    delete_structures(structures);
}