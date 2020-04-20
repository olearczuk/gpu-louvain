#include <iostream>
#include "modularity_optimisation.hpp"

int main() {
    float min_gain = 0.001;
    data_structures structures = read_input_data();
    optimise_modularity(min_gain, structures);

    for (int i = 0; i < structures.V; i++) {
        int vertex = structures.vertices[i];
        std::cout << vertex + 1 << " " << structures.vertex_community[vertex] + 1 << "\n";
    }

    delete_structures(structures);
}