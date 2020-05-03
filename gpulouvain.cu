#include "utils.cuh"
#include "modularity_optimisation.cuh"
#include <iostream>

int main() {
    auto host_struct = read_input_data();
    device_structures dev_struct;
    copy_structures(host_struct, dev_struct);
	optimise_modularity(0.1, dev_struct, host_struct);

    delete_structures(host_struct, dev_struct);
}
