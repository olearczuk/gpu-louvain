#ifndef __MODULARITY_OPTIMISATION__CUH__
#define __MODULARITY_OPTIMISATION__CUH__
#include "utils.cuh"

__constant__ float M;

/**
 * Function responsible for executing 1 phase (modularity optimisation)
 * @param min_gain    minimum gain for going to next iteration of this phase
 * @param dev_struct  structures kept in device memory
 * @param host_struct structures kept in host memory
 * @return information whether any changes were applied
 */
bool optimise_modularity(float min_gain, device_structures& dev_struct, host_structures& host_struct);


#endif /* __MODULARITY_OPTIMISATION__CUH__ */