#ifndef __MODULARITY_OPTIMISATION__CUH__
#define __MODULARITY_OPTIMISATION__CUH__
#include "utils.cuh"

__constant__ float M;

/**
 * Function responsible for executing 1 phase (modularity optimisation)
 * @param minGain          minimum gain for going to next iteration of this phase
 * @param deviceStructures structures kept in device memory
 * @param hostStructures   structures kept in host memory
 * @return information whether any changes were applied
 */
bool optimiseModularity(float minGain, device_structures& deviceStructures, host_structures& hostStructures);


#endif /* __MODULARITY_OPTIMISATION__CUH__ */