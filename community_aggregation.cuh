#ifndef LOUVIAN_COMMUNITY_AGGREGATION_CUH
#define LOUVIAN_COMMUNITY_AGGREGATION_CUH

#include "utils.cuh"

/**
 * Transforms every community into single vertex.
 * @param deviceStructures structures kept in device memory
 * @param hostStructures   structures kept in host memory
 */
void aggregateCommunities(device_structures &deviceStructures, host_structures &hostStructures);

#endif //LOUVIAN_COMMUNITY_AGGREGATION_CUH
