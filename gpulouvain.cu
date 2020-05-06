#include "utils.cuh"
#include "modularity_optimisation.cuh"
#include "community_aggregation.cuh"

int main() {
    auto hostStructures = readInputData();
    device_structures deviceStructures;
	copyStructures(hostStructures, deviceStructures);
	optimiseModularity(0.1, deviceStructures, hostStructures);
//	aggregateCommunities(deviceStructures, hostStructures);

	deleteStructures(hostStructures, deviceStructures);
}
