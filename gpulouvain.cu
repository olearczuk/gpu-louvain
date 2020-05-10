#include "utils.cuh"
#include "modularity_optimisation.cuh"
#include "community_aggregation.cuh"



int main() {
    auto hostStructures = readInputData();
    device_structures deviceStructures;
	copyStructures(hostStructures, deviceStructures);
	for (;;) {
		if (!optimiseModularity(0.001, deviceStructures, hostStructures))
			break;
		break;
		aggregateCommunities(deviceStructures, hostStructures);
	}
//	printOriginalToCommunity(deviceStructures, hostStructures);
}
