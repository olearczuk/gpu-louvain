#include "utils.cuh"
#include "modularity_optimisation.cuh"
#include "community_aggregation.cuh"



int main() {
    auto hostStructures = readInputData();
    device_structures deviceStructures;
	copyStructures(hostStructures, deviceStructures);
	initM(hostStructures);
	for (;;) {
		if (!optimiseModularity(0.001, deviceStructures, hostStructures))
			break;
		aggregateCommunities(deviceStructures, hostStructures);
	}
	int V;
	HANDLE_ERROR(cudaMemcpy(&V, deviceStructures.V, sizeof(int), cudaMemcpyDeviceToHost));
	printf("%f\n", calculateModularity(V, hostStructures.M, deviceStructures));
	printOriginalToCommunity(deviceStructures, hostStructures);
}
