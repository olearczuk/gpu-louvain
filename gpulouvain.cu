#include "utils.cuh"
#include "modularity_optimisation.cuh"
#include "community_aggregation.cuh"


int main(int argc, char *argv[]) {
	char *fileName;
	float minGain;
	bool isVerbose;
	parseCommandLineArgs(argc, argv, &minGain, &isVerbose, &fileName);

    auto hostStructures = readInputData(fileName);
    device_structures deviceStructures;
    aggregation_phase_structures aggregationPhaseStructures;

    cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	copyStructures(hostStructures, deviceStructures, aggregationPhaseStructures);
	initM(hostStructures);
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float memoryTime;
	HANDLE_ERROR(cudaEventElapsedTime(&memoryTime, start, stop));

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	for (;;) {
		if (!optimiseModularity(minGain, deviceStructures, hostStructures))
			break;
		break;
		aggregateCommunities(deviceStructures, hostStructures, aggregationPhaseStructures);
	}
	int V;
	HANDLE_ERROR(cudaMemcpy(&V, deviceStructures.V, sizeof(int), cudaMemcpyDeviceToHost));
	printf("%f\n", calculateModularity(V, hostStructures.M, deviceStructures));
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float algorithmTime;
	HANDLE_ERROR(cudaEventElapsedTime(&algorithmTime, start, stop));
	printf("%f %f\n", algorithmTime, algorithmTime + memoryTime);
	if (isVerbose)
		printOriginalToCommunity(deviceStructures, hostStructures);
	deleteStructures(hostStructures, deviceStructures, aggregationPhaseStructures);
}
