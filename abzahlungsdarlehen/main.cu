#include <stdio.h>
#include <stdlib.h>

typedef struct {
	int year;
	float rest_jahresanfang;
	float rest_jahresende;
	float tilgung;
	float zins;
	float rate;
} abzahlungsdarlehen_kernel_result_t;

__global__ void abzahlungsdarlehen_kernel(abzahlungsdarlehen_kernel_result_t* result, int anzahl_jahre, float kreditsumme, float zinssatz) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < anzahl_jahre + 1) {
		result[i].year = i;
		result[i].tilgung = kreditsumme / anzahl_jahre;
		result[i].rest_jahresanfang = kreditsumme - (result[i].tilgung * (i - 1));
		result[i].rest_jahresende = kreditsumme - (result[i].tilgung * i);
		result[i].zins = (result[i].rest_jahresanfang * zinssatz) / 100;
		result[i].rate = result[i].tilgung + result[i].zins;
	}
}

int main() {
	int anzahl_jahre = 5;
	float kreditsumme = 20000;
	float zinssatz = 7.75;

#if 1
	char input[100] = { 0 };
	printf("Kreditsumme [€]:\n");
	fgets(input, 100, stdin);
	kreditsumme = atof(input);

	printf("Laufzeit [Jahre]:\n");
	fgets(input, 100, stdin);
	anzahl_jahre = atoi(input);

	printf("Zinssatz [%%]:\n");
	fgets(input, 100, stdin);
	int idx = 0;
	while (input[idx] != '\n') {
		if (input[idx] == ',') {
			input[idx] = '.';
		}
		idx++;
	}
	zinssatz = atof(input);
#endif

	abzahlungsdarlehen_kernel_result_t* result = nullptr;
	cudaMallocManaged(&result, anzahl_jahre * sizeof(abzahlungsdarlehen_kernel_result_t));

	dim3 threads(1, 1, 1);
	dim3 blocks(anzahl_jahre + 1, 1, 1);
	abzahlungsdarlehen_kernel<<<blocks, threads>>>(result, anzahl_jahre, kreditsumme, zinssatz);
	cudaDeviceSynchronize();

	float tilgung_total = 0;
	float zins_total = 0;
	float rate_total = 0;

	printf("\n");
	printf("Jahr    Rest (JA)    Tilgung     Rest (JE)      Zins        Rate\n");
	printf("----------------------------------------------------------------\n");
	for (int i = 1; i < anzahl_jahre + 1; i++) {
		printf("%4d %11.2f %11.2f %11.2f %11.2f %11.2f\n", result[i].year, result[i].rest_jahresanfang, result[i].rest_jahresende, result[i].tilgung, result[i].zins, result[i].rate);
		tilgung_total += result[i].tilgung;
		zins_total += result[i].zins;
		rate_total += result[i].rate;
	}
	printf("----------------------------------------------------------------\n");

	// print total with proper formatting
	printf("%4s %11s %11.2f %11s %11.2f %11.2f\n", "", "", tilgung_total, "", zins_total, rate_total);

	cudaFree(result);
	return 0;
}