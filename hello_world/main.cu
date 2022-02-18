#include <stdio.h>

#define ARRAY_SIZE 10

__global__ void add_kernel(float (*a) [ARRAY_SIZE][ARRAY_SIZE], float (*b)[ARRAY_SIZE][ARRAY_SIZE]) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < ARRAY_SIZE && j < ARRAY_SIZE) {
		(*a)[i][j] += (*b)[i][j];
	}
}

int main() {
	printf("Hello, World!\n");

	float (*a)[ARRAY_SIZE][ARRAY_SIZE];
	float (*b)[ARRAY_SIZE][ARRAY_SIZE];

	cudaMallocManaged(&a, sizeof(float) * ARRAY_SIZE * ARRAY_SIZE);
	cudaMallocManaged(&b, sizeof(float) * ARRAY_SIZE * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++) {
		for (int j = 0; j < ARRAY_SIZE; j++) {
			(*a)[i][j] = 1;
			(*b)[i][j] = 1;
		}
	}

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(ARRAY_SIZE / threadsPerBlock.x + 1, ARRAY_SIZE / threadsPerBlock.y + 1);

	printf("numBlocks: %d %d\n", numBlocks.x, numBlocks.y);
	printf("threadsPerBlock: %d %d\n", threadsPerBlock.x, threadsPerBlock.y);

	add_kernel<<<numBlocks, threadsPerBlock>>>(a, b);
	cudaDeviceSynchronize();

	for (int i = 0; i < ARRAY_SIZE; i++) {
		for (int j = 0; j < ARRAY_SIZE; j++) {
			printf("%f ", (*a)[i][j]);
		}
		printf("\n");
	}

	return 0;
}