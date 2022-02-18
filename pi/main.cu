#include <stdio.h>
#include <assert.h>

__global__ void pi_part_kernel(double* out, unsigned long long out_len, int offset) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < out_len) {
		if (i % 2 == 0) {
			out[i] = 4.0 / ((i + offset) * 2 + 1);
		} else {
			out[i] = -4.0 / ((i + offset) * 2 + 1);
		}
	}
}

void print_progress(int max, int curr) {
	int width = 80;
	int progress_percent = (curr * 100) / max;

	char progress_bar[width + 3] = {0};
	for (int i = 0; i < width; i++) {
		progress_bar[i + 1] = ' ';

		if (i < (width * progress_percent) / 100) {
			if (i < (width * (progress_percent - 1)) / 100) {
				progress_bar[i + 1] = '=';
			} else {
				progress_bar[i + 1] = '>';
			}
		}
	}

	progress_bar[0] = '[';
	progress_bar[width + 1] = ']';
	progress_bar[width + 2] = '\0';

	fprintf(stdout, "\r%d%% %s\r", progress_percent, progress_bar);
	fflush(stdout);
}

int main() {
#if 0
	unsigned long long num_iters = 0;
	int num_sub_iters = 0;

	char input[512];
	printf("Enter number of iterations: ");
	fgets(input, sizeof(input), stdin);
	num_iters = atoll(input);

	printf("Enter number of iterration length: ");
	fgets(input, sizeof(input), stdin);
	num_sub_iters = atoi(input);
#else
	unsigned long long num_iters = 100000000;
	int num_sub_iters = 10000000;
#endif

	printf("Number of iterations: %llu\n", num_iters);
	printf("Number of sub iterations: %d\n", num_sub_iters);

	double* pi_part;
	cudaMallocManaged(&pi_part, sizeof(double) * num_sub_iters);

	double pi;
	for (int i = 0; i < num_iters / num_sub_iters + 1; i++) {
		dim3 threads(1, 1, 1);
		dim3 blocks(num_sub_iters, 1, 1);
		pi_part_kernel<<<blocks, threads>>>(pi_part, num_iters, i * num_sub_iters);
		cudaDeviceSynchronize();

		for (int j = 0; j < num_sub_iters; j++) {
			pi += pi_part[j];
		}

		print_progress(num_iters / num_sub_iters, i);
	}

	printf("DEVICE: the value of PI is %.100Lf\n", pi);

	cudaFree(pi_part);

	return 0;
}
