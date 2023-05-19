#include <stdio.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>


cudaError_t runCuda(int upper_bound, unsigned int* cpu_pre_primes, int cpu_pre_primes_count);

unsigned int count;
unsigned int* cpu_primes; //cuda primes >D
unsigned int* gpu_truth_out;

unsigned int max_prime;
unsigned int truth_table_size; // count of uint in truth_table (max_accused_value / 32)

//using namespace std;
using namespace std::chrono;


bool is_it_prime_tough(unsigned int number);
unsigned int* get_primes(unsigned int count);

bool inline is_it_prime_though(unsigned int number, unsigned int* primes) {
	unsigned int rt = sqrt(number);
	for (size_t n = 0; primes[n] <= rt; n++)
	{
		if (number % primes[n] == 0) {
			return false;
		}
	}
}

unsigned int* get_primes(unsigned int until, unsigned int& n) {
	unsigned int* primes = new unsigned int[until + 1];

	primes[0] = 2;
	primes[1] = 3;
	primes[2] = 5;

	unsigned int accused = 7;
	n = 3;
	while (accused < until)
	{
		if (is_it_prime_though(accused, primes)) {
			primes[n] = accused;
			n++;
		}
		accused += 4;

		if (is_it_prime_though(accused, primes)) {
			primes[n] = accused;
			n++;
		}
		accused += 2;
	}
	return primes;
}

unsigned int max_prime_value(unsigned int n) {
	float k = 10.273;
	float ln = log(n);
	float lln = log(log(n));

	return ceil(n * (ln + lln - 1 + (lln - 2.0) / ln - (lln * lln - 6.0 * lln + k) / (2.0 * ln * ln))) + 10000u;
}

void get_prime_array()
{
	size_t n = 0;
	std::fstream file;
	file.open("primes" + std::to_string(count) + ".txt", std::ios_base::out);

	if (!file.is_open())
	{
		std::cout << "Unable to open the file.\n";
		return;
	}

	for (size_t i = 0; i < truth_table_size; i++)
	{
		for (size_t j = 0; j < 32; j++)
		{
			if (!(1 & (gpu_truth_out[i] >> (31 - j))))
			{
				//std::cout << (2 + (i * 32 + j)) << ", ";
				uint64_t prime = (2 + (i * 32 + j));
				//prime = prime << 1; prime |= 1;
				file << prime << ",";
				n++;
			}

			if (n >= count)
			{
				return;
			}
		}
	}

	file.close();
}

int main()
{
	std::cout << "How many primes you want?" << std::endl;
	std::cin >> count;

	auto startTime = high_resolution_clock::now();

	count = std::max(count, 5u);

	max_prime = max_prime_value(count);
	std::cout << max_prime << " is the upper bound." << std::endl;

	truth_table_size = ceil(max_prime / 32.0) + 1;
	gpu_truth_out = (unsigned int*)malloc(truth_table_size * sizeof(int));

	auto start_cpu = high_resolution_clock::now();
	unsigned int precalculated_prime_count;
	unsigned int* cpu_primes = get_primes(ceil(sqrt(max_prime)) + 1, precalculated_prime_count);
	std::cout << "The cpu took " << duration_cast<microseconds>(high_resolution_clock::now() - start_cpu).count() / 1000000.0 << "sec." << std::endl;

	auto start_gpu = high_resolution_clock::now();
	cudaError_t cudaStatus = runCuda(max_prime, cpu_primes, precalculated_prime_count);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	std::cout << "The gpu took " << duration_cast<microseconds>(high_resolution_clock::now() - start_gpu).count() / 1000000.0 << "sec." << std::endl;

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	for (size_t i = 0; i < truth_table_size; i++)
	{
		//std::cout << gpu_truth_out[i] << ", ";
	}

	std::cout << "The calculation took " << duration_cast<milliseconds>(high_resolution_clock::now() - startTime).count() / 1000.0 << "sec." << std::endl;

	get_prime_array();

	return 0;
}


__global__ void gpu_calc(int upper_bound, const unsigned int* cpu_primes, unsigned int* truth_table)
{
	unsigned int accused = 0;

	unsigned int int_pos = threadIdx.x;
	unsigned int pos = 0;

	for (size_t i = 0; accused <= upper_bound; i++)
	{
		pos = blockIdx.x + i * gridDim.x;
		accused = blockDim.x * pos + int_pos + 2;

		bool is_prime = true;
		float root = sqrtf((float)accused);
		unsigned int curr_prime = 2;
		for (size_t j = 0; curr_prime <= root; j++)
		{
			curr_prime = cpu_primes[j];
			if (accused % curr_prime == 0)
			{
				is_prime = false;
				break;
			}
		}

		if (is_prime)
		{
			truth_table[pos] |= 1 << (sizeof(int) * 8 - 1 - int_pos);
			// 1 << (sizeof(int) * 8 - 1 - int_pos)
			//atomicOr(truth_table + (pos), 1 << (sizeof(int) * 8 - 1 - int_pos));
		}

	}
}

#define GET_SECTION_ID(x) x >> 5
#define GET_INNER_ID(x) 31 - (x & 31)

__global__ void gpu_sieve(const int upper_bound, const int cpu_prime_count, const unsigned int* cpu_primes, unsigned int* truth_table)
{
	unsigned int num_of_threads = gridDim.x * blockDim.x;

	for (int prime_id = threadIdx.x + blockDim.x * blockIdx.x; prime_id < cpu_prime_count; prime_id += num_of_threads)
	{
		uint64_t prime = (uint64_t)cpu_primes[prime_id];
		uint64_t sq_prime = prime * prime;

		for (uint64_t index = sq_prime - 2; index <= upper_bound; index += prime)
		{
			atomicOr(truth_table + (GET_SECTION_ID(index)), unsigned int(1) << GET_INNER_ID(index));//minden ami NEM prime az 1 aka true. -> minden ami 0 AZ prime
		}
	}
}




// Helper function for using CUDA to add vectors in parallel.
cudaError_t runCuda(int upper_bound, unsigned int* cpu_pre_primes, int cpu_pre_primes_count)
{
	cudaError_t cudaStatus;
	unsigned int* gpu_pre_primes; //The pre generated primes from the cpu UPLOADED TO GPU
	unsigned int* gpu_truth_table;



	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	/* CREATE MEMORY */
	cudaStatus = cudaMalloc((void**)&gpu_pre_primes, cpu_pre_primes_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&gpu_truth_table, truth_table_size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	/* ADD DATA TO MEMORY */

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(gpu_pre_primes, cpu_pre_primes, cpu_pre_primes_count * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! cudaMemcpyHostToDevice\n");
		goto Error;
	}

	//free(cpu_pre_primes);

	cudaMemset(gpu_truth_table, 0, sizeof(int) * truth_table_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!\n");
		goto Error;
	}

	/* LAUNCH KERNEL*/
	//gpu_calc << <512, 32 >> > (upper_bound, gpu_pre_primes, gpu_truth_table);
	gpu_sieve << <512, 32 >> > (upper_bound, cpu_pre_primes_count, gpu_pre_primes, gpu_truth_table);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "gpu_calc launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching gpu_calc!\n", cudaStatus);
		goto Error;
	}


	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(gpu_truth_out, gpu_truth_table, truth_table_size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! cudaMemcpyDeviceToHost\n");
		goto Error;
	}

Error:
	cudaFree(cpu_pre_primes);
	cudaFree(gpu_truth_table);

	return cudaStatus;
}

