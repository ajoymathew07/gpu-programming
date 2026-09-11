#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

__global__ void dkernel(const int* A, const int* B, const int* C, const int* D, int* E, int p, int q, int r) {
	__shared__ int sA[32][33];
	__shared__ int sB[32][33];
	__shared__ int sC[32][33];
	__shared__ int sD[32][33];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i0 = blockIdx.y * 32;
	int j0 = blockIdx.x * 32;
	int row = i0 + ty;
	int col = j0 + tx;

	int sum = 0;
	int numTiles = (q + 32 - 1) / 32;

	for(int t = 0; t < numTiles; ++t) {
		int t0 = t * 32;

		sA[ty][tx] = (t0 + ty < q && i0 + tx < p) ? A[(t0 + ty) * p + (i0 + tx)] : 0;
		sB[ty][tx] = (t0 + ty < q && j0 + tx < r) ? B[(t0 + ty) * r + (j0 + tx)] : 0;
		sC[ty][tx] = (i0 + ty < p && t0 + tx < q) ? C[(i0 + ty) * q + (t0 + tx)] : 0;
		sD[ty][tx] = (j0 + ty < r && t0 + tx < q) ? D[(j0 + ty) * q + (t0 + tx)] : 0;
		__syncthreads();

		for(int k = 0; k < 32; ++k) {
			sum += sA[k][ty] * sB[k][tx];
			sum += sC[ty][k] * sD[tx][k];
		}
		__syncthreads();
	}

	if (row < p && col < r) {
		E[row * r + col] = sum;
	}
}

// function to compute the output matrix
void compute(int p, int q, int r, int *h_matrixA, int *h_matrixB,
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;

	// allocate memory...
	cudaMalloc(&d_matrixA, q * p * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, q * p * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
	dim3 block(32, 32);
	dim3 grid((r + block.x - 1) / block.x, (p + block.y - 1) / block.y);
	dkernel<<<grid, block>>>(d_matrixA, d_matrixB, d_matrixC, d_matrixD, d_matrixE, p, q, r);
	
	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}



int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;

    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n");
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(q * p * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, q, p);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	compute(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
