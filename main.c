#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include "cblas.h"
#include "papi.h"
#include "mpi.h"
#include<string.h>
//#define N 16


//Function Declarations
void fillArray(double*);
void printMatrix(double* matrix);
void resetMatrix(double*);
void compareMatrices(double *matrix1, double *matrix2, int size);
double calculateGFLOPS(double);
void matrixMuliply();



//matrix multiplication functions
//void dgemmIJK();

//Global Variables
int N=0;
int P;
int tag = 999;

int sizeArray[] = {64,128,256,512,1024,2048};
//int pArray[]= {8,16,32,64};

//int sizeArray[] = {64,128,256,512,1028,2048}; //{64,128,256,512,1028,2048 }

double * A;
double * B;
double * C;
double * C1;

// double * localA;
// double * localB;
// double * localC;




/***********************************************************
 * 										main function
************************************************************/
int main() {

	//srand initialization
	srand(time(NULL));
	MPI_Init(NULL, NULL);

	for(int i=0;i<6;i++){

		N= sizeArray[i];

		matrixMuliply();
	}
	
	MPI_Finalize();





	//system("pause");
	return 0;
}

/***********************************************************
 * Function: printMatrices
 * Prints the matrix row by row
************************************************************/

void matrixMuliply(){

	double  start;
	double end;
	double cpu_time_used;
	double sum = 0;
	int alpha = 1;
	int beta = 1;
	//int startPoint;
	//int endPoint;
	int processRank;

	MPI_Status status;
	MPI_Request request;

	/***************** MPI Environment ****************************/
	//MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &P);
	MPI_Comm_rank(MPI_COMM_WORLD, &processRank);


	//startPoint = processRank * (N / P);
	//endPoint =	(processRank+1) * (N / P);



	double * localA = (double * )(malloc(N* (N/P) * sizeof(double)));
	double * localB = (double * )(malloc(N* (N/P) * sizeof(double)));
	double * localC = (double * )(malloc(N* (N/P) * sizeof(double)));
	//double * localC1 = (double * )(malloc(N* (N/P) * sizeof(double)));

		start = MPI_Wtime();
	if(processRank == 0){

		A = (double *)(malloc(N*N * sizeof(double)));
		B = (double *)(malloc(N*N * sizeof(double)));
		C = (double *)(malloc(N*N * sizeof(double)));
		C1 = (double *)(malloc(N*N * sizeof(double)));

		fillArray(A);
		fillArray(B);
		//resetMatrix(C);

		}

		MPI_Scatter(
		A,
		N*N/P,
		MPI_DOUBLE,
		localA,
		N*N/P,
		MPI_DOUBLE,
		0,
		MPI_COMM_WORLD);

		MPI_Scatter(
		B,
		N*N/P,
		MPI_DOUBLE,
		localB,
		N*N/P,
		MPI_DOUBLE,
		0,
		MPI_COMM_WORLD);

		MPI_Scatter(
		C,
		N*N/P,
		MPI_DOUBLE,
		localC,
		N*N/P,
		MPI_DOUBLE,
		0,
		MPI_COMM_WORLD);


			//Calculation
			for(int s=processRank; s<processRank+P; s++){
				for(int i=0;i<N;i++){
					for(int j=0;j<N/P; j++){
						for(int k =0; k < N/P; k++){
							localC[i+N*j] += localA[i+ N*k] + localB[N/P*(s%P) + k+N*j];
						}
					}
				}

				int leftN= ((processRank - 1) + P) % P;
				int rightN= ((processRank + 1) + P) % P;

				double *localA_RECV_TEMP = (double * )(malloc(N* (N/P) * sizeof(double)));
				MPI_Sendrecv(localA, N*N/P, MPI_DOUBLE, leftN, tag, localA_RECV_TEMP,
															 N*N/P,	MPI_DOUBLE, rightN, tag, MPI_COMM_WORLD, &status);

				memcpy(localA, localA_RECV_TEMP, N*N/P * sizeof(double));
				free(localA_RECV_TEMP);
			}


			MPI_Gather(
								localA,
								N*N/P,
								MPI_DOUBLE,
								A,
								N*N/P,
								MPI_DOUBLE,
								0,
								MPI_COMM_WORLD);


			MPI_Gather(
								localB,
								N*N/P,
								MPI_DOUBLE,
								B,
								N*N/P,
								MPI_DOUBLE,
								0,
								MPI_COMM_WORLD);

			MPI_Gather(
								localC,
								N*N/P,
								MPI_DOUBLE,
								C,
								N*N/P,
								MPI_DOUBLE,
								0,
								MPI_COMM_WORLD);


	end = MPI_Wtime();
	cpu_time_used = ((double)(end - start));

 //freeing the dynamic memory

	free(localA);
	free(localB);
	free(localC);

	if (processRank == 0) {

		//Matrix Verification Using CBLAS

		//Computing Matrix Multiplication using CBLAS
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A, N, B, N, beta, C1, N);


		//Verifying the result
		compareMatrices(C, C1,N);

		printf("**************************************************************************************************************\n\n");

		printf("Avg execution time with Ctimer\n Size (NxN) %d \t\t %lf\n\n", N, (cpu_time_used));
		printf("GFLOPS:\t\t %lf\n\n",calculateGFLOPS(cpu_time_used) );
		printf("**************************************************************************************************************\n\n");

		free(A);
		free(B);
		free(C);
		free(C1);
	}

//	MPI_Finalize();

}

/***********************************************************
 * Function: printMatrices
 * Prints the matrix row by row
************************************************************/

void printMatrix(double * matrix) {

	printf("**************************************************************************************************************\n\n");
	for (int i = 0; i < N; i++) {

		for (int j = 0; j < N; j++) {

			printf("%lf  |", matrix[i*N + j]);
		}

		printf("\n");
	}
	printf("**************************************************************************************************************\n\n");
}


/***********************************************************
 * Function: fillArray
 * fills the matrices using random numbers in a row major format
************************************************************/
void  fillArray(double * matrix) {

	for (int i = 0; i < N; i++) {

		for (int j = 0; j < N; j++) {

			double r = (double)rand() / RAND_MAX * 2.0;      //float in range -1 to 1
			matrix[i + j*N] = r;
		}
	}
}

/***********************************************************
 * Function: resetMatrix
 * assignes zero to each element of the matrix
************************************************************/
void resetMatrix(double* matrix) {

	for (int i = 0; i < N; i++) {

		for (int j = 0; j < N; j++) {

			matrix[i + j * N] = 0;
		}
	}
}

/***********************************************************
 * Function: calculateGFLOPS
 * Calculates GFlops
************************************************************/
double	calculateGFLOPS(double time) {

	double gflop = (2.0 * N*N*N) / (time * 1e+9);

	return gflop;
}


/***********************************************************
 * Function: compareMatrices
 * Compares two matrices
************************************************************/
void compareMatrices(double *matrix1, double *matrix2,int size) {

	for (int i = 0; i < size;i ++) {

		for (int j = 0; j < size; j++) {

			if (matrix1[i*size + j] == matrix2[i*size + j]) {
					//do nothing
			}
			else {
				printf("\nMatrices are equal!\n");
				return;
			}
		}
	}
	printf("\nMatrices are not equal!\n");
}


//
// /*******************************************************************************
//  *						Matrix Multiplication using IJK algorithm
//  *******************************************************************************/
// void dgemmIJK() {
//
// 	//////////////************   PAPI CODE   ************//////////////////////////////
//
// 	int Events[NUM_EVENTS] = { PAPI_FP_OPS, PAPI_TOT_CYC, PAPI_L1_TCM, PAPI_LST_INS }; //level 1 cache misses
//
// 	int EventSet = PAPI_NULL;
//
// 	long long values[NUM_EVENTS];
//
// 	int retval;
//
// 	/* Initialize the Library*/
//
// 	retval = PAPI_library_init(PAPI_VER_CURRENT);
//
// 	/* Allocate space for the new eventset and do setup */
// 	retval = PAPI_create_eventset(&EventSet);
//
//
// 	/* Add Flops and total cycles to the eventset*/
// 	retval = PAPI_add_events(EventSet, Events, NUM_EVENTS);
//
// 	//////////////************   PAPI CODE   ************////////////////////////////
//
// 	int alpha, beta;
// 	alpha = beta = 1.0;
//
// 	clock_t  start;
// 	clock_t end;
//
// 	double cpu_time_used;
// 	double sum = 0;
//
//
// 	for (int ctr1 = 0; ctr1 < 6; ctr1++) {
//
// 		//memory allocation for matrices
//
// 		N = sizeArray[ctr1];
//
// 		//memory allocation for array pointers
// 		A = (double *)(malloc(N*N * sizeof(double)));
// 		B = (double *)(malloc(N*N * sizeof(double)));
// 		C = (double *)(malloc(N*N * sizeof(double)));
//
// 		C1 = (double *)(malloc(N*N * sizeof(double)));
//
// 			//filling the matrix
// 			fillArray(A);
// 			fillArray(B);
//
// 			//reset C1
// 			resetMatrix(C1);
//
// 	///////////////////////////////////////////////////////
//
// 		/* Start the counters */
// 		retval = PAPI_start(EventSet);
//
// 	///////////////////////////////////////////////////////
//
// 		for (int ctr2 = 0; ctr2 < 3; ctr2++) {
//
// 			//reset C
// 			resetMatrix(C);
//
// 			start = clock();
//
// 			//matrix multiplication
//
// 			for (int i = 0; i < N; i++) {
//
// 				for (int j = 0; j < N; j++) {
//
// 					double cij = C[i*N + j];
//
// 					for (int k = 0; k < N; k++) {
//
// 						cij = cij + A[i*N + k] * B[k*N + j];
// 						C[i*N + j] = cij;
// 					}
// 				}
// 			}
//
// 			end = clock();
//
// 			cpu_time_used = ((double)(end - start));
//
// 			sum += cpu_time_used;
//
//
// 		}
//
// 	/////////////////////////////////////////////////////////////////
//
// 		/*Stop counters and store results in values */
//
// 		retval = PAPI_stop(EventSet, values);
//
// 	////////////////////////////////////////////////////////////////
//
//
// 		//Matrix Verification Using CBLAS
//
// 		//Computing Matrix Multiplication using CBLAS
// 		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A, N, B, N, beta, C1, N);
//
// 		//Verifying the result
// 		compareMatrices(C,C1);
//
//
// 		printf("**************************************************************************************************************\n\n");
//
// 		printf("Avg execution time with Ctimer\n Size (NxN) %d \t\t %lf\n\n", N, (sum / 3.0));
// 		printf("GFLOPS:\t\t %lf\n\n",calculateGFLOPS(sum,size) );
// 		sum = 0;
//
// 		printf("\n\n______________________________________________________________________________________________________________\n\n");
//
// 		printf("PAPI Data/n");
//
// 		for (int ctr = 0; ctr < NUM_EVENTS; ctr++) {
//
// 			printf("/////////////////////////////////////\n");
// 			printf("%lld\n", values[ctr]);
// 			printf("/////////////////////////////////////\n");
// 		}
//
//
// 		printf("**************************************************************************************************************\n\n");
//
// 		//freeing the dynamic memory
// 		free(A);
// 		free(B);
// 		free(C);
// 		free(C1);
//
// 	}
//
// }
