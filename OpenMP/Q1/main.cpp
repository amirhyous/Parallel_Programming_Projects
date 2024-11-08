#include 	<stdio.h>
#include    <math.h>
#include	<time.h>
#include 	<sys/time.h>
#include 	<stdlib.h>
#include 	"omp.h"	
#include 	<limits.h>

typedef struct timeval timeVal;

#define		VECTOR_SIZE		  	1048576
#define		THREADS_NUM			4

float randF()
{
    
    return (float)rand() / RAND_MAX * 100.00;
}
float* generateFloatArr(float* arr, int size)
{
    srand(time(NULL));
    for(int i = 0; i < size; i++) {
        arr[i] = randF();
    }
    return arr;
}

int main (void)
{
	const int arrLen = 1 << 20;
    float fVec1[arrLen];
    
	timeVal start, end, start1, end1;

    generateFloatArr(fVec1, arrLen);

//serial

	float min_serial[4];
    float min_index_serial[4];
    float min_of_min_serial, min_of_min_index_serial;

	min_serial[0] = fVec1[0];
	min_serial[1] = fVec1[1];
	min_serial[2] = fVec1[2];
	min_serial[3] = fVec1[3];

    gettimeofday(&start, NULL);

	for (long i = 4; i < VECTOR_SIZE; i+=4)
		if(fVec1[i] < min_serial[0])
		{
			min_serial[0] = fVec1[i];
			min_index_serial[0] = i;
		}
	for (long i = 4; i < VECTOR_SIZE; i+=4)
		if(fVec1[i+1] < min_serial[1])
		{
			min_serial[1] = fVec1[i+1];
			min_index_serial[1] = i+1;
		}
	for (long i = 4; i < VECTOR_SIZE; i+=4)
		if(fVec1[i+2] < min_serial[2])
		{
			min_serial[2] = fVec1[i+2];
			min_index_serial[2] = i+2;
		}
	for (long i = 4; i < VECTOR_SIZE; i+=4)
		if(fVec1[i+3] < min_serial[3])
		{
			min_serial[3] = fVec1[i+3];
			min_index_serial[3] = i+3;
		}
	
    min_of_min_serial = min_serial[0];
    for (int i = 1; i < 4; i++)
		if (min_serial[i] < min_of_min_serial)
		{
			min_of_min_serial = min_serial[i];
			min_of_min_index_serial = min_index_serial[i];
		}

	gettimeofday(&end, NULL);

    long time1 = end.tv_usec - start.tv_usec;

//parallel

    gettimeofday(&start1, NULL);
	float min_parallel;
	float min_index_parallel;
	float min_local;
	float min_index_local;

	float Inf = std::numeric_limits<float>::infinity();  
	min_parallel = Inf;

	#pragma omp parallel num_threads(THREADS_NUM) shared(min_parallel, min_index_parallel) private(min_local, min_index_local)
	{
		min_local = fVec1[(VECTOR_SIZE / THREADS_NUM) * omp_get_thread_num()];
		min_index_local = (VECTOR_SIZE / THREADS_NUM) * omp_get_thread_num();
		#pragma omp for
			for(int i = 0; i < VECTOR_SIZE; i++)
				if(fVec1[i] < min_local)
				{
					min_local = fVec1[i];
					min_index_local = i;
				}

		#pragma omp critical
			for(int i = 0; i < THREADS_NUM; i++)
				if(min_local < min_parallel)
				{
					min_parallel = min_local;
					min_index_parallel = min_index_local;
				}
	}


	gettimeofday(&end1, NULL);
    long time2 = end1.tv_usec - start1.tv_usec;


	printf ("\nSerial Run time = %0.3f \n", (float) time1);
	printf ("Parallel Run time = %0.3f \n", (float) time2);
	printf ("Speedup = %f\n\n", (float) (time1)/(float) time2);
    printf("serial min index: %0.3f\n", min_of_min_index_serial);
    printf("serial min: %f\n\n", min_of_min_serial);
    printf("parallel min index: %0.3f\n", min_index_parallel);
    printf("parallel min: %f\n", min_parallel);
    
	return 0;
}
