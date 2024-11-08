#include 	<stdio.h>
// #include 	"sse2neon.h"
#include    <math.h>
#include	<time.h>
#include 	<sys/time.h>
#include 	<stdlib.h>
//#include "ipp.h"
#include <x86intrin.h>


typedef struct timeval timeVal;

#define		VECTOR_SIZE		  	1048576

typedef union {
	__m128 	int128;
	float	SPFP[4];
} intVec;

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

	__m128 vec, index, incr, min, min_index, mask;
    gettimeofday(&start1, NULL);

	index = _mm_set_ps (3.0, 2.0, 1.0, 0.0);
    min_index = _mm_set_ps (3.0, 2.0, 1.0, 0.0);
	incr  = _mm_set1_ps (4.0);
    min = _mm_load_ps(&fVec1[0]);
	for (long i = 4; i < VECTOR_SIZE; i+=4) {
		vec   = _mm_loadu_ps (&fVec1[i]);
		min   = _mm_min_ps (vec, min);
		index = _mm_add_ps (index, incr);
        min_index = _mm_blendv_ps(min_index, index, _mm_cmpeq_ps(vec, min));
	}

    float min_of_min_parallel;
    float min_of_min_parallel_index;
    float min_parallel[4];
    intVec min_index_Vec;
    min_index_Vec.int128 = min_index;
    _mm_store_ps(min_parallel, min);
    min_of_min_parallel = min_parallel[0];
	for (int i = 1; i < 4; i++)
		if (min_parallel[i] < min_of_min_parallel)
		{
			min_of_min_parallel = min_parallel[i];
			min_of_min_parallel_index = min_index_Vec.SPFP[i];
		}

	gettimeofday(&end1, NULL);
    long time2 = end1.tv_usec - start1.tv_usec;


	printf ("\nSerial Run time = %f \n", (float) time1);
	printf ("Parallel Run time = %f \n", (float) time2);
	printf ("Speedup = %f\n\n", (float) (time1)/(float) time2);
    printf("serial min index: %f\n", min_of_min_index_serial);
    printf("serial min: %f\n", min_of_min_serial);
    printf("parallel min index: %f\n", min_of_min_parallel_index);
    printf("parallel min: %f\n", min_of_min_parallel);
    
	return 0;
}
