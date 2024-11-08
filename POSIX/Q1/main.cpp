#include 	<stdio.h>
#include    <math.h>
#include	<time.h>
#include 	<sys/time.h>
#include 	<stdlib.h>
#include 	<limits.h>
#include "unistd.h"
#include "pthread.h"

typedef struct timeval timeVal;

#define		VECTOR_SIZE		  	1048576
#define		THREADS_NUM			4

float array [VECTOR_SIZE];
    

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

//Parallen Fuction
typedef struct{
	int start_index;
	int end_index;
} in_param_t;

float min_parallel ;
int min_index_parallel;
pthread_mutex_t lock;

void *find_local_min_parallel(void *arg)
{
	in_param_t *inp = (in_param_t *) arg;
	float local_min = array[inp->start_index];
	int local_min_index;

	for (int i=inp->start_index; i<inp->end_index; i++)
		if (array[i] < local_min)
		{
			local_min = array[i];
			local_min_index = i;
		}

	pthread_mutex_lock (&lock);
	if (local_min < min_parallel)
	{
		min_parallel = local_min;
		min_index_parallel = local_min_index;
	}

	pthread_mutex_unlock (&lock);
	
	pthread_exit (NULL);
}

int main (void)
{
	timeVal start, end, start1, end1;

    generateFloatArr(array, VECTOR_SIZE);

//Serial

	float min_serial[4];
    int min_index_serial[4];
    float min_of_min_serial;
	int min_of_min_index_serial;

	min_serial[0] = array[0];
	min_serial[1] = array[1];
	min_serial[2] = array[2];
	min_serial[3] = array[3];

    gettimeofday(&start, NULL);

	for (long i = 4; i < VECTOR_SIZE; i+=4)
		if(array[i] < min_serial[0])
		{
			min_serial[0] = array[i];
			min_index_serial[0] = i;
		}
	for (long i = 4; i < VECTOR_SIZE; i+=4)
		if(array[i+1] < min_serial[1])
		{
			min_serial[1] = array[i+1];
			min_index_serial[1] = i+1;
		}
	for (long i = 4; i < VECTOR_SIZE; i+=4)
		if(array[i+2] < min_serial[2])
		{
			min_serial[2] = array[i+2];
			min_index_serial[2] = i+2;
		}
	for (long i = 4; i < VECTOR_SIZE; i+=4)
		if(array[i+3] < min_serial[3])
		{
			min_serial[3] = array[i+3];
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

//Parallel

    gettimeofday(&start1, NULL);
	
	pthread_t th[THREADS_NUM];
	
	in_param_t in_param [THREADS_NUM] =  
		{{0, VECTOR_SIZE/4}, {VECTOR_SIZE/4, VECTOR_SIZE/2},
		 {VECTOR_SIZE/2, 3 * VECTOR_SIZE/4}, {3 * VECTOR_SIZE/4, VECTOR_SIZE}};

	min_parallel = array[0];

	for (int i = 0; i < THREADS_NUM; i++)
		pthread_create (&th[i], NULL, find_local_min_parallel, (void *) &in_param[i]);

	for (int i = 0; i < THREADS_NUM; i++)
		pthread_join (th[i], NULL);

	gettimeofday(&end1, NULL);
    long time2 = end1.tv_usec - start1.tv_usec;

	printf ("\nSerial Run time = %0.3f \n", (float) time1);
	printf ("Parallel Run time = %0.3f \n", (float) time2);
	printf ("Speedup = %f\n\n", (float) (time1)/(float) time2);
    printf("serial min index: %d\n", min_of_min_index_serial);
    printf("serial min: %f\n\n", min_of_min_serial);
    printf("parallel min index: %d\n", min_index_parallel);
    printf("parallel min: %f\n", min_parallel);
    
	return 0;
}
