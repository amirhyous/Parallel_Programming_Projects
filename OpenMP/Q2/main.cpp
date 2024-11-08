#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include "omp.h"
#include <sys/time.h>

#define MAX_RAND_FLOAT 100.0
#define THREADS_NUM 4

#define SPEED_UP(SERIAL, PARALLEL) \
        (float)(SERIAL) / (float)(PARALLEL)

typedef struct timeval timeVal;

void printResult(float avg, float stdev, long execTime)
{
    printf("Average Value : %.3f\n", avg);
    printf("Result Standard Deviation : %.3f\n", stdev);
    printf("Execution Time : %li us\n", execTime);
}

float randF()
{
    
    return (float)rand() / RAND_MAX * MAX_RAND_FLOAT;
}

float* generateFloatArr(float* arr, int size)
{
    srand(time(NULL));
    for(int i = 0; i < size; i++) {
        arr[i] = randF();
    }
    return arr;
}

long stdevSerial(float* arr, int size)
{
    timeVal start, stop;
    // ------------------ execution started ----------------- 
    gettimeofday(&start, NULL);
    float avg = 0, stdev = 0;
    for(int i = 0; i < 4; i++)
    {
        float sum = 0;
        for(int j = 0; j < size; j+=4)
        {
            sum += (float)(arr[j + i]);
        }
        avg += sum;
    }
    avg /= size;

    for (int i = 0; i < 4; i++)
    {
        float diff = 0;
        for(int j = 0; j  < size; j+=4)
        {
            diff += pow((arr[j + i] - avg),2);
        }
        stdev += diff;
    }

    stdev /= (size);
    stdev = sqrt(stdev);

    gettimeofday(&stop, NULL);
    // ------------------ execution finished ----------------- 
    
    long execTime = stop.tv_usec - start.tv_usec;
    
    printResult(avg, stdev, execTime);

    return execTime;

}

long stdevParallel(float* arr, int size)
{
    timeVal start, end;
    // ------------------ execution started ----------------- 
    float avg = 0, stdev = 0, sum = 0, diff = 0;
    gettimeofday(&start, NULL);
    #pragma omp parallel num_threads(THREADS_NUM) shared(avg, stdev, sum, diff) 
    {
        #pragma omp for reduction(+:sum)
            for(int i = 0; i < size; i++)
                sum += (float)(arr[i]);
        #pragma omp barrier
        #pragma omp single
            avg = sum / size;
        
        #pragma omp for reduction(+:diff)
            for(int i = 0; i  < size; i++)
                diff += pow((arr[i] - avg),2);
        #pragma omp barrier
        #pragma omp single
        {
            stdev = diff / (size);
            stdev = sqrt(stdev);
        }    
    }

    gettimeofday(&end, NULL);
    // ------------------ execution finished ----------------- 
    

    long execTime = end.tv_usec - start.tv_usec;

    // printf("average value : %f\n", average);
    // printf("result standard deviation : %f\n", standardDeviation);
    // printf("execution time : %li us\n", execTime);
    
    printResult(avg, stdev, execTime);
    return execTime;

    
}

int main()
{
    const int arrLen = 1 << 20;
    float arr[arrLen];
    
    generateFloatArr(arr, arrLen);

    printf("* serial execution : \n");
    long serialExecTime = stdevSerial(arr, arrLen);
    printf("\n");

    printf("* parallel execution : \n");
    long parallelExecTime = stdevParallel(arr, arrLen);
    
    printf("\n\nSpeed up : %f\n", SPEED_UP(serialExecTime, parallelExecTime)); 
}







