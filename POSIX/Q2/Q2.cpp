#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include "pthread.h"
#include "unistd.h"
#include <sys/time.h>

#define MAX_RAND_FLOAT 100.0
#define THREADS_NUM 8

#define SPEED_UP(SERIAL, PARALLEL) \
        (float)(SERIAL) / (float)(PARALLEL)

typedef struct timeval timeVal;


typedef struct ParallelArgs
{
    int threadNum;
    int startIndex;
    int stopIndex;
    int totalSize;
    float* numbers;
    float* resultAvg;
    float* resultStdev;
    pthread_mutex_t* avgMutex; 
    pthread_barrier_t* barrier;

} ParallelArgs;


void printResult(float avg, float stdev, long execTime)
{
    printf("Average Value : %f\n", avg);
    printf("Result Standard Deviation : %f\n", stdev);
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


void* calculateStdev(void* arg)
{
    ParallelArgs* pArgs = (ParallelArgs*)arg;
    // printf("Thread Number %d starts working\nCalculating  [%d , %d) \n\n", 
    //             pArgs->threadNum, pArgs->startIndex, pArgs->stopIndex);

    // average calculation :
    float sum;
    for(int i = pArgs->startIndex; i < pArgs->stopIndex; i++)
    {
        sum += pArgs->numbers[i];
    }
    sum /= pArgs->totalSize;
    
    pthread_mutex_lock(pArgs->avgMutex);
    *(pArgs->resultAvg) += sum;
    pthread_mutex_unlock(pArgs->avgMutex);
    // printf("Avg job done by T%d\n\n", pArgs->threadNum);

    float diff = 0.0, temp = 0;
    pthread_barrier_wait(pArgs->barrier);

    // stdev calculation

    for(int i = pArgs->startIndex; i < pArgs->stopIndex; i++)
    {
        temp = pArgs->numbers[i] - (*(pArgs->resultAvg));
        temp *= temp;
        diff += temp;
    }
    // printf("stdev part calculated by T%d : %f\n\n", pArgs->threadNum, diff);

    pthread_mutex_lock(pArgs->avgMutex);
    *(pArgs->resultStdev) += diff;
    pthread_mutex_unlock(pArgs->avgMutex);


    pthread_exit(NULL);
}




long stdevSerial(float* arr, int size)
{
    timeVal start, stop;
    int chunkSize = size/THREADS_NUM;
    int startIndex, stopIndex;
    // ------------------ execution started ----------------- 
    gettimeofday(&start, NULL);
    float avg = 0, stdev = 0, sum = 0, diff = 0;

    for(int i = 0; i < THREADS_NUM; i++)
    {
        sum = 0;
        startIndex = i*chunkSize;
        if(i == THREADS_NUM-1)
            stopIndex = size;
        else
            stopIndex = (i+1)*chunkSize;

        for(int j = startIndex; j < stopIndex; j++)
        {
            sum += arr[j];
        }
        sum /= size;

        avg += sum;
    }

    float temp;
    for(int i = 0; i < THREADS_NUM; i++)
    {
        diff = 0;
        startIndex = i*chunkSize;

        if(i == THREADS_NUM-1)
            stopIndex = size;
        else
            stopIndex = (i+1)*chunkSize;

        for(int j = startIndex; j < stopIndex; j++)
        {
            temp = arr[j] - avg;
            temp *= temp;
            diff += temp;
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
    int chunkSize = size/THREADS_NUM;
    int startIndex, stopIndex;
    int order = 0;

    gettimeofday(&start, NULL);

    pthread_t threads[THREADS_NUM];
    pthread_mutex_t mutex;
    pthread_barrier_t barrier;
    ParallelArgs tArguments[THREADS_NUM];
    
    pthread_mutex_init(&mutex, NULL);
    pthread_barrier_init(&barrier, NULL, THREADS_NUM);

    for(int i = 0; i < THREADS_NUM; i++)
    {
        startIndex = i*chunkSize;
        stopIndex;
        if(i == THREADS_NUM-1)
            stopIndex = size;
        else
            stopIndex = (i+1)*chunkSize;
        
        tArguments[i] = {i, startIndex, stopIndex, size, arr,
                         &avg, &stdev, &mutex, &barrier};

        pthread_create(&threads[i], NULL, calculateStdev, (void*)(&(tArguments[i])));
    }


    for(int i = 0; i < THREADS_NUM; i++)
        pthread_join(threads[i], NULL);
    

    stdev /= size;
    stdev = sqrt(stdev);

    pthread_mutex_destroy(&mutex);

    gettimeofday(&end, NULL);
    // ------------------ execution finished ----------------- 
    

    long execTime = end.tv_usec - start.tv_usec;

    printResult(avg, stdev, execTime);
    return execTime;

    
}

int main()
{
    const int arrLen = 1 << 20;
    float arr[arrLen];
    

    // For single Execution test Uncomment Below :


    // generateFloatArr(arr, arrLen);

    // printf("* parallel execution : \n");
    // long parallelExecTime = stdevParallel(arr, arrLen);
    // printf("\n");


    // printf("* serial execution : \n");
    // long serialExecTime = stdevSerial(arr, arrLen);

    
    // printf("\n\nSpeed up : %f\n", SPEED_UP(serialExecTime, parallelExecTime)); 

    // ----------------------------------------------------------------------------------------


    // For Multiple Executions Uncomment below :

    for(int i = 0; i < 1000; i++)
    {
        generateFloatArr(arr, arrLen);

        printf("* parallel execution : \n");
        long parallelExecTime = stdevParallel(arr, arrLen);
        printf("\n");


        printf("* serial execution : \n");
        long serialExecTime = stdevSerial(arr, arrLen);

        
        printf("\n\nSpeed up : %f\n", SPEED_UP(serialExecTime, parallelExecTime)); 
        printf("\n*************\n");
    }

    // ----------------------------------------------------------------------
}







