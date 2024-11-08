#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <x86intrin.h>
#include <sys/time.h>

#include "../lib/primitives.h"

#define MAX_RAND_FLOAT 100.0

typedef struct timeval timeVal;

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

    stdev /= (size - 1);
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
    gettimeofday(&start, NULL);
    
    __m128 vec;
    __m128 avg = _mm_set1_ps(0.0f), stdev = _mm_set1_ps(0.0f);
    __m128 divisor = _mm_set1_ps((float)(size));
    __m128 stdevDivVal = _mm_set1_ps((float)(size - 1));

    for (int i = 0; i < size; i+=4)
    {
        vec = _mm_loadu_ps(&arr[i]);
        avg = _mm_add_ps(avg, vec);
    }
    avg = _mm_hadd_ps(avg, avg);
    avg = _mm_hadd_ps(avg, avg);
    avg = _mm_div_ps(avg, divisor);

    for (int i = 0; i < size; i+=4)
    {
        vec = _mm_loadu_ps(&arr[i]);
        vec = _mm_sub_ps(vec, avg);     // xi - avg
        vec = _mm_mul_ps(vec, vec);     // (xi - avg)^2
        stdev = _mm_add_ps(stdev, vec);
    }
    stdev = _mm_hadd_ps(stdev, stdev);
    stdev = _mm_hadd_ps(stdev, stdev);
    stdev = _mm_div_ps(stdev, stdevDivVal);
    stdev = _mm_sqrt_ps(stdev);

    float average = _mm_cvtss_f32(avg);
    float standardDeviation = _mm_cvtss_f32(stdev);

    gettimeofday(&end, NULL);
    // ------------------ execution finished ----------------- 
    

    long execTime = end.tv_usec - start.tv_usec;

    // printf("average value : %f\n", average);
    // printf("result standard deviation : %f\n", standardDeviation);
    // printf("execution time : %li us\n", execTime);
    
    printResult(average, standardDeviation, execTime);
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