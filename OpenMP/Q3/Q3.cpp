#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>
#include "x86intrin.h"

#include <sys/time.h>
#include <cmath>

#include "../lib/primitives.h"

#define IMAGE1_ADDRESS "../img/q3_img01.png"
#define IMAGE2_ADDRESS "../img/q3_img02.png"

#define THREADS_NUM 8

typedef struct timeval timeVal;

cv::Mat imgAdditionSerial(cv::Mat img1, cv::Mat img2, long* execTime)
{
    cv::Mat resultImg(img1.rows, img1.cols, CV_8U);
    unsigned char* resultImgData = (unsigned char*) resultImg.data;
    unsigned char* img1Data = (unsigned char*) img1.data;
    unsigned char* img2Data = (unsigned char*) img2.data;
    timeVal start, end;

    // ----------------------- calculation started ---------------------------/
    gettimeofday(&start, NULL);
    for (int row = 0; row < img1.rows; row++)
        for (int col = 0; col < img1.cols; col++)
        {
            int idx = row * img1.cols + col;
            resultImgData[idx] = abs(img1Data[idx] - img2Data[idx]);     
        } 
    gettimeofday(&end, NULL);
    // ----------------------- calculation finished ---------------------------/
    *execTime = end.tv_usec - start.tv_usec;
    return resultImg;  
}

cv::Mat imgAdditionParallel(cv::Mat img1, cv::Mat img2, long* execTime)
{
    cv::Mat resultImg(img1.rows, img1.cols, CV_8U);

    unsigned char* resImgData = (unsigned char*) resultImg.data;
    unsigned char* img1Data = (unsigned char*) img1.data;
    unsigned char* img2Data = (unsigned char*) img2.data;
    timeVal start, end;

    // ----------------------- calculation started ---------------------------/
    gettimeofday(&start, NULL);
    int row = 0, col = 0;

    #pragma omp parallel for schedule (dynamic) private(row, col) shared(resImgData, img1Data, img2Data)
        for (row = 0; row < img1.rows; row++) {
            for (col = 0; col < img1.cols; col++)
            {
                int idx = row * img1.cols + col;
                resImgData[idx] = abs(img1Data[idx] - img2Data[idx]);     
            } 
        }

    gettimeofday(&end, NULL);
    // ----------------------- calculation finished ---------------------------/
    *execTime = end.tv_usec - start.tv_usec;
    return resultImg;
}

int main()
{
    const int numOfExec = 5000;
    long serialExecTime, parallelExecTime;
    cv::Mat img1 = cv::imread(IMAGE1_ADDRESS);
    cv::Mat img2 = cv::imread(IMAGE2_ADDRESS);

    float speedupAvg = 0.0;
    float speedup = 0.0;

    for(int i = 0; i < numOfExec; i++)
    {
        speedup = 0;
        cv::Mat resultImgSerial = imgAdditionSerial(img1, img2, &serialExecTime);
        cv::Mat resultImgParallel = imgAdditionParallel(img1, img2, &parallelExecTime);

        speedup = SPEED_UP(serialExecTime, parallelExecTime);

        if(speedup < 0)
            continue;
        
        speedupAvg += speedup;
        
        printf("Serial exec. Time : %li us\n", serialExecTime);
        printf("Parallel exec. Time : %li us\n", parallelExecTime);
        printf("-> speedup : %0.4f\n", speedup);
        printf("\n");

    }
    
    speedupAvg /= (float)numOfExec;
    
    printf("***********\n");
    printf("Average Speedup after %i executions : %0.4f\n", numOfExec, speedupAvg);
    printf("***********\n");

    return 0;

}