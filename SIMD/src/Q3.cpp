#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "x86intrin.h"
#include <sys/time.h>
#include <cmath>

#include "../lib/primitives.h"

#define IMAGE1_ADDRESS "../img/q3_img01.png"
#define IMAGE2_ADDRESS "../img/q3_img02.png"

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
    __m128i vec1, vec2, res;

    for(int row = 0; row < img1.rows; row++)
    {
        for(int col = 0; col < img1.cols; col += 16)
        {
            int idx = row * img1.cols + col;
            vec1 = _mm_loadu_si128((const __m128i*)(&img1Data[idx]));
            vec2 = _mm_loadu_si128((const __m128i*)(&img2Data[idx]));
            res = _mm_abs_epi16(_mm_sub_epi16(vec1, vec2));
            _mm_storeu_si128((__m128i*)(&resImgData[idx]), res);

        }
    }   
    gettimeofday(&end, NULL);
    // ----------------------- calculation finished ---------------------------/
    *execTime = end.tv_usec - start.tv_usec;
    return resultImg;
}

int main()
{
    long serialExecTime, parallelExecTime;
    cv::Mat img1 = cv::imread(IMAGE1_ADDRESS);
    cv::Mat img2 = cv::imread(IMAGE2_ADDRESS);
    
    cv::Mat resultImgSerial = imgAdditionSerial(img1, img2, &serialExecTime);
    cv::Mat resultImgParallel = imgAdditionParallel(img1, img2, &parallelExecTime);
    
    printf("Serial exec. Time : %li us\n", serialExecTime);
    printf("Parallel exec. Time : %li us\n", parallelExecTime);
    printf("-> speedup : %0.4f\n", SPEED_UP(serialExecTime, parallelExecTime));


    

    return 0;

}