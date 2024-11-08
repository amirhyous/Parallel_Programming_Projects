#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "x86intrin.h"
#include <sys/time.h>

#include "../lib/primitives.h"

#define IMAGE1_ADDRESS "../img/q4_img01.png"
#define IMAGE2_ADDRESS "../img/q4_img02.png"
#define ALPHA_INVERSED 4


typedef struct timeval timeVal;

cv::Mat imgAdditionSerial(cv::Mat img1, cv::Mat img2, long* execTime, int alphaInversed = ALPHA_INVERSED)
{
    cv::Mat resultImg(img1.rows, img1.cols, CV_8U);
    unsigned char* resultImgData = (unsigned char*) resultImg.data;
    unsigned char* img1Data = (unsigned char*) img1.data;
    unsigned char* img2Data = (unsigned char*) img2.data;

    timeVal start, end;

    // ----------------------- calculation started ---------------------------/
    gettimeofday(&start, NULL);
    for (int row = 0; row < img1.rows; row++)
    {
        for (int col = 0; col < img1.cols; col++)
        {
            int idx1 = row * img1.cols + col;
            int res = img1Data[idx1];

            if(row <= img2.rows && col <= img2.cols)
            {
                int idx2 = row * img2.cols + col;
                res += (img2Data[idx2] >> 2);
                
                if(res > 255)
                    res = 255;
                
            }

            resultImgData[idx1] = res;
            
        }
        
    }
    gettimeofday(&end, NULL);
    // ----------------------- calculation finished ---------------------------/
    *execTime = end.tv_usec - start.tv_usec;

    return resultImg;
    
}

cv::Mat imgAdditionParallel(cv::Mat img1, cv::Mat img2, long* execTime, int alphaInversed = ALPHA_INVERSED)
{
    cv::Mat resultImg(img1.rows, img1.cols, CV_8U);

    unsigned char* resImgData = (unsigned char*) resultImg.data;
    unsigned char* img1Data = (unsigned char*) img1.data;
    unsigned char* img2Data = (unsigned char*) img2.data;
    timeVal start, end;

    // ----------------------- calculation started ---------------------------/
    gettimeofday(&start, NULL);
    __m128i res = _mm_set1_epi8(0);
    __m128i b = _mm_set1_epi8(alphaInversed);

    for(int row = 0; row < img1.rows; row++)
    {
        for(int col = 0; col < img1.cols; col += 16)
        {
            int idx1 = row * img1.cols + col;
            res = _mm_loadu_si128((const __m128i*)(&img1Data[idx1]));

            if(row <= img2.rows && col + 16 <= img2.cols)
            {
                int idx2 = row * img2.cols + col;
                b = _mm_loadu_si128((const __m128i*)(&img2Data[idx2]));

                b = _mm_srli_epi16(b,2);
                b = _mm_and_si128(b, _mm_set1_epi16(0x3F3F));

                res = _mm_adds_epu8(res, b);
                
            }

            _mm_storeu_si128((__m128i*)(&resImgData[idx1]), res);

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
    cv::Mat img1 = cv::imread(IMAGE1_ADDRESS, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(IMAGE2_ADDRESS, cv::IMREAD_GRAYSCALE);
    
    

    cv::Mat resultImgSerial = imgAdditionSerial(img1, img2, &serialExecTime);
    cv::Mat resultImgParallel = imgAdditionParallel(img1, img2, &parallelExecTime);
    
    // uncomment to check any conflicts between two resulted images :

    // for (int i = 0; i < img1.rows; i++)
    // {
    //     for (int j = 0; j < img1.cols; j++)
    //     {
    //         int idx = i*img1.cols + j;
    //         if(resultImgParallel.data[idx] != resultImgSerial.data[idx])
    //         {
    //             printf("conflicts : row %i, col %i -> serial : %i, parallel : %i\n", i, j, resultImgSerial.data[idx], resultImgParallel.data[idx]);
    //         }
    //     }
        
    // }
    
    printf("Serial exec. Time : %li us\n", serialExecTime);
    printf("Parallel exec. Time : %li us\n", parallelExecTime);
    printf("-> speedup : %0.4f\n", SPEED_UP(serialExecTime, parallelExecTime));


    cv::namedWindow("serial", cv::WINDOW_AUTOSIZE);
    cv::imshow("serial", resultImgSerial);

    cv::namedWindow("parallel", cv::WINDOW_AUTOSIZE);
    cv::imshow("parallel", resultImgParallel);
    cv::waitKey(0);

    return 0;

}