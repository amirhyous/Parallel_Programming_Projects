#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
//#include "x86intrin.h"

#include <sys/time.h>
#include <cmath>
#include "unistd.h"
#include "pthread.h"


#include "../lib/primitives.h"

#define IMAGE1_ADDRESS "../img/q3_img01.png"
#define IMAGE2_ADDRESS "../img/q3_img02.png"

#define THREADS_NUM 4

typedef struct timeval timeVal;

cv::Mat img1;
cv::Mat img2;

cv::Mat img_abs_serial(long* execTime)
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

//Parallel:

typedef struct{
	int start_index;
	int end_index;
} in_param_t;

void *img_abs(void *arg)
{
	in_param_t *inp = (in_param_t *) arg;
    
    unsigned char* img1Data = (unsigned char*) img1.data;
    unsigned char* img2Data = (unsigned char*) img2.data;
    unsigned char* resImgData_parallel = new unsigned char[img1.rows * img1.cols];

    for (int row = inp->start_index; row < inp->end_index; row++)
        for (int col = 0; col < img1.cols; col++)
        {
            int idx = row * img1.cols + col;
            resImgData_parallel[idx] = abs(img1Data[idx] - img2Data[idx]);     
        } 
	pthread_exit (NULL);
}

void img_abs_parallel(long* execTime)
{
    timeVal start, end;

    // ----------------------- calculation started ---------------------------/
    gettimeofday(&start, NULL);

    pthread_t th[THREADS_NUM];
	
	in_param_t in_param [THREADS_NUM] =  
		{{0, img1.rows/4}, {img1.rows/4, img1.rows/2},
		 {img1.rows/2, 3 * img1.rows/4}, {3 * img1.rows/4, img1.rows}};

	for (int i = 0; i < THREADS_NUM; i++)
		pthread_create (&th[i], NULL, img_abs, (void *) &in_param[i]);

	for (int i = 0; i < THREADS_NUM; i++)
		pthread_join (th[i], NULL);

    gettimeofday(&end, NULL);
    // ----------------------- calculation finished ---------------------------/
    *execTime = end.tv_usec - start.tv_usec;
}


int main()
{
    const int numOfExec = 10;
    long serialExecTime, parallelExecTime;
    img1 = cv::imread(IMAGE1_ADDRESS);
    img2 = cv::imread(IMAGE2_ADDRESS);

    float speedupAvg = 0.0;
    float speedup = 0.0;

    for(int i = 0; i < numOfExec; i++)
    {
        speedup = 0;
        cv::Mat resultImgSerial = img_abs_serial(&serialExecTime);
        img_abs_parallel(&parallelExecTime);

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