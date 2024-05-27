//
// Created by be on 24-5-27.
//
#include "opencv2/opencv.hpp"
#ifndef WOOL_WOOL_H
#define WOOL_WOOL_H

using namespace std;
using namespace cv;

// 清除干扰区域函数
void Clear_Interference(Mat src, Mat &dst, double min_area);

// 绘制图像的边缘轮廓
void Draw_Outline(Mat src, Mat dst);

Mat Pretreat_Img(Mat src);

// 对二值图像处理，过滤异常轮廓，选出正常可测量轮廓
vector<vector<Point>> Filter_Outline(Mat src);

Mat Calculate_Wool_Dim(Mat src, Mat dst, vector<vector<Point>> contours);

vector<vector<double>> Calculate_Dim(vector<Mat> Contours_choose, vector<vector<Point>> &all_longline_contours);

#endif //WOOL_WOOL_H
