#include "opencv2/opencv.hpp"
#include "wool.h"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    if (argc != 2){
        cout << "请重新输入图像路径" << endl;
        return 1;
    }
    string img_dir = argv[1];
    Mat image_read = imread(img_dir);												    // 将图片转化为Mat格式
	if (image_read.empty())
	{
		cout << "当前图像读取失败" << endl;
		return 1;
	}
	imshow("Source", image_read);													    // 显示原图像
	Mat image_resize;
	Size dsize = Size(660, 500);
	resize(image_read, image_resize, dsize, 0, 0, INTER_AREA);                          // 缩小图像成660×500
	Rect m_select = Rect(10, 10, 650, 490);                                             // 裁剪图像边缘
	Mat crop_img = image_resize(m_select);												// 裁剪图像边缘									
	cvtColor(crop_img, crop_img, COLOR_BGR2GRAY);		                                // 转化为灰度图，不然会报错
	imshow("灰度转化", crop_img);		
	Mat lpt_image, process_image;
	lpt_image = Pretreat_Img(crop_img);													// 获取封闭图像
	process_image = lpt_image.clone();													// 保存膨胀后的图片
	floodFill(lpt_image, Point(0, 0), Scalar(0));                                       // 填充样本外部
	imshow("填充图",lpt_image);
	Draw_Outline(lpt_image, process_image);												// 绘制图像轮廓边缘并进行泛洪填充
	//imshow("floodFill", process_image);
	bitwise_not(process_image, process_image);                                          // 反转图像像素值
	//imshow("预处理后",process_image);
	vector<vector<Point>> contours;
	contours = Filter_Outline(process_image);											// 剔除异常图像，并选出正常轮廓
	Mat show_dim = Calculate_Wool_Dim(crop_img, process_image, contours);
	imshow("显示直径", show_dim);
	waitKey(0);
}


