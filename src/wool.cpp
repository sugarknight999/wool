//
// Created by be on 24-5-27.
//
#include "wool.h"

/**
* @brief  Progress_Img						 图像预处理
* @param  src:crop_img                       输入图像矩阵
* @param  dst                                输出结果
* @return lpt_image				             封闭的图像
* @经过调整，中值滤波的的滤波范围的效果：7 > 5 > 3
*/
Mat Pretreat_Img(Mat src)
{
    Mat image0, image1, image2, image3, lpt_image;									    // 定义预处理中间变量
    adaptiveThreshold(src, image0, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 35, 5);	// 自适应二值化
    //imshow("自适应二值化",image0);
    //medianBlur(image0, image1, 3);														// 中值滤波保证图像的平滑
    //GaussianBlur(image0, image1, cv::Size(5, 5), 3, 3);
    bilateralFilter(image0, image1, 9, 50, 25 / 2);
    //imshow("中值滤波",image1);
    //imshow("高斯滤波", image1);
    //imshow("双边滤波", image1);
    Clear_Interference(image1, image2, 1000);											// 去除干扰区域
    //imshow("去除干扰区域", image2);
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(6, 6));						// 6 * 6核函数
    erode(image2, image3, element);														// 腐蚀
    //imshow("腐蚀图", image3);
    dilate(image3, lpt_image, element);													// 膨胀
    //imshow("膨胀图", lpt_image);
    return lpt_image;
}

/**
* @brief  Draw_Outline						 绘制图像边缘轮廓
* @param  src:lpt_image, dst:process_image   输入图像矩阵(lpt_image是已经泛洪填充后的  process_image是前膨胀结束后的)
* @return progross_image		             封闭的图像
*/
void Draw_Outline(Mat src, Mat dst)
{	Point lpt;
    vector<vector<Point>> contoursPoint;
    //vector<Vec<int, 4>> hierarchy;
    /*
       findContours(image, countours，hierarchy, mode，method, offset);一般使用image,countours，mode，method即可
       image:单通道图像矩阵，可以是灰度图，建议二值图像（最好是Canny/拉普拉斯等边缘检测算子处理后的）
       countours:"vector<vector<Point>>contours"一个双重向量，向量内每个元素保存了一组由连续的point点，每一组point点集就是一个轮廓构成的点的集合的向量
       hierarchy:"vector<Vec<int,4>>hierarchy",，定义了一个“向量内每一个元素包含了四个int型变量”的向量
       mode:定义轮廓的检索模式——RETR_EXTERNAL:内轮廓不重复计算
       method：定义轮廓的近似方法——CHAIN_APPROX_SIMPLE:仅保留拐点信息
    */
    findContours(src, contoursPoint, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    /*
        contoursPoint[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
    */
    Mat Contours = Mat::zeros(src.size(), CV_8UC1);										 // 建立一个lpt_image.size大小的空矩阵
    cout << "【检测到的轮廓数量是：】" << contoursPoint.size() << endl;					 // 打印一下检测到的轮廓的数量
    for (int i = 0; i < contoursPoint.size(); i++)                                       // 对每一个检测到轮廓进行循环
    {
        for (int j = 0; j < contoursPoint[i].size(); j++)
        {
            Point P = Point(contoursPoint[i][j].x, contoursPoint[i][j].y);               // 提取第i个轮廓第j个像素点的x，y至P集合中
            Contours.at<uchar>(P) = 255;												 // 向量contours内保存的所有轮廓点集
        }
    }
    imshow("The Contours Points", Contours);                                             // 得到的点

    for (int i = 0; i < contoursPoint.size(); i++)
    {
        int min_x = 1000, min_j;
        for (int j = 0; j < contoursPoint[i].size(); j++)								 // 遍历此轮廓所有的点，找到最左侧的点
        {
            if (contoursPoint[i][j].x < min_x)
            {
                min_x = contoursPoint[i][j].x;
                min_j = j;
            }
        }
        lpt = Point(contoursPoint[i][min_j].x, contoursPoint[i][min_j].y);			     // lpt：最左侧的点
        cout << "【第" << i << "个轮廓的最左侧定位点坐标为：】" << lpt.x << " " << lpt.y << endl;	 // 打印日志输出最左侧点的坐标
        floodFill(dst, lpt, Scalar(0));										 // 泛洪填充，填充黑色
    }
}

/**
* @brief  Clear_Interference				 清除干扰区域函数
* @param  src                                输入图像矩阵
* @param  dst                                输出结果
* @return min_area                           设定的最小面积清除阈值
*/
void Clear_Interference(Mat src, Mat &dst, double min_area)
{
    // 备份复制
    dst = src.clone();
    vector<vector<Point>> contours;  // 创建轮廓容器
    vector<Vec4i> hierarchy;

    // 寻找轮廓的函数
    // 第四个参数CV_RETR_EXTERNAL，表示寻找最外围轮廓
    // 第五个参数CV_CHAIN_APPROX_NONE，表示保存物体边界上所有连续的轮廓点到contours向量内
    findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point());

    if (!contours.empty() && !hierarchy.empty())
    {
        vector<vector<Point> >::const_iterator itc = contours.begin();
        // 遍历所有轮廓
        while (itc != contours.end())
        {
            // 定位当前轮廓所在位置
            Rect rect = boundingRect(Mat(*itc));
            // contourArea函数计算连通区面积
            double area = contourArea(*itc);
            // 若面积小于设置的阈值
            if (area < min_area)
            {
                // 遍历轮廓所在位置所有像素点
                for (int i = rect.y; i < rect.y + rect.height; i++)
                {
                    uchar *output_data = dst.ptr<uchar>(i);
                    for (int j = rect.x; j < rect.x + rect.width; j++)
                    {
                        // 将连通区的值置0
                        if (output_data[j] == 0)
                        {
                            output_data[j] = 255;
                        }
                    }
                }
            }
            itc++;
        }
    }
}

/**
* @brief  Filter_Outline					 图像轮廓筛选(面积筛选，长宽比筛选）
* @param  src：process_image                 输入图像矩阵
* @筛选剔除掉面积小于2000的轮廓或者长宽比小于7的轮廓
*/
vector<vector<Point>> Filter_Outline(Mat src)
{
    /*----------------------------------------------------面积筛选-----------------------------------------------------*/
    vector<vector<Point>> contours;
    //vector< Vec<int, 4>> hierarchy1;
    findContours(src, contours, RETR_TREE, CHAIN_APPROX_NONE, Point());// 对二值图像再进行一次轮廓提取，剔除异常轮廓
    cout << "【面积筛选前总共轮廓个数为】：" << (int)contours.size() << endl;
    for (int i = 0; i < (int)contours.size(); i++)
    {
        double Area = contourArea(contours[i]);
        cout << "【用轮廓面积计算函数计算出来的第" << i << "个轮廓的面积为：" << Area << endl;
    }
    vector <vector<Point>>::iterator iter1 = contours.begin();
    for (; iter1 != contours.end();)
    {
        double Area = contourArea(*iter1);
        if (Area < 2000)                                                                // 筛选剔除掉面积小于2000的轮廓
        {
            iter1 = contours.erase(iter1);
        }
        else
        {
            ++iter1;
        }
    }

    cout << "【面积筛选后总共轮廓个数为：" << (int)contours.size() << endl;
    for (int i = 0; i < (int)contours.size(); i++)
    {
        double Area = contourArea(contours[i]);
        cout << "【用轮廓面积计算函数计算出来的第" << i << "个轮廓的面积为：】" << Area << endl;
    }
    /*--------------------------------------------------长宽比筛选-----------------------------------------------------*/
    for (int j = 0; j < (int)contours.size(); ++j)										// 遍历所有轮廓
    {
        RotatedRect rotateRect = minAreaRect(contours[j]);								// 轮廓最小外接矩形
        if (rotateRect.size.height > rotateRect.size.width)
        {
            cout << "【用最小矩形函数计算出来的第" << j << "个轮廓的长宽比为：" << (double)(rotateRect.size.height / rotateRect.size.width) << endl;
        }
        else
        {
            cout << "【用最小矩形函数计算出来的第" << j << "个轮廓的长宽比为：" << (double)(rotateRect.size.width / rotateRect.size.height) << endl;
        }
    }

    vector <vector<Point>>::iterator iter2 = contours.begin();
    for (; iter2 != contours.end();)
    {
        RotatedRect rotateRect = minAreaRect(*iter2);
        double min_Areareat;															//定义长宽比
        if (rotateRect.size.height > rotateRect.size.width)
        {
            min_Areareat = rotateRect.size.height / rotateRect.size.width;
        }
        else
        {
            min_Areareat = rotateRect.size.width / rotateRect.size.height;
        }
        if (min_Areareat < 6)                                                         // 筛选掉长宽比小于7的纤维图像
        {	iter2 = contours.erase(iter2);	}
        else
        {	++iter2;  }
    }

    for (int j = 0; j < (int)contours.size(); ++j)									  // 遍历所有轮廓
    {
        RotatedRect rotateRect = minAreaRect(contours[j]);							  // 轮廓最小外接矩形
        double rotateRect_h = rotateRect.size.height;
        double rotateRect_w = rotateRect.size.width;
        if (rotateRect_h > rotateRect_w)
        {
            cout << "【用最小矩形函数筛选出来的第" << j << "个轮廓的长宽比为：" << (double)(rotateRect_h / rotateRect_w) << endl;
        }
        else
        {
            cout << "【用最小矩形函数筛选出来的第" << j << "个轮廓的长宽比为：" << (double)(rotateRect_w / rotateRect_h) << endl;
        }
    }

    Mat Contours = Mat::zeros(src.size(), CV_8UC1);							             // 建立一个lpt_image.size大小的空矩阵
    for (int i = 0; i < contours.size(); i++)                                       // 对每一个检测到轮廓进行循环
    {
        for (int j = 0; j < contours[i].size(); j++)
        {
            Point P = Point(contours[i][j].x, contours[i][j].y);               // 提取第i个轮廓第j个像素点的x，y至P集合中
            Contours.at<uchar>(P) = 255;												 // 向量contours内保存的所有轮廓点集
        }
    }
    imshow("筛选之后的图片", Contours);                                             // 打印过滤之后的图像
    return contours;
    //Mat imageContours = crop_img.clone();
    //cvtColor(imageContours, imageContours, COLOR_GRAY2RGB);						   // 将二值化图片转化为彩色图
    //drawContours(imageContours, contours, -1, Scalar(0, 0, 255), 3);				   // 在初始图上画出所有筛选后的轮廓，-1 表示所有轮廓
    //imshow("imageContours", imageContours);
}

/**
* @brief  Calculate_Wool_Dim                 计算中心点并进行直径的计算
* @param  src:crop_img, dst:process_image    输入长边图像向量
* @return show_dim		                     输出结果
*/
Mat Calculate_Wool_Dim(Mat src, Mat dst, vector<vector<Point>> contours)
{
    /*---------------------------------求所有轮廓的最小外接矩形与最小外接矩形的中心点-------------------------------*/

    vector<double> longline;															// 定义矩形的长边
    vector<Point> tcpv;																	// 中心坐标集合
    Mat show_dim = src.clone();															// show_dim显示直径
    cvtColor(show_dim, show_dim, COLOR_GRAY2RGB);										// 转化为RGB图像
    for (int c = 0; c < contours.size(); ++c)											// 遍历所有轮廓
    {
        RotatedRect rotateRect = minAreaRect(contours[c]);								// 轮廓最小外接矩形
        Point tcp = Point(rotateRect.center.x, rotateRect.center.y);					// 最小外接矩形的中心点坐标
        tcpv.push_back(tcp);															// 在tcpv最后添加该中心坐标点
        Point2f rect_points[4];															// 定义外接矩形的四个点
        rotateRect.points(rect_points);													// 外接矩形四个顶点坐标放入rect_points
        for (int i = 0; i < 4; i++)														// 四条边进行四次循环
        {
            line(show_dim, rect_points[i], rect_points[(i + 1) % 4], Scalar(0, 255, 255), 2);	// line四条边
        }

        if (rotateRect.size.height > rotateRect.size.width)								// 得到矩形的长边
        {
            longline.push_back(rotateRect.size.height);									// 在longline中添加长边
        }
        else
        {
            longline.push_back(rotateRect.size.width);									// 在longline中添加长边
        }
    }
    /*--------------------------------------------计算中心点并进行直径的计算----------------------------------------*/

    vector<Mat> Contours_choose;
    // vector<Mat> global_Contours_choose;
    Mat Contours_choose_in_one = Mat::zeros(dst.size(), CV_8UC1);						// 建立process_image大小的空矩阵Contours_choose_in_one，所有主干的集合
    vector<vector<Point>> longline_contours;
    for (int i = 0; i < contours.size(); i++)											// 遍历所有轮廓
    {
        Mat Mempty = Mat::zeros(dst.size(), CV_8UC1);									// 建立process_image大小的空矩阵Mempty
        vector<Point> longlinecontours;													// 矩形的长边点的集合，筛选之后的
        for (int j = 0; j < contours[i].size(); j++)
        {
            double dist2tcp = sqrt(pow(contours[i][j].x - tcpv[i].x, 2) + pow(contours[i][j].y - tcpv[i].y, 2));	// 计算每个轮廓点和中心的距离
            if (dist2tcp < (0.3 * longline[i]))																// 如果小于0.3的长边则为主干
            {
                Point p_coutours = Point(contours[i][j].x, contours[i][j].y);
                longlinecontours.push_back(p_coutours);									// push_back函数向向量longlinecontours内保存长边的轮廓点集，共contours[i].sizee()个（一维）
                Mempty.at<uchar>(p_coutours) = 255;
                Contours_choose_in_one.at<uchar>(p_coutours) = 255;
            }
        }
        Contours_choose.push_back(Mempty);								          		// push_back函数向向量Contours_choose内保存的该主干轮廓点集，共contours.size()个
        // global_Contours_choose = Contours_choose;
        longline_contours.push_back(longlinecontours);									// push_back函数向向量longline_contours内保存长边的轮廓点集，共contours.sizee()个（二维）
        string str = "Point of Contours_choose" + to_string(i);
        //imshow(str,Contours_choose[i]);
    }
    imshow("Point of Contours_choose", Contours_choose_in_one);						// show一下所有的主干

    vector<vector<double>> mindist;														// 平均距离
    vector<vector<Point>> all_longline_contours;
    cout << "222222222222222222222" << endl;
    mindist = Calculate_Dim(Contours_choose, all_longline_contours);
    cout << "33333333333333333333" << endl;
    for (int i = 0; i < contours.size(); ++i)
    {
        stringstream sdtream, sstream;

        sdtream << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(1) << mindist[2][i];	// 输出向右对齐的小数点后1位的浮点数
        sstream << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(2) << mindist[3][i];	// 输出向右对齐的小数点后2位的浮点数
        putText(show_dim, "D_um: " + sdtream.str() + " um", Point(tcpv[i].x - 140, tcpv[i].y + 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100, 100, 250), 2, 2);
        putText(show_dim, "St_D: " + sstream.str() + " um", Point(tcpv[i].x - 140, tcpv[i].y + 50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250, 100, 100), 2, 2);
        circle(show_dim, tcpv[i], 5, Scalar(0, 0, 0), -1);
    }

    //vector<double> average_mindist;														// 平均距离
    //vector<double> variance_mindist;													// 距离方差
    //vector<vector<Point>> all_longline_contours;
    //average_mindist, variance_mindist = Calculate_Dim(Contours_choose, all_longline_contours);

    //for (int i = 0; i < contours.size(); ++i)
    //{
    //	stringstream sdtream, sstream;
    //	sstream << setiosflags(ios::fixed) << setprecision(2) << variance_mindist[i];
    //	sdtream << setiosflags(ios::fixed) << setprecision(1) << average_mindist[i];	// 输出对齐的小数点后1位的浮点数


    //	putText(show_dim, "D： " + sdtream.str() + " um", Point(tcpv[i].x - 140, tcpv[i].y + 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 100, 0), 2, 2);
    //	putText(show_dim, "σ：" + sstream.str() + " um", Point(tcpv[i].x - 140, tcpv[i].y + 50), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(100, 100, 100), 2, 2);
    //	circle(show_dim, tcpv[i], 5, Scalar(0, 0, 0), -1);
    //}
    Mat imageC = src.clone();
    // 将二值化图片转化为彩色图，并在初始图上画出轮廓。
    cvtColor(imageC, imageC, COLOR_GRAY2RGB);
    drawContours(imageC, all_longline_contours, -1, Scalar(0, 0, 255), 3);

    drawContours(show_dim, contours, -1, Scalar(0, 0, 255), 3);

    imshow("显示长边轮廓", imageC);
    return show_dim;
}

/**
* @brief  Calculate_Dim                      直径计算函数
* @param  src                                输入长边图像向量
* @return average_mindist                    输出结果
*/
vector<vector<double>> Calculate_Dim(vector<Mat> Contours_choose, vector<vector<Point>> &all_longline_contours)
{
    vector<vector<double>> mindist;
    vector<double> average_mindist;
    vector<double> variance_mindist;
    for (int i = 0; i < Contours_choose.size(); i++)
    {
        vector<vector<Point>> longline_contours;										// 生成一个point型的二维数组变量contours
        //vector<Vec4i> hierarchy2;														// 生成一个Vec4i型的一维数组hierarchy,里面放的是4个int的整数
        findContours(Contours_choose[i], longline_contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
        for (int i = 0; i < (int)longline_contours.size(); i++)
        {
            vector<Point> longline_contours_i;
            for (int j = 0; j < (int)longline_contours[i].size(); j++)
            {
                longline_contours_i.push_back(longline_contours[i][j]);
            }
            all_longline_contours.push_back(longline_contours_i);
        }
        vector<int> pt0x, pt0y, pt1x, pt1y, pt0px, pt0py;
        for (int j = 0; j < longline_contours[0].size(); j++)
        {
            int x = longline_contours[0][j].x;
            int y = longline_contours[0][j].y;
            pt0x.push_back(x);															// 第0条长边包含点的x坐标——pt0x
            pt0y.push_back(y);															// 第0条长边包含点的y坐标——pt0y
        }
        for (int j = 0; j < longline_contours[1].size(); j++)
        {
            int x = longline_contours[1][j].x;
            int y = longline_contours[1][j].y;
            pt1x.push_back(x);															// 第1条长边包含点的x坐标——pt1x
            pt1y.push_back(y);															// 第1条长边包含点的y坐标——pt1y
        }

        double dist, all_mindist = 0;													// 做循环，第0条长边的每个点与第1条长边的每个点求最短距离，求和，最后求平均
        vector<double> col_mindist;														// 记录第0条长边的每个点与第1条长边的每个点求最短距离
        for (int i = 0; i < pt0x.size(); i++)											// 第一层循环
        {
            double mindist = 10000;
            int x, y;
            for (int j = 0; j < pt1x.size(); j++)										// 第二层循环
            {
                dist = sqrt(pow(pt0x[i] - pt1x[j], 2) + pow(pt0y[i] - pt1y[j], 2));		// 第二层循环是求第0条长边的点与第1条长边的每个点的最小距离
                if (dist < mindist)
                {
                    mindist = dist;
                    x = pt1x[j];
                    y = pt1y[j];
                }
            }
            all_mindist = all_mindist + mindist;										// 最小距离求和
            col_mindist.push_back(mindist);
            Point pt0, pt1;
            pt0.x = pt0x[i], pt0.y = pt0y[i];
            pt1.x = x, pt1.y = y;
        }
        double average_dist = (all_mindist) / (pt0x.size());							// 求平均
        cout << all_mindist / pt0x.size() << endl;
        average_mindist.push_back(average_dist);										// 最大的for循环结束，将该幅图的平均距离


        double varia, all_varia = 0;													// 循环-第0条长边每个点与对应的第1条长边每个点的最短距离的标准差，查看纤维直径的波动标准
        for (int i = 0; i < col_mindist.size(); i++)										// 求每个距离数据减去平均值后的平方之和
        {
            varia = (col_mindist[i] - average_dist)*(col_mindist[i] - average_dist);
            all_varia = all_varia + varia;
        }
        double vaira_dist = sqrt(all_varia / pt0x.size());								// 求标准差
        cout << sqrt(all_varia / pt0x.size()) << endl;
        variance_mindist.push_back(vaira_dist);
        mindist.push_back(average_mindist);
        mindist.push_back(variance_mindist);

    }																					// 最大的for循环结束，得到将该幅图的平均距离和直径标准差


    return mindist;
}