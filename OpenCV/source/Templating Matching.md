# 模板匹配

## 模板匹配介绍

<img src="https://github.com/Einstellung/OpenCV_learning/blob/master/OpenCV/images/Templating%20Matching/1.png?raw=true"  width = "400"> 


- 模板匹配就是在整个图像区域发现与给定子图像匹配的小块区域。
- 所以模板匹配首先需要一个模板图像T（给定的子图像）
- 另外需要一个待检测的图像-源图像S
- 工作方法，在带检测图像上，从左到右，从上向下计算模板图像与重叠子图像的匹配度，匹配程度越大，两者相同的可能性越大。

## 模板匹配用到的算法
OpenCV中包含了六种模板匹配的算法：

<img src="https://github.com/Einstellung/OpenCV_learning/blob/master/OpenCV/images/Templating%20Matching/2.png?raw=true"  width = "500"> 


<img src="https://github.com/Einstellung/OpenCV_learning/blob/master/OpenCV/images/Templating%20Matching/3.png?raw=true"  width = "400"> 


<img src="https://github.com/Einstellung/OpenCV_learning/blob/master/OpenCV/images/Templating%20Matching/4.png?raw=true"  width = "700"> 
![image]()

## API介绍

```c++
matchTemplate(

InputArray image,// 源图像，必须是8-bit或者32-bit浮点数图像

InputArray templ,// 模板图像，类型与输入图像一致

OutputArray result,// 输出结果，必须是单通道32位浮点数，假设源图像WxH,模板图像wxh,
	             则结果必须为W-w+1, H-h+1的大小。(w是宽，h是高)
int method,//使用的匹配方法，一般推荐使用归一化的方法

InputArray mask=noArray()//(optional)
)
```

![image](https://github.com/Einstellung/OpenCV_learning/blob/master/OpenCV/images/Templating%20Matching/5.png?raw=true)

## 代码演示

```c++
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;

Mat src, temp, dst;

int match_method = CV_TM_SQDIFF;
int max_track = 5;

void Match_Demo(int, void*);

int main(int argc, char** argv) {
	src = imread("D:/temp/6.bmp");
	temp = imread("D:/temp/temp.png");
	if (!src.data || !temp.data) {
		printf("could not load image...\n");
		return -1;
	}

	namedWindow("input image", CV_WINDOW_NORMAL);
	namedWindow("output image", CV_WINDOW_NORMAL);
	namedWindow("template match-demo", CV_WINDOW_NORMAL);
	imshow("input image", src);
	const char* trackbar_title = "Match Algo Type";
	createTrackbar(trackbar_title, "output image", &match_method, max_track, Match_Demo);
	Match_Demo(0, 0);  //先调用一下，保证初始值不为空

	waitKey(0);
	return 0;
}

void Match_Demo(int, void*)
{
	int width = src.cols - temp.cols + 1;
	int height = src.rows - temp.rows + 1;
	Mat result(width, height, CV_32FC1);  //必须是32位浮点数

	matchTemplate(src, temp, result, match_method, Mat()); //到时候从trackbar上面获取match_method

	//下面是归一化，把结果变成0到1之间
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	//下面要找出模板匹配最大值最小值的位置，也就是和哪个位置匹配
	Point minLoc; // 找出最小值的位置
	Point maxLoc; // 找出最大值的位置
	double min, max;
	//OpenCV提供API来找出最大最小值的位置
	minMaxLoc(result, &min, &max, &minLoc, &maxLoc, Mat());

	//用矩形框来把最大最小值来标出来
	src.copyTo(dst); //在dst上面进行绘制工作
	Point temLoc;
	if (match_method == CV_TM_SQDIFF ||  match_method == CV_TM_SQDIFF_NORMED)  //对于这两种方法而言，应该是最小值来匹配。其他方法是最大值来匹配
	{
		temLoc = minLoc;
	}
	else {
		temLoc = maxLoc;
	}

	rectangle(dst, Rect(temLoc.x, temLoc.y, temp.cols, temp.rows), Scalar(0, 0, 255), 2, 8);  //在最终输出图像上绘制一个矩形
	rectangle(result, Rect(temLoc.x, temLoc.y, temp.cols, temp.rows), Scalar(0, 0, 255), 2, 8); //在result结果上面输出一个矩形

	imshow("output image", result);
	imshow("template match-demo", dst);
}
```
