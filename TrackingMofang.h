#pragma once
#include <opencv.hpp>
#include <iostream>

using namespace cv;

class Mofang
{
public:
	Mat Mofang_TBGR(Mat Mofang, std::string color);
	Mat Mofang_THSV(Mat Mofang, std::string color);
private:
};
