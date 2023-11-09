#include "TrackingMofang.h"
#include <vector>

Mat Mofang::Mofang_TBGR(Mat Mofang, std::string color)
{
    Mat Result(Size(Mofang.cols, Mofang.rows), CV_8UC1, Scalar(0));
    if (Mofang.empty() || Result.empty())
    {
        Mat NO(Size(255, 255), CV_8UC1, Scalar(255));
        std::cout << "Fail to open Path, please check." << std::endl;
        return NO;
    }
    int main_color = 0;
    int color_1 = 0;
    int color_2 = 0;
    if (color == "BLUE")
    {
        main_color = 0;
        color_1 = 1;
        color_2 = 2;
        for (int x = 0; x < Mofang.rows; x++)
        {
            for (int y = 0; y < Mofang.cols; y++)
            {
                //BGR
                if (double(Mofang.at<cv::Vec3b>(x, y)[main_color]) / Mofang.at<cv::Vec3b>(x, y)[color_2] >= 1.5 &&
                    Mofang.at<cv::Vec3b>(x, y)[main_color] >= 60)
                {
                    Result.at<uchar>(x, y) = 255;
                }
            }
        }
    }
    else if (color == "GREEN")
    {
        main_color = 1;
        color_1 = 0;
        color_2 = 2;
        for (int x = 0; x < Mofang.rows; x++)
        {
            for (int y = 0; y < Mofang.cols; y++)
            {
                //BGR
                if (double(Mofang.at<cv::Vec3b>(x, y)[main_color]) / Mofang.at<cv::Vec3b>(x, y)[color_1] >= 1.5 &&
                    double(Mofang.at<cv::Vec3b>(x, y)[main_color]) / Mofang.at<cv::Vec3b>(x, y)[color_2] >= 1.5 &&
                    Mofang.at<cv::Vec3b>(x, y)[main_color] >= 60)
                {
                    Result.at<uchar>(x, y) = 255;
                }
            }
        }
    }
    else if (color == "RED")
    {
        main_color = 2;
        color_1 = 0;
        color_2 = 1;
        for (int x = 0; x < Mofang.rows; x++)
        {
            for (int y = 0; y < Mofang.cols; y++)
            {
                //BGR
                if (double(Mofang.at<cv::Vec3b>(x, y)[main_color]) / Mofang.at<cv::Vec3b>(x, y)[color_1] >= 2.0 &&
                    double(Mofang.at<cv::Vec3b>(x, y)[main_color]) / Mofang.at<cv::Vec3b>(x, y)[color_2] >= 2.0 &&
                    Mofang.at<cv::Vec3b>(x, y)[main_color] >= 60 && Mofang.at<cv::Vec3b>(x, y)[color_1] <= 80)
                {
                    Result.at<uchar>(x, y) = 255;
                }
            }
        }
    }
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(Result, contours, hierarchy, 0, 2, Point());
    for (int i = 0; i < contours.size(); i++)
    {
        Rect rect = boundingRect(contours[i]);
        if (rect.area() < 200)
        {
            continue;
        }
        Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
        circle(Mofang, center, 2, Scalar(0, 250, 0), 3);
    }
    drawContours(Mofang, contours, -1, Scalar(255), 2, 8);
    /*imshow("Mofang",Bian_Result);
    imshow("Result",F_Result);*/
    return Mofang;
}

Mat Mofang::Mofang_THSV(Mat Mofang, std::string color)
{
    Mat Result(Size(Mofang.cols, Mofang.rows), CV_8UC1, Scalar(0));
    if (Mofang.empty() || Result.empty())
    {
        Mat NO(Size(255, 255), CV_8UC1, Scalar(255));
        std::cout << "Fail to open Path, please check." << std::endl;
        return NO;
    }
    int main_color = 0;
    int color_1 = 0;
    int color_2 = 0;
    if (color == "BLUE")
    {
        main_color = 0;
        color_1 = 1;
        color_2 = 2;
        for (int x = 0; x < Mofang.rows; x++)
        {
            for (int y = 0; y < Mofang.cols; y++)
            {
                //HSV
                if (Mofang.at<cv::Vec3b>(x, y)[main_color] > 190)
                {
                    Result.at<uchar>(x, y) = 255;
                }
            }
        }
    }
    else if (color == "GREEN")
    {
        main_color = 1;
        color_1 = 0;
        color_2 = 2;
        for (int x = 0; x < Mofang.rows; x++)
        {
            for (int y = 0; y < Mofang.cols; y++)
            {
                //BGR
                if (double(Mofang.at<cv::Vec3b>(x, y)[main_color]) / Mofang.at<cv::Vec3b>(x, y)[color_1] >= 1.5 &&
                    double(Mofang.at<cv::Vec3b>(x, y)[main_color]) / Mofang.at<cv::Vec3b>(x, y)[color_2] >= 1.5 &&
                    Mofang.at<cv::Vec3b>(x, y)[main_color] >= 60)
                {
                    Result.at<uchar>(x, y) = 255;
                }
            }
        }
    }
    else if (color == "RED")
    {
        main_color = 2;
        color_1 = 0;
        color_2 = 1;
        for (int x = 0; x < Mofang.rows; x++)
        {
            for (int y = 0; y < Mofang.cols; y++)
            {
                //BGR
                if (double(Mofang.at<cv::Vec3b>(x, y)[main_color]) / Mofang.at<cv::Vec3b>(x, y)[color_1] >= 2.0 &&
                    double(Mofang.at<cv::Vec3b>(x, y)[main_color]) / Mofang.at<cv::Vec3b>(x, y)[color_2] >= 2.0 &&
                    Mofang.at<cv::Vec3b>(x, y)[main_color] >= 60 && Mofang.at<cv::Vec3b>(x, y)[color_1] <= 80)
                {
                    Result.at<uchar>(x, y) = 255;
                }
            }
        }
    }
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(Result, contours, hierarchy, 0, 2, Point());
    drawContours(Mofang, contours, -1, Scalar(255), 2, 8);
    /*imshow("Mofang",Bian_Result);
    imshow("Result",F_Result);*/
    return Mofang;
}
