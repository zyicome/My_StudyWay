#include "TrackingMofang.h"

int main()
{
	Mat a = imread("C:/Users/zyb/Desktop/OpenCV_Pictureandfile/MofanBianbie_LIANG.jpg", IMREAD_COLOR);
	cv::VideoCapture video(0);
	if (video.isOpened())
	{
		std::cout << "摄像头打开成功!" << std::endl;
		Mat Mofang_TP(Size(640,480),CV_8UC3);
		Mat Result(Size(640, 480), CV_8UC3);
		Mofang mofang;
		bool quit = false;
		while (!quit)
		{
			video.read(Mofang_TP);
			resize(Mofang_TP, Mofang_TP, Size(1024, 625));
			GaussianBlur(Mofang_TP, Mofang_TP, Size(3, 3), 0, 0);
			//cvtColor(Mofang_TP, Mofang_TP, COLOR_BGR2HSV);
			Result = mofang.Mofang_TBGR(Mofang_TP, "GREEN");
			imshow("Mofang", Mofang_TP);
			if (waitKey(10) == 27)
			{
				quit = true;
				std::cout << "摄像头关闭!" << std::endl;
				video.release();
				destroyAllWindows();
			}
		}
	}
	else
	{
		std::cout << "请检测摄像头配置情况!" << std::endl;
		return -1;
	}
}

