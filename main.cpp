#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <fstream>
#include <iomanip>

using namespace std;
using namespace cv;

int main()
{
	ifstream ifs;
	ifs.open("MyData_path.txt", ios::in);
	if (!ifs.is_open())
	{
		return 0;
	}
	string img_path;
	vector<Point2f> img_corners_p;
	vector<vector<Point2f>> img_corners_ps;
	while (getline(ifs,img_path))
	{
		Mat img_init = imread(img_path, IMREAD_GRAYSCALE);
		resize(img_init, img_init, Size(1024, 628));
		//imshow("img_init",img_init);
		waitKey(0);
		Mat img_f;
		bool is_findcorners = findChessboardCorners(img_init, Size(4,6), img_corners_p);
		if (is_findcorners == 0)
		{
			cout << "未找到角点，请切换图像!" << endl;
			//return 0;
		}
		else
		{
			find4QuadCornerSubpix(img_init, img_corners_p, Size(5,5));
			drawChessboardCorners(img_init, Size(4, 6), img_corners_p, true);
			//imshow("img_init", img_init);
			img_corners_ps.push_back(img_corners_p);
		}
	}
	ifs.close();
	vector<Point3f> world_p;
	vector<vector<Point3f>> world_ps;
	float BroadSize_x = 10;
	float BroadSize_y = 10;
	for (int epoch = 0; epoch < img_corners_ps.size(); epoch++)
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 6; j++)
			{
				float w_x = BroadSize_x * i;
				float w_y = BroadSize_y * j;
				float w_z = 0;
				Point3f w_p = Point3f(w_x, w_y, w_z);
				world_p.push_back(w_p);
			}
		}
		world_ps.push_back(world_p);
		world_p.clear();
	}
	Mat camera_in = Mat(3, 3, CV_32FC1, Scalar::all(0));
	Mat camera_ji = Mat(1, 5, CV_32FC1, Scalar::all(0));
	vector<Mat> rvecs;
	vector<Mat> tvecs;
	/*cout << world_ps.size() << endl;
	cout << world_ps[0].size() << endl;
	cout << world_ps[1].size() << endl;
	cout << world_ps[2].size() << endl;
	cout << world_ps[3].size() << endl;
	cout << world_ps[4].size() << endl;
	cout << world_ps[5].size() << endl;
	cout << world_ps[6].size() << endl;
	cout << img_corners_ps.size() << endl;
	cout << img_corners_ps[0].size() << endl;*/

	calibrateCamera(world_ps, img_corners_ps, Size(1024, 628), camera_in, camera_ji, rvecs, tvecs);
	cout << "camera_in:" << endl << camera_in << endl;
	cout << "camera_ji:" << endl << camera_ji << endl;
	ofstream ofs;
	ofs.open("camera_data.txt", ios::out);
	if (!ofs.is_open())
	{
		cout << "保存文件打开失败，相机参数未保存!" << endl;
		return 0;
	}
	ofs << "camera_in:" << endl;
	ofs << camera_in << endl;
	ofs << "camera_ji:" << endl;
	ofs << camera_ji << endl;
	ofs.close();
	vector<Point3f> object_ps;
	object_ps.push_back(Point3f(0, 0, 0));
	object_ps.push_back(Point3f(10, 0, 0));
	object_ps.push_back(Point3f(10, 10, 0));
	object_ps.push_back(Point3f(0, 10, 0));
	vector<Point2f> image_ps;
	image_ps.push_back(Point2f(100, 150));
	image_ps.push_back(Point2f(300, 150));
	image_ps.push_back(Point2f(300, 300));
	image_ps.push_back(Point2f(100, 300));
	Mat rvec;
	Mat tvec;
	Mat RotationMatrix;
	solvePnP(object_ps, image_ps, camera_in, camera_ji, rvec, tvec);
	Rodrigues(rvec, RotationMatrix);
	double distance;
	distance = sqrt(pow(tvec.at<double>(0, 0), 2) + pow(tvec.at<double>(1, 0), 2) + pow(tvec.at<double>(2, 0), 2)) / 10;
	double angle;
	Point3f camera_p = Point3f(1, 0, 0);
	/*Mat tvec_t = tvec.t();
	cout << "tvec_t:" << endl;
	cout << tvec_t << endl;*/
	Point3f object_image_p = Point3f(tvec);
	double a = fabs(camera_p.x * object_image_p.x
		+ camera_p.y * object_image_p.y
		+ camera_p.z * object_image_p.z);
	double b = norm(object_image_p);
	double c = norm(camera_p);
	angle = acos(a / (b * c)) * (180 / 3.1415926);
	cout << "rvec" << ":" << endl;
	cout << rvec << endl;
	cout << "tvec" << ":" << endl;
	cout << tvec << endl;
	cout << "RotationMatrix:" << endl;
	cout << RotationMatrix << endl;
	cout << "物体距离相机的距离为：" << endl;
	cout << distance << endl;
	cout << "a:" << endl;
	cout << a << endl;
	cout << "b:" << endl;
	cout << b << endl;
	cout << "c:" << endl;
	cout << c << endl;
	cout << "物体与相机的夹角为：" << endl;
	cout << angle << endl;
 	return 0;
}
