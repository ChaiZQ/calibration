#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <fstream> //文件输入输出流
#include <stdio.h>

using namespace cv;
using namespace std;

//namespace eyeHand{
	bool find(int i,std::vector<int> noise_);
	
	Mat MatrixCombine(Mat upper, Mat lower);

	double vec_norm(Mat a);

	bool check(vector<pair<int,int>> points, pair<int,int> new_points);

	double getDistance(double x,double y,Mat line);

	Mat ransac(int numOfp,vector<double> samples, int k,double threshold_In, double& average);

	void readRobotData(string robotFileName, vector<Mat>&RobotPose, vector<Mat>&RobotPosition,int nFrame,vector<int> failedIndex);

	void saveEyeHand(Mat eyehandTranslation, Mat eyehandRotation, int numSamples);

	void varify(Mat EH_translation, 
		Mat EyeHandRotation,
		vector<Mat> RobotPosition,
		vector<Mat> RobotPose, 
		vector<Mat> cameraPosition,
		vector<Mat> cameraPose,double Varify_X = 0.0,double Varify_Y = 0.0, double Varify_R = 200.0);

	void eyeHandCalibraion(string cameraFileName,string robotFileName,vector<int> failedIndex);
//}
