#include "../include/eyeHand.h"

bool find(int i,std::vector<int> noise_)
{   
	int j = 0;
	while(j < noise_.size() )
	{
		if( i == noise_[j])
			return true;
		j++;
	}
	return false;
};

/*----------------------------
 * 功能 : 矩阵拼接
 *----------------------------
 * 函数 : MatrixCombine
 * 返回 : 拼接后的矩阵
 *
 * 参数 : upper		        [in]	待拼接的上部矩阵
 * 参数 : lower   	        [in]	待拼接的下部矩阵
 * 参数 : combinedMat   	[in]	拼接后的矩阵
 */
Mat MatrixCombine(Mat upper, Mat lower)
{   
	int c_new;
	int r_new;
	c_new = upper.cols ;				//列数不变
	r_new = upper.rows + lower.rows;    //行数相加
	Mat combinedMat(Size(c_new,r_new),CV_64FC1); 
	for(int i = 0; i < r_new ;i++)
	{	
		double* dataCombined = combinedMat.ptr<double>(i);

		if(i < (upper.rows))
		{
			double* data_upper = upper.ptr<double>(i);
			for(int j = 0;j < c_new;j ++)
			{   
				dataCombined[j] = data_upper[j];
			}
		}
		else 
		{
			double* data_lower = lower.ptr<double>(i-(upper.rows));
			for(int j = 0;j < c_new;j ++)
			{   
				dataCombined[j] = data_lower [j];
			}
		}
	}
	return combinedMat;
}

/*----------------------------
 * 功能 : 求3元向量长度
 *----------------------------
 * 函数 : vec_norm
 * 返回 : 输入向量长度
 *
 * 参数 : a		        [in]	输入向量
 * 参数 :    	        [out]	向量长度
 */
double vec_norm(Mat a)
{
	return sqrt( a.at<double>(0)*a.at<double>(0) + a.at<double>(1) * a.at<double>(1) + a.at<double>(2) * a.at<double>(2));
}

/*----------------------------
 * 功能 : 判断生成的随机数点，是否重复
 *----------------------------
 * 函数 : check
 * 返回 : bool，true表示重复，false表示不重复
 *
 * 参数 : index     	   [in]	已生成的点下标，vector
 * 参数 : k         	   [in]	需要比较的点
  */
bool check(vector<pair<int,int>> points, pair<int,int> new_points)
{
	for(int i = 0; i < points.size(); i++)
	{
		if( new_points.first == points[i].first &&  new_points.second == points[i].second)
			return true;
		else if(new_points.second == points[i].first &&  new_points.first == points[i].second)
			return true;
	}
	return false;
}
/*----------------------------
 * 功能 : 输入样本点，返回拟合直线参数
 *----------------------------
 * 函数 : getLine
 * 返回 : 拟合直线参数：a0 a1
 *
 * 参数 : samplePoints		[in]	样本，vector
 * 参数 : sampleWeights		[in]	样本权重，vector
 * 参数 : Mat(2*1)	        [out]	a0 a1,存储在Mat中
 */
template<class T>
Mat getLine(vector<T> samplePoints,vector<T> sampleWeights = vector<T>(1) )
{
	Mat left = (Mat_<T>(2,2) << 0, 0, 0, 0);
	Mat right = (Mat_<T>(2,1) << 0, 0);

    if(sampleWeights.size() == 1)
		sampleWeights = vector<T>(samplePoints.size(),1);

	for(int i = 0; i < samplePoints.size();i ++)
	{
		left.at<T>(0,0) += sampleWeights[i];
		left.at<T>(0,1) += (sampleWeights[i]) * i;
		left.at<T>(1,0) += (sampleWeights[i]) * i;
		left.at<T>(1,1) += (sampleWeights[i]) * i * i;
		right.at<T>(0,0) += (sampleWeights[i]) * (samplePoints[i]);
		right.at<T>(1,0) += (sampleWeights[i]) * (samplePoints[i]) * i;
		
	}
	return left.inv() * right;
}

/*----------------------------
 * 功能 : 计算点到直线距离
 *----------------------------
 * 函数 : getDistance
 * 返回 : 点到直线距离
 *
 * 参数 : y		           [in]	点y坐标
 * 参数 : x		           [in]	点x坐标（vector下标）
 * 参数 : line	           [in]	直线参数（Mat）
 * 参数 : distance	       [out]点到直线距离
 */
double getDistance(double x,double y,Mat line)
{
	return abs(line.at<double>(1) * x - y + line.at<double>(0)) / sqrt(line.at<double>(1) * line.at<double>(1) +1);
}
/*----------------------------
 * 功能 : Ransac
 *----------------------------
 * 函数 : ransac
 * 返回 : 拟合直线参数
 *
 * 参数 : numOfp		   [in]	观测样本点数
 * 参数 : samples		   [in]	样本vector
 * 参数 : k  	           [in]	迭代次数
 * 参数 : threshold_In	   [in] 判断为内点的阈值
 * 参数 : average	       [in] 内点取值均值，引用
 * 参数 : Result	       [in] 拟合直线参数（Mat）
 */

Mat ransac(int numOfp,vector<double> samples, int k,double threshold_In, double& average)
{
	RNG rng;						                     //随机数生成器
	vector<double> weights(numOfp,1);
//	double threshold_In = 1;							//判断为内点的距离阈值
//	int k = 200;										//迭代次数 
	int num_ofInliners = 0;   						    //内点数目
	Mat Result = (Mat_<double>(2,1) << 0, 0);			//迭代结果直线

	if (samples.size() < 2)
	{
		cout<<"Too little samples!"<<endl;
		return Result;
	}

	vector<pair<int,int>> randomP_array;         //随机点对容器 
	vector<int> p1_array;                       //随机选取点容器 1
	vector<int> p2_array;                       //随机选取点容器 2

	for(int i = 0; i < k; i++)
	{
		// 随机选取两点，计算方程
		int p1 = 0;
		int p2 = 0;
		pair<int,int> p(0,0);
 		while(p1 == p2 || check(randomP_array,p))
 		{
 			p1 = rng.uniform(0,numOfp);  //随机选取两个点（下标）
 			p2 = rng.uniform(0,numOfp); 
 		}
		p.first = p1;
		p.second = p2;
	    randomP_array.push_back(p);
		vector<double> linePara;
		vector<double> lineWeight;
		linePara.push_back(samples[p1]);
		linePara.push_back(samples[p2]);
		lineWeight.push_back(1);
		lineWeight.push_back(1); 
		Mat x = (Mat_<double>(2,1) << 0, 0);
		x = getLine(linePara,lineWeight);

		// 计算点到直线距离，判断是否为内点
		vector<double> inliners;          //内点容器
		for(int j = 0; j < numOfp; j++)
		{
			if(getDistance(j,samples[j],x) < threshold_In)
				inliners.push_back(samples[j]);
		}
		// 根据内点重新估计模型
		Mat lineTemp = getLine(inliners);
		if(inliners.size() > num_ofInliners)
		{ 
			Result = lineTemp;
			num_ofInliners = inliners.size();
			//计算内点取值均值
			double sum = 0;
			for (vector<double>::iterator iter = inliners.begin();iter != inliners.end();iter++)  
			{  
				sum += *iter;  
			} 
			average = sum / inliners.size();
			//cout<<Result<<endl;
			//cout<<num_ofInliners<<endl;
		}
	}
   return Result;
}

void readRobotData(string robotFileName, vector<Mat>&RobotPose, vector<Mat>&RobotPosition,int nFrame,vector<int> failedIndex)
{
	RobotPose.clear();
	RobotPosition.clear();
	ifstream fin(robotFileName); 
	const int LINE_LENGTH = 100; 
	char str[LINE_LENGTH];  
	int k = 2;					//读取起始地址  p and ( occupies the 1st and 2nd places
	int poseLength = 0;
	int valid_pose = 0;
	while(fin.getline(str,LINE_LENGTH))
	{   
		if(find(valid_pose,failedIndex))
		{
			valid_pose ++;
			continue;
		}
		Mat RobotPose_data(Size(1,3),CV_64FC1); 
		Mat RobotPosition_data(Size(1,3),CV_64FC1); 
		k = 2;
	    poseLength = 0;
		double* pose = new double[6];    //位姿数组
		for(int i = 0;i < 6; i++ )
		{	
			char buffer[100];
			while(str[k] != ',')
			{  
				buffer[k-2-poseLength] = str[k];
				k++;
				if(str[k] == ']')
					break;
			}
			poseLength = k-1;
			switch (i)
			{	case 0: 
				RobotPosition_data.at<double>(0) = atof(buffer) * 1000.0;  //注意机器人输出的位置，单位是m
				break;
				case 1:
				RobotPosition_data.at<double>(1) = atof(buffer) * 1000.0;
				break;
				case 2:
				RobotPosition_data.at<double>(2) = atof(buffer) * 1000.0;
				break;
				case 3:
				RobotPose_data.at<double>(0) = atof(buffer);
				break;
				case 4:
				RobotPose_data.at<double>(1) = atof(buffer);
				break;
				case 5:
				RobotPose_data.at<double>(2) = atof(buffer);
				break;
			}
			k++;
		}
		RobotPosition.push_back(RobotPosition_data);
		RobotPose.push_back(RobotPose_data);
		delete[] pose;
		valid_pose ++;
		if(RobotPose.size() == nFrame - failedIndex.size())
			break;
	}
}

void eyeHandCalibraion(string cameraFileName,string robotFileName,vector<int> failedIndex)
{
	FileStorage fs(cameraFileName,FileStorage::READ);
	int NumOfImg = 0;
	Mat extrinsic_parameters;
	fs["nframes"] >> NumOfImg;
	fs["extrinsic_parameters"] >> extrinsic_parameters;

	vector<Mat> RobotPose;
	vector<Mat> cameraPose;
	vector<Mat> RobotPosition;
	vector<Mat> cameraPosition;

	vector<Mat> EyeHandR;
	vector<Mat> EyeHandAxisAngle;//axis angle表示的eye hand姿态
	vector<Mat> EyeHandP_L12;//求解位置 方程左边（求逆）
	vector<Mat> EyeHandP_L23;//求解位置 方程左边（求逆）
	vector<Mat> EyeHandP_R12;//求解位置 方程右边
	vector<Mat> EyeHandP_R23;//求解位置 方程右边

	//// ----------------------read camera parameters----------------------------
	for(int i = 0;i < NumOfImg ; i++)
	{
		Mat temp_pose = Mat::zeros(Size(1,3),CV_64FC1);
		Mat temp_position = Mat::zeros(Size(1,3),CV_64FC1);

		temp_pose.at<double>(0,0) = extrinsic_parameters.at<double>(i,0);
		temp_pose.at<double>(0,1) = extrinsic_parameters.at<double>(i,1);
		temp_pose.at<double>(0,2) = extrinsic_parameters.at<double>(i,2);

		temp_position.at<double>(0,0) = extrinsic_parameters.at<double>(i,3);
		temp_position.at<double>(0,1) = extrinsic_parameters.at<double>(i,4);
		temp_position.at<double>(0,2) = extrinsic_parameters.at<double>(i,5);

		cameraPosition.push_back(temp_position );
		cameraPose.push_back(temp_pose);
		cout << i << "  " << temp_pose << endl;
	}

	//// ----------------------read robot parameters----------------------------
	readRobotData(robotFileName,RobotPose,RobotPosition,NumOfImg,failedIndex);
	cout << "robot position" << endl;
	for(int i = 0; i < NumOfImg; i++){
		cout << i << "   " << RobotPosition[i] << endl;
	}

	//// ----------------------求解手眼矩阵变量定义----------------------------
	Mat RobotPose_1(Size(1,3),CV_64FC1);			 // i-1次机器人姿态，axis/angle
	Mat RobotPose_2(Size(1,3),CV_64FC1);			// i次机器人姿态，axis/angle
	Mat RobotPose_3(Size(1,3),CV_64FC1);			// i+1次机器人姿态，axis/angle
	Mat RobotPosition_1(Size(1,3),CV_64FC1);        // i-1次机器人位置，
	Mat RobotPosition_2(Size(1,3),CV_64FC1);        // i次机器人位置，
	Mat RobotPosition_3(Size(1,3),CV_64FC1);        // i+1次机器人位置，

	Mat RobotPose_Matrix1(Size(3,3),CV_64FC1);		// i-1次机器人姿态，33矩阵
	Mat RobotPose_Matrix2(Size(3,3),CV_64FC1);		// i次机器人姿态，33矩阵
	Mat RobotPose_Matrix3(Size(3,3),CV_64FC1);		// i+1次机器人姿态，33矩阵

	Mat cameraPose_1(Size(1,3),CV_64FC1);			// i-1次camera姿态，axis/angle
	Mat cameraPose_2(Size(1,3),CV_64FC1);			// i次camera姿态，axis/angle
	Mat cameraPose_3(Size(1,3),CV_64FC1);			// i+1次camera姿态，axis/angle
	Mat cameraPosition_1(Size(1,3),CV_64FC1);        // i-1次camera位置，
	Mat cameraPosition_2(Size(1,3),CV_64FC1);        // i次camera位置，
	Mat cameraPosition_3(Size(1,3),CV_64FC1);        // i+1次camera位置，


	Mat cameraPose_Matrix1(Size(3,3),CV_64FC1);		// i-1次camera姿态，33矩阵
	Mat cameraPose_Matrix2(Size(3,3),CV_64FC1);		// i次camera姿态，33矩阵
	Mat cameraPose_Matrix3(Size(3,3),CV_64FC1);		// i+1次camera姿态，33矩阵
	
	int a,b,c;
	Mat L,R,L_temp,R_temp;//用于求解最小2乘的参数

	vector<double> position_Result_x;
	vector<double> position_Result_y;
	vector<double> position_Result_z;
	vector<double> orientation_Result_Rx;
	vector<double> orientation_Result_Ry;
	vector<double> orientation_Result_Rz;

	//// -------------------------------求解手眼矩阵：计算所有组合------------------------------------------，
	for(a = 0; a <  RobotPosition.size(); a++)
	{
		for(b = a +1; b < RobotPosition.size(); b++)
		{  
			for(c = b+1; c < RobotPosition.size(); c++)
			{
						cameraPose_1 = cameraPose[a];
						cameraPose_2 = cameraPose[b];
						cameraPose_3 = cameraPose[c];
						cameraPosition_1 = cameraPosition[a];
						cameraPosition_2 = cameraPosition[b];
						cameraPosition_3 = cameraPosition[c];

						RobotPose_1 = RobotPose[a];
						RobotPose_2 = RobotPose[b];
						RobotPose_3 = RobotPose[c];
						RobotPosition_1 = RobotPosition[a];
						RobotPosition_2 = RobotPosition[b];
						RobotPosition_3 = RobotPosition[c];
						
						Rodrigues(RobotPose_1,RobotPose_Matrix1);
						Rodrigues(RobotPose_2,RobotPose_Matrix2);
						Rodrigues(RobotPose_3,RobotPose_Matrix3);

						Rodrigues(cameraPose_1,cameraPose_Matrix1);
						Rodrigues(cameraPose_2,cameraPose_Matrix2);
						Rodrigues(cameraPose_3,cameraPose_Matrix3);


						Mat R_Left1(Size(3,3),CV_64FC1); 
						Mat R_Right1(Size(3,3),CV_64FC1); 
						Mat R_Left2(Size(3,3),CV_64FC1); 
						Mat R_Right2(Size(3,3),CV_64FC1); 

						Mat R_Left_angle1(Size(1,3),CV_64FC1); 
						Mat R_Right_angle1(Size(1,3),CV_64FC1);
						Mat R_Left_angle2(Size(1,3),CV_64FC1); 
						Mat R_Right_angle2(Size(1,3),CV_64FC1);

						Mat pL1(Size(1,3),CV_64FC1);
						Mat pL2(Size(1,3),CV_64FC1);
						Mat pL3(Size(1,3),CV_64FC1);

						Mat pR1(Size(1,3),CV_64FC1);
						Mat pR2(Size(1,3),CV_64FC1);
						Mat pR3(Size(1,3),CV_64FC1);

						R_Left1 = RobotPose_Matrix1.t() * RobotPose_Matrix2;
						R_Left2 = RobotPose_Matrix2.t() * RobotPose_Matrix3;
						R_Right1 = cameraPose_Matrix1 * cameraPose_Matrix2.t();
						R_Right2 = cameraPose_Matrix2 * cameraPose_Matrix3.t();

						Rodrigues(R_Left1,R_Left_angle1);
						Rodrigues(R_Left2,R_Left_angle2);
						Rodrigues(R_Right1,R_Right_angle1);
						Rodrigues(R_Right2,R_Right_angle2);

						Mat LeftMatrix(Size(3,3),CV_64FC1);
						Mat RightMatrix(Size(3,3),CV_64FC1);
						Mat LeftCross(Size(1,3),CV_64FC1); 
						Mat RightCross(Size(1,3),CV_64FC1); 

						int i,j;
						for(i = 0;i < 3; i++)
						{
							for(j = 0;j < 3; j++)
								if (j == 0)
									{
										LeftMatrix.at<double>(i,j) = R_Left_angle1.at<double>(i) / vec_norm(R_Left_angle1);
										RightMatrix.at<double>(i,j) = R_Right_angle1.at<double>(i) / vec_norm(R_Right_angle1);
									}
								else if (j ==1)
									{
									LeftMatrix.at<double>(i,j) = R_Left_angle2.at<double>(i) / vec_norm(R_Left_angle2);
									RightMatrix.at<double>(i,j) = R_Right_angle2.at<double>(i) / vec_norm(R_Right_angle2);
									}
								else 
								{   LeftCross = R_Left_angle1.cross(R_Left_angle2);
									RightCross = R_Right_angle1.cross(R_Right_angle2);
									LeftMatrix.at<double>(i,j) = LeftCross.at<double>(i) / vec_norm(LeftCross);
									RightMatrix.at<double>(i,j) = RightCross.at<double>(i) / vec_norm(RightCross);
								}

						}
					
						Mat EyeHandMatrix(Size(3,3),CV_64FC1); 
						Mat EyeHand_AxisAngle(Size(1,3),CV_64FC1); 
						Mat EyeHand_position(Size(1,3),CV_64FC1); 

						Mat E = Mat::eye(3,3,CV_64FC1);
						Mat temp;

						//求解手眼矩阵旋转向量
						EyeHandMatrix = LeftMatrix * RightMatrix.inv();

						pL1 = RobotPose_Matrix1.t() * RobotPosition_2 - RobotPose_Matrix1.t() * RobotPosition_1;
						pR1 = (-1) * cameraPose_Matrix1 * cameraPose_Matrix2.t() * cameraPosition_2 + cameraPosition_1;

						pL2 = RobotPose_Matrix2.t() * RobotPosition_3 - RobotPose_Matrix2.t() * RobotPosition_2;
						pR2 = (-1) * cameraPose_Matrix2 * cameraPose_Matrix3.t() * cameraPosition_3 + cameraPosition_2;

						temp = R_Left2 - E;
						
						EyeHandR.push_back(EyeHandMatrix);
						Rodrigues(EyeHandMatrix,EyeHand_AxisAngle);

						//输出各种组合的标定结果，用于比较
			/*				printf("%d %d %d\n",a,b,c);
						printf("%f\n",EyeHand_AxisAngle.at<double>(0) * 180.0 / CV_PI);
						printf("%f\n",EyeHand_AxisAngle.at<double>(1) * 180.0 / CV_PI);
						printf("%f\n",EyeHand_AxisAngle.at<double>(2)* 180.0 / CV_PI);
						printf("\n");*/
						//write(EyeHand_AxisAngle,true,"Orientation.txt");
						//printf("%d %d %d\n",a,b,c);

						//手眼矩阵旋转向量放入vector容器，供求平均值
						EyeHandAxisAngle.push_back(EyeHand_AxisAngle);

						//手眼矩阵旋转向量,各个分量放入vector容器，供RANSAC
						orientation_Result_Rx.push_back(EyeHand_AxisAngle.at<double>(0));
						orientation_Result_Ry.push_back(EyeHand_AxisAngle.at<double>(1));
						orientation_Result_Rz.push_back(EyeHand_AxisAngle.at<double>(2));

						Mat L12(Size(3,3),CV_64FC1); 
						Mat L23(Size(3,3),CV_64FC1); 
						Mat R12(Size(3,3),CV_64FC1); 
						Mat R23(Size(3,3),CV_64FC1); 

						R12 = EyeHandMatrix * pR1 - pL1;
						R23 = EyeHandMatrix * pR2 - pL2;
						L12 = R_Left1 - E;
						L23 = R_Left2 - E;
						
					   //将求解平移向量的矩阵方程，左右矩阵依次拼接
						if(a == 0 && b ==1 && c ==2)  //初始化矩阵，然后拼接
						{	
							L = L12; R = R23;
						}
						L_temp = MatrixCombine(L12,L23);
						L = MatrixCombine(L,L_temp);
						R_temp = MatrixCombine(R12,R23);
						R = MatrixCombine(R,R_temp);

					//	L_temp = MatrixCombine(L12,L23);
					//	R_temp = MatrixCombine(R12,R23);

				     	Mat Result_(Size(1,3),CV_64FC1); 
						solve(L_temp,R_temp,Result_,DECOMP_QR);  
						position_Result_x.push_back(Result_.at<double>(0));
						position_Result_y.push_back(Result_.at<double>(1));
						position_Result_z.push_back(Result_.at<double>(2));
						//write(Result_,true,"Position.txt");
			}
		}
	} 
	cout << "A" << endl;
	
	//利用RANSAC剔除外点
	double positionResultX, positionResultY, positionResultZ,orientationResultRx,orientationResultRy,orientationResultRz;
	Mat temp_X = ransac(position_Result_x.size(),position_Result_x,1000,0.5,positionResultX);
	Mat temp_Y = ransac(position_Result_y.size(),position_Result_y,1000,0.5,positionResultY);
	Mat temp_Z = ransac(position_Result_z.size(),position_Result_z,1000,0.5,positionResultZ);
	Mat temp_Rx = ransac(orientation_Result_Rx.size(),orientation_Result_Rx,1000,0.01,orientationResultRx);
	Mat temp_Ry = ransac(orientation_Result_Ry.size(),orientation_Result_Ry,1000,0.01,orientationResultRy);
	Mat temp_Rz = ransac(orientation_Result_Rz.size(),orientation_Result_Rz,1000,0.01,orientationResultRz);

	Mat EH_translation(Size(1,3),CV_64FC1); 
	Mat EH_rotation(Size(1,3),CV_64FC1); 

	EH_translation.at<double>(0) =  positionResultX;
	EH_translation.at<double>(1) =  positionResultY;
	EH_translation.at<double>(2) =  positionResultZ;

	EH_rotation.at<double>(0) =  orientationResultRx;
	EH_rotation.at<double>(1) =  orientationResultRy;
	EH_rotation.at<double>(2) =  orientationResultRz;
	
	cout << EH_translation << endl;
	cout << EH_rotation << endl;
	cout << "Position:" << endl;
}