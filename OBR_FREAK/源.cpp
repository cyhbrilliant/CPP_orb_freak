//#include <iostream>   
//#include <opencv.hpp>
//#include "opencv2/core/core.hpp"   
//#include "opencv2/features2d/features2d.hpp"   
//#include "opencv2/highgui/highgui.hpp"   
//#include <time.h>
//#include <vector>   
//
//using namespace cv;  
//using namespace std;  
//
//
//
////
////void KeyPoint2Mat(Mat *mat,vector<KeyPoint> &kp)
////{
////	*mat=Mat::zeros(7,kp.size(),CV_32FC1);
////	//cout<<*mat<<endl;
////	cout<<kp.size()<<endl;
////	for (int i=0;i<kp.size();i++)
////	{
////		mat->at<float>(0,i)=kp.at(i).pt.x;
////		mat->at<float>(1,i)=kp.at(i).pt.y;
////		mat->at<float>(2,i)=kp.at(i).angle;
////		mat->at<float>(3,i)=kp.at(i).class_id;
////		mat->at<float>(4,i)=kp.at(i).octave;
////		mat->at<float>(5,i)=kp.at(i).response;
////		mat->at<float>(6,i)=kp.at(i).size;
////	}
////	
////}
////
////void Mat2KeyPoint(Mat *mat,vector<KeyPoint> &kpx)
////{
////	cout<<mat->cols<<endl;
////	KeyPoint kp;
////	for (int i=0;i<mat->cols;i++)
////	{
////		kp.pt.x=mat->at<float>(0,i);
////		kp.pt.y=mat->at<float>(1,i);
////		kp.angle=mat->at<float>(2,i);
////		kp.class_id=mat->at<float>(3,i);
////		kp.octave=mat->at<float>(4,i);
////		kp.response=mat->at<float>(5,i);
////		kp.size=mat->at<float>(6,i);
////		kpx.push_back(kp);
////	}
////	
////
////}
//
//
//
//void orbKeyPoint(Mat &img,vector<KeyPoint> &keypoint,Mat &des)
//{
//	Ptr<FeatureDetector> dector =ORB::create(200,1.2f,3,31,0,2,ORB::FAST_SCORE,31,20);
//
//	dector->detect(img,keypoint);
//	dector->compute(img,keypoint,des);
//
//	/*long be=clock();
//	dector->detectAndCompute(img,Mat(),keypoint,des);
//	long en=clock();*/
//
//	//cout<<en-be<<"ms"<<endl;
//	//===预览角点检测
//	//drawKeypoints(img, keypoint, img, Scalar::all(255), DrawMatchesFlags::DRAW_OVER_OUTIMG);  
//	//imshow("FAST feature", img);  
//	//cvWaitKey(0);  
//
//}
//
//
//void copyKeyPoint(vector<Point2f> &keypre,vector<Point2f> &keynext)
//{
//	for (int i=0;i<keypre.size();i++)
//	{
//		Point2f temp;
//		temp.x=keypre.at(i).x;
//		temp.y=keypre.at(i).y;
//		keynext.push_back(temp);
//	}
//}
//
//
//int main()  
//{  
//	
//
//	
//	////测试模式1
//	//Mat img1=imread("test_module3.jpg",0);
//	//Mat img2=imread("test_module4.jpg",0);
//
//	////特征点
//	//vector<KeyPoint> keypoint1;
//	//vector<KeyPoint> keypoint2;
//
//	////描述子
//	//Mat describe1;
//	//Mat describe2;
//
//	//orbKeyPoint(img1,keypoint1,describe1);
//	//orbKeyPoint(img2,keypoint2,describe2);
//
//	//vector<DMatch> Vmatch;
//	//Ptr<DescriptorMatcher> matcher=DescriptorMatcher::create("BruteForce-Hamming");
//	//matcher->match(describe1,describe2,Vmatch);
//
//	//Mat img_matches;
//	//namedWindow("Mathc",0);
//	//drawMatches(img1,keypoint1,img2,keypoint2,Vmatch,img_matches,Scalar::all(-1),CV_RGB(255,255,255),Mat(),4);  
//	//imshow("Mathc",img_matches); 
//
//	//
//	//vector<Point2f> kp1;
//	//vector<Point2f> kp2;
//	//for (int i=0;i<keypoint1.size();i++)
//	//{
//	//	//Point2f p1(keypoint1.at(i).pt.x,keypoint1.at(i).pt.y);
//	//	Point2f p1(keypoint1.at(i).pt.y,keypoint1.at(i).pt.x);
//	//	kp1.push_back(p1);
//
//	//	int idx=Vmatch.at(i).trainIdx;
//	//	//Point2f p2(keypoint2.at(idx).pt.x,keypoint2.at(idx).pt.y);
//	//	Point2f p2(keypoint2.at(idx).pt.y,keypoint2.at(idx).pt.x);
//	//	kp2.push_back(p2);
//	//}
//
//
//
//	//Mat H;
//	//Mat a;
//	//H=findHomography(kp1,kp2,RANSAC,3);
//
//	////H=Mat::eye(3,3,CV_32FC1);
//	//cout<<H<<endl;
//	//Mat img_transform;
//
//	//warpPerspective(img2,img_transform,H,Size(1280,720));
//	//namedWindow("trans");
//	//imshow("trans",img_transform); 
//	//waitKey(0);
//
//
//
//	////测试模式2
//
//	//VideoCapture cap(0);
//	//cap.set( CV_CAP_PROP_FRAME_WIDTH,1280);
//	//cap.set( CV_CAP_PROP_FRAME_HEIGHT,720);
//
//	//Mat img1=imread("lab.jpg",0);
//	//Mat describe1;
//	//vector<KeyPoint> keypoint1;
//	//orbKeyPoint(img1,keypoint1,describe1);
//
//	//Mat preimg;
//	//vector<Point2f> prekey;
//	//vector<Point2f> nowkey;
//	//vector<uchar> statue;
//	//vector<double> err;
//	//for (int i=0;i<prekey.size();i++)
//	//{
//	//	Point2f temp;
//	//	temp.x=keypoint1.at(i).pt.x;
//	//	temp.y=keypoint1.at(i).pt.y;
//	//	prekey.push_back(temp);
//	//}
//	//img1.copyTo(preimg);
//
//	//while(true)
//	//{
//	//	Mat img2;
//	//	Mat img2_gray;
//	//	vector<KeyPoint> keypoint2;
//	//	Mat describe2;
//
//	//	cap>>img2;
//	//	cvtColor(img2,img2_gray,CV_RGB2GRAY);
//	//	cout<<img2_gray.type()<<preimg.type();
//
//	//	calcOpticalFlowPyrLK(preimg,img2_gray,prekey,nowkey,statue,err);
//
//
//	//	img2_gray.copyTo(preimg);
//	//	copyKeyPoint(nowkey,prekey);
//
//	//	//orbKeyPoint(img2_gray,keypoint2,describe2);
//
//	//	//if (keypoint2.size()>2)
//	//	//{
//
//	//	//}
//
//	//	//vector<vector<DMatch>> Vmatch;
//	//	//Ptr<DescriptorMatcher> matcher=DescriptorMatcher::create("BruteForce-Hamming");
//	//
//	//	//matcher->knnMatch(describe1,describe2,Vmatch,2);
//
//	//	//Mat img_matches;
//	//	//namedWindow("Mathc",0);
//	//	//
//	//	//
//	//	//vector<DMatch> KNmatch;
//	//	////cout<<Vmatch.at(0).at(0).distance/Vmatch.at(0).at(1).distance<<endl;
//	//	//long b=clock();
//	//	//for(int i=0;i<Vmatch.size();i++)
//	//	//{
//	//	//	//drawMatches(img1,keypoint1,img2,keypoint2,Vmatch.at(0),img_matches,Scalar::all(-1),CV_RGB(255,255,255),Mat(),4);
//	//	//	float ratio=Vmatch.at(i).at(0).distance/Vmatch.at(i).at(1).distance;
//	//	//	if (ratio<0.9)
//	//	//	{
//	//	//		KNmatch.push_back(Vmatch.at(i).at(0));
//	//	//	}
//	//	//	
//
//
//	//	for (int i=0;i<nowkey.size();i++)
//	//	{
//	//		circle(img2,cvPoint(nowkey.at(i).x,nowkey.at(i).y),2,CV_RGB(255,255,255),2);
//	//	}
//
//
//
//	//	vector<Point2f> kp1;
//	//	vector<Point2f> kp2;
//	//	//for (int i=0;i<KNmatch.size();i++)
//	//	//{
//	//	//	////Point2f p1(keypoint1.at(i).pt.x,keypoint1.at(i).pt.y);
//	//	//	//Point2f p1(keypoint1.at(i).pt.x,keypoint1.at(i).pt.y);
//	//	//	//kp1.push_back(p1);
//	//	//	//
//	//	//	//int idx=Vmatch.at(i).at(0).trainIdx;
//	//	//	////Point2f p2(keypoint2.at(idx).pt.x,keypoint2.at(idx).pt.y);
//	//	//	//Point2f p2(keypoint2.at(idx).pt.x,keypoint2.at(idx).pt.y);
//	//	//	//kp2.push_back(p2);
//
//
//	//	//	int quaryx=KNmatch.at(i).queryIdx;
//	//	//	int trainx=KNmatch.at(i).trainIdx;
//
//	//	//	float mx=keypoint1.at(quaryx).pt.x;
//	//	//	float my=keypoint1.at(quaryx).pt.y;
//	//	//	float kx=keypoint2.at(trainx).pt.x;
//	//	//	float ky=keypoint2.at(trainx).pt.y;
//
//	//	//	circle(img2,cvPoint(kx,ky),2,CV_RGB(255,255,255),2);
//
//	//	//	Point2f p1;
//	//	//	Point2f p2;
//	//	//	p1.x=mx;
//	//	//	p1.y=my;
//	//	//	p2.x=kx;
//	//	//	p2.y=ky;
//	//	//	kp1.push_back(p1);
//	//	//	kp2.push_back(p2);
//	//	//}
//
//	//	
//	//	Mat H;
//	//	Mat no=Mat::eye(3,3,CV_32FC1);
//	//	H=findHomography(kp1,kp2,RANSAC,3);
//	///*	H=findHomography(kp1,kp2);*/
//	//	//H=Mat::eye(3,3,CV_32FC1);
//	//	cout<<H<<endl;
//	//	Mat img_transform;
//
//	//	Point2f Wrectangle[4];
//	//	Wrectangle[0].x=-300;
//	//	Wrectangle[0].y=-300;
//	//	Wrectangle[1].x=-300;
//	//	Wrectangle[1].y=300;
//	//	Wrectangle[2].x=300;
//	//	Wrectangle[2].y=300;
//	//	Wrectangle[3].x=300;
//	//	Wrectangle[3].y=-300;
//
//
//
//	//	for (int i=0;i<4;i++)
//	//	{
//	//		Wrectangle[i].x+=640;
//	//		Wrectangle[i].y+=360;
//	//	}
//	//	vector<Point2f> WrectangleP;
//	//	vector<Point2f> WrectangleP_t;
//	//	for (int i=0;i<4;i++)
//	//	{
//	//		WrectangleP.push_back(Wrectangle[i]);
//	//	}
//
//
//
//
//	//	perspectiveTransform(WrectangleP,WrectangleP_t,H);
//
//	//	for (int i=0;i<3;i++)
//	//	{
//	//		line(img2,cvPoint(WrectangleP_t.at(i).x,WrectangleP_t.at(i).y),cvPoint(WrectangleP_t.at(i+1).x,WrectangleP_t.at(i+1).y),CV_RGB(255,255,255),2);
//	//	}
//	//	line(img2,cvPoint(WrectangleP_t.at(3).x,WrectangleP_t.at(3).y),cvPoint(WrectangleP_t.at(0).x,WrectangleP_t.at(0).y),CV_RGB(255,255,255),2);
//
//	//	namedWindow("trans");
//	//	imshow("trans",img2); 
//	//	
//	//	 
//
//	//	waitKey(20);
//	//}
//
//	Mat pre,next,frame;
//	VideoCapture cap;
//	cap.open(0);
//	cap.set( CV_CAP_PROP_FRAME_WIDTH,320);
//	cap.set( CV_CAP_PROP_FRAME_HEIGHT,240);
//	vector<Point2f> prepoint,nextpoint;
//	vector<uchar> state;
//	vector<float>err;
//	int con=0;
//
//	vector<Point2f> inters;
//	vector<uchar> stateinters;
//	vector<Point2f> storePoint;
//
//	Mat describe1;
//	vector<KeyPoint> keypoint1;
//
//	
//
//	Point2f Wrectangle[4];
//	Wrectangle[0].x=-60;
//	Wrectangle[0].y=-60;
//	Wrectangle[1].x=-60;
//	Wrectangle[1].y=60;
//	Wrectangle[2].x=60;
//	Wrectangle[2].y=60;
//	Wrectangle[3].x=60;
//	Wrectangle[3].y=-60;
//	for (int i=0;i<4;i++)
//	{
//		Wrectangle[i].x+=160;
//		Wrectangle[i].y+=120;
//	}
//	vector<Point2f> WrectangleP;
//	vector<Point2f> WrectangleP_t;
//	for (int i=0;i<4;i++)
//	{
//		WrectangleP_t.push_back(Wrectangle[i]);
//	}
//	
//
//
//	if(!cap.isOpened())cout<<"摄像头未打开，请打开摄像头！！！"<<endl;
//	for(;;)
//	{
//		
//		cap>>frame;
//		if(frame.empty())break;
//
//		cvtColor(frame,next,CV_BGR2GRAY);
//
//
//		
//
//		if(!next.empty()&&!pre.empty())
//		{
//			/*con++;
//		
//				orbKeyPoint(pre,keypoint1,describe1);
//				for (int i=0;i<keypoint1.size();i++)
//				{
//					Point2f temp;
//					temp.x=keypoint1.at(i).pt.x;
//					temp.y=keypoint1.at(i).pt.y;
//					prepoint.push_back(temp);
//				}*/
//
//			
//			
//
//			long be1=clock();
//			goodFeaturesToTrack(pre,prepoint,100,0.001,10,Mat(),3,false,0.04);
//			long be2=clock();
//			cornerSubPix(pre,prepoint,Size(10,10),Size(-1,-1),TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03));
//			long be3=clock();
//			calcOpticalFlowPyrLK(pre,next,prepoint,nextpoint,state,err,Size(31,31),3);
//			long be4=clock();
//
//
//			cout<<be2-be1<<"  "<<be3-be2<<"  "<<be4-be3<<endl;
//			vector<Point2f> keyp1;
//			vector<Point2f> keyp2;
//
//			for(int i=0;i<state.size();i++)
//			{
//				if(state[i]!=0)
//				{
//					line(frame,Point((int)prepoint[i].x,(int)prepoint[i].y),Point((int)nextpoint[i].x,(int)nextpoint[i].y),Scalar::all(-1));
//					Point2f temp1;
//					Point2f temp2;
//					temp1.x=prepoint[i].x;
//					temp1.y=prepoint[i].y;
//					temp2.x=nextpoint[i].x;
//					temp2.y=nextpoint[i].y;
//					keyp1.push_back(temp1);
//					keyp2.push_back(temp2);
//
//				}
//			}
//
//			Mat H=findHomography(keyp1,keyp2,RANSAC);
//
//			/*cout<<H<<endl;*/
//
//		
//			perspectiveTransform(WrectangleP_t,WrectangleP_t,H);
//
//			for (int i=0;i<3;i++)
//			{
//				line(frame,cvPoint(WrectangleP_t.at(i).x,WrectangleP_t.at(i).y),cvPoint(WrectangleP_t.at(i+1).x,WrectangleP_t.at(i+1).y),CV_RGB(255,255,255),2);
//			}
//			line(frame,cvPoint(WrectangleP_t.at(3).x,WrectangleP_t.at(3).y),cvPoint(WrectangleP_t.at(0).x,WrectangleP_t.at(0).y),CV_RGB(255,255,255),2);
//
//			
//
//	/*		copyKeyPoint(nextpoint,prepoint);*/
//
//
//
//			/*		for (int i=0;i<inters.size();i++)
//			{
//			Point2f
//			if ()
//			{
//			}
//			}
//
//
//			Mat H=findHomography(inters,nextpoint,RANSAC,3);*/
//
//			namedWindow("frame",0);
//			imshow("frame",frame);
//			waitKey(1);
//		}
//		next.copyTo(pre);
//
//
//	}
//
//
//	
//	
//	system("pause");
//	return 0;
//	
//} 
//
//
//
//
//
//
////
////void describe_sita(int *descirbeSita,int *dsnum,int r)
////{
////	(*dsnum)=0;
////	for (int i=-r;i<=r;i++)
////	{
////		for (int j=-r;j<=r;j++)
////		{
////			if ((i*i+j*j)<=(r*r+1))
////			{
////				*(descirbeSita+(*dsnum)*2+0)=i;
////				*(descirbeSita+(*dsnum)*2+1)=j;
////				(*dsnum)++;
////			}
////		}
////	}
////}
//
//
//
//
////void OrbFreak_checkPoint(Mat &img,Mat &des,vector<KeyPoint> &kp)
////{
////	 
////	FAST(img,kp,40);
////
////
////	Ptr<FeatureDetector> dector =ORB::create(50,2.0f,1,31,0,2,ORB::FAST_SCORE,31,20);
////	dector->detect(img,kp);
////	for(int i=0;i<kp.size();i++)
////	{
////		cout<<"oct"<<kp.at(i).octave;
////		cout<<"    res"<<kp.at(i).response;
////		cout<<"    size"<<kp.at(i).size<<"    ang"<<kp.at(i).angle<<endl;
////	}
////	
////	dector->compute(img,kp,des);
////	dector->detectAndCompute(img,NULL,kp,des,false);
////	
////	FAST(img,kp,40);
////
////
////	cout<<kp.size()<<endl;	
////	vector<KeyPoint>::iterator i=kp.begin();
////	kp.erase(i);
////	cout<<kp.size();	
////
////
////	=====
////
////	harris取前n个点
////
////	=====
////
////
////
////	for (int i=0;i<kp.size();i++)
////	{
////		*(keyPoint+3*i+0)=kp.at(i).pt.x;
////		*(keyPoint+3*i+1)=kp.at(i).pt.y;
////	}
////
////	*num=kp.size();//=============长度赋值
////	drawKeypoints(img, kp, img, Scalar::all(255), DrawMatchesFlags::DRAW_OVER_OUTIMG);  
////	imshow("FAST feature", img);  
////	cvWaitKey(0);  
////
////}
//
//
//
////
////void calDirection(Mat &img,float *keyPoint,int *num,int *describeSita,int *dsnum)
////{
////	int cx,cy;
////	int x,y;
////	float m01=0;//y
////	float m10=0;//x
////	for(int i=0;i<*num;i++)
////	{
////		cx=*(keyPoint+3*i+0);
////		cy=*(keyPoint+3*i+1);
////		for (int m=0;m<*dsnum;m++)
////		{
////			x=*(describeSita+2*m+0);
////			y=*(describeSita+2*m+1);
////			
////			m01+=y*img.at<uchar>(y+cy,x+cx);
////			m10+=x*img.at<uchar>(y+cy,x+cx);
////		}
////		*(keyPoint+3*i+2)=fastAtan2(m01,m10);
////	}
////}
//
//
//
//
////VideoCapture cap(0);
//
////cap.set( CV_CAP_PROP_FRAME_WIDTH,1280);
////cap.set( CV_CAP_PROP_FRAME_HEIGHT,720);
////
//
////Mat img1=imread("test_module3.jpg",0);
////Mat img2;
////Mat img2_gray;
//
//
////vector<KeyPoint> kp1;
////vector<KeyPoint> kp2;
////Mat des1;
////Mat des2;
//
////OrbFreak_checkPoint(img1,des1,kp1);
//
////while(true)
////{
////	cap>>img2;
//
////	long beginx=clock();
//
////	cvtColor(img2,img2_gray,CV_RGB2GRAY);
//
////	OrbFreak_checkPoint(img2_gray,des2,kp2); 
//
////vector<DMatch> matches;
////Ptr<DescriptorMatcher> matchx=DescriptorMatcher::create("BruteForce-Hamming");
//////Ptr<DescriptorMatcher> matchx=DescriptorMatcher::create("BruteForce");
////matchx->match(des1,des2,matches);
////long endx=clock();
////cout<<endx-beginx<<"ms"<<endl;
////Mat img_matches;  
////namedWindow("Mathc",0);
////drawMatches(img1,kp1,img2_gray,kp2,matches,img_matches,Scalar::all(-1),CV_RGB(255,255,255),Mat(),4);  
////imshow("Mathc",img_matches); 
//
//
////waitKey(20);
////}
//
//
//
//
////Mat img1=imread("test0.jpg",0);
////Mat img2=imread("test1.jpg",0);
////float img1_keyPoint[2000*3];//x,y,sita
////float img2_keyPoint[2000*3];
//
//////===角度描述范围
////int descirbeSita[100*2];//x,y
////int dsnum[1];
////describe_sita(descirbeSita,dsnum,3);
//////======
//
////int img1_num[1];
////int img2_num[1];
////OrbFreak_checkPoint(img1,img1_keyPoint,img1_num); 
////calDirection(img1,img1_keyPoint,img1_num,descirbeSita,dsnum);
//
//
//
////OrbFreak_checkPoint(img1,des1,kp1);
////OrbFreak_checkPoint(img2,des2,kp2);
////cout<<des1.rows<<"  "<<des1.cols<<endl;
////vector<DMatch> matches;
////Ptr<DescriptorMatcher> matchx=DescriptorMatcher::create("BruteForce-Hamming");
////matchx->match(des1,des2,matches);
//////cout<<des1;
//////cout<<matches.at(1).queryIdx;
////cout<<des1.rows<<"  "<<des1.cols<<endl;
////Mat img_matches;  
//////cout<<matches.at(198).queryIdx;
////namedWindow("Mathc",0);
////drawMatches(img1,kp1,img2,kp2,matches,img_matches,Scalar::all(-1),CV_RGB(255,255,255),Mat(),4);  
////imshow("Mathc",img_matches); 
////waitKey(0);
//
//
//




#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

static void help()
{
	// print a welcome message, and the OpenCV version
	cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
		"Using OpenCV version " << CV_VERSION << endl;
	cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
	cout << "\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - auto-initialize tracking\n"
		"\tc - delete all the points\n"
		"\tn - switch the \"night\" mode on/off\n"
		"To add/remove a feature point click it\n" << endl;
}

Point2f point;
bool addRemovePt = false;

static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
	if( event == EVENT_LBUTTONDOWN )
	{
		point = Point2f((float)x, (float)y);
		addRemovePt = true;
	}
}

int main( int argc, char** argv )
{
	help();

	VideoCapture cap;
	TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
	Size subPixWinSize(10,10), winSize(31,31);

	const int MAX_COUNT = 500;
	bool needToInit = false;
	bool nightMode = false;

	if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
		cap.open(argc == 2 ? argv[1][0] - '0' : 0);
	else if( argc == 2 )
		cap.open(argv[1]);

	if( !cap.isOpened() )
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}


	namedWindow( "LK Demo", 1 );
	setMouseCallback( "LK Demo", onMouse, 0 );

	Mat gray, prevGray, image, frame;
	vector<Point2f> points[2];
	
	for(;;)
	{
		cap >> frame;
		if( frame.empty() )
			break;

		frame.copyTo(image);
		cvtColor(image, gray, COLOR_BGR2GRAY);

		if( nightMode )
			image = Scalar::all(0);

		if( needToInit )
		{
			// automatic initialization
			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
			addRemovePt = false;
		}
		else if( !points[0].empty() )
		{
			vector<uchar> status;
			vector<float> err;
			if(prevGray.empty())
				gray.copyTo(prevGray);
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
				3, termcrit, 0, 0.001);
			size_t i, k;
			for( i = k = 0; i < points[1].size(); i++ )
			{
				if( addRemovePt )
				{
					if( norm(point - points[1][i]) <= 5 )
					{
						addRemovePt = false;
						continue;
					}
				}

				if( !status[i] )
					continue;

				points[1][k++] = points[1][i];
				circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
			}
			points[1].resize(k);
		}

		if( addRemovePt && points[1].size() < (size_t)MAX_COUNT )
		{
			vector<Point2f> tmp;
			tmp.push_back(point);
			cornerSubPix( gray, tmp, winSize, Size(-1,-1), termcrit);
			points[1].push_back(tmp[0]);
			addRemovePt = false;
		}
	

		needToInit = false;
		imshow("LK Demo", image);

		char c = (char)waitKey(10);
		if( c == 27 )
			break;
		switch( c )
		{
		case 'r':
			needToInit = true;
			break;
		case 'c':
			points[0].clear();
			points[1].clear();
			break;
		case 'n':
			nightMode = !nightMode;
			break;
		}

		std::swap(points[1], points[0]);
		cv::swap(prevGray, gray);
	}

	return 0;
}
