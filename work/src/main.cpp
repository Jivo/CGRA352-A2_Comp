
// std
#include <iostream>

// opencv
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>

#include "nnf.hpp"
#include "reconstruction.hpp"
#include "gauss_pyr.hpp"

#include "globalVars.cpp"


using namespace cv;
using namespace std;

Rect findMaskBounds(const Mat& mask);
void swapPatch(const Mat& source, Mat& target, const Mat& mask, Point translation);

int main( int argc, char** argv ) {
	if( argc != 3) {
		cout << "Wrong number of arguments: " << argc<<" REQUIRED: 3"<<endl;
		return -1;}
	
	Mat source, mask;
	source = imread(argv[1], CV_LOAD_IMAGE_COLOR); 
	mask = imread(argv[2], CV_LOAD_IMAGE_COLOR);

	if(!source.data || !mask.data) {
		cout << "Could not open or find the image" << std::endl;
		return -1;}
	
	Mat target = source.clone();
	swapPatch(source, target, mask, Point(-270, 0));

	//The pyramid of source and target images
	vector<Mat> sourceGaus;  getGauss(source, sourceGaus, PYRAMID_SIZE);
	vector<Mat> targetGaus;  getGauss(target, targetGaus, PYRAMID_SIZE);

	//Pyramid of BORDER source and target images, with extensions to make patching easier.
	vector<Mat> sourceGausBorder;  generateBorders(sourceGaus, sourceGausBorder);
	vector<Mat> targetGausBorder;  generateBorders(targetGaus, targetGausBorder);
	


	/****************************
	           START
	*****************************/

	Mat nnf = Mat::zeros(targetGaus.at(PYRAMID_SIZE-1).rows, 
		targetGaus.at(PYRAMID_SIZE - 1).cols, CV_32SC2);
	
	Mat cost(nnf.rows, nnf.cols, CV_32F);
	getNNF(nnf, cost, sourceGaus.at(PYRAMID_SIZE - 1), targetGaus.at(PYRAMID_SIZE - 1),
		sourceGausBorder.at(PYRAMID_SIZE - 1), targetGausBorder.at(PYRAMID_SIZE - 1));
	
	int stop = 2;

	for (int i = PYRAMID_SIZE - 1; i >= stop; i--) {
		iterateNNF(nnf, cost, sourceGaus.at(i), targetGaus.at(i),
			sourceGausBorder.at(i), targetGausBorder.at(i));
		if (i != stop) {
			nnf = upSample(nnf, Size(targetGaus.at(i - 1).cols, targetGaus.at(i - 1).rows));
			getCost(cost, nnf, sourceGausBorder.at(i - 1), targetGausBorder.at(i - 1));

		}
	}

	/****************************
	           FINISH
	*****************************/


	namedWindow("smallest", WINDOW_AUTOSIZE);
	imshow("smallest", targetGaus.at(4));
	namedWindow("2nd", WINDOW_AUTOSIZE);
	imshow("2nd", targetGaus.at(3));
	namedWindow("3rd", WINDOW_AUTOSIZE);
	imshow("3rd", targetGaus.at(2));
	namedWindow("4th", WINDOW_AUTOSIZE);
	imshow("4th", targetGaus.at(1));
	namedWindow("5th", WINDOW_AUTOSIZE);
	imshow("5th", targetGaus.at(0));

	Mat reconstr = reconstruct(nnf, sourceGausBorder.at(stop));
	reconstr = reconstr(Rect(CENTER, nnf.size()));

	Mat nnfImg = nnf2img(nnf, sourceGaus.at(stop).size(),false);

	imwrite("work/res/saves/final.jpg", reconstr);
	namedWindow("reconstruction", WINDOW_AUTOSIZE);
	imshow("reconstruction", reconstr);

	imwrite("work/res/saves/nnf.jpg", nnfImg);
	namedWindow("nnf", WINDOW_AUTOSIZE);
	imshow("nnf", nnfImg);

	waitKey(0);
}

/****************************
          SWAP PATCH
*****************************/
void swapPatch(const Mat& source, Mat& target, const Mat& mask, Point off) {
	Rect r1 = findMaskBounds(mask);
	Rect r2 = Rect(r1.x + off.x, r1.y + off.y, r1.width, r1.height);
	Mat retarget = source(r1);
	Mat refill = source(r2);
	retarget.copyTo(target(r2));
	refill.copyTo(target(r1));
}

/****************************
       FIND MASK BOUNDS
*****************************/
Rect findMaskBounds(const Mat& mask) {
	Point p1(-1, -1), p2(-1, -1);
	int i = -1;
	for (int row = 0; row < mask.rows; row++) {
		for (int col = 0; col < mask.cols; col++) {
			int val1 = mask.at<Vec3b>(row, col)[0];
			int val2 = mask.at<Vec3b>((mask.rows - 1 - row), (mask.cols - 1 - col))[0];
			if (mask.at<Vec3b>(row, col)[2] > 0 && i == -1) { 
				i++;
			}
			if (val1 > 0 && p1.x == -1) {
				p1 = Point(col, row);}
			if (val2 > 0 && p2.x == -1) {
				p2 = Point(mask.cols - 1 - col,
					mask.rows - 1 - row);}
		}}
	return Rect(p1, Size(p2.x - p1.x, p2.y - p1.y));
}
