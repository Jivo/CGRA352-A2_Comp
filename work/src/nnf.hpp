 //std
#include <iostream>

// opencv
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;

void getNNF(Mat&nnf, Mat&cost, const Mat&source, const Mat&target, const Mat&sourceBorder, const Mat&targetBorder);
void iterateNNF(Mat&nnf, Mat &cost, const Mat&source, const Mat&target,
	const Mat&sourceBorder, const Mat&targetBorder);
void initialize(const Mat&source, const Mat&target, const Mat&sourceBorder, const Mat&targetBorder, Mat&nnf, Mat&cost);
void propagate(int iteration, Mat &cost, Mat &nnf, Size s, const Mat&sourceBorder, const Mat&targetBorder);
void randSearch(int step, int radius, Mat &nnf, Mat &cost, const Mat&source, const Mat&sourceBorder, const Mat& targetBorder);
Mat upSample(const Mat&nnf, Size s);
void getCost(Mat&cost, const Mat& nnf, const Mat&sourceBorder, const Mat&targetBorder);
Mat nnf2img(Mat nnf, Size s, bool absolute);
Point randomPoint(int radius, Point origin);