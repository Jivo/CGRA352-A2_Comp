// std
#include <iostream>

// opencv
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;
Mat reconstruct(const Mat&nnf, const Mat&sourceBorder);
Mat nnf2img(Mat nnf, Size s, bool absolute);