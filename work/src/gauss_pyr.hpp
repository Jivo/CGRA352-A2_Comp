// opencv
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

void getGauss(const Mat&m, vector<Mat>& holder, int num);
void generateBorders(const vector<Mat>& source, vector<Mat>& dst);
