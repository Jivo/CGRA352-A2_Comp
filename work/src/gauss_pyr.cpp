#include "gauss_pyr.hpp"
#include "globalVars.cpp"

/****************************
        GET GAUSS
*****************************/
void getGauss(const Mat&m, vector<Mat>& holder, int num) {
	holder.push_back(m.clone());
	Mat current;
	for (int i = 0; i < num; i++) {
		current = holder[i];
		Mat half = current.clone();
		pyrDown(current, half);
		holder.push_back(half);
	}
}
/****************************
      GENERATE BORDERS
*****************************/
void generateBorders(const vector<Mat>& source, vector<Mat>& dst) {
	int left = CENTER.x, right = SIZE - 1 - CENTER.x,
		top = CENTER.y, bottom = SIZE - 1 - CENTER.y;
	for (int i = 0; i < source.size(); i++) {
		Mat current = source.at(i).clone();
		copyMakeBorder(current, current, top, bottom, left, right, BORDER_CONSTANT);
		dst.push_back(current);
	}
}