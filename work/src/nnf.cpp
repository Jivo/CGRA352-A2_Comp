#include "nnf.hpp"
#include "reconstruction.hpp"
#include "globalVars.cpp"

/****************************
GET NNF
*****************************/
void getNNF(Mat&nnf, Mat &cost, const Mat&source, const Mat&target,
	const Mat&sourceBorder, const Mat&targetBorder) {
	Mat newTarget;

	//initialize(source, target, sourceBorder,
	//	targetBorder, nnf, cost);

	newTarget = targetBorder;
	for (int i = 0; i < MAX_ITERATIONS; i++) {
		propagate(i, cost, nnf, Size(source.cols, source.rows), sourceBorder, newTarget);
		randSearch(1, max(source.rows, source.cols), nnf, cost, source, sourceBorder, newTarget);
	}
}
/****************************
ITERATE NNF
*****************************/
void iterateNNF(Mat&nnf, Mat &cost, const Mat&source, const Mat&target,
	const Mat&sourceBorder, const Mat&targetBorder) {
	Mat newTarget;

	for (int i = 0; i < MAX_ITERATIONS; i++) {
		newTarget = reconstruct(nnf, sourceBorder);
		propagate(i, cost, nnf, Size(source.cols, source.rows), sourceBorder, newTarget);
		randSearch(1, max(source.rows, source.cols), nnf, cost, source, sourceBorder, newTarget);
	}
}
/*****************************
INITIALIZE
******************************/
void initialize(const Mat&source, const Mat&target, const Mat&sourceBorder,
	const Mat&targetBorder, Mat&nnf, Mat&cost) {
	for (int row = 0; row < nnf.rows; row++) {
		for (int col = 0; col < nnf.cols; col++) {

			double r1 = (double)rand() / RAND_MAX;
			double r2 = (double)rand() / RAND_MAX;
			if (r1 >= 1)r1 = 0.999;
			if (r2 >= 1)r2 = 0.999;
			int offsetRow = int(r1*source.rows) - row;
			int offsetCol = int(r2*source.cols) - col;

			nnf.at<Vec2i>(row, col) = Vec2i(offsetRow, offsetCol);
			if (offsetCol + col >= source.cols || offsetCol + col < 0)cout << offsetCol + col << endl;
			if (offsetRow + row >= source.rows || offsetRow + row < 0)cout << offsetRow + row << endl;

			Mat sourcePatch = sourceBorder(Rect(col + offsetCol, row + offsetRow, SIZE, SIZE));
			Mat targetPatch = targetBorder(Rect(col, row, SIZE, SIZE));

			cost.at<float>(row, col) = norm(sourcePatch, targetPatch);
		}
	}
}

/*****************************
PROPAGATE
******************************/
void propagate(int iteration, Mat &cost, Mat &nnf, Size s, const Mat&sourceBorder, const Mat&targetBorder) {
	Rect bounds(Point(0, 0), s);
	int propDir = iteration % 2 == 0 ? -1 : 1;

	int startRow = (nnf.rows - 1) - (iteration % 2)*(nnf.rows - 1);
	int startCol = (nnf.cols - 1) - (iteration % 2)*(nnf.cols - 1);
	int endRow = nnf.rows - 1 - startRow;
	int endCol = nnf.cols - 1 - startCol;

	for (int row = startRow; row != endRow; row += propDir) {
		for (int col = startCol; col != endCol; col += propDir) {
			float *c = &cost.at<float>(row, col);
			Vec2i *o = &nnf.at<Vec2i>(row, col);

			int neighbourX = col - propDir;
			int neighbourY = row - propDir;

			if (neighbourX >= 0 && neighbourX < nnf.cols) {
				Vec2i *oX = &nnf.at<Vec2i>(row, neighbourX);
				Point p(col + (*oX)[1], row + (*oX)[0]);
				if (cost.at<float>(row, neighbourX) < *c && bounds.contains(p)) {
					*o = nnf.at<Vec2i>(row, neighbourX);
					*c = norm(sourceBorder(Rect((*o)[1] + col, (*o)[0] + row, SIZE, SIZE)),
						targetBorder(Rect(col, row, SIZE, SIZE)));
				}
			}
			if (neighbourY >= 0 && neighbourY < nnf.rows) {
				Vec2i *oY = &nnf.at<Vec2i>(neighbourY, col);
				Point p(col + (*oY)[1], row + (*oY)[0]);
				if (cost.at<float>(neighbourY, col) < *c && bounds.contains(p)) {
					*o = nnf.at<Vec2i>(neighbourY, col);
					*c = norm(sourceBorder(Rect((*o)[1] + col, (*o)[0] + row, SIZE, SIZE)),
						targetBorder(Rect(col, row, SIZE, SIZE)));
				}
			}
		}
	}
}
/*****************************
RANDOM SEARCH
******************************/
void randSearch(int step, int radius, Mat &nnf, Mat &cost, const Mat&source, const Mat&sourceBorder, const Mat& targetBorder) {
	Rect bounds(0, 0, source.cols, source.rows);
	int currentRadius = radius;

	for (int row = 0; row < nnf.rows; row++) {
		for (int col = 0; col < nnf.cols; col += step) {
			if (col >= nnf.cols) continue;
			currentRadius = radius;
			while (currentRadius > 1) {

				Vec2i offset = nnf.at<Vec2i>(row, col);
				Point newOff = randomPoint(currentRadius, Point(offset[1], offset[0]));
				if (newOff.y + row < 0) {
					int newY = -(newOff.y + row);
					newOff.y = newY - row;
				}
				if (newOff.y + row >= source.rows) {
					int newY = 2 * source.rows - 1 - (newOff.y + row);
					newOff.y = newY - row;
				}
				if (newOff.x + col < 0) {
					int newX = -(newOff.x + col);
					newOff.x = newX - col;
				}
				if (newOff.x + col >= source.cols) {
					int newX = 2 * source.cols - 1 - (newOff.x + col);
					newOff.x = newX - col;
				}

				if (!bounds.contains(newOff + Point(col, row))) { currentRadius /= 2; continue; };

				float newNorm = norm(sourceBorder(Rect(col + newOff.x, row + newOff.y, SIZE, SIZE)),
					targetBorder(Rect(col, row, SIZE, SIZE)));

				if (newNorm < cost.at<float>(row, col)) {
					cost.at<float>(row, col) = newNorm;
					nnf.at<Vec2i>(row, col) = Vec2i(newOff.y, newOff.x);
				}
				currentRadius /= 2;
			}
		}
	}
}

/****************************
UP SAMPLE
*****************************/
Mat upSample(const Mat&nnf, Size s) {
	Mat newNNF(s.height, s.width, nnf.type());
	for (int row = 0; row < newNNF.rows; row++) {
		for (int col = 0; col < newNNF.cols; col++) {
			Vec2i offset = nnf.at<Vec2i>(row / 2, col / 2) * 2;
			offset[0] += col % 2;
			offset[1] += row % 2;
			if (offset[0] + row >= s.height) { offset[0]--; }
			if (offset[1] + col >= s.width) { offset[1]--; }

			newNNF.at<Vec2i>(row, col) = offset;
		}
	}
	return newNNF;
}

/*****************************
GET COST
******************************/
void getCost(Mat&cost, const Mat& nnf, const Mat&sourceBorder, const Mat&targetBorder) {
	cost = Mat(nnf.rows, nnf.cols, CV_32F);
	for (int row = 0; row < cost.rows; row++) {
		for (int col = 0; col < cost.cols; col++) {
			Vec2i offset = nnf.at<Vec2i>(row, col);
			Rect targetIndex(col, row, SIZE, SIZE);
			Rect sourceIndex(col + offset[1], row + offset[0], SIZE, SIZE);


			cost.at<float>(row, col) = norm(sourceBorder(sourceIndex), targetBorder(targetIndex));
		}
	}
}

/*****************************
RANDOM POINT
******************************/
Point randomPoint(int radius, Point origin) {
	double t = (2 * CV_PI)*((double)rand() / RAND_MAX);
	double z = ((double)rand() / RAND_MAX)*radius +
		((double)rand() / RAND_MAX)*radius;
	double r = z > radius ? 2 * radius - z : z;
	return Point(r*cos(t) + origin.x, r*sin(t) + origin.y);
}