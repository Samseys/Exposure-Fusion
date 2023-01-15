#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

void loadImages(vector<Mat> &images, string folder);
void laplacianBlending(Mat &result, vector<Mat> images, vector<Mat> weightMap);
void generateGaussianPyramid(vector<Mat> &result, Mat image, int levels);
void generateLaplacianPyramid(vector<Mat> &result, Mat image, int levels);
void pyramidUp(Mat input, Mat &output, int odd[2]);
void pyramidDown(Mat input, Mat& output);
void collapse(Mat &result, vector<Mat> laplacianPyramid, int pyrLevels);
void computeWeightMaps(vector<Mat> &result, vector<Mat> images, double weights[]);
void getContrastMetric(Mat &result, Mat image);
void getSaturationMetric(Mat &result, Mat image);
void getWellExposednessMetric(Mat &result, Mat image);