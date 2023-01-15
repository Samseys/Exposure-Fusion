#include "main.hpp"
string folder;
int main(int argc, char **argv) {
    if (argc == 1) {
        printf("Specify the folder containing the input images.\n");
        return EXIT_FAILURE;
    }
    folder = argv[1];

    if (!fs::is_directory(folder)) {
        printf("The given path does not specify a folder. (Remove the last \\ if present)\n");
        return EXIT_FAILURE;
    }

    vector<Mat> images;
    loadImages(images, folder);

    if (images.size() == 0) {
        printf("The folder is empty.\n");
        return EXIT_FAILURE;
    }

    vector<Mat> weightMaps;

    // adjust the weights for contrast, saturation and well exposedness
    double weights[] = {1, 1, 1};
    computeWeightMaps(weightMaps, images, weights);

    Mat result;
    laplacianBlending(result, images, weightMaps);
    result.convertTo(result, CV_8UC1, 255);

    fs::create_directories(folder + "\\out");
    imwrite(folder + "\\out\\" + "out.jpg", result);
    return EXIT_SUCCESS;
}

void loadImages(vector<Mat> &images, string folder) {
    for (const auto &item : fs::directory_iterator(folder)) {
        fs::path path = item.path();
        if (!path.has_extension()) continue;
        string extension = path.extension().string();
        if (!(extension == ".jpg" || extension == ".png" || extension == ".jpeg"))
            continue;
        Mat image = imread(samples::findFile(path.string()));
        image.convertTo(image, CV_64FC3, 1.0 / 255);
        images.push_back(image);
    }
}

void computeWeightMaps(vector<Mat> &result, vector<Mat> images, double weights[]) {
    Mat totalWeight = Mat::zeros(images[0].size(), CV_64FC1);
    for (Mat image : images) {
        Mat contrastMetric;
        getContrastMetric(contrastMetric, image);
        Mat saturationMetric;
        getSaturationMetric(saturationMetric, image);
        Mat wellExposednessMetric;
        getWellExposednessMetric(wellExposednessMetric, image);

        pow(contrastMetric, weights[0], contrastMetric);
        pow(saturationMetric, weights[1], saturationMetric);
        pow(wellExposednessMetric, weights[2], wellExposednessMetric);

        Mat weightMap;
        weightMap = contrastMetric.mul(saturationMetric).mul(wellExposednessMetric) + 1e-12;

        result.push_back(weightMap);
        totalWeight += weightMap;
    }

    for (int i = 0; i < result.size(); i++)
        result[i] /= totalWeight;
}

void getContrastMetric(Mat &result, Mat image) {
    image.convertTo(result, CV_8UC3, 255);
    cvtColor(result, result, COLOR_BGR2GRAY);
    result.convertTo(result, CV_64FC1, 1.0 / 255);
    GaussianBlur(result, result, Size(5, 5), 1);
    Laplacian(result, result, CV_64FC1, 5);
    result = abs(result);
}

void getSaturationMetric(Mat &result, Mat image) {
    result = Mat(image.size(), CV_64FC1);
    for (int x = 0; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {
            double mean = (image.at<Vec3d>(x, y)[0] + image.at<Vec3d>(x, y)[1] + image.at<Vec3d>(x, y)[2]) / 3;
            double variance = pow(image.at<Vec3d>(x, y)[0] - mean, 2) +
                              pow(image.at<Vec3d>(x, y)[1] - mean, 2) +
                              pow(image.at<Vec3d>(x, y)[2] - mean, 2);

            result.at<double>(x, y) = sqrt(variance / 3);
        }
    }
}

void getWellExposednessMetric(Mat &result, Mat image) {
    result = Mat(image.size(), CV_64FC1);
    double sigma = 0.2;
    for (int x = 0; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {
            double blue = exp(-(pow(image.at<Vec3d>(x, y)[0] - 0.5, 2)) / (2 * sigma * sigma));
            double green = exp(-(pow(image.at<Vec3d>(x, y)[1] - 0.5, 2)) / (2 * sigma * sigma));
            double red = exp(-(pow(image.at<Vec3d>(x, y)[2] - 0.5, 2)) / (2 * sigma * sigma));
            result.at<double>(x, y) = blue * green * red;
        }
    }
}

void laplacianBlending(Mat &result, vector<Mat> images, vector<Mat> weightMap) {
    int pyrLevels = (int)(log(min({images[0].rows, images[0].cols})) / log(2));
    vector<Mat> laplacianResult;
    generateGaussianPyramid(laplacianResult, Mat::zeros(images[0].size(), CV_64FC3), pyrLevels);
    for (int i = 0; i < images.size(); i++) {
        vector<Mat> gaussianPyramidWeights;
        generateGaussianPyramid(gaussianPyramidWeights, weightMap[i], pyrLevels);

        vector<Mat> laplacianPyramidImage;
        generateLaplacianPyramid(laplacianPyramidImage, images[i], pyrLevels);

        for (int k = 0; k < laplacianResult.size(); k++) {
            for (int x = 0; x < laplacianResult[k].rows; x++) {
                for (int y = 0; y < laplacianResult[k].cols; y++) {
                    laplacianResult[k].at<Vec3d>(x, y) += laplacianPyramidImage[k].at<Vec3d>(x, y) * gaussianPyramidWeights[k].at<double>(x, y);
                }
            }
        }
    }

    collapse(result, laplacianResult, pyrLevels);
}

void generateGaussianPyramid(vector<Mat> &result, Mat image, int levels) {
    result.push_back(image.clone());
    for (int i = 1; i < levels; i++) {
        image = result[i - 1];
        Mat toAppend;
        pyramidDown(image, toAppend);
        result.push_back(toAppend);
    }
}

void generateLaplacianPyramid(vector<Mat> &result, Mat image, int levels) {
    vector<Mat> gaussianPyramid;
    generateGaussianPyramid(gaussianPyramid, image, levels);

    for (int i = 0; i < levels - 1; i++) {
        Mat pyrLevel = gaussianPyramid[i].clone();
        Mat pyrLevelUpscaled;
        int odd[2];
        odd[0] = pyrLevel.cols - gaussianPyramid[i + 1].cols * 2;
        odd[1] = pyrLevel.rows - gaussianPyramid[i + 1].rows * 2;
        pyramidUp(gaussianPyramid[i + 1], pyrLevelUpscaled, odd);
        pyrLevel -= pyrLevelUpscaled;
        result.push_back(pyrLevel);
    }
    result.push_back(gaussianPyramid[levels - 1]);
}

void pyramidDown(Mat input, Mat &output) {
    input = input.clone();
    double gaussianKernelArray[5] = {0.0625, 0.25, 0.375, 0.25, 0.0625};
    Mat gaussianKernel = Mat(Size(5, 1), CV_64F, &gaussianKernelArray);
    sepFilter2D(input, input, -1, gaussianKernel, gaussianKernel.t());
    int channels = input.channels();
    output = Mat(Size((input.cols + 1) / 2, (input.rows + 1) / 2), input.type());
    switch (channels) {
        case 1:
            for (int x = 0; x < output.rows; x++) {
                for (int y = 0; y < output.cols; y++) {
                    output.at<double>(x, y) = input.at<double>(2 * x, 2 * y);
                }
            }
            break;
        case 3:
            for (int x = 0; x < output.rows; x++) {
                for (int y = 0; y < output.cols; y++) {
                    output.at<Vec3d>(x, y) = input.at<Vec3d>(2 * x, 2 * y);
                }
            }
            break;
        default:
            break;
    }
}

void pyramidUp(Mat input, Mat &output, int odd[2]) {
    int channels = input.channels();
    copyMakeBorder(input, input, 1, 1, 1, 1, BORDER_REPLICATE);
    output = Mat::zeros(Size(2 * input.cols, 2 * input.rows), input.type());
    switch (channels) {
        case 1:
            for (int x = 0; x < input.rows; x++) {
                for (int y = 0; y < input.cols; y++) {
                    output.at<double>(2 * x, 2 * y) = input.at<double>(x, y);
                }
            }
            break;
        case 3:
            for (int x = 0; x < input.rows; x++) {
                for (int y = 0; y < input.cols; y++) {
                    output.at<Vec3d>(2 * x, 2 * y) = input.at<Vec3d>(x, y);
                }
            }
            break;
        default:
            break;
    }

    double gaussianKernelArray[5] = {0.0625, 0.25, 0.375, 0.25, 0.0625};
    Mat gaussianKernel = Mat(Size(5, 1), CV_64F, &gaussianKernelArray);
    sepFilter2D(output, output, -1, gaussianKernel, gaussianKernel.t());
    output *= 4;
    Mat aux = output.colRange(2, output.cols - (2 - odd[0])).rowRange(2, output.rows - (2 - odd[1]));
    output = aux.clone();
}

void collapse(Mat &result, vector<Mat> laplacianPyramid, int pyrLevels) {
    result = laplacianPyramid[pyrLevels - 1];
    for (int i = pyrLevels - 2; i >= 0; i--) {
        int odd[2];
        odd[0] = laplacianPyramid[i].cols - result.cols * 2;
        odd[1] = laplacianPyramid[i].rows - result.rows * 2;
        pyramidUp(result, result, odd);
        result += laplacianPyramid[i];
    }

    for (int x = 0; x < result.rows; x++) {
        for (int y = 0; y < result.cols; y++) {
            for (int i = 0; i < 3; i++) {
                if (result.at<Vec3d>(x, y)[i] > 1)
                    result.at<Vec3d>(x, y)[i] = 1;
                else if (result.at<Vec3d>(x, y)[i] < 0)
                    result.at<Vec3d>(x, y)[i] = 0;
            }
        }
    }
}