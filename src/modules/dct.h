#include <stdio.h>
#include <fftw3.h>
#include <time.h>
#include <fftw3.h>
#include <vector>
#include <utility>
#include <iostream>
#include <algorithm>
#include "config.h"
// include opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
// Assuming Matrix is defined as follows
template <typename T>
using Matrix = vector<vector<T>>;

//SADCT
pair<Matrix<double>, Matrix<bool>> sadct(const Matrix<double> &data, const Matrix<bool> &mask);
Matrix<double> saidct(const Matrix<double> &data, const Matrix<bool> &mask);
//Delta-SADCT
pair<Matrix<double>, Matrix<bool>> delta_sadct(const Matrix<double> &data, const Matrix<bool> &mask);
Matrix<double> delta_saidct(const Matrix<double> &data, const Matrix<bool> &mask);

cv::Mat saidct(const cv::Mat &data, const cv::Mat &mask);
pair<cv::Mat, cv::Mat> sadct(const cv::Mat &data, const cv::Mat &mask);