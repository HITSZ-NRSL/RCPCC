#ifndef SERIALIZE_H
#define SERIALIZE_H
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iostream>
#include "config.h"

using namespace std;
template <typename T>
using Matrix = vector<vector<T>>;
//[yaw_precision,pitch_precision,plane_threshold,quantization_step]
extern double quantization_dict[16][4];
// 序列化cv::Mat对象到std::ostringstream的函数
void serializeMatToStream(const cv::Mat &mat, std::ostringstream &oss);

// 反序列化cv::Mat对象从std::istringstream的函数
void deserializeMatFromStream(std::istringstream &iss, cv::Mat &mat);

// 序列化std::vector<cv::Vec4f>对象到std::ostringstream的函数
void serializeVec4fVectorToStream(const std::vector<cv::Vec4f> &vec, std::ostringstream &oss);

// 反序列化std::vector<cv::Vec4f>对象从std::istringstream的函数
void deserializeVec4fVectorFromStream(std::istringstream &iss, std::vector<cv::Vec4f> &vec);

// 序列化Matrix<double>对象到std::ostringstream的函数
void serializeMatrixToStream(const Matrix<double> &mat, std::ostringstream &oss);

// 反序列化Matrix<double>对象从std::istringstream的函数
void deserializeMatrixFromStream(std::istringstream &iss, Matrix<double> &mat);

// 序列化std::vector<int>对象到std::ostringstream的函数
void serializeIntVectorToStream(const std::vector<int> &vec, std::ostringstream &oss);

// 反序列化std::vector<int>对象从std::istringstream的函数
void deserializeIntVectorFromStream(std::istringstream &iss, std::vector<int> &vec);

// 将所有数据序列化到一个std::vector<char>中的函数
std::vector<char> serializeData(const cv::Mat &b_mat, const int *idx_sizes,
                                const std::vector<cv::Vec4f> &coefficients, const cv::Mat &occ_mat,
                                const std::vector<int> &tile_fit_lengths, const Matrix<double> &dct_mat);

// 从std::vector<char>反序列化所有数据的函数
bool deserializeData(const std::vector<char> &data,
                     cv::Mat &b_mat, int *idx_sizes,
                     std::vector<cv::Vec4f> &coefficients, cv::Mat &occ_mat,
                     std::vector<int> &tile_fit_lengths, Matrix<double> &dct_mat);

/****量化编码*****/

std::vector<char> q_serializeData(const int q_level, const cv::Mat &b_mat, const int *idx_sizes,
                                  const std::vector<cv::Vec4f> &coefficients, const cv::Mat &occ_mat,
                                  const std::vector<int> &tile_fit_lengths, const Matrix<double> &dct_mat);

bool q_deserializeData(const std::vector<char> &data,
                       float &pitch_precision, float &yaw_precision,
                       cv::Mat &b_mat, int *idx_sizes,
                       std::vector<cv::Vec4f> &coefficients, cv::Mat &occ_mat,
                       std::vector<int> &tile_fit_lengths, Matrix<double> &dct_mat);

#endif