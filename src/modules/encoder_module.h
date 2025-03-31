#ifndef ENCODER_MODULE_H
#define ENCODER_MODULE_H

#include "encoder.h"
#include "dct.h"
#include "config.h"
class EncoderModule
{
public:
  cv::Mat f_mat;//range image value mat
  cv::Mat b_mat;
  cv::Mat occ_mat;
  Matrix<double> dct_mat;
  std::vector<cv::Vec4f> coefficients;//plane coefficients
  std::vector<int> tile_fit_lengths;
  std::vector<float> unfit_nums;
  float pitch_precision;
  float yaw_precision;
  float threshold;//threshold for plane fitting
  int tile_size;
  int row;
  int col;
  int idx_sizes[2];//size of img divided by tile size
  int q_level;//quantization level 

public:
  EncoderModule(float pitch_precision, float yaw_precision, float threshold,
                int tile_size,int q_level=-1);
  EncoderModule(int tile_size,int q_level);
  
  ~EncoderModule();
  void encode(std::vector<point_cloud> &pcloud_data);
  std::vector<char> packData();
  std::vector<char> encodeToData(std::vector<point_cloud> &pcloud_data,bool use_compress=false);
};
#endif