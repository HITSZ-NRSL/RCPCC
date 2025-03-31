#include "encoder_module.h"
#include "io.h"
#include "pcc_module.h"
#include "serializer.h"
#include "binary_compressor.h"
EncoderModule::EncoderModule(float pitch_precision, float yaw_precision, float threshold,
                             int tile_size, int q_level) : pitch_precision(pitch_precision), yaw_precision(yaw_precision),
                                                           threshold(threshold), tile_size(tile_size), q_level(q_level)
{
    cout << "precision: " << pitch_precision << " " << yaw_precision << endl;
    row = (VERTICAL_DEGREE / yaw_precision);
    row = ((row + tile_size - 1) / tile_size) * tile_size;
    col = HORIZONTAL_DEGREE / pitch_precision + tile_size;
    col = ((col + tile_size - 1) / tile_size) * tile_size;
    f_mat = cv::Mat(row, col, CV_32FC4, cv::Scalar(0.f, 0.f, 0.f, 0.f));
    b_mat = cv::Mat(row / tile_size, col / tile_size, CV_32SC1, 0.f);
    occ_mat = cv::Mat(row / tile_size, col / tile_size, CV_32SC1, 0.f);
    dct_mat = Matrix<double>(row, std::vector<double>(col, 0));
    tile_fit_lengths = std::vector<int>();
    coefficients = std::vector<cv::Vec4f>();
    unfit_nums = std::vector<float>();
    idx_sizes[0] = row / tile_size;
    idx_sizes[1] = col / tile_size;
}
EncoderModule::EncoderModule(int tile_size, int q_level):tile_size(tile_size),q_level(q_level)
{   
    yaw_precision=quantization_dict[q_level][0];
    pitch_precision=quantization_dict[q_level][1];
    threshold=quantization_dict[q_level][2];
    row = (VERTICAL_DEGREE / yaw_precision);
    row = ((row + tile_size - 1) / tile_size) * tile_size;
    col = HORIZONTAL_DEGREE / pitch_precision + tile_size;
    col = ((col + tile_size - 1) / tile_size) * tile_size;
    f_mat = cv::Mat(row, col, CV_32FC4, cv::Scalar(0.f, 0.f, 0.f, 0.f));
    b_mat = cv::Mat(row / tile_size, col / tile_size, CV_32SC1, 0.f);
    occ_mat = cv::Mat(row / tile_size, col / tile_size, CV_32SC1, 0.f);
    dct_mat = Matrix<double>(row, std::vector<double>(col, 0));
    tile_fit_lengths = std::vector<int>();
    coefficients = std::vector<cv::Vec4f>();
    unfit_nums = std::vector<float>();
    idx_sizes[0] = row / tile_size;
    idx_sizes[1] = col / tile_size;
}

EncoderModule::~EncoderModule() {}
void EncoderModule::encode(std::vector<point_cloud> &pcloud_data)
{
    PccResult pcc_res;
    /*******************************************************************/
    // initialization

    double proj_time, fit_time;
    float psnr, total_pcloud_size;

    /*******************************************************************/
    // convert range map
    std::cout << "CURRENT pcloud size: " << pcloud_data.size() << std::endl;
    // Characterize Range Map
    // floating map;

    proj_time = map_projection(f_mat, pcloud_data, pitch_precision, yaw_precision, 'e');

    // cv::Mat value_mat = cv::Mat::zeros(row, col, CV_32FC1);
    // for (int i = 0; i < row; i++)
    // {
    //     for (int j = 0; j < col; j++)
    //     {
    //         value_mat.at<float>(i, j) = f_mat.at<cv::Vec4f>(i, j)[0];
    //     }
    // }
    // cv::imshow("value_mat", value_mat);
    // cv::waitKey(0);
    pcc_res.proj_times->push_back(proj_time);

    // compute compression rate: bit-per-point (bpp)
    pcc_res.compression_rate->push_back(8.0f * f_mat.cols * f_mat.rows / pcloud_data.size());

    // loss error compute;
    // psnr = compute_loss_rate<cv::Vec4f>(*f_mat, pcloud_data, pitch_precision, yaw_precision);

    // update the info;
    pcc_res.loss_rate->push_back(psnr);

    std::cout << "Loss rate [PSNR]: " << psnr << " Compression rate: "
              << pcc_res.compression_rate->back() << " bpp." << std::endl;

    /*******************************************************************/
    // fitting range map
    int mat_div_tile_sizes[] = {row / tile_size, col / tile_size};

    // cv::Mat *b_mat = new cv::Mat(row / tile_size, col / tile_size, CV_32SC1, 0.f);
    // cv::Mat *occ_mat = new cv::Mat(row / tile_size, col / tile_size, CV_32SC1, 0.f);

    // encode the occupatjon map
    encoder::encode_occupation_mat(f_mat, occ_mat, tile_size, mat_div_tile_sizes);

    fit_time = encoder::single_channel_encode(f_mat, b_mat, mat_div_tile_sizes, coefficients,
                                              unfit_nums, tile_fit_lengths,
                                              threshold, tile_size);
    cout << "fit_time" << fit_time << endl;

    // point cloud mask
    cv::Mat mask = cv::Mat::zeros(row, col, CV_32SC1);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            if (f_mat.at<cv::Vec4f>(i, j)[0] != 0)
            {
                mask.at<int>(i, j) = 1;
            }
        }
    }
    // clear the fit points tiles
    for (int r_idx = 0; r_idx < mat_div_tile_sizes[0]; r_idx++)
    {
        for (int c_idx = 0; c_idx < mat_div_tile_sizes[1]; c_idx++)
        {
            if (b_mat.at<int>(r_idx, c_idx) == 1)
            {
                for (int i = 0; i < tile_size; i++)
                {
                    for (int j = 0; j < tile_size; j++)
                    {
                        mask.at<int>(r_idx * tile_size + i, c_idx * tile_size + j) = 0;
                    }
                }
            }
        }
    }
    // sadct
    Matrix<double> value_map(row, std::vector<double>(col, 0));
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            value_map[i][j] = f_mat.at<cv::Vec4f>(i, j)[0];
        }
    }
    Matrix<bool> mask_map(row, std::vector<bool>(col, false));
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            mask_map[i][j] = mask.at<int>(i, j) == 1 ? true : false;
        }
    }
    // SADCT
    clock_t start = clock();
    #ifdef USE_SADCT
    auto coeff_res = sadct(value_map, mask_map);
    #else
    auto coeff_res = delta_sadct(value_map, mask_map);
    #endif


 
    clock_t end = clock();
    cout << "SADCT time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    dct_mat = coeff_res.first;


    pcc_res.fit_times->push_back(fit_time);
}

// 序列化数据
std::vector<char> EncoderModule::packData()
{
    std::vector<char> data;
    if (q_level != -1)
    {
        data = q_serializeData(q_level, b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths, dct_mat);
    }
    else
    {
        data = serializeData(b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths, dct_mat);
    }
    return data;
}

std::vector<char> EncoderModule::encodeToData(std::vector<point_cloud> &pcloud_data, bool use_compress)
{
    encode(pcloud_data);
    clock_t start = clock();
    //ANCHOR:use_compress
    auto temp = use_compress ? compressData(packData()) : packData();
    clock_t end = clock();
    cout << "compress time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    return temp;
}
