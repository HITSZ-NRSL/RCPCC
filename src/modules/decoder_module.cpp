#include "decoder_module.h"
DecoderModule::DecoderModule(float pitch_precision, float yaw_precision, int tile_size) : pitch_precision(pitch_precision), yaw_precision(yaw_precision), tile_size(tile_size)
{

    row = (VERTICAL_DEGREE / yaw_precision);
    row = ((row + tile_size - 1) / tile_size) * tile_size;
    col = HORIZONTAL_DEGREE / pitch_precision + tile_size;
    col = ((col + tile_size - 1) / tile_size) * tile_size;
    r_mat = cv::Mat(row, col, CV_32FC1, 0.f);
    b_mat = cv::Mat(row / tile_size, col / tile_size, CV_32SC1, 0.f);
    occ_mat = cv::Mat(row / tile_size, col / tile_size, CV_32SC1, 0.f);
    dct_mat = Matrix<double>(row, std::vector<double>(col, 0.0));
    unfit_mask_mat = Matrix<bool>(row, std::vector<bool>(col, false));
    unfit_nums = std::vector<float>(row * col, 0.f);
}
DecoderModule::DecoderModule(const std::vector<char> &data, int tile_size, bool use_compress, int ksample) : tile_size(tile_size)
{
    bool flag;
    if (use_compress)
    {
        flag = q_deserializeData(decompressData(data), yaw_precision, pitch_precision, b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths, dct_mat);
    }
    else
    {
        flag = deserializeData(data, b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths, dct_mat);
    }

    if (!flag)
    {
        cout << "deserialize data failed" << endl;
        return;
    }
    row = idx_sizes[0] * tile_size;
    col = idx_sizes[1] * tile_size;
    r_mat = cv::Mat(row, col, CV_32FC1, 0.f);
    unfit_mask_mat = Matrix<bool>(row, std::vector<bool>(col, false));
    unfit_nums = std::vector<float>(row * col, 0.f);
    if (ksample >1)//anchor
    {
        decode(b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths, dct_mat, ksample);
        cout << "ksample:" << ksample << endl;
    }
    else
    {
        decode(b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths, dct_mat);
    }
}

DecoderModule::~DecoderModule()
{
}
void DecoderModule(const std::vector<char> &data, int tile_size, bool use_compress, int ksamlple)
{
}
void DecoderModule::decode(cv::Mat &b_mat, const int *idx_sizes,
                           std::vector<cv::Vec4f> &coefficients, cv::Mat &occ_mat,
                           std::vector<int> &tile_fit_lengths,
                           Matrix<double> &dct_mat)
{ // decode的r_mat是float类型的
    // decoder::single_channel_decode(r_mat, b_mat, idx_sizes,
    //                                coefficients, occ_mat,
    //                                tile_fit_lengths,
    //                                unfit_nums, tile_size);
    decoder::single_channel_decode(r_mat, b_mat, idx_sizes,
                                   coefficients, occ_mat,
                                   tile_fit_lengths,
                                   tile_size, dct_mat);
    cout << "tile_fit_lengths size: " << tile_fit_lengths.size() << endl;
    cout << "coefficients size: " << coefficients.size() << endl;
    cout << "b_mat size: " << b_mat.size() << endl;
    cout << "r_mat size: " << r_mat.size() << endl;
    cout << "occ_mat size: " << occ_mat.size() << endl;

    // cv::Mat mask = cv::Mat::zeros(row, col, CV_32FC1);
    // for (int i = 0; i < row; i++)
    // {
    //     for (int j = 0; j < col; j++)
    //     {

    //         mask.at<float>(i, j) = r_mat.at<float>(i, j);
    //     }
    // }
    // cv::imshow("r_mat", mask);
    // cv::waitKey(0);

    restore_pcloud(r_mat, pitch_precision, yaw_precision, restored_pcloud);
    cout << "pointcloud size: " << restored_pcloud.size() << endl;
}
// ksample version
void DecoderModule::decode(cv::Mat &b_mat, const int *idx_sizes,
                           std::vector<cv::Vec4f> &coefficients, cv::Mat &occ_mat,
                           std::vector<int> &tile_fit_lengths,
                           Matrix<double> &dct_mat, int ksamlple)
{
    Matrix<float> extra_pc;
    decoder::single_channel_decode(r_mat, b_mat, idx_sizes,
                                   coefficients, occ_mat,
                                   tile_fit_lengths,
                                   tile_size, dct_mat, ksamlple, extra_pc);
    cout << "tile_fit_lengths size: " << tile_fit_lengths.size() << endl;
    cout << "coefficients size: " << coefficients.size() << endl;
    cout << "b_mat size: " << b_mat.size() << endl;
    cout << "r_mat size: " << r_mat.size() << endl;
    cout << "occ_mat size: " << occ_mat.size() << endl;

    // cv::Mat mask = cv::Mat::zeros(row, col, CV_32FC1);
    // for (int i = 0; i < row; i++)
    // {
    //     for (int j = 0; j < col; j++)
    //     {

    //         mask.at<float>(i, j) = r_mat.at<float>(i, j);
    //     }
    // }
    // cv::imshow("r_mat", mask);
    // cv::waitKey(0);

    restore_pcloud(r_mat, pitch_precision, yaw_precision, restored_pcloud);
    restore_extra_pcloud(pitch_precision, yaw_precision, restored_pcloud, extra_pc);
    cout<<"Extra pointcloud size: "<<extra_pc.size()<<endl;

    cout << "pointcloud size: " << restored_pcloud.size() << endl;
}

void DecoderModule::unpackdata(const std::vector<char> &data)
{
    deserializeData(data, b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths, dct_mat);
}

void DecoderModule::decodeFromData(const std::vector<char> &data, bool use_compress)
{
    if (use_compress)
    {
        unpackdata(decompressData(data));
    }
    else
    {
        unpackdata(data);
    }

    decode(b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths, dct_mat);
}