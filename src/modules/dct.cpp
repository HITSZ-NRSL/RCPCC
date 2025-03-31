// this code is for the DCT module
// author: Zoratt. mail:1720049199@qq.com
// date: 2024-8-23
#include "dct.h"

#include <opencv2/opencv.hpp>
#include <fftw3.h>

std::pair<cv::Mat, cv::Mat> sadct(const cv::Mat &data, const cv::Mat &mask)
{
    assert(data.type() == CV_32FC4);
    assert(mask.type() == CV_32SC1);

    int rows = data.rows;
    int cols = data.cols;

    cv::Mat temp_img(rows, cols, CV_64F, cv::Scalar(0.0));
    cv::Mat mask_(rows, cols, CV_32SC1, cv::Scalar(0));

    std::vector<double> col_data(rows);
    std::vector<double> dct_col(rows);

    // 对列进行Shift和DCT
    for (int j = 0; j < cols; ++j)
    {
        int N = 0;
        for (int i = 0; i < rows; ++i)
        {
            if (mask.at<int>(i, j))
            {
                col_data[N++] = static_cast<double>(data.at<cv::Vec4f>(i, j)[0]);
            }
        }

        if (N > 0)
        {
            fftw_plan dct_plan = fftw_plan_r2r_1d(N, col_data.data(), dct_col.data(), FFTW_REDFT10, FFTW_ESTIMATE);
            fftw_execute(dct_plan);
            fftw_destroy_plan(dct_plan);

            for (int i = 0; i < N; ++i)
            {
                temp_img.at<double>(i, j) = dct_col[i];
                mask_.at<int>(i, j) = 1;
            }
        }
    }

    cv::Mat temp_img_new(rows, cols, CV_64F, cv::Scalar(0.0));
    cv::Mat mask_final(rows, cols, CV_32SC1, cv::Scalar(0));

    std::vector<double> row_data(cols);
    std::vector<double> dct_row(cols);

    // 对行进行Shift和DCT
    for (int i = 0; i < rows; ++i)
    {
        int N = 0;
        for (int j = 0; j < cols; ++j)
        {
            if (mask_.at<int>(i, j))
            {
                row_data[N++] = temp_img.at<double>(i, j);
            }
        }

        if (N > 0)
        {
            fftw_plan dct_plan = fftw_plan_r2r_1d(N, row_data.data(), dct_row.data(), FFTW_REDFT10, FFTW_ESTIMATE);
            fftw_execute(dct_plan);
            fftw_destroy_plan(dct_plan);

            for (int j = 0; j < N; ++j)
            {
                temp_img_new.at<double>(i, j) = dct_row[j];
                mask_final.at<int>(i, j) = 1;
            }
        }
    }

    return std::make_pair(temp_img_new, mask_final);
}
cv::Mat saidct(const cv::Mat &data, const cv::Mat &mask)
{
    assert(data.type() == CV_32FC4);
    assert(mask.type() == CV_32SC1);

    int rows = mask.rows;
    int cols = mask.cols;
    cv::Mat coeff(rows, cols, CV_64F, cv::Scalar(0.0));

    cv::Mat _mask(rows, cols, CV_32SC1, cv::Scalar(0));
    cv::Mat _mask1(rows, cols, CV_32SC1, cv::Scalar(0));
    cv::Mat index_mask1(rows, cols, CV_32SC2, cv::Scalar(-1, -1));
    cv::Mat index_mask(rows, cols, CV_32SC2, cv::Scalar(-1, -1));
    cv::Mat index_mask2(rows, cols, CV_32SC2, cv::Scalar(-1, -1));

    // Step 1: Generate masks and index maps
    for (int j = 0; j < cols; ++j)
    {
        int l = 0;
        for (int i = 0; i < rows; ++i)
        {
            if (mask.at<int>(i, j))
            {
                _mask1.at<int>(l, j) = 1;
                index_mask1.at<cv::Vec2i>(l, j) = cv::Vec2i(i, j);
                l++;
            }
        }
    }

    for (int i = 0; i < rows; ++i)
    {
        int l = 0;
        for (int j = 0; j < cols; ++j)
        {
            if (_mask1.at<int>(i, j))
            {
                _mask.at<int>(i, l) = 1;
                index_mask.at<cv::Vec2i>(i, l) = index_mask1.at<cv::Vec2i>(i, j);
                index_mask2.at<cv::Vec2i>(i, l) = cv::Vec2i(i, j);
                l++;
            }
        }
    }

    // Step 2: Fill coefficients to the original dimensions
    for (int i = 0; i < data.rows; ++i)
    {
        for (int j = 0; j < data.cols; ++j)
        {
            coeff.at<double>(i, j) = static_cast<double>(data.at<cv::Vec4f>(i, j)[0]);
        }
    }

    // Step 3: Perform IDCT on each row
    cv::Mat temp_coeff(rows, cols, CV_64F, cv::Scalar(0.0));
    for (int i = 0; i < rows; ++i)
    {
        int valid_num = cv::countNonZero(_mask.row(i));
        if (valid_num > 0)
        {
            std::vector<double> row_data(valid_num);
            for (int j = 0; j < valid_num; ++j)
            {
                row_data[j] = coeff.at<double>(i, j);
            }

            std::vector<double> idct_row(valid_num);
            fftw_plan idct_plan = fftw_plan_r2r_1d(valid_num, row_data.data(), idct_row.data(), FFTW_REDFT01, FFTW_ESTIMATE);
            fftw_execute(idct_plan);
            fftw_destroy_plan(idct_plan);

            for (int j = 0; j < valid_num; ++j)
            {
                temp_coeff.at<double>(i, j) = idct_row[j] / (2.0 * valid_num);
            }
        }
    }
    coeff = temp_coeff;

    // Step 4: Map IDCT rows back to column positions
    temp_coeff.setTo(0);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (_mask.at<int>(i, j))
            {
                cv::Vec2i index = index_mask2.at<cv::Vec2i>(i, j);
                temp_coeff.at<double>(index[0], index[1]) = coeff.at<double>(i, j);
            }
        }
    }
    coeff = temp_coeff;

    // Step 5: Perform IDCT on each column
    temp_coeff.setTo(0);
    for (int j = 0; j < cols; ++j)
    {
        int valid_num = 0;
        for (int i = 0; i < rows; ++i)
        {
            if (_mask1.at<int>(i, j))
                valid_num++;
        }

        if (valid_num > 0)
        {
            std::vector<double> col_data(valid_num);
            for (int i = 0; i < valid_num; ++i)
            {
                col_data[i] = coeff.at<double>(i, j);
            }

            std::vector<double> idct_col(valid_num);
            fftw_plan idct_plan = fftw_plan_r2r_1d(valid_num, col_data.data(), idct_col.data(), FFTW_REDFT01, FFTW_ESTIMATE);
            fftw_execute(idct_plan);
            fftw_destroy_plan(idct_plan);

            for (int i = 0; i < valid_num; ++i)
            {
                temp_coeff.at<double>(i, j) = idct_col[i] / (2.0 * valid_num);
            }
        }
    }
    coeff = temp_coeff;

    // Step 6: Map IDCT columns back to original positions
    cv::Mat real_result(rows, cols, CV_64F, cv::Scalar(0.0));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (_mask1.at<int>(i, j))
            {
                cv::Vec2i index = index_mask1.at<cv::Vec2i>(i, j);
                real_result.at<double>(index[0], index[1]) = coeff.at<double>(i, j);
            }
        }
    }

    return real_result;
}
// now use

pair<Matrix<double>, Matrix<bool>> sadct(const Matrix<double> &data, const Matrix<bool> &mask)
{
    int rows = data.size();
    int cols = data[0].size();
    Matrix<double> temp_img(rows, vector<double>(cols, 0.0));
    Matrix<bool> mask_(rows, vector<bool>(cols, false));

    vector<double> col_data(rows);
    vector<double> dct_col(rows);

    // Shift and apply DCT on columns
    for (int j = 0; j < cols; ++j)
    {
        int N = 0;
        for (int i = 0; i < rows; ++i)
        {
            if (mask[i][j])
            {
                col_data[N++] = data[i][j];
            }
        }

        if (N > 0)
        {
            fftw_plan dct_plan = fftw_plan_r2r_1d(N, col_data.data(), dct_col.data(), FFTW_REDFT10, FFTW_ESTIMATE);
            fftw_execute(dct_plan);
            fftw_destroy_plan(dct_plan);

            for (int i = 0; i < N; ++i)
            {
                temp_img[i][j] = dct_col[i] / (SQRT_2 * sqrt(N));
                mask_[i][j] = true;
            }
        }
    }

    Matrix<double> temp_img_new(rows, vector<double>(cols, 0.0));
    Matrix<bool> mask_final(rows, vector<bool>(cols, false));

    vector<double> row_data(cols);
    vector<double> dct_row(cols);

    // Shift and apply DCT on rows
    for (int i = 0; i < rows; ++i)
    {
        int N = 0;
        for (int j = 0; j < cols; ++j)
        {
            if (mask_[i][j])
            {
                row_data[N++] = temp_img[i][j];
            }
        }

        if (N > 0)
        {
            fftw_plan dct_plan = fftw_plan_r2r_1d(N, row_data.data(), dct_row.data(), FFTW_REDFT10, FFTW_ESTIMATE);
            fftw_execute(dct_plan);
            fftw_destroy_plan(dct_plan);

            for (int j = 0; j < N; ++j)
            {
                temp_img_new[i][j] = dct_row[j] / (SQRT_2 * sqrt(N));
                mask_final[i][j] = true;
            }
        }
    }

    return make_pair(temp_img_new, mask_final);
}

Matrix<double> saidct(const Matrix<double> &data, const Matrix<bool> &mask)
{
    int rows = mask.size();
    int cols = mask[0].size();
    Matrix<double> coeff(rows, vector<double>(cols, 0.0));

    Matrix<bool> _mask(rows, vector<bool>(cols, false));
    Matrix<bool> _mask1(rows, vector<bool>(cols, false));
    Matrix<pair<int, int>> index_mask1(rows, vector<pair<int, int>>(cols, {-1, -1}));
    Matrix<pair<int, int>> index_mask(rows, vector<pair<int, int>>(cols, {-1, -1}));
    Matrix<pair<int, int>> index_mask2(rows, vector<pair<int, int>>(cols, {-1, -1}));

    // Step 1: Generate masks and index maps
    for (int j = 0; j < cols; ++j)
    {
        int l = 0;
        for (int i = 0; i < rows; ++i)
        {
            if (mask[i][j])
            {
                _mask1[l][j] = true;
                index_mask1[l][j] = {i, j};
                l++;
            }
        }
    }

    for (int i = 0; i < rows; ++i)
    {
        int l = 0;
        for (int j = 0; j < cols; ++j)
        {
            if (_mask1[i][j])
            {
                _mask[i][l] = true;
                index_mask[i][l] = index_mask1[i][j];
                index_mask2[i][l] = {i, j};
                l++;
            }
        }
    }

    // Step 2: Fill coefficients to the original dimensions
    for (int i = 0; i < data.size(); ++i)
    {
        for (int j = 0; j < data[i].size(); ++j)
        {
            coeff[i][j] = data[i][j];
        }
    }

    // Step 3: Perform IDCT on each row
    Matrix<double> temp_coeff(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i)
    {
        int valid_num = std::count(_mask[i].begin(), _mask[i].end(), true);
        if (valid_num > 0)
        {
            vector<double> row_data(valid_num);
            for (int j = 0; j < valid_num; ++j)
            {
                row_data[j] = coeff[i][j];
            }

            // Create FFTW plan for IDCT
            vector<double> idct_row(valid_num);
            fftw_plan idct_plan = fftw_plan_r2r_1d(valid_num, row_data.data(), idct_row.data(), FFTW_REDFT01, FFTW_ESTIMATE);
            fftw_execute(idct_plan);
            fftw_destroy_plan(idct_plan);

            // Scale the result
            for (int j = 0; j < valid_num; ++j)
            {
                temp_coeff[i][j] = idct_row[j] / (SQRT_2 * sqrt(valid_num));
            }
        }
    }
    coeff = temp_coeff;

    // Step 4: Map IDCT rows back to column positions
    temp_coeff.assign(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (_mask[i][j])
            {
                temp_coeff[index_mask2[i][j].first][index_mask2[i][j].second] = coeff[i][j];
            }
        }
    }
    coeff = temp_coeff;

    // Step 5: Perform IDCT on each column
    temp_coeff.assign(rows, vector<double>(cols, 0.0));
    for (int j = 0; j < cols; ++j)
    {
        int valid_num = 0;
        for (int i = 0; i < rows; ++i)
        {
            if (_mask1[i][j])
                valid_num++;
        }

        if (valid_num > 0)
        {
            vector<double> col_data(valid_num);
            for (int i = 0; i < valid_num; ++i)
            {
                col_data[i] = coeff[i][j];
            }

            // Create FFTW plan for IDCT
            vector<double> idct_col(valid_num);
            fftw_plan idct_plan = fftw_plan_r2r_1d(valid_num, col_data.data(), idct_col.data(), FFTW_REDFT01, FFTW_ESTIMATE);
            fftw_execute(idct_plan);
            fftw_destroy_plan(idct_plan);

            // Scale the result
            for (int i = 0; i < valid_num; ++i)
            {
                temp_coeff[i][j] = idct_col[i] / (SQRT_2 * sqrt(valid_num));
            }
        }
    }
    coeff = temp_coeff;

    // Step 6: Map IDCT columns back to original positions
    Matrix<double> real_result(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (_mask1[i][j])
            {
                real_result[index_mask1[i][j].first][index_mask1[i][j].second] = coeff[i][j];
            }
        }
    }
    return real_result;
}

pair<Matrix<double>, Matrix<bool>> delta_sadct(const Matrix<double> &data, const Matrix<bool> &mask)
{
    // make a copy of data
    Matrix<double> temp_img = data;
    // mean
    double sum = 0;
    int cnt = 0;
    for (int i = 0; i < data.size(); ++i)
    {
        for (int j = 0; j < data[i].size(); ++j)
        {
            if (mask[i][j])
            {
                sum += data[i][j];
                cnt += 1;
            }
        }
    }
    sum /= cnt;
    for (int i = 0; i < data.size(); ++i)
    {
        for (int j = 0; j < data[i].size(); ++j)
        {
            if (mask[i][j])
            {
                temp_img[i][j] -= sum;
            }
        }
    }
    double scale = 10000;
    auto res = sadct(temp_img, mask);
    res.first[0][0] = sum * scale;
    // 打印10*10
    // float max = 0;
    // int pos[2] = {0, 0};
    // for (int i = 0; i < res.first.size(); ++i)
    // {
    //     for (int j = 0; j < res.first.size(); ++j)
    //     {
    //         cout << res.first[i][j] << " ";
    //         if (fabs(res.first[i][j]) > max)
    //         {

    //             max = fabs(res.first[i][j]);
    //             pos[0] = i;
    //             pos[1] = j;
    //         }
    //     }
    //     cout << endl;
    // }
    // cout << "max:" << max << "(" << pos[0] << "," << pos[1] << ")" << endl;
    return res;
}

Matrix<double> delta_saidct(const Matrix<double> &data, const Matrix<bool> &mask)
{
    int rows = mask.size();
    int cols = mask[0].size();
    double scale = 10000;
    double mean = data[0][0] / scale;
    Matrix<double> coeff(rows, vector<double>(cols, 0.0));

    Matrix<bool> _mask(rows, vector<bool>(cols, false));
    Matrix<bool> _mask1(rows, vector<bool>(cols, false));
    Matrix<pair<int, int>> index_mask1(rows, vector<pair<int, int>>(cols, {-1, -1}));
    Matrix<pair<int, int>> index_mask(rows, vector<pair<int, int>>(cols, {-1, -1}));
    Matrix<pair<int, int>> index_mask2(rows, vector<pair<int, int>>(cols, {-1, -1}));

    // Step 1: Generate masks and index maps
    for (int j = 0; j < cols; ++j)
    {
        int l = 0;
        for (int i = 0; i < rows; ++i)
        {
            if (mask[i][j])
            {
                _mask1[l][j] = true;
                index_mask1[l][j] = {i, j};
                l++;
            }
        }
    }

    for (int i = 0; i < rows; ++i)
    {
        int l = 0;
        for (int j = 0; j < cols; ++j)
        {
            if (_mask1[i][j])
            {
                _mask[i][l] = true;
                index_mask[i][l] = index_mask1[i][j];
                index_mask2[i][l] = {i, j};
                l++;
            }
        }
    }

    // Step 2: Fill coefficients to the original dimensions
    for (int i = 0; i < data.size(); ++i)
    {
        for (int j = 0; j < data[i].size(); ++j)
        {
            coeff[i][j] = data[i][j];
        }
    }
    // set the DC component to zero
    coeff[0][0] = 0;

    // Step 3: Perform IDCT on each row
    Matrix<double> temp_coeff(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i)
    {
        int valid_num = std::count(_mask[i].begin(), _mask[i].end(), true);
        if (valid_num > 0)
        {
            vector<double> row_data(valid_num);
            for (int j = 0; j < valid_num; ++j)
            {
                row_data[j] = coeff[i][j];
            }

            // Create FFTW plan for IDCT
            vector<double> idct_row(valid_num);
            fftw_plan idct_plan = fftw_plan_r2r_1d(valid_num, row_data.data(), idct_row.data(), FFTW_REDFT01, FFTW_ESTIMATE);
            fftw_execute(idct_plan);
            fftw_destroy_plan(idct_plan);

            // Scale the result
            for (int j = 0; j < valid_num; ++j)
            {
                temp_coeff[i][j] = idct_row[j] / (SQRT_2 * sqrt(valid_num));
            }
        }
    }
    coeff = temp_coeff;

    // correct DC component error
    double nominator = 0;
    double denominator = 0;
    for (int j = 0; j < cols; ++j)
    {
        int valid_num = 0;
        for (int i = 0; i < rows; ++i)
        {
            if (_mask[i][j])
            {
                valid_num++;
            }
        }
        if (valid_num > 0)
        {
            double factor = SQRT_2 * sqrt(valid_num);
            nominator += factor * coeff[0][j];
            denominator += factor;
        }
    }
    double e_dc = nominator / denominator;
    for (int j = 0; j < cols; ++j)
    {
        int valid_num = 0;
        if (_mask[0][j])
        {
            coeff[0][j] += 0;
        }
    }

    // Step 4: Map IDCT rows back to column positions
    temp_coeff.assign(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (_mask[i][j])
            {
                temp_coeff[index_mask2[i][j].first][index_mask2[i][j].second] = coeff[i][j];
            }
        }
    }
    coeff = temp_coeff;

    // Step 5: Perform IDCT on each column
    temp_coeff.assign(rows, vector<double>(cols, 0.0));
    for (int j = 0; j < cols; ++j)
    {
        int valid_num = 0;
        for (int i = 0; i < rows; ++i)
        {
            if (_mask1[i][j])
                valid_num++;
        }

        if (valid_num > 0)
        {
            vector<double> col_data(valid_num);
            for (int i = 0; i < valid_num; ++i)
            {
                col_data[i] = coeff[i][j];
            }

            // Create FFTW plan for IDCT
            vector<double> idct_col(valid_num);
            fftw_plan idct_plan = fftw_plan_r2r_1d(valid_num, col_data.data(), idct_col.data(), FFTW_REDFT01, FFTW_ESTIMATE);
            fftw_execute(idct_plan);
            fftw_destroy_plan(idct_plan);

            // Scale the result
            for (int i = 0; i < valid_num; ++i)
            {
                temp_coeff[i][j] = idct_col[i] / (SQRT_2 * sqrt(valid_num));
            }
        }
    }
    coeff = temp_coeff;

    // Step 6: Map IDCT columns back to original positions
    Matrix<double> real_result(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (_mask1[i][j])
            {
                real_result[index_mask1[i][j].first][index_mask1[i][j].second] = coeff[i][j];
            }
        }
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (mask[i][j])
            {
                real_result[i][j] += mean;
            }
        }
    }
    return real_result;
}