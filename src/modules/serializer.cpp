
#include "serializer.h"
using namespace std;
// pitch yaw threshold quantization_step
double quantization_dict[16][4] = {
    {0.25, 0.5, 0.1, 0.1},  // 0x
    {0.25, 0.5, 0.2, 0.20}, // 0x
    {0.25, 0.5, 0.4, 0.20}, // good
    {0.5, 1.0, 0.1, 0.2},   // good
    {0.5, 1.0, 0.2, 0.2},   // good
    {1.0, 2.0, 0.4, 0.20},  // 1x
};
// {1.0, 2.0, 0.1, 0.00005},   // 5
//     {1.0, 2.0, 0.5, 0.0005},    // 6
//     {2.0, 2.0, 0.5, 0.001}}
cv::Mat get_dct_mask(const cv::Mat &occ_mat, int tile_size, const cv::Mat &b_mat)
{
    // Shift and apply DCT on columns
    int rows = occ_mat.rows;
    int cols = occ_mat.cols;
    int tt2 = tile_size * tile_size;
    cv::Mat raw_occ_mat(rows * tile_size, cols * tile_size, CV_32SC1, cv::Scalar(0));
    for (int r_idx = 0; r_idx < rows; r_idx++)
    {
        for (int c_idx = 0; c_idx < cols; c_idx++)
        {
            if (b_mat.at<int>(r_idx, c_idx) == 0)
            {
                int occ_code = occ_mat.at<int>(r_idx, c_idx);
                for (int i = 0; i < tt2; i++)
                {
                    if (((occ_code >> i) & 1) == 1)
                    {
                        raw_occ_mat.at<int>(r_idx * tile_size + i / tile_size, c_idx * tile_size + i % tile_size) = 1;
                    }
                }
            }
        }
    }
    cv::Mat dct_mask(raw_occ_mat.rows, raw_occ_mat.cols, CV_32SC1, cv::Scalar(0));
    for (int j = 0; j < raw_occ_mat.cols; ++j)
    {
        int l = 0;
        for (int i = 0; i < raw_occ_mat.rows; ++i)
        {
            if (raw_occ_mat.at<int>(i, j) == 1)
            {
                dct_mask.at<int>(l, j) = 1;
                l++;
            }
        }
    }
    cv::Mat dct_mask_final(dct_mask.rows, dct_mask.cols, CV_32SC1, cv::Scalar(0));
    for (int i = 0; i < dct_mask.rows; ++i)
    {
        int l = 0;
        for (int j = 0; j < dct_mask.cols; ++j)
        {
            if (dct_mask.at<int>(i, j) == 1)
            {
                dct_mask_final.at<int>(i, l) = 1;
                l++;
            }
        }
    }
    return dct_mask_final;
}
void serialize_bMat_toStream(const cv::Mat &mat, std::ostringstream &oss)
{
    int cnt = 0;
    char code = 0;
    for (int row = 0; row < mat.rows; row++)
    {
        for (int col = 0; col < mat.cols; col++)
        {
            if (cnt == 8)
            {
                oss.write(&code, sizeof(code));
                cnt = 0;
                code = 0;
            }
            int status = mat.at<int>(row, col) > 0 ? 1 : 0;
            code += (status << cnt);
            cnt++;
        }
    }
    if (cnt > 0)
    {
        oss.write(&code, sizeof(code));
    }
}

void deserialize_bMat_fromStream(cv::Mat &mat, std::istringstream &iss, int rows, int cols)
{
    mat.create(rows, cols, CV_32SC1);
    int cnt = 0;
    char code = 0;
    iss.read(&code, 1);
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            if (cnt == 8)
            {
                iss.read(&code, sizeof(code));
                cnt = 0;
            }
            if (code >> cnt & 1 == 1)
            {
                mat.at<int>(row, col) = 1;
            }
            else
            {
                mat.at<int>(row, col) = 0;
            }
            cnt++;
        }
    }
}

void serialize_occMat_toStream(const cv::Mat &occ_mat, std::ostringstream &out_stream)
{
    unsigned short code;
    for (int row = 0; row < occ_mat.rows; row++)
    {
        for (int col = 0; col < occ_mat.cols; col++)
        {
            code = (unsigned short)occ_mat.at<int>(row, col);
            out_stream.write(reinterpret_cast<const char *>(&code), sizeof(code));
        }
    }
}

void deserialize_occMat_fromStream(cv::Mat &occ_mat, std::istringstream &in_stream, int rows, int cols)
{
    occ_mat.create(rows, cols, CV_32SC1);
    unsigned short code = 0;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            in_stream.read(reinterpret_cast<char *>(&code), sizeof(code));
            occ_mat.at<int>(row, col) = code;
        }
    }
}

// 序列化单个cv::Mat对象到std::stringstream
void serializeMatToStream(const cv::Mat &mat, std::ostringstream &oss)
{
    int type = mat.type();
    int rows = mat.rows;
    int cols = mat.cols;
    size_t elemSize = mat.elemSize();

    oss.write(reinterpret_cast<const char *>(&type), sizeof(int));
    oss.write(reinterpret_cast<const char *>(&rows), sizeof(int));
    oss.write(reinterpret_cast<const char *>(&cols), sizeof(int));
    oss.write(reinterpret_cast<const char *>(&elemSize), sizeof(size_t));
    oss.write(reinterpret_cast<const char *>(mat.data), elemSize * mat.total());
}

// 反序列化单个cv::Mat对象从std::stringstream
void deserializeMatFromStream(std::istringstream &iss, cv::Mat &mat)
{
    int type, rows, cols;
    size_t elemSize;

    iss.read(reinterpret_cast<char *>(&type), sizeof(int));
    iss.read(reinterpret_cast<char *>(&rows), sizeof(int));
    iss.read(reinterpret_cast<char *>(&cols), sizeof(int));
    iss.read(reinterpret_cast<char *>(&elemSize), sizeof(size_t));

    mat.create(rows, cols, type);
    iss.read(reinterpret_cast<char *>(mat.data), elemSize * mat.total());
}

// 序列化std::vector<cv::Vec4f>对象到std::stringstream
void serializeVec4fVectorToStream(const std::vector<cv::Vec4f> &vec, std::ostringstream &oss)
{
    size_t size = vec.size();
    oss.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
    for (const auto &v : vec)
    {
        oss.write(reinterpret_cast<const char *>(&v), sizeof(cv::Vec4f));
    }
}

// 反序列化std::vector<cv::Vec4f>对象从std::stringstream
void deserializeVec4fVectorFromStream(std::istringstream &iss, std::vector<cv::Vec4f> &vec)
{
    size_t size;
    iss.read(reinterpret_cast<char *>(&size), sizeof(size_t));
    vec.resize(size);
    for (size_t i = 0; i < size; ++i)
    {
        iss.read(reinterpret_cast<char *>(&vec[i]), sizeof(cv::Vec4f));
    }
}

// 序列化Matrix<double>对象到std::stringstream
void serializeMatrixToStream(const Matrix<double> &mat, std::ostringstream &oss)
{
    size_t rows = mat.size();
    oss.write(reinterpret_cast<const char *>(&rows), sizeof(size_t));
    for (const auto &row : mat)
    {
        size_t cols = row.size();
        oss.write(reinterpret_cast<const char *>(&cols), sizeof(size_t));
        oss.write(reinterpret_cast<const char *>(row.data()), cols * sizeof(double));
    }
}

// 反序列化Matrix<double>对象从std::stringstream
void deserializeMatrixFromStream(std::istringstream &iss, Matrix<double> &mat)
{
    size_t rows;
    iss.read(reinterpret_cast<char *>(&rows), sizeof(size_t));
    mat.resize(rows);
    for (size_t i = 0; i < rows; ++i)
    {
        size_t cols;
        iss.read(reinterpret_cast<char *>(&cols), sizeof(size_t));
        std::vector<double> row(cols);
        iss.read(reinterpret_cast<char *>(row.data()), cols * sizeof(double));
        mat[i] = row;
    }
}

// 序列化std::vector<int>对象到std::stringstream
void serializeIntVectorToStream(const std::vector<int> &vec, std::ostringstream &oss)
{
    size_t size = vec.size();
    oss.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
    for (auto data : vec)
    {
        unsigned short quantized_d = (unsigned short)data;
        oss.write(reinterpret_cast<const char *>(&quantized_d), sizeof(quantized_d));
    }
}

// 反序列化std::vector<int>对象从std::stringstream
void deserializeIntVectorFromStream(std::istringstream &iss, std::vector<int> &vec)
{
    size_t size;
    iss.read(reinterpret_cast<char *>(&size), sizeof(size_t));
    vec.resize(size);
    for (size_t i = 0; i < size; ++i)
    {
        unsigned short quantized_d;
        iss.read(reinterpret_cast<char *>(&quantized_d), sizeof(quantized_d));
        vec[i] = quantized_d;
    }
}

// 将所有数据序列化到一个std::vector<char>中
/*序列化的数据有:
q_level(int):设置对应的pitch precision ,yaw precision, threshold,quantization step
b_mat(int) idx_sizes(int int) coefficients(float) occ_mat(int) tile_fit_lengths(int) dct_mat(double->int)
*
*
*
*/
std::vector<char> serializeData(const cv::Mat &b_mat, const int *idx_sizes,
                                const std::vector<cv::Vec4f> &coefficients, const cv::Mat &occ_mat,
                                const std::vector<int> &tile_fit_lengths, const Matrix<double> &dct_mat)
{
    std::ostringstream oss(std::ios::binary);
    // 包头校验
    oss.write("N", sizeof(char));
    serializeMatToStream(b_mat, oss);
    for (int i = 0; i < 2; ++i)
    {
        oss.write(reinterpret_cast<const char *>(idx_sizes + i), sizeof(int));
    }
    serializeVec4fVectorToStream(coefficients, oss);
    serializeMatToStream(occ_mat, oss);
    serializeIntVectorToStream(tile_fit_lengths, oss);
    serializeMatrixToStream(dct_mat, oss);
    std::string data_str = oss.str();
    return std::vector<char>(data_str.begin(), data_str.end());
}

// 从std::vector<char>反序列化所有数据
bool deserializeData(const std::vector<char> &data,
                     cv::Mat &b_mat, int *idx_sizes,
                     std::vector<cv::Vec4f> &coefficients, cv::Mat &occ_mat,
                     std::vector<int> &tile_fit_lengths, Matrix<double> &dct_mat)
{
    std::string data_str(data.begin(), data.end());
    std::istringstream iss(data_str, std::ios::binary);
    // 检查包头
    char header;
    iss.read(reinterpret_cast<char *>(&header), sizeof(char));
    if (header != 'N')
    {
        return false;
    }

    deserializeMatFromStream(iss, b_mat);
    for (int i = 0; i < 2; ++i)
    {
        iss.read(reinterpret_cast<char *>(idx_sizes + i), sizeof(int));
    }
    deserializeVec4fVectorFromStream(iss, coefficients);
    deserializeMatFromStream(iss, occ_mat);
    deserializeIntVectorFromStream(iss, tile_fit_lengths);
    deserializeMatrixFromStream(iss, dct_mat);
    return true;
}
// Zigzag 编码
uint32_t zigzag_encode(int32_t n)
{
    return (n << 1) ^ (n >> 31);
}

// Zigzag 解码
int32_t zigzag_decode(uint32_t n)
{
    return (n >> 1) ^ -(n & 1);
}
// 量化并序列化为int类型
void serialize_dct_toStream(const Matrix<double> &mat, std::ostringstream &oss, const float quantization_step, const cv::Mat &occ_mat, const cv::Mat &b_mat)
{
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    oss.write(reinterpret_cast<const char *>(&rows), sizeof(size_t));
    oss.write(reinterpret_cast<const char *>(&cols), sizeof(size_t));
    // for (const auto &row : mat)
    // {
    //     for (const auto &val : row)
    //     {
    //         int quantized_val = static_cast<int>(val / quantization_step);
    //         oss.write(reinterpret_cast<const char *>(&quantized_val), sizeof(int));
    //     }
    // }
    std::vector<int> indices;
    cv::Mat dct_mask = get_dct_mask(occ_mat, 4, b_mat);
    for (int i = 0; i < dct_mask.rows; i++)
    {
        for (int j = 0; j < dct_mask.cols; j++)
        {
            if (dct_mask.at<int>(i, j) == 1)
            {
                indices.push_back(i * cols + j);
            }
        }
    }
    for (auto idx : indices)
    {
        int row = idx / cols;
        int col = idx % cols;
#ifndef USE_ZIGZAG
        int quantized_val = static_cast<int>(mat[row][col] / quantization_step);
        oss.write(reinterpret_cast<const char *>(&quantized_val), sizeof(int));
#else
        uint32_t quantized_val = zigzag_encode(static_cast<int>(mat[row][col] / quantization_step));
        oss.write(reinterpret_cast<const char *>(&quantized_val), sizeof(uint32_t));
#endif
    }
}
// 反量化并反序列化为double类型
void deserialize_dct_fromStream(std::istringstream &iss, Matrix<double> &mat, const float quantization_step, const cv::Mat &occ_mat, const cv::Mat &b_mat)
{
    size_t rows, cols;
    iss.read(reinterpret_cast<char *>(&rows), sizeof(size_t));
    iss.read(reinterpret_cast<char *>(&cols), sizeof(size_t));
    mat.resize(rows, std::vector<double>(cols, 0.0));

    // for (size_t i = 0; i < rows; ++i)
    // {
    //     std::vector<double> row(cols);
    //     for (size_t j = 0; j < cols; ++j)
    //     {
    //         int quantized_val;
    //         iss.read(reinterpret_cast<char *>(&quantized_val), sizeof(int));
    //         row[j] = quantized_val * quantization_step;
    //     }
    //     mat[i] = row;
    // }
    std::vector<int> indices;
    cv::Mat dct_mask = get_dct_mask(occ_mat, 4, b_mat);
    for (int i = 0; i < dct_mask.rows; i++)
    {
        for (int j = 0; j < dct_mask.cols; j++)
        {
            if (dct_mask.at<int>(i, j) == 1)
            {
                indices.push_back(i * cols + j);
            }
        }
    }

    for (auto idx : indices)
    {
        int row = idx / cols;
        int col = idx % cols;

#ifndef USE_ZIGZAG
        int quantized_val;
        iss.read(reinterpret_cast<char *>(&quantized_val), sizeof(int));
        mat[row][col] = quantized_val * quantization_step;
#else
        uint32_t quantized_val;
        iss.read(reinterpret_cast<char *>(&quantized_val), sizeof(uint32_t));
        mat[row][col] = zigzag_decode(quantized_val) * quantization_step;
#endif
    }
}

std::vector<char> q_serializeData(const int q_level, const cv::Mat &b_mat, const int *idx_sizes,
                                  const std::vector<cv::Vec4f> &coefficients, const cv::Mat &occ_mat,
                                  const std::vector<int> &tile_fit_lengths, const Matrix<double> &dct_mat)
{
    std::ostringstream oss(std::ios::binary);
    // 添加包头检查字符
    oss.write("Q", sizeof(char));
    float parameters[4] = {quantization_dict[q_level][0], quantization_dict[q_level][1], quantization_dict[q_level][2], quantization_dict[q_level][3]};
    oss.write(reinterpret_cast<const char *>(parameters), sizeof(float) * 4);
    // pitch yaw threshold quantization_step
    int now_size = oss.str().size();
    uint16_t bmat_row = b_mat.rows, bmat_col = b_mat.cols;
    oss.write(reinterpret_cast<const char *>(&bmat_row), sizeof(uint16_t));
    oss.write(reinterpret_cast<const char *>(&bmat_col), sizeof(uint16_t));
    serialize_bMat_toStream(b_mat, oss);

    cout << " bmat size:" << oss.str().size() - now_size << endl;
    now_size = oss.str().size();
    for (int i = 0; i < 2; ++i)
    {
        oss.write(reinterpret_cast<const char *>(idx_sizes + i), sizeof(int));
    }

    serializeVec4fVectorToStream(coefficients, oss);
    cout << " coeff size :" << oss.str().size() - now_size << endl;
    now_size = oss.str().size();
    serialize_occMat_toStream(occ_mat, oss);
    cout << " occ size :" << oss.str().size() - now_size << endl;
    now_size = oss.str().size();
    serializeIntVectorToStream(tile_fit_lengths, oss);
    cout << " tile  length size :" << oss.str().size() - now_size << endl;
    now_size = oss.str().size();
    serialize_dct_toStream(dct_mat, oss, quantization_dict[q_level][3], occ_mat, b_mat);
    cout << " dct size :" << oss.str().size() - now_size << endl;
    now_size = oss.str().size();
    std::string data_str = oss.str();
    cout << "total size: " << data_str.size() << endl;
    return std::vector<char>(data_str.begin(), data_str.end());
}
// 从std::vector<char>反序列化所有数据
bool q_deserializeData(const std::vector<char> &data,
                       float &pitch_precision, float &yaw_precision,
                       cv::Mat &b_mat, int *idx_sizes,
                       std::vector<cv::Vec4f> &coefficients, cv::Mat &occ_mat,
                       std::vector<int> &tile_fit_lengths, Matrix<double> &dct_mat)
{
    std::string data_str(data.begin(), data.end());
    std::istringstream iss(data_str, std::ios::binary);
    // 检查包头
    char header;
    iss.read(reinterpret_cast<char *>(&header), sizeof(char));
    if (header != 'Q')
    {
        return false;
    }
    // pitch yaw threshold quantization_step
    float parameters[4];
    iss.read(reinterpret_cast<char *>(parameters), sizeof(float) * 4);
    pitch_precision = parameters[0];
    yaw_precision = parameters[1];
    uint16_t b_mat_row, b_mat_col;
    iss.read(reinterpret_cast<char *>(&b_mat_row), sizeof(uint16_t));
    iss.read(reinterpret_cast<char *>(&b_mat_col), sizeof(uint16_t));
    deserialize_bMat_fromStream(b_mat, iss, b_mat_row, b_mat_col);
    for (int i = 0; i < 2; ++i)
    {
        iss.read(reinterpret_cast<char *>(idx_sizes + i), sizeof(int));
    }
    deserializeVec4fVectorFromStream(iss, coefficients);
    deserialize_occMat_fromStream(occ_mat, iss, b_mat_row, b_mat_col);
    deserializeIntVectorFromStream(iss, tile_fit_lengths);
    deserialize_dct_fromStream(iss, dct_mat, parameters[3], occ_mat, b_mat);
    return true;
}