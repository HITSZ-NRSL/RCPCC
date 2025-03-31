#include <binary_compressor.h>
// Function to read a file into a vector of bytes
std::vector<char> readFile(const std::string &filePath)
{
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        throw std::runtime_error("Error reading file: " + filePath);
    }

    return buffer;
}

// Function to write compressed data to a file
void writeFile(const std::string &filePath, const std::vector<char> &data)
{
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    file.write(data.data(), data.size());
    if (!file)
    {
        throw std::runtime_error("Error writing file: " + filePath);
    }
}

// Function to compress data using Zstandard
std::vector<char> compressData(const std::vector<char> &data)
{   
    
    size_t compressedSize = ZSTD_compressBound(data.size());
    std::vector<char> compressedData(compressedSize);

    size_t actualSize = ZSTD_compress(compressedData.data(), compressedSize, data.data(), data.size(), ZSTD_maxCLevel());
    if (ZSTD_isError(actualSize))
    {
        throw std::runtime_error("Compression failed: " + std::string(ZSTD_getErrorName(actualSize)));
    }

    compressedData.resize(actualSize);
    //计算压缩比
    std::cout << "binary Compression ratio: " << (double)data.size() / actualSize << std::endl;
    return compressedData;
}

std::vector<char> decompressData(const std::vector<char> &compressedData)
{
    unsigned long long decompressedSize = ZSTD_getFrameContentSize(compressedData.data(), compressedData.size());
    if (decompressedSize == ZSTD_CONTENTSIZE_ERROR)
    {
        throw std::runtime_error("Not a valid compressed frame.");
    }
    else if (decompressedSize == ZSTD_CONTENTSIZE_UNKNOWN)
    {
        throw std::runtime_error("Original size unknown.");
    }

    std::vector<char> decompressedData(decompressedSize);

    size_t actualSize = ZSTD_decompress(decompressedData.data(), decompressedSize, compressedData.data(), compressedData.size());
    if (ZSTD_isError(actualSize))
    {
        throw std::runtime_error("Decompression failed: " + std::string(ZSTD_getErrorName(actualSize)));
    }

    return decompressedData;
}