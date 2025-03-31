#ifndef BINARY_COMPRESSOR_H
#define BINARY_COMPRESSOR_H
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <zstd.h>
std::vector<char> readFile(const std::string &filePath);
void writeFile(const std::string &filePath, const std::vector<char> &data);
std::vector<char> compressData(const std::vector<char> &data);
std::vector<char> decompressData(const std::vector<char> &compressedData);

#endif