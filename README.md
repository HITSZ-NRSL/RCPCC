# RCPCC 🚀
<div align="center"><img src="imgs/banner.png" width=100% /></div>

## [ICRA 2025] Real-Time LiDAR Point Cloud Compression and Transmission for Resource-Constrained Robots 🤖

**Authors**: [Yuhao Cao](https://github.com/ZorAttC), [Yu Wang](https://github.com/wangyu-060070), and [Haoyao Chen](https://ieeexplore.ieee.org/author/37600762500)  
**Affiliation**: [Networked Robotics and Systems Lab, HITSZ](https://www.nrs-lab.com/)

---

### Introduction 🌟
This repository provides an efficient, real-time LiDAR point cloud compression algorithm tailored for mechanically scanned LiDARs (e.g., Velodyne-64e). The algorithm leverages spatial structure of point clouds for surface fitting and employs Discrete Cosine Transform (DCT) to balance precision and compression ratio. It achieves blazing-fast compression speeds (e.g., 41.06ms for encoding and 11.35ms for decoding on an i7-124650H) and delivers high compression ratios. 

> **Note**: Due to the transmission framework’s heavy reliance on specific communication setups, we’ve open-sourced only the compression module.  

For a deep dive, check out our paper:  
📄 [Real-Time LiDAR Point Cloud Compression and Transmission for Resource-Constrained Robots](https://arxiv.org/abs/2502.06123)

---

### How to Compile 🛠️
First, install the required libraries:  
```bash
sudo apt update && sudo apt install -y libopencv-dev libfftw3-dev libzstd-dev libpcl-dev libboost-all-dev
```

Then, clone and build the code:  
```bash
git clone https://github.com/HITSZ-NRSL/RCPCC.git
mkdir build && cd build
cmake ../src && make
```

---

### How to Use 🎮
We’ve included a handy test program:  
```bash
./build/example ./test_file/0000000000.bin 0
```
This command compresses a `.bin` point cloud file with compression level 0. Once launched, the program visualizes both the original and compressed point clouds side by side.  

- **Compression Levels**: Choose between 0 and 5. Higher numbers = higher compression ratio (but lower precision).  

---

### Code Notes 📝
#### Compression Levels  
In `src/modules/serializer.cpp`:  
```cpp
double quantization_dict[16][4] = {
    {0.25, 0.5, 0.1, 0.1}, 
    {0.25, 0.5, 0.2, 0.20},
    {0.25, 0.5, 0.4, 0.20}, 
    {0.5, 1.0, 0.1, 0.2},   
    {0.5, 1.0, 0.2, 0.2},   
    {1.0, 2.0, 0.4, 0.20}, 
};
```
These define the compression levels. Parameters represent:  
- Pitch resolution  
- Yaw resolution  
- Surface fitting threshold  
- DCT quantization step  
*Larger values = higher compression, lower accuracy.*

#### Configuration  
In `src/utils/config.h`:  
```cpp
#define ROW_OFFSET 32.0f
#define COL_OFFSET 180.0f

#define VERTICAL_DEGREE (32.0f + 5.0f)
#define HORIZONTAL_DEGREE (180.0f + 180.0f)
```
Adjust these Field of View (FOV) parameters based on your LiDAR setup. The defaults work well for Velodyne-64e.  
> **Tip**: Parameters don’t need to match physical specs exactly, but improper settings may cause truncation (missing points) or redundant areas (lower compression efficiency).

---

### Thanks 🙏
Explore raw data from [KITTI](http://www.cvlibs.net/datasets/kitti/) for more experiments! The Velodyne data is plug-and-play—just specify the file path and compression level.  

We’re deeply inspired by:  
🌟 [Real-Time Spatio-Temporal LiDAR Point Cloud Compression](https://github.com/horizon-research/Real-Time-Spatio-Temporal-LiDAR-Point-Cloud-Compression)  

### Citation 📚
If you find this codebase helpful for your research or projects, please cite our paper:
```latex
@article{cao2025realtime,
  title={Real-Time LiDAR Point Cloud Compression and Transmission for Resource-Constrained Robots},
  author={Cao, Yuhao and Wang, Yu and Chen, Haoyao},
  journal={arXiv preprint arXiv:2502.06123},
  year={2025}
}
```
