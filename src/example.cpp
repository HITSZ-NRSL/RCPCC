#include <iostream>
#include <fstream>
#include <iomanip>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>  
#include <chrono>  
#include "modules/encoder_module.h"
#include "modules/decoder_module.h"
#include "../utils/struct.h"
#include "../utils/utils.h"

// Visualization function: Create two separate windows and set initial camera position
void visualizePointClouds(const pcl::PointCloud<pcl::PointXYZ>::Ptr& original_cloud,
                         const pcl::PointCloud<pcl::PointXYZ>::Ptr& restored_cloud) {
    // Create Visualizer for original point cloud
    pcl::visualization::PCLVisualizer::Ptr original_viewer(new pcl::visualization::PCLVisualizer("Original Point Cloud"));
    original_viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> original_color(original_cloud, 255, 0, 0);
    original_viewer->addPointCloud<pcl::PointXYZ>(original_cloud, original_color, "original_cloud");
    original_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original_cloud");
    original_viewer->addCoordinateSystem(1.0);
    original_viewer->initCameraParameters();

    // Set initial camera position for original point cloud window
    original_viewer->setCameraPosition(
        0, 0, 30,    // Camera position (x, y, z)
        0, 0, 0,     // Focal point (x, y, z)
        0, 1, 0      // Up direction (x, y, z)
    );

    // Create Visualizer for restored point cloud
    pcl::visualization::PCLVisualizer::Ptr restored_viewer(new pcl::visualization::PCLVisualizer("Restored Point Cloud"));
    restored_viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> restored_color(restored_cloud, 0, 255, 0);
    restored_viewer->addPointCloud<pcl::PointXYZ>(restored_cloud, restored_color, "restored_cloud");
    restored_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "restored_cloud");
    restored_viewer->addCoordinateSystem(1.0);
    restored_viewer->initCameraParameters();

    // Set initial camera position for restored point cloud window
    restored_viewer->setCameraPosition(
        0, 0, 30,    // Camera position (x, y, z)
        0, 0, 0,     // Focal point (x, y, z)
        0, 1, 0      // Up direction (x, y, z)
    );

    // Main loop to keep both windows running simultaneously
    while (!original_viewer->wasStopped() && !restored_viewer->wasStopped()) {
        original_viewer->spinOnce(100);
        restored_viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <q_level>" << std::endl;
        return -1;
    }

    int q_level = std::stoi(argv[2]);

    // Load point cloud using project's loading functions
    std::vector<point_cloud> pcloud_data;

    std::string filename = argv[1];
    std::string extension = filename.substr(filename.find_last_of(".") + 1);

    if (extension == "bin")
    {
        load_pcloud(filename, pcloud_data);
    }
    else if (extension == "ply")
    {
        load_pcloud_ply(filename, pcloud_data);
    }
    else
    {
        std::cerr << "Unsupported file format: " << extension << std::endl;
        return -1;
    }

    // Convert original data to PCL format
    pcl::PointCloud<pcl::PointXYZ>::Ptr original_pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    original_pcl_cloud->width = pcloud_data.size();
    original_pcl_cloud->height = 1;
    original_pcl_cloud->is_dense = true;
    original_pcl_cloud->points.resize(pcloud_data.size());
    
    for (size_t i = 0; i < pcloud_data.size(); ++i)
    {
        original_pcl_cloud->points[i].x = pcloud_data[i].x;
        original_pcl_cloud->points[i].y = pcloud_data[i].y;
        original_pcl_cloud->points[i].z = pcloud_data[i].z;
    }

    // Encode point cloud
    EncoderModule encoder(4, q_level);
    std::vector<char> encoded_data = encoder.encodeToData(pcloud_data, true);

    // Get original file size
    std::ifstream in_file(argv[1], std::ios::binary | std::ios::ate);
    size_t original_size = in_file.tellg();
    in_file.close();

    // Calculate compression ratio
    double compression_ratio = static_cast<double>(original_size) / encoded_data.size();

    std::cout << "Compression results:" << std::endl;
    std::cout << "  Original size: " << original_size << " bytes" << std::endl;
    std::cout << "  Compressed size: " << encoded_data.size() << " bytes" << std::endl;
    std::cout << "  Compression ratio: " << std::fixed << std::setprecision(2) << compression_ratio << ":1" << std::endl;

    // Decode point cloud
    DecoderModule decoder(encoded_data, 4, true, q_level);
    auto restored_pcloud = decoder.restored_pcloud;

    // Convert restored data to PCL format
    pcl::PointCloud<pcl::PointXYZ>::Ptr restored_pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    restored_pcl_cloud->width = restored_pcloud.size();
    restored_pcl_cloud->height = 1;
    restored_pcl_cloud->is_dense = true;
    restored_pcl_cloud->points.resize(restored_pcloud.size());

    for (size_t i = 0; i < restored_pcloud.size(); ++i)
    {
        restored_pcl_cloud->points[i].x = restored_pcloud[i].x;
        restored_pcl_cloud->points[i].y = restored_pcloud[i].y;
        restored_pcl_cloud->points[i].z = restored_pcloud[i].z;
    }

    // Save as PLY
    std::string output_filename = "decompressed.ply";
    pcl::io::savePLYFileASCII(output_filename, *restored_pcl_cloud);
    std::cout << "Saved decompressed point cloud to " << output_filename << std::endl;

    // Visualize both point clouds
    std::cout << "Opening PCL Viewer... (Close the viewer window to exit)" << std::endl;
    visualizePointClouds(original_pcl_cloud, restored_pcl_cloud);

    return 0;
}