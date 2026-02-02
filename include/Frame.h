#pragma once
#include "common_include.h"

struct Frame
{
    // 使用 shared_ptr 智能指针管理，防止内存泄漏
    typedef std::shared_ptr<Frame> Ptr;
  
    int id;
    double timestamp;
    cv::Mat img; // 左图

    Eigen::Matrix4d T_w_c = Eigen::Matrix4d::Identity(); // 位姿

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    std::vector<long unsigned int> track_ids; // ID
    std::vector<cv::Point3f> map_points;      // 3D点

    // 构造函数
    Frame(int id, double time, const cv::Mat &image);

    // 工厂模式创建帧 (更现代的写法)
    static Frame::Ptr createFrame(int id, double time, const cv::Mat &image);
};