#pragma once
#include "common_include.h"
#include "Frame.h"

class Backend
{
public:
    Backend();

    // 外部接口
    void AddFrame(Frame::Ptr frame);
    void Optimize();
    Eigen::Matrix4d GetCurrentPose();

private:
    std::deque<Frame::Ptr> sliding_window_;
    int window_size_ = 7;

    // --- [新增] 相机内参 ---
    double fx = 458.654;
    double fy = 457.296;
    double cx = 367.215;
    double cy = 248.375;
};