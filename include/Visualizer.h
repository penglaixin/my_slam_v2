#pragma once
#include "common_include.h"
#include "Frame.h"

class Visualizer
{
public:
    Visualizer();

    // 唯一的入口：接收当前帧
    void AddCurrentFrame(Frame::Ptr frame);

    // 显示画面 (在主线程调用)
    void ShowResult();

private:
    // 存储历史轨迹点 (用于画红线)
    std::vector<cv::Point3f> path_history_;

    // 当前帧的指针
    Frame::Ptr curr_frame_ = nullptr;

    // 画布
    cv::Mat traj_img_;
};