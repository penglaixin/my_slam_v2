#pragma once
#include "common_include.h"
#include "Frame.h"
#include "Backend.h"
#include "LoopClosing.h"

class Frontend
{
public:
    Frontend();
    Frame::Ptr GetCurrentFrame();

    // 处理新的一帧 (这是唯一的入口)
    // 返回：当前帧的位姿 (用于画图)
    Eigen::Matrix4d Track(double time, const cv::Mat &img_l, const cv::Mat &img_r);

    // 注入回环模块
    void SetLoopCloser(LoopClosing::Ptr loop_closer) { loop_closer_ = loop_closer; }

private:
    // 内部函数
    void ExtractFeatures();
    void TrackLastFrame();
    void StereoMapping();
    bool NeedKeyFrame();
    
    // 成员变量
    Frame::Ptr curr_frame_ = nullptr;
    Frame::Ptr last_frame_ = nullptr;
    Frame::Ptr last_keyframe_ = nullptr; // 用于判断关键帧

    cv::Mat img_l_, img_r_;
    cv::Ptr<cv::ORB> orb_;

    // 记录上一帧的速度，用于恒速模型
    Eigen::Matrix4d last_velocity_ = Eigen::Matrix4d::Identity();

    // 后端指针
    shared_ptr<Backend> backend_;

    // ID 计数器
    long unsigned int global_landmark_id_ = 0;

    // 参数
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375, b = 0.11;

    // 回环检测模块指针
    LoopClosing::Ptr loop_closer_ = nullptr;
};