#pragma once
#include "common_include.h"
#include "Frame.h"
#include <DBoW3/DBoW3.h> // 引入 DBoW3
#include <g2o/types/sba/types_six_dof_expmap.h> // 位姿图优化

class LoopClosing
{
public:
    typedef std::shared_ptr<LoopClosing> Ptr;

    // 构造函数：需要传入字典文件的路径
    LoopClosing(std::string vocab_path);

    // 唯一的入口：接收关键帧
    void AddKeyFrame(Frame::Ptr frame);

    void SaveTrajectory(const std::string& filename);

private:
    // 执行回环修正的新函数
    void CorrectLoop(int loop_frame_index, Frame::Ptr curr_frame);

    // DBoW3 的核心组件
    DBoW3::Vocabulary vocab_;
    DBoW3::Database db_;

    // 所有的关键帧数据库 (用于回溯)
    std::vector<Frame::Ptr> all_keyframes_;
};