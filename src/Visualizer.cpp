#include "Visualizer.h"

Visualizer::Visualizer()
{
    // 初始化画布 (800x800 黑色背景)
    traj_img_ = cv::Mat::zeros(800, 800, CV_8UC3);
}

void Visualizer::AddCurrentFrame(Frame::Ptr frame)
{
    curr_frame_ = frame;

    // 获取当前位姿 (World -> Camera)
    // T_w_c 的 (0,3) 是 x, (1,3) 是 y, (2,3) 是 z
    Eigen::Matrix4d Twc = curr_frame_->T_w_c;

    // 存入历史轨迹 (这里我们取 x 和 z 作为俯视图平面)
    // 注意：具体用哪两个轴取决于你的坐标系定义，EuRoC通常 x-z 是俯视图
    cv::Point3f pos(Twc(0, 3), Twc(1, 3), Twc(2, 3));

    path_history_.push_back(pos);
}

void Visualizer::ShowResult()
{
    if (curr_frame_ == nullptr)
        return;

    // -----------------------------------------
    // 1. 画轨迹 (相机跟随模式)
    // -----------------------------------------
    // 每次重画，清空画布
    traj_img_.setTo(cv::Scalar(0, 0, 0));

    // 获取当前相机中心
    Eigen::Matrix4d Twc = curr_frame_->T_w_c;
    double center_x = Twc(0, 3);
    double center_z = Twc(2, 3); // 假设 Z 是前进方向

    double scale = 30.0; // 缩放比例

    // 遍历历史轨迹画点
    for (size_t i = 0; i < path_history_.size(); i++)
    {
        auto &p = path_history_[i];

        // 坐标转换：把世界坐标映射到 800x800 的窗口中心 (400,400)
        int draw_x = int((p.x - center_x) * scale) + 400;
        int draw_y = 400 - int((p.z - center_z) * scale); // y轴反转

        if (draw_x >= 0 && draw_x < 800 && draw_y >= 0 && draw_y < 800)
        {
            cv::circle(traj_img_, cv::Point(draw_x, draw_y), 1, cv::Scalar(0, 0, 255), -1);
        }
    }

    // 画当前相机位置 (绿色大点)
    cv::circle(traj_img_, cv::Point(400, 400), 5, cv::Scalar(0, 255, 0), -1);

    // 写上帧号
    cv::putText(traj_img_, "Frame: " + std::to_string(curr_frame_->id),
                cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

    cv::imshow("Trajectory", traj_img_);

    // -----------------------------------------
    // 2. 画特征点 (Features)
    // -----------------------------------------
    cv::Mat img_show;
    // 如果是灰度图，转成彩色以便画彩色圈
    if (curr_frame_->img.channels() == 1)
    {
        cv::cvtColor(curr_frame_->img, img_show, cv::COLOR_GRAY2BGR);
    }
    else
    {
        img_show = curr_frame_->img.clone();
    }

    // 遍历特征点画圈
    for (size_t i = 0; i < curr_frame_->keypoints.size(); i++)
    {
        // 如果有 ID (track_ids[i] != -1)，画绿色；否则画红色
        if (curr_frame_->track_ids[i] != -1)
        {
            cv::circle(img_show, curr_frame_->keypoints[i].pt, 2, cv::Scalar(0, 255, 0), -1);
        }
        else
        {
            cv::circle(img_show, curr_frame_->keypoints[i].pt, 2, cv::Scalar(0, 0, 255), -1);
        }
    }

    cv::imshow("Features", img_show);

    // 必须有 waitKey 才能刷新图像
    cv::waitKey(1);
}