#include "Frontend.h"

// ==========================================
// 1. 构造函数：初始化参数
// ==========================================
Frontend::Frontend()
{
    // 对应 main.cpp: cv::Ptr<cv::ORB> orb = cv::ORB::create(3000);
    orb_ = cv::ORB::create(3000);

    // 初始化后端
    backend_ = std::make_shared<Backend>();
}

// ==========================================
// 2. 主入口 Track：指挥整个流程
// ==========================================
Eigen::Matrix4d Frontend::Track(double time, const cv::Mat &img_l, const cv::Mat &img_r)
{
    img_l_ = img_l;
    img_r_ = img_r;

    // --- 构造当前帧 ---
    // 对应 main.cpp: Frame curr_frame; ...
    static int frame_cnt = 0;
    curr_frame_ = Frame::createFrame(frame_cnt++, time, img_l);

    // --- 提取特征 ---
    ExtractFeatures();

    // --- 跟踪 (PnP) ---
    if (last_frame_)
    {
        // 如果有上一帧，就进行跟踪
        TrackLastFrame();
    }
    else
    {
        // 对应 main.cpp: else { curr_frame.T_w_c = Identity(); }
        curr_frame_->T_w_c = Eigen::Matrix4d::Identity();
    }

    // --- 建图 (Stereo Mapping) ---
    StereoMapping();

    // --- 关键帧策略 & 后端 ---
    if (NeedKeyFrame())
    {
        // 把当前帧加入后端
        backend_->AddFrame(curr_frame_);
        // 触发优化
        backend_->Optimize();

        // 更新关键帧
        last_keyframe_ = curr_frame_;

        // 关键帧才使用回环检测
        if (loop_closer_)
        {
            loop_closer_->AddKeyFrame(curr_frame_);
        }
    }

    // --- 更新上一帧 ---
    // 对应 main.cpp: last_frame = curr_frame;
    last_frame_ = curr_frame_;

    // 返回最新位姿用于画图 (优先用后端优化过的)
    return backend_->GetCurrentPose();
}

// ==========================================
// 3. 提取特征
// ==========================================
void Frontend::ExtractFeatures()
{
    // 对应 main.cpp: orb->detectAndCompute(...)
    // 修改：curr_frame 变成了 curr_frame_->
    orb_->detectAndCompute(img_l_, cv::noArray(), curr_frame_->keypoints, curr_frame_->descriptors);

    // 初始化 ID
    curr_frame_->track_ids.resize(curr_frame_->keypoints.size(), -1);
    // 初始化 map_points (注意这里要是 vector<Point3f>)
    curr_frame_->map_points.resize(curr_frame_->keypoints.size(), cv::Point3f(0, 0, 0));
}

// ==========================================
// 4. 跟踪上一帧 (PnP)
// ==========================================
void Frontend::TrackLastFrame()
{
    // 对应 main.cpp: 阶段 1: Tracking

    // 1.1 匹配
    cv::BFMatcher matcher(cv::NORM_HAMMING, true); // 开启交叉验证
    std::vector<cv::DMatch> matches;
    // 修改：last_frame.descriptors -> last_frame_->descriptors
    matcher.match(last_frame_->descriptors, curr_frame_->descriptors, matches);

    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;

    // 1.2 筛选匹配 & 继承 ID
    for (auto &m : matches)
    {
        // 修改：m.distance > 80 (放宽后的阈值)
        if (m.distance > 80)
            continue;

        int last_idx = m.queryIdx;
        int curr_idx = m.trainIdx;

        // 修改：访问成员用 ->
        long unsigned int old_id = last_frame_->track_ids[last_idx];

        if (old_id != -1)
        {
            // 继承 ID 和 坐标
            curr_frame_->track_ids[curr_idx] = old_id;
            curr_frame_->map_points[curr_idx] = last_frame_->map_points[last_idx];

            pts_3d.push_back(last_frame_->map_points[last_idx]);
            pts_2d.push_back(curr_frame_->keypoints[curr_idx].pt);
        }
    }

    // 1.3 解 PnP
    if (pts_3d.size() > 10)
    {
        cv::Mat rvec, tvec;
        cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        std::vector<int> inliers;

        cv::solvePnPRansac(pts_3d, pts_2d, K, cv::Mat(), rvec, tvec,
                           false, 100, 4.0, 0.99, inliers, cv::SOLVEPNP_EPNP);

        if (inliers.size() < 10)
        {
            // Tracking Lost: 恒速模型
            // std::cout << "Tracking LOST! Using Constant Velocity." << std::endl;
            curr_frame_->T_w_c = last_frame_->T_w_c * last_velocity_;
        }
        else
        {
            // Tracking Success
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            Eigen::Matrix3d R_cw;
            Eigen::Vector3d t_cw;
            // OpenCV Mat -> Eigen
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    R_cw(r, c) = R.at<double>(r, c);
            t_cw << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);

            Eigen::Matrix4d T_c_w = Eigen::Matrix4d::Identity();
            T_c_w.block<3, 3>(0, 0) = R_cw;
            T_c_w.block<3, 1>(0, 3) = t_cw;

            Eigen::Matrix4d calculated_T_w_c = T_c_w.inverse();

            // 限速器（Velocity Gating）
            Eigen::Vector3d pos_curr = calculated_T_w_c.block<3, 1>(0, 3);
            Eigen::Vector3d pos_last = last_frame_->T_w_c.block<3, 1>(0, 3);
            double dist = (pos_curr - pos_last).norm();

            // PnP 正常，更新位姿
            curr_frame_->T_w_c = calculated_T_w_c;

            // 如果发生跳变(比如 > 0.5)，说明可能回环了，重置速度为 0
            if (dist > 0.5)
            {
                last_velocity_ = Eigen::Matrix4d::Identity();
            }
            else
            {
                // 更新速度: velocity = T_last.inv * T_curr
                last_velocity_ = last_frame_->T_w_c.inverse() * curr_frame_->T_w_c;
            }
        }
    }
    else
    {
        // 匹配点太少，直接用恒速模型
        curr_frame_->T_w_c = last_frame_->T_w_c * last_velocity_;
    }
}

// ==========================================
// 5. 双目建图 (Stereo Mapping)
// ==========================================
void Frontend::StereoMapping()
{
    // 对应 main.cpp: 阶段 2: Stereo Mapping

    // 2.1 提取右图特征
    std::vector<cv::KeyPoint> kp_r;
    cv::Mat des_r;
    orb_->detectAndCompute(img_r_, cv::noArray(), kp_r, des_r);

    // 2.2 匹配
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(curr_frame_->descriptors, des_r, matches);

    for (auto &m : matches)
    {
        if (m.distance > 80)
            continue; // 阈值统一放宽

        int curr_idx = m.queryIdx;
        int right_idx = m.trainIdx;

        // 如果是老点，跳过
        if (curr_frame_->track_ids[curr_idx] != -1)
            continue;

        cv::Point2f pt_l = curr_frame_->keypoints[curr_idx].pt;
        cv::Point2f pt_r = kp_r[right_idx].pt;

        // 视差计算
        double disparity = pt_l.x - pt_r.x;

        // 过滤条件 (你调好的参数)
        if (disparity < 3.0)
            continue;

        double Z = fx * b / disparity;
        if (Z < 0.1 || Z > 15.0)
            continue; // 深度过滤

        // 转世界坐标 (最重要的公式)
        double X = (pt_l.x - cx) * Z / fx;
        double Y = (pt_l.y - cy) * Z / fy;

        Eigen::Vector3d p_cam(X, Y, Z);
        Eigen::Matrix3d R_wc = curr_frame_->T_w_c.block<3, 3>(0, 0);
        Eigen::Vector3d t_wc = curr_frame_->T_w_c.block<3, 1>(0, 3);
        Eigen::Vector3d p_world = R_wc * p_cam + t_wc;

        // 发 ID
        // 修改：使用成员变量 global_landmark_id_
        curr_frame_->track_ids[curr_idx] = global_landmark_id_++;
        curr_frame_->map_points[curr_idx] = cv::Point3f(p_world.x(), p_world.y(), p_world.z());
    }
}

// ==========================================
// 6. 关键帧判断
// ==========================================
bool Frontend::NeedKeyFrame()
{
    if (!last_keyframe_)
        return true; // 第一帧

    // 计算运动距离
    Eigen::Vector3d t_curr = curr_frame_->T_w_c.block<3, 1>(0, 3);
    Eigen::Vector3d t_last = last_keyframe_->T_w_c.block<3, 1>(0, 3);
    double dist = (t_curr - t_last).norm();

    // 移动超过 0.2 米则为关键帧
    return dist > 0.2;
}

Frame::Ptr Frontend::GetCurrentFrame()
{
    return curr_frame_;
}