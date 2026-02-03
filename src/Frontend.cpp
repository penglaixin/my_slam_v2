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
    // 提取 ORB 特征
    orb_->detectAndCompute(img_l_, cv::noArray(), curr_frame_->keypoints, curr_frame_->descriptors);

    // 亚像素优化 (Sub-pixel Refinement)
    // 只有当特征点数量足够多时才做，防止崩溃
    if (curr_frame_->keypoints.size() > 0)
    {
        // A. 提取 Point2f 坐标
        std::vector<cv::Point2f> points;
        for (auto &kp : curr_frame_->keypoints)
        {
            points.push_back(kp.pt);
        }

        // B. 执行亚像素优化
        // 注意：cornerSubPix 对 ORB 这类角点非常有效，但对边缘点可能效果一般
        // 这一步会修改 points 里的坐标，使其变成小数 (如 123.45)
        cv::cornerSubPix(img_l_, points, cv::Size(5, 5), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 0.03));

        // C. 把优化后的坐标塞回 KeyPoints
        for (size_t i = 0; i < curr_frame_->keypoints.size(); i++)
        {
            curr_frame_->keypoints[i].pt = points[i];
        }
    }

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

    // 1.1 使用 knnMatch 找最近的 2 个点
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(last_frame_->descriptors, curr_frame_->descriptors, knn_matches, 2);

    std::vector<cv::DMatch> matches;
    const float ratio_thresh = 0.7f; // ORB-SLAM 经典参数

    for (auto &m : knn_matches)
    {
        if (m.size() < 2)
            continue;

        // 核心逻辑：如果最近距离 < 0.7 * 次近距离，才认为是好匹配
        if (m[0].distance < ratio_thresh * m[1].distance)
        {
            matches.push_back(m[0]);
        }
    }

    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;

    // 1.2 筛选匹配 & 继承 ID
    for (auto &m : matches)
    {
        // 修改：m.distance > 50 (放宽后的阈值)
        if (m.distance > 100)
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
                           false, 100, 2.0, 0.99, inliers, cv::SOLVEPNP_EPNP);

        if (inliers.size() < 15)
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

            curr_frame_->T_w_c = calculated_T_w_c;

            // 限速器（Velocity Gating）
            Eigen::Vector3d pos_curr = calculated_T_w_c.block<3, 1>(0, 3);
            Eigen::Vector3d pos_last = last_frame_->T_w_c.block<3, 1>(0, 3);
            double dist = (pos_curr - pos_last).norm();

            // 如果发生跳变(比如 > 0.5)，说明可能回环了，重置速度为 0
            if (dist > 2.0)
            {
                // std::cout << "PnP Exploded (Dist=" << dist << "m)! Rejecting." << std::endl;
                // 拒绝错误结果，使用恒速模型平滑过渡
                curr_frame_->T_w_c = last_frame_->T_w_c * last_velocity_;
            }
            else if (dist > 0.5)
            {
                // std::cout << "Loop Jump Detected (" << dist << "m)! Reset Velocity." << std::endl;
                // 接受位置
                curr_frame_->T_w_c = calculated_T_w_c;
                // 速度归零，防止下一帧飞出去
                last_velocity_ = Eigen::Matrix4d::Identity();
            }
            else
            {
                // PnP 结果合理，接受
                curr_frame_->T_w_c = calculated_T_w_c;

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
    // 阶段 2: Stereo Mapping

    // 2.1 提取右图特征
    std::vector<cv::KeyPoint> kp_r;
    cv::Mat des_r;
    orb_->detectAndCompute(img_r_, cv::noArray(), kp_r, des_r);

    // 右图亚像素优化
    if (kp_r.size() > 0)
    {
        std::vector<cv::Point2f> points_r;
        for (auto &kp : kp_r)
            points_r.push_back(kp.pt);

        cv::cornerSubPix(img_r_, points_r, cv::Size(5, 5), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 0.03));

        for (size_t i = 0; i < kp_r.size(); i++)
            kp_r[i].pt = points_r[i];
    }

    // 2.2 匹配
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(curr_frame_->descriptors, des_r, matches);

    for (auto &m : matches)
    {
        if (m.distance > 50)
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
        if (Z < 0.1 || Z > 5.0)
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

    // 获取当前帧和上一个关键帧的位姿
    Eigen::Matrix4d T_curr = curr_frame_->T_w_c;
    Eigen::Matrix4d T_last_kf = last_keyframe_->T_w_c;

    // 平移检测
    Eigen::Vector3d t_curr = curr_frame_->T_w_c.block<3, 1>(0, 3);
    Eigen::Vector3d t_last = last_keyframe_->T_w_c.block<3, 1>(0, 3);
    double dist = (t_curr - t_last).norm();

    // 旋转检测 [计算相对旋转，看转了多少度]
    Eigen::Matrix3d R_curr = T_curr.block<3, 3>(0, 0);
    Eigen::Matrix3d R_last = T_last_kf.block<3, 3>(0, 0);

    // 相对旋转 R_rel = R_last.inv * R_curr
    Eigen::Matrix3d R_rel = R_last.inverse() * R_curr;

    // 将旋转矩阵转换为 轴角 (AxisAngle) 来计算角度
    // norm() 得到的就是旋转的弧度值
    double rot_angle = Eigen::AngleAxisd(R_rel).angle();

    // 跟踪质量检测
    // 如果当前跟踪的内点太少，说明旧点快跟丢了，赶紧插入关键帧补充新点
    //bool tracking_weak = (num_inliers_ < 40);

    if (dist > 0.2 || rot_angle > 0.1)
    {
        // std::cout << "New Keyframe: Dist=" << dist << "m, Angle=" << rot_angle << "rad" << std::endl;
        return true;
    }

    return false;
}

Frame::Ptr Frontend::GetCurrentFrame()
{
    return curr_frame_;
}