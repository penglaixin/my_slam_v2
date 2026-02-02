#include "Backend.h"

// 构造函数
Backend::Backend()
{
    // 如果需要，可以在这里初始化内参，或者直接在头文件赋值
}

void Backend::AddFrame(Frame::Ptr frame)
{
    sliding_window_.push_back(frame);
    if (sliding_window_.size() > window_size_)
    {
        sliding_window_.pop_front();
    }
}

Eigen::Matrix4d Backend::GetCurrentPose()
{
    if (sliding_window_.empty())
        return Eigen::Matrix4d::Identity();
    return sliding_window_.back()->T_w_c;
}

// --- 核心优化函数 ---
void Backend::Optimize()
{
    // 1. 检查帧数，少于2帧不优化
    if (sliding_window_.size() < 2)
        return;

    // 2. 初始化 g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    // optimizer.setVerbose(true); // 调试时可打开

    // 3. 添加顶点：相机位姿 (Pose)
    // 注意：这里遍历的是 sliding_window_ (成员变量)
    for (size_t i = 0; i < sliding_window_.size(); i++)
    {
        g2o::VertexSE3Expmap *v_pose = new g2o::VertexSE3Expmap();
        v_pose->setId(i); // ID 0 ~ 6

        // [修改点] 使用 -> 访问指针成员
        Eigen::Matrix4d T_c_w = sliding_window_[i]->T_w_c.inverse();
        Eigen::Matrix3d R = T_c_w.block<3, 3>(0, 0);
        Eigen::Vector3d t = T_c_w.block<3, 1>(0, 3);

        v_pose->setEstimate(g2o::SE3Quat(R, t));

        // 固定第一帧
        if (i == 0)
            v_pose->setFixed(true);

        optimizer.addVertex(v_pose);
    }

    // 4. 添加顶点：路标点 (Point)
    std::map<int, int> map_point_id_to_g2o_id;
    int g2o_point_id_counter = sliding_window_.size();

    for (size_t i = 0; i < sliding_window_.size(); i++)
    {
        Frame::Ptr f = sliding_window_[i]; // [修改点] 取出来的是指针

        for (size_t k = 0; k < f->track_ids.size(); k++)
        { // [修改点] 用 ->
            int id = f->track_ids[k];
            if (id == -1)
                continue;

            if (map_point_id_to_g2o_id.find(id) == map_point_id_to_g2o_id.end())
            {
                g2o::VertexPointXYZ *v_point = new g2o::VertexPointXYZ();
                v_point->setId(g2o_point_id_counter);

                // [修改点] 用 ->
                cv::Point3f &p = f->map_points[k];
                v_point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
                v_point->setMarginalized(true);

                optimizer.addVertex(v_point);

                map_point_id_to_g2o_id[id] = g2o_point_id_counter;
                g2o_point_id_counter++;
            }
        }
    }

    // 5. 添加边 (Edge)
    for (size_t i = 0; i < sliding_window_.size(); i++)
    {
        Frame::Ptr f = sliding_window_[i]; // [修改点] 指针

        for (size_t k = 0; k < f->keypoints.size(); k++)
        {
            int id = f->track_ids[k];
            if (id == -1)
                continue;

            int point_vertex_id = map_point_id_to_g2o_id[id];

            g2o::EdgeSE3ProjectXYZ *edge = new g2o::EdgeSE3ProjectXYZ();
            edge->setVertex(0, optimizer.vertex(point_vertex_id));
            edge->setVertex(1, optimizer.vertex(i));

            // [修改点] 用 -> 访问 keypoints
            edge->setMeasurement(Eigen::Vector2d(f->keypoints[k].pt.x, f->keypoints[k].pt.y));
            edge->setInformation(Eigen::Matrix2d::Identity());

            // 相机内参 (使用类成员变量 fx, fy...)
            edge->fx = fx;
            edge->fy = fy;
            edge->cx = cx;
            edge->cy = cy;

            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            edge->setRobustKernel(rk);
            rk->setDelta(1.0);

            optimizer.addEdge(edge);
        }
    }

    // 6. 执行优化
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // 7. 写回结果

    // A. 写回位姿
    for (size_t i = 0; i < sliding_window_.size(); i++)
    {
        g2o::VertexSE3Expmap *v_pose = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(i));
        g2o::SE3Quat se3 = v_pose->estimate().inverse();

        // [修改点] 用 -> 写回
        sliding_window_[i]->T_w_c.block<3, 3>(0, 0) = se3.rotation().toRotationMatrix();
        sliding_window_[i]->T_w_c.block<3, 1>(0, 3) = se3.translation();
    }

    // B. 写回点坐标
    for (size_t i = 0; i < sliding_window_.size(); i++)
    {
        Frame::Ptr f = sliding_window_[i]; // [修改点] 指针

        for (size_t k = 0; k < f->track_ids.size(); k++)
        {
            int id = f->track_ids[k];
            if (id == -1)
                continue;

            int g2o_id = map_point_id_to_g2o_id[id];
            g2o::VertexPointXYZ *v_point = static_cast<g2o::VertexPointXYZ *>(optimizer.vertex(g2o_id));
            Eigen::Vector3d p_opt = v_point->estimate();

            // [修改点] 用 -> 写回
            f->map_points[k] = cv::Point3f(p_opt.x(), p_opt.y(), p_opt.z());
        }
    }
}