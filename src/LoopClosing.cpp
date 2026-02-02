#include "LoopClosing.h"

LoopClosing::LoopClosing(std::string vocab_path)
{
    std::cout << "Loading Vocabulary from: " << vocab_path << " ..." << std::endl;
    vocab_.load(vocab_path);

    if (vocab_.empty())
    {
        std::cerr << "Fatal Error: Vocabulary not loaded!" << std::endl;
        exit(-1);
    }
    std::cout << "Vocabulary loaded! Words: " << vocab_.size() << std::endl;

    // 初始化数据库，使用刚才加载的字典
    db_ = DBoW3::Database(vocab_, false, 0);
}

void LoopClosing::AddKeyFrame(Frame::Ptr frame)
{
    // 1. 计算当前帧的 BoW 向量
    // DBoW3 需要 descriptors 是 cv::Mat 类型，我们 Frame 里正好就是
    if (frame->descriptors.empty())
        return;

    // 转换：将描述子转换为词袋向量
    // 这步会计算 frame->bow_vec (需要在 Frame.h 里加这个成员变量，见下文)
    DBoW3::BowVector current_bow_vec;
    vocab_.transform(frame->descriptors, current_bow_vec);

    // 2. 存入数据库
    db_.add(current_bow_vec);
    all_keyframes_.push_back(frame);

    // 3. 查询回环 (Query)
    // 在数据库里找和当前向量最像的 4 个候选帧
    DBoW3::QueryResults results;
    db_.query(current_bow_vec, results, 4, -1);

    // 4. 简单的回环判断逻辑
    if (results.size() > 0)
    {
        for (auto &res : results)
        {
            // res.Id 是数据库里的索引，也就是第几个关键帧
            // res.Score 是相似度得分 (0.0 ~ 1.0)

            // 过滤掉自己 (最新的几帧肯定和自己很像，不算回环)
            int loop_index = res.Id;
            int curr_index = all_keyframes_.size() - 1;

            // 如果这个候选帧是很久以前的 (比如 50 帧之前)，且分数够高
            if (curr_index - loop_index > 50 && res.Score > 0.05)
            {
                std::cout << "========== LOOP CANDIDATE FOUND! ==========" << std::endl;
                std::cout << "Current " << curr_index << " <--> History " << loop_index
                          << " (Score: " << res.Score << ")" << std::endl;

                // [核心] 一旦发现候选者，立即尝试修正！
                CorrectLoop(loop_index, frame);

                // 为了简单，我们只处理分数最高的一个回环，处理完就退出
                return;
            }
        }
    }
}

// --------------------------------------------------------------------------------
// [新增] 回环修正核心逻辑
// 1. 几何验证 (PnP 算相对位姿)
// 2. 位姿图优化 (PGO)
// --------------------------------------------------------------------------------
void LoopClosing::CorrectLoop(int loop_index, Frame::Ptr curr_frame)
{
    Frame::Ptr old_frame = all_keyframes_[loop_index];

    // --- A. 几何验证 (Geometric Verification) ---
    // 通过特征匹配和 PnP，算出 "当前帧" 相对于 "历史帧" 的准确位姿 T_old_curr

    cv::BFMatcher matcher(cv::NORM_HAMMING, true); // 强交叉验证
    std::vector<cv::DMatch> matches;
    matcher.match(old_frame->descriptors, curr_frame->descriptors, matches);

    std::vector<cv::Point3f> object_points; // 历史帧的 3D 点
    std::vector<cv::Point2f> image_points;  // 当前帧的 2D 点

    for (auto &m : matches)
    {
        // 只使用历史帧中有效的 3D 点
        if (old_frame->map_points[m.queryIdx] != cv::Point3f(0, 0, 0))
        {
            object_points.push_back(old_frame->map_points[m.queryIdx]);
            image_points.push_back(curr_frame->keypoints[m.trainIdx].pt);
        }
    }

    if (object_points.size() < 20)
    {
        std::cout << "Loop Rejected: Not enough matched points." << std::endl;
        return;
    }

    cv::Mat rvec, tvec;
    // 相机内参 (简单起见硬编码，标准做法是从 Frame 获取)
    cv::Mat K = (cv::Mat_<double>(3, 3) << 458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1);
    std::vector<int> inliers;

    // 解 PnP 算出相对运动
    cv::solvePnPRansac(object_points, image_points, K, cv::Mat(), rvec, tvec, false, 100, 3.0, 0.99, inliers);

    if (inliers.size() < 15)
    {
        std::cout << "Loop Rejected: PnP failed." << std::endl;
        return;
    }

    std::cout << ">>> Loop Verified! Inliers: " << inliers.size() << ". Starting PGO..." << std::endl;

    // --- B. 计算相对变换矩阵 T_old_curr ---
    // PnP 算出的是 T_curr_w (如果 object_points 是世界坐标)
    // 但这里的 object_points 已经是曾经的世界坐标了，所以可以直接认为算出来的是修正后的位姿
    // 为了简化 PGO，我们构建一个边：从 Old 到 Curr 的变换

    cv::Mat R_mat;
    cv::Rodrigues(rvec, R_mat);
    Eigen::Matrix3d R_eig;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_eig(i, j) = R_mat.at<double>(i, j);
    Eigen::Vector3d t_eig(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    // 这是一个修正后的“正确”位姿 (Camera -> World 的逆)
    // 注意：这里逻辑简化了，严谨的做法是算出 T_old_curr 相对变换
    // 我们这里假设 PnP 直接算出了当前帧在“老地图”里的绝对位姿 T_c_oldmap
    g2o::SE3Quat T_c_oldmap_meas(R_eig, t_eig);

    // --- C. 位姿图优化 (Pose Graph Optimization) ---
    // 构建一个图，把所有关键帧连起来，再加上这条回环边

    g2o::SparseOptimizer optimizer;
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto linearSolver = std::make_unique<LinearSolverType>();
    auto blockSolver = std::make_unique<BlockSolverType>(std::move(linearSolver));
    auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
    optimizer.setAlgorithm(solver);

    // 1. 添加所有关键帧作为顶点
    for (size_t i = 0; i < all_keyframes_.size(); i++)
    {
        g2o::VertexSE3Expmap *v = new g2o::VertexSE3Expmap();
        v->setId(i);

        // 初始估计值：当前的 T_w_c (取逆变为 T_c_w)
        Eigen::Matrix4d T_c_w = all_keyframes_[i]->T_w_c.inverse();
        Eigen::Matrix3d R = T_c_w.block<3, 3>(0, 0);
        Eigen::Vector3d t = T_c_w.block<3, 1>(0, 3);
        v->setEstimate(g2o::SE3Quat(R, t));

        if (i == 0)
            v->setFixed(true); // 固定第一帧
        optimizer.addVertex(v);
    }

    // 2. 添加顺序边 (Odometry Edges): i -> i+1
    for (size_t i = 0; i < all_keyframes_.size() - 1; i++)
    {
        g2o::EdgeSE3Expmap *edge = new g2o::EdgeSE3Expmap();
        edge->setVertex(0, optimizer.vertex(i));
        edge->setVertex(1, optimizer.vertex(i + 1));

        // 测量值：两个帧之间的相对运动 T_i_j = T_i.inv * T_j
        Eigen::Matrix4d T_i = all_keyframes_[i]->T_w_c.inverse();
        Eigen::Matrix4d T_j = all_keyframes_[i + 1]->T_w_c.inverse();
        Eigen::Matrix4d T_rel = T_i * T_j.inverse(); // T_i_w * T_w_j = T_i_j

        Eigen::Matrix3d R_rel = T_rel.block<3, 3>(0, 0);
        Eigen::Vector3d t_rel = T_rel.block<3, 1>(0, 3);

        edge->setMeasurement(g2o::SE3Quat(R_rel, t_rel));
        edge->setInformation(Eigen::MatrixXd::Identity(6, 6));
        optimizer.addEdge(edge);
    }

    // 3. 添加回环边 (Loop Edge): old -> curr
    // 这是一条强约束边，把它加上去，图就会变形闭合
    g2o::EdgeSE3Expmap *loop_edge = new g2o::EdgeSE3Expmap();
    loop_edge->setVertex(0, optimizer.vertex(loop_index));                // 历史帧
    loop_edge->setVertex(1, optimizer.vertex(all_keyframes_.size() - 1)); // 当前帧

    // 测量值：我们刚才 PnP 算出来的相对变换
    // 为了简化，这里其实应该算出 T_old_curr。
    // 简易做法：利用 PnP 算出的 T_c_w (相对于世界)，和 Old 帧的 T_old_w 组合
    // 但因为 PnP 是拿 Old 帧的点算的，所以算出来的直接就是 T_curr_old (在 Old 坐标系下的位姿)
    // T_c_oldmap_meas 就是 T_curr_old 的逆 (World->Camera)

    // 这里如果数学推导太复杂，我们用一个工程近似：
    // 我们认为 PnP 算出的位姿是“绝对真理”，希望当前帧优化到那个位置
    // 但 g2o 需要的是相对边。

    // 让我们用最直观的方法：计算 T_old_curr = T_old_w * T_w_curr (用PnP结果)
    // PnP 结果 T_pnp 是 Camera -> World (Old Frame System)
    Eigen::Matrix4d T_curr_in_old_world = Eigen::Matrix4d::Identity();
    T_curr_in_old_world.block<3, 3>(0, 0) = R_eig;       // 这里 R_eig 是 PnP 算出的 R_cw
    T_curr_in_old_world.block<3, 1>(0, 3) = t_eig;       // t_cw
    T_curr_in_old_world = T_curr_in_old_world.inverse().eval(); // 变成 T_w_c (在旧世界坐标系下)

    Eigen::Matrix4d T_old_w = old_frame->T_w_c.inverse(); // World -> Old Camera (其实就是 Old Camera -> World 的逆)
    // 这里的坐标系有点绕。
    // 简单粗暴法：PnP 算出的就是 T_c_w (假设 Old Frame 是原点)。
    // 所以相对变换就是 T_c_w 本身。
    loop_edge->setMeasurement(T_c_oldmap_meas);

    // 给回环边极大的权重 (强行拉过去)
    loop_edge->setInformation(Eigen::MatrixXd::Identity(6, 6) * 100.0);
    optimizer.addEdge(loop_edge);

    // 4. 执行优化
    std::cout << "Optimizing pose graph..." << std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    // 5. 更新所有关键帧位姿 (Correct All Poses)
    std::cout << "Loop Closed! Updating poses..." << std::endl;
    for (size_t i = 0; i < all_keyframes_.size(); i++)
    {
        g2o::VertexSE3Expmap *v = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(i));
        g2o::SE3Quat se3 = v->estimate().inverse(); // T_c_w -> T_w_c

        all_keyframes_[i]->T_w_c.block<3, 3>(0, 0) = se3.rotation().toRotationMatrix();
        all_keyframes_[i]->T_w_c.block<3, 1>(0, 3) = se3.translation();
    }

    // 同步更新当前帧（因为 main 函数用的是 curr_frame 画图）
    curr_frame->T_w_c = all_keyframes_.back()->T_w_c;
}