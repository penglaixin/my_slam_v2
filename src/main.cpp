#include "Frontend.h"
#include "Visualizer.h"
#include "LoopClosing.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>

using namespace std;

// 读取文件的辅助函数 (保持你原来的实现即可，为了版面我就不重复贴具体实现了)
void LoadImages(const string &strPathToCam, vector<string> &vstrImageFilenames, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cerr << "Usage: ./run_slam path_to_cam0 path_to_cam1" << endl;
        return -1;
    }

    // 1. 读取数据
    vector<string> left_images, right_images;
    vector<double> timestamps;
    LoadImages(argv[1], left_images, timestamps);
    LoadImages(argv[2], right_images, timestamps);

    if (left_images.empty())
        return -1;

    // 2. 初始化系统
    Frontend slam_system; // 初始化前端

    std::string vocab_path = "/home/plxslam/my_slam_v2/config/ORBvoc.txt"; // 词典路径
    LoopClosing::Ptr loop_closer = std::make_shared<LoopClosing>(vocab_path);

    slam_system.SetLoopCloser(loop_closer); // 注入回环检测模块

    Visualizer viewer; // 可视化器

    // -----------------------------------------
    // 创建文件 trajectory.txt
    // -----------------------------------------
    ofstream f_traj("trajectory.txt");
    f_traj << fixed << setprecision(9); // 设置精度，保证时间戳和坐标足够精确

    // 3. 主循环
    for (size_t i = 0; i < left_images.size(); i++)
    {
        cv::Mat img_l = cv::imread(left_images[i], 0);
        cv::Mat img_r = cv::imread(right_images[i], 0);
        if (img_l.empty())
            continue;

        // --- 核心算法 ---
        slam_system.Track(timestamps[i], img_l, img_r);

        // --- 可视化 ---
        // 获取当前帧 (包含最新的位姿和特征点)
        Frame::Ptr current_frame = slam_system.GetCurrentFrame();

        // 喂给 Visualizer
        viewer.AddCurrentFrame(current_frame);
        viewer.ShowResult(); // 这里面会弹窗 Trajectory 和 Features

        // --- 打印当前位姿 (X坐标) ---
        // 注意：不要用 Track 的返回值，直接看 Frame 的位姿
        cout << "Frame " << i << " X: " << current_frame->T_w_c(0, 3)
             << " Z: " << current_frame->T_w_c(2, 3) << endl;

        Eigen::Matrix3d R = current_frame->T_w_c.block<3, 3>(0, 0);
        Eigen::Vector3d t = current_frame->T_w_c.block<3, 1>(0, 3);
        Eigen::Quaterniond q(R);

        // 写入 trajectory.txt
        f_traj << current_frame->timestamp << " "
               << t.x() << " " << t.y() << " " << t.z() << " "
               << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }
    
    f_traj.close(); // 循环结束后关闭文件
    return 0;
}

// 记得把 LoadImages 函数的实现贴在 main 函数后面
void LoadImages(const string &strPathToCam, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strPathToCam + "/data.csv");
    if (!f.is_open())
        return;
    string s0;
    getline(f, s0);
    while (!f.eof())
    {
        string s;
        getline(f, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            string t_str, filename;
            getline(ss, t_str, ',');
            getline(ss, filename, ',');
            if (filename.empty())
                filename = t_str + ".png";
            while (!filename.empty() && (filename.back() == '\r' || filename.back() == '\n'))
                filename.pop_back();
            vstrImageFilenames.push_back(strPathToCam + "/data/" + filename);
            vTimestamps.push_back(stod(t_str) / 1e9);
        }
    }
}