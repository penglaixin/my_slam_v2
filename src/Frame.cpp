#include "Frame.h"

Frame::Frame(int id, double time, const cv::Mat &image)
    : id(id), timestamp(time), img(image) {}

Frame::Ptr Frame::createFrame(int id, double time, const cv::Mat &image)
{
    return std::make_shared<Frame>(id, time, image);
}