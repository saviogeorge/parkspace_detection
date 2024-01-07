#ifndef YOLOV5_H
#define YOLOV5_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>

struct Image
{
    cv::Mat img;
    std::vector<bool> box_intersect_list;
};

class Yolov5Model
{

    public:

    std::vector<cv::Mat> pre_process(cv::Mat &input_image, cv::dnn::Net &net);

    Image post_process(cv::Mat &input_image, std::vector<cv::Mat> &outputs) ;

    private:

    void get_classification_list(std::vector<std::string> &class_list);
    void draw_label(cv::Mat& input_image, std::string label, int left, int top);
    void draw_bounding_boxes(cv::Mat &input_image, std::vector<cv::Rect> &box_list, std::vector<float> &confidence_list, std::vector<int> &box_indices, std::vector<int> &class_ids);
    std::vector<bool> intersect_list(std::vector<cv::Rect> &box_list, std::vector<int> &box_indices);

};



#endif