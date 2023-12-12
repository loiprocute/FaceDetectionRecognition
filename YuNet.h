#pragma once

#include "opencv2/opencv.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

class YuNet
{
public:
    YuNet(const std::string& model_path,
        const cv::Size& input_size = cv::Size(320, 320),
        float conf_threshold = 0.6f,
        float nms_threshold = 0.3f,
        int top_k = 5000,
        int backend_id = 0,
        int target_id = 0);
    void setBackendAndTarget(int backend_id, int target_id);
    /* Overwrite the input size when creating the model. Size format: [Width, Height].
    */
    void setInputSize(const cv::Size& input_size);

    cv::Mat infer(const cv::Mat image);
private:
    cv::Ptr<cv::FaceDetectorYN> model;
    std::string model_path_;
    cv::Size input_size_;
    float conf_threshold_;
    float nms_threshold_;
    int top_k_;
    int backend_id_;
    int target_id_;
};

// Function to find the most frequent element in a vector of strings
std::string mostFrequentElement(const std::vector<std::string>& vec);

std::string get_recognition(cv::flann::Index& index, std::vector<std::string>& labels, cv::Mat& feature);

std::tuple<cv::Mat, std::vector<std::string>> get_feature(YuNet detector, cv::Ptr<cv::FaceRecognizerSF> faceRecognizer, cv::flann::Index& index, std::vector<std::string>& labels, cv::Mat image);
// Function to save a vector of cv::Mat to a file
void saveMat(const cv::Mat mat, const std::string& filename);

cv::Mat readMat(const std::string& filename);

cv::Mat convertMatVectorToMat(const std::vector<cv::Mat>& matVector);
