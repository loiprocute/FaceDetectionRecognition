#include "opencv2/opencv.hpp"

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <windows.h>

using namespace std;
using namespace cv;

const std::map<std::string, int> str2backend{
    {"opencv", cv::dnn::DNN_BACKEND_OPENCV}, {"cuda", cv::dnn::DNN_BACKEND_CUDA},
    {"timvx",  cv::dnn::DNN_BACKEND_TIMVX},  {"cann", cv::dnn::DNN_BACKEND_CANN}
};
const std::map<std::string, int> str2target{
    {"cpu", cv::dnn::DNN_TARGET_CPU}, {"cuda", cv::dnn::DNN_TARGET_CUDA},
    {"npu", cv::dnn::DNN_TARGET_NPU}, {"cuda_fp16", cv::dnn::DNN_TARGET_CUDA_FP16}
};

class YuNet
{
public:
    YuNet(const std::string& model_path,
        const cv::Size& input_size = cv::Size(320, 320),
        float conf_threshold = 0.6f,
        float nms_threshold = 0.3f,
        int top_k = 5000,
        int backend_id = 0,
        int target_id = 0)
        : model_path_(model_path), input_size_(input_size),
        conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
        top_k_(top_k), backend_id_(backend_id), target_id_(target_id)
    {
        model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
    }

    void setBackendAndTarget(int backend_id, int target_id)
    {
        backend_id_ = backend_id;
        target_id_ = target_id;
        model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
    }

    /* Overwrite the input size when creating the model. Size format: [Width, Height].
    */
    void setInputSize(const cv::Size& input_size)
    {
        input_size_ = input_size;
        model->setInputSize(input_size_);
    }

    cv::Mat infer(const cv::Mat image)
    {
        cv::Mat res;
        model->detect(image, res);
        return res;
    }

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


cv::Mat get_feature(YuNet detector, Ptr<FaceRecognizerSF> faceRecognizer, string input_path) {
    auto image = cv::imread(input_path);
    // Inference
    detector.setInputSize(image.size());
    auto faces = detector.infer(image);
    if (faces.rows == 0) {
        cv::Mat emptyMat;
        return emptyMat;
    }
    // Aligning and cropping facial image through the first face of faces detected.
    Mat aligned_face;
    faceRecognizer->alignCrop(image, faces.row(0), aligned_face);
    // Run feature extraction with given aligned_face
    Mat feature;
    faceRecognizer->feature(aligned_face, feature);
    return feature;
}


// Function to save a vector of cv::Mat to a file
void saveMat(const cv::Mat mat, const std::string& filename) {
    std::ofstream outputFile(filename, std::ios::binary);
    if (outputFile.is_open()) {
        int rows = mat.rows;
        int cols = mat.cols;
        int type = mat.type();

        outputFile.write(reinterpret_cast<char*>(&rows), sizeof(int));
        outputFile.write(reinterpret_cast<char*>(&cols), sizeof(int));
        outputFile.write(reinterpret_cast<char*>(&type), sizeof(int));

        if (!mat.empty()) {
            outputFile.write(reinterpret_cast<const char*>(mat.data), mat.total() * mat.elemSize());
            std::cout << "Matrix data has been written to " << filename << " successfully!" << std::endl;
        }
        else {
            std::cerr << "Matrix is empty! Cannot write data to file." << std::endl;
        }

        outputFile.close();
    }
    else {
        std::cerr << "Could not open the file " << filename << " for writing!" << std::endl;
    }
}

cv::Mat readMat(const std::string& filename) {
    std::ifstream inputFile(filename, std::ios::binary);
    cv::Mat result;

    if (inputFile.is_open()) {
        int rows, cols, type;

        inputFile.read(reinterpret_cast<char*>(&rows), sizeof(int));
        inputFile.read(reinterpret_cast<char*>(&cols), sizeof(int));
        inputFile.read(reinterpret_cast<char*>(&type), sizeof(int));

        result.create(rows, cols, type);

        if (!result.empty()) {
            inputFile.read(reinterpret_cast<char*>(result.data), result.total() * result.elemSize());
            std::cout << "Matrix data has been read from " << filename << " successfully!" << std::endl;
        }
        else {
            std::cerr << "Failed to create matrix for data reading!" << std::endl;
        }

        inputFile.close();
    }
    else {
        std::cerr << "Could not open the file " << filename << " for reading!" << std::endl;
    }

    return result;
}

string replace(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos)
        return "**";
    str.replace(start_pos, from.length(), to);
    return str;
}

// Function to find files in a directory and its subdirectories
std::tuple<std::vector<std::string>, std::vector<std::string>> FindFilesInDirectory(const std::string& directory, const std::string& subdir, const std::string& format) {
    std::vector<std::string> filePaths;
    std::vector<std::string> subdirs;
    WIN32_FIND_DATAA fileData;
    //std::string searchPath = directory + "\\*.jpg"; // Path to .jpg files
    std::string searchPath = directory + format; // Path to .jpg files
    HANDLE hFind = FindFirstFileA(searchPath.c_str(), &fileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            std::string fileName = fileData.cFileName;
            if ((fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
                filePaths.push_back(directory + "\\" + fileName);
                subdirs.push_back(subdir);
            }
        } while (FindNextFileA(hFind, &fileData) != 0);
        FindClose(hFind);
    }

    return std::make_tuple(filePaths, subdirs);
}

// Function to find files in the Database directory and its subdirectories
std::tuple<std::vector<std::string>, std::vector<std::string>> FindFilesInDatabaseDirectory(const std::string& format) {
    std::vector<std::string> allFilePaths;
    std::vector<std::string> allSubfolder;
    std::string databaseDirectory = "Database"; // Path to the Database directory
    WIN32_FIND_DATAA dirData;
    std::string searchDirPath = databaseDirectory + "\\*";

    HANDLE hDir = FindFirstFileA(searchDirPath.c_str(), &dirData);
    if (hDir != INVALID_HANDLE_VALUE) {
        do {
            if ((dirData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
                strcmp(dirData.cFileName, ".") != 0 &&
                strcmp(dirData.cFileName, "..") != 0) {

                std::string subdirPath = databaseDirectory + "\\" + dirData.cFileName;
                auto filesAndSubdirs = FindFilesInDirectory(subdirPath, dirData.cFileName, format);
                std::vector<std::string> filesInSubdir = std::get<0>(filesAndSubdirs);
                std::vector<std::string> subdir = std::get<1>(filesAndSubdirs);
                allFilePaths.insert(allFilePaths.end(), filesInSubdir.begin(), filesInSubdir.end());
                allSubfolder.insert(allSubfolder.end(), subdir.begin(), subdir.end());
            }
        } while (FindNextFileA(hDir, &dirData) != 0);
        FindClose(hDir);
    }

    return std::make_tuple(allFilePaths, allSubfolder);
}


int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv,
        "{help  h           |                                   | Print this message}"
        "{input i           |                                   | Set input to a certain image, omit if using camera}"
        "{model m           | resources/face_detection_yunet_2023mar.onnx  | Set path to the model}"
        "{backend b         | opencv                            | Set DNN backend}"
        "{target t          | cpu                               | Set DNN target}"
        "{save s            | false                             | Whether to save result image or not}"
        "{vis v             | false                             | Whether to visualize result image or not}"
        /* model params below*/
        "{conf_threshold    | 0.8                               | Set the minimum confidence for the model to identify a face. Filter out faces of conf < conf_threshold}"
        "{nms_threshold     | 0.3                               | Set the threshold to suppress overlapped boxes. Suppress boxes if IoU(box1, box2) >= nms_threshold, the one of higher score is kept.}"
        "{top_k             | 5000                              | Keep top_k bounding boxes before NMS. Set a lower value may help speed up postprocessing.}"
    );
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    std::string input_path = parser.get<std::string>("input");
    std::string model_path = parser.get<std::string>("model");
    std::string backend = parser.get<std::string>("backend");
    std::string target = parser.get<std::string>("target");
    bool save_flag = parser.get<bool>("save");
    bool vis_flag = parser.get<bool>("vis");

    // model params
    float conf_threshold = parser.get<float>("conf_threshold");
    float nms_threshold = parser.get<float>("nms_threshold");
    int top_k = parser.get<int>("top_k");
    const int backend_id = str2backend.at(backend);
    const int target_id = str2target.at(target);

    // Instantiate YuNet
    YuNet model(model_path, cv::Size(320, 320), conf_threshold, nms_threshold, top_k, backend_id, target_id);
    Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create("resources/face_recognition_sface_2021dec.onnx", "");

    auto Browser_folder = FindFilesInDatabaseDirectory("\\*.jpg");
    std::vector<std::string> filePaths = std::get<0>(Browser_folder);
    std::vector<std::string> subfolders = std::get<1>(Browser_folder);

    for (size_t i = 0; i < filePaths.size(); ++i) {
        cv:Mat feature = get_feature(model, faceRecognizer, filePaths[i]);
        if (feature.empty()) {
            cout << filePaths[i]  << endl;
            subfolders.erase(subfolders.begin() + i);
        }
        else {
            string tmp_path = replace(filePaths[i], "jpg", "bin");
            saveMat(feature, tmp_path);
            std::cout << "File: " << filePaths[i] << " in Subfolder: " << subfolders[i] << std::endl;
        }
        
    }

    return 0;
}


