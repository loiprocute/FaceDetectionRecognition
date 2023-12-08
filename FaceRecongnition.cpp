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

cv::Mat visualize(const cv::Mat& image, const cv::Mat& faces, float fps = -1.f)
{
    static cv::Scalar box_color{ 0, 255, 0 };
    static std::vector<cv::Scalar> landmark_color{
        cv::Scalar(255,   0,   0), // right eye
        cv::Scalar(0,   0, 255), // left eye
        cv::Scalar(0, 255,   0), // nose tip
        cv::Scalar(255,   0, 255), // right mouth corner
        cv::Scalar(0, 255, 255)  // left mouth corner
    };
    static cv::Scalar text_color{ 0, 255, 0 };

    auto output_image = image.clone();

    if (fps >= 0)
    {
        cv::putText(output_image, cv::format("FPS: %.2f", fps), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
    }

    for (int i = 0; i < faces.rows; ++i)
    {
        // Draw bounding boxes
        int x1 = static_cast<int>(faces.at<float>(i, 0));
        int y1 = static_cast<int>(faces.at<float>(i, 1));
        int w = static_cast<int>(faces.at<float>(i, 2));
        int h = static_cast<int>(faces.at<float>(i, 3));
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);

        // Confidence as text
        float conf = faces.at<float>(i, 14);
        cv::putText(output_image, cv::format("%.4f", conf), cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);

        // Draw landmarks
        for (int j = 0; j < landmark_color.size(); ++j)
        {
            int x = static_cast<int>(faces.at<float>(i, 2 * j + 4)), y = static_cast<int>(faces.at<float>(i, 2 * j + 5));
            cv::circle(output_image, cv::Point(x, y), 2, landmark_color[j], 2);
        }
    }
    return output_image;
}


cv::Mat visualize_w_recog(const cv::Mat& image, const cv::Mat& faces, vector<string>& recognitons, float fps = -1.f)
{
    static cv::Scalar box_color{ 0, 255, 0 };
    static std::vector<cv::Scalar> landmark_color{
        cv::Scalar(255,   0,   0), // right eye
        cv::Scalar(0,   0, 255), // left eye
        cv::Scalar(0, 255,   0), // nose tip
        cv::Scalar(255,   0, 255), // right mouth corner
        cv::Scalar(0, 255, 255)  // left mouth corner
    };
    static cv::Scalar text_color{ 0, 255, 0 };

    auto output_image = image.clone();

    if (fps >= 0)
    {
        cv::putText(output_image, cv::format("FPS: %.2f", fps), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
    }

    for (int i = 0; i < faces.rows; ++i)
    {
        // Draw bounding boxes
        int x1 = static_cast<int>(faces.at<float>(i, 0));
        int y1 = static_cast<int>(faces.at<float>(i, 1));
        int w = static_cast<int>(faces.at<float>(i, 2));
        int h = static_cast<int>(faces.at<float>(i, 3));
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);

        // Confidence as text
        float conf = faces.at<float>(i, 14);
        cv::putText(output_image, cv::format("%.4f", conf), cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);

        //Recongize as Text
        string label = recognitons[i];
        cv::putText(output_image, label, cv::Point(x1+w, y1 + h), cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);
        
        // Draw landmarks
        for (int j = 0; j < landmark_color.size(); ++j)
        {
            int x = static_cast<int>(faces.at<float>(i, 2 * j + 4)), y = static_cast<int>(faces.at<float>(i, 2 * j + 5));
            cv::circle(output_image, cv::Point(x, y), 2, landmark_color[j], 2);
        }
    }
    return output_image;
}

std::vector<std::string> readFromFile(const std::string& filename) {
    std::ifstream file(filename); // M? file
    std::vector<std::string> result; // Vector ?? l?u d? li?u t? file

    if (file.is_open()) { // Ki?m tra xem file m? th�nh c�ng ch?a
        std::string line;
        while (std::getline(file, line)) { // ??c t?ng d�ng t? file
            result.push_back(line); // Th�m d�ng v�o vector
        }
        file.close(); // ?�ng file sau khi ?� ??c xong
    }
    else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }

    return result;
}


// Function to find the most frequent element in a vector of strings
std::string mostFrequentElement(const std::vector<std::string>& vec) {
    std::unordered_map<std::string, int> freqMap;

    // Count occurrences of each string in the vector
    for (const std::string& str : vec) {
        freqMap[str]++;
    }

    // Find the string with the maximum occurrences
    std::string mostFrequent;
    int maxFrequency = 0;

    for (const auto& pair : freqMap) {
        if (pair.second > maxFrequency) {
            mostFrequent = pair.first;
            maxFrequency = pair.second;
        }
    }

    return mostFrequent;
}


string get_recognition(cv::flann::Index& index, std::vector<std::string>& labels, Mat& feature) {


    int k = 3;
    cv::Mat indicesMat;
    cv::Mat distsMat;

    vector<string> filter_path;
    //std::cout << "feature: " << feature.size() << std::endl;
    //std::cout << "indicesMat: " << indicesMat.size() << std::endl;
    //std::cout << "distsMat: " << distsMat.size() << std::endl;

    index.knnSearch(feature, indicesMat, distsMat, k, cv::flann::SearchParams());

    //cout << indicesMat.rows << endl;
    for (int i = 0; i < indicesMat.rows; ++i) {
        for (int j = 0; j < k; ++j) {
            float dist = distsMat.at<float>(i, j);
            if (dist < 70) {
                int idx = indicesMat.at<int>(i, j);
                cout << labels[idx] << " - score : " << dist << endl;
                filter_path.push_back(labels[idx]);
            }
            
        }
    }
    if (filter_path.size() == 0) {
        return "unknow";
    }
    std::string mostFrequent = mostFrequentElement(filter_path);
    return mostFrequent;


}

std::tuple<cv::Mat, std::vector<std::string>> get_feature(YuNet detector, Ptr<FaceRecognizerSF> faceRecognizer, cv::flann::Index& index, std::vector<std::string>& labels, Mat image) {
    //auto image = cv::imread(input_path);
    // Inference
    detector.setInputSize(image.size());
    vector<string> recognitons;
    auto faces = detector.infer(image);

    if (faces.rows == 0) {
        cv::Mat emptyMat;
        return std::make_tuple(emptyMat, recognitons);
    }
    else {
        for (int i = 0; i < faces.rows; i++) {
            Mat aligned_face;
            faceRecognizer->alignCrop(image, faces.row(0), aligned_face);
            // Run feature extraction with given aligned_face
            Mat feature;
            faceRecognizer->feature(aligned_face, feature);
            recognitons.push_back(get_recognition(index, labels, feature));

        }
        return std::make_tuple(faces, recognitons);
    }
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

// Function to convert a vector of cv::Mat to a single cv::Mat by concatenating vertically (along rows)
cv::Mat convertMatVectorToMat(const std::vector<cv::Mat>& matVector) {
    cv::Mat result;

    if (matVector.empty()) {
        std::cerr << "Input vector is empty!" << std::endl;
        return result;
    }

    cv::vconcat(matVector, result); // Vertical concatenation

    return result;
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


    //Load database vector Embedding
    auto Browser_folder = FindFilesInDatabaseDirectory("\\*.bin");
    std::vector<std::string> filePaths = std::get<0>(Browser_folder);
    std::vector<std::string> subfolders = std::get<1>(Browser_folder);
    std::vector<cv::Mat> loadedMatVector;

    //std::cout << "Files found:" << std::endl;
    for (size_t i = 0; i < filePaths.size(); ++i) {
    cv:Mat feature = readMat(filePaths[i]);
        loadedMatVector.push_back(feature.clone());
    }

    cv::Mat dataset = convertMatVectorToMat(loadedMatVector);
    cv::flann::Index index(dataset, cv::flann::KDTreeIndexParams());

    int device_id = 0;
    auto cap = cv::VideoCapture(device_id);
    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    model.setInputSize(cv::Size(w, h));

    auto tick_meter = cv::TickMeter();
    cv::Mat frame;
    while (cv::waitKey(1) < 0)
    {
        bool has_frame = cap.read(frame);
        if (!has_frame)
        {
            std::cout << "No frames grabbed! Exiting ...\n";
            break;
        }

        // Inference
        tick_meter.start();
        auto feature_detection = get_feature(model, faceRecognizer, index, subfolders, frame);
        auto faces = std::get<0>(feature_detection);
        auto recoginitions = std::get<1>(feature_detection);
        tick_meter.stop();
        // Draw results on the input image
        auto res_image = visualize_w_recog(frame, faces, recoginitions, (float)tick_meter.getFPS());
        // Visualize in a new window
        cv::imshow("YuNet Demo", res_image);
        tick_meter.reset();

        // // Inference
        //tick_meter.start();
        //auto faces = model.infer(frame);
        //tick_meter.stop();
        //// Draw results on the input image
        //auto res_image = visualize(frame, faces,(float)tick_meter.getFPS());
        //// Visualize in a new window
        //cv::imshow("YuNet Demo", res_image);

        tick_meter.reset();
    }



    return 0;
}


