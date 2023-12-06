#include "opencv2/opencv.hpp"

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

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


std::vector<std::string> readFromFile(const std::string& filename) {
    std::ifstream file(filename); // Mở file
    std::vector<std::string> result; // Vector để lưu dữ liệu từ file

    if (file.is_open()) { // Kiểm tra xem file mở thành công chưa
        std::string line;
        while (std::getline(file, line)) { // Đọc từng dòng từ file
            result.push_back(line); // Thêm dòng vào vector
        }
        file.close(); // Đóng file sau khi đã đọc xong
    }
    else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }

    return result;
}

Mat get_feature(YuNet detector, Ptr<FaceRecognizerSF> faceRecognizer, string input_path) {
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


string replace(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return "**";
    str.replace(start_pos, from.length(), to);
    return str;
}

//int main() {
//    vector<cv::String> fn;
//    glob("noface/*.jpg", fn, false);
//    size_t count = fn.size(); //number of png files in images folder
//
//    vector<Mat> images_feature;
//    vector <string> all_paths;
//    for (size_t i = 0; i < count; i++) {
//        string path = replace(fn[i], "jpg", "bin");
//        cout << path << endl;
//        cout << fn[i] << endl;
//    }
//}



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

    vector<cv::String> fn;
    cv::glob("Database/*/*.jpg", fn, false);
    //size_t count = fn.size(); //number of png files in images folder
    vector <string> all_paths;
    for (size_t i = 0; i < fn.size(); i++) {
    cv:Mat feature = get_feature(model, faceRecognizer, fn[i]);
        if (feature.empty()) {
            cout << fn[i] << endl;
            //continue;
        }
        else {
            string path = replace(fn[i], "jpg", "bin");
            all_paths.push_back(fn[i]);
            saveMat(feature, path);
        }
    }
    return 0;
}

//int main() {
//    vector <string> all_paths = readFromFile("all_paths.txt");
//    std::vector<cv::Mat> loadedMatVector = loadMatVector("mat_vector_data.bin");
//    
//    /*for (const auto& element : loadedMatVector) {
//        std::cout << "Element: " << element << std::endl;
//    }*/
//    cout << "size vector : " << loadedMatVector.size() << endl;
//    cout << "size path : " << all_paths.size() << endl;
//    int i = 4;
//    clock_t begin = clock();
//    cv::Mat queryMat = loadedMatVector[i].clone();
//    cout << "search item : " << all_paths[i] << endl;
//
//    cv::Mat dataset = convertMatVectorToMat(loadedMatVector);
//    cv::flann::Index index(dataset, cv::flann::KDTreeIndexParams());
//
//
//    int k = 5;
//
//
//    cv::Mat indicesMat;
//    cv::Mat distsMat;   
//
//    index.knnSearch(queryMat, indicesMat, distsMat, k, cv::flann::SearchParams());
//    for (int i = 0; i < indicesMat.rows; ++i) {
//        std::cout << "Query point " << i << ":\n";
//        for (int j = 0; j < k; ++j) {
//            int index = indicesMat.at<int>(i, j);
//            float dist = distsMat.at<float>(i, j);
//            std::cout << "Nearest neighbor " << j + 1 << ": Index = " << index << ", Distance = " << dist << "\n";
//            std::cout << "path :" << all_paths[index] << endl;
//        }
//        std::cout << "-----------------\n";
//    }
//    clock_t end = clock(); //ghi lại thời gian lúc sau
//    cout << "Time run: " << (float)(end - begin) / CLOCKS_PER_SEC << " s" << endl;
//
//    return 0;
//}
