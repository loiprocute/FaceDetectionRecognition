#include "opencv2/opencv.hpp"
#include "YuNet.h"
#include "Browser_folder.h"


const std::map<std::string, int> str2backend{
    {"opencv", cv::dnn::DNN_BACKEND_OPENCV}, {"cuda", cv::dnn::DNN_BACKEND_CUDA},
    {"timvx",  cv::dnn::DNN_BACKEND_TIMVX},  {"cann", cv::dnn::DNN_BACKEND_CANN}
};
const std::map<std::string, int> str2target{
    {"cpu", cv::dnn::DNN_TARGET_CPU}, {"cuda", cv::dnn::DNN_TARGET_CUDA},
    {"npu", cv::dnn::DNN_TARGET_NPU}, {"cuda_fp16", cv::dnn::DNN_TARGET_CUDA_FP16}
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


cv::Mat visualize_w_recog(const cv::Mat& image, const cv::Mat& faces, std::vector<std::string>& recognitons, float fps = -1.f)
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
        std::string label = recognitons[i];
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
    cv::Ptr<cv::FaceRecognizerSF> faceRecognizer = cv::FaceRecognizerSF::create("resources/face_recognition_sface_2021dec.onnx", "");


    //Load database vector Embedding
    auto Browser_folder = FindFilesInDatabaseDirectory("\\*.bin");
    std::vector<std::string> filePaths = std::get<0>(Browser_folder);
    std::vector<std::string> subfolders = std::get<1>(Browser_folder);
    std::vector<cv::Mat> loadedMatVector;

    //std::cout << "Files found:" << std::endl;
    for (size_t i = 0; i < filePaths.size(); ++i) {
    cv::Mat feature = readMat(filePaths[i]);
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
        std::tuple<cv::Mat, std::vector<std::string>>  feature_detection = get_feature(model, faceRecognizer, index, subfolders, frame);
        cv::Mat faces = std::get<0>(feature_detection);
        std::vector<std::string> recoginitions = std::get<1>(feature_detection);
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


