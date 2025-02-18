#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
// 读取图像并进行预处理
cv::Mat preprocess_image(const std::string& image_path) {
    // 读取图像
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        exit(1);
    }

    // 调整图像大小为28x28（MNIST 图像大小）
    cv::resize(image, image, cv::Size(28, 28));

    // 图像归一化：将像素值从 [0, 255] 转换到 [0.0, 1.0]
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    return image;
}

int main() {
    // 模型路径
    std::string model_path = "../../../../extension/onnx/mnist-8.onnx";  // 修改为实际的模型路径

    // 初始化 ONNX Runtime 环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");

    // 创建会话选项
    Ort::SessionOptions session_options;

    // 加载 ONNX 模型
    Ort::Session session(env, model_path.c_str(), session_options);

    // 输入输出节点名（根据你的模型进行调整）
    const char* input_name = "Input3";  // 根据实际输入节点名称调整
    const char* output_name = "Plus214_Output_0";  // 根据实际输出节点名称调整

    // 读取并预处理图像
    cv::Mat image = preprocess_image("../../../../extension/onnx/test/images/2_58.png");  // 输入图像路径

    // 确保图像数据是连续的
    if (!image.isContinuous()) {
        image = image.clone();
    }

    // 创建输入张量
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, 1, 28, 28};  // 批量大小 1, 1 通道, 28x28 图像

    // 手动将 cv::Mat 数据复制到 std::vector<float>
    std::vector<float> input_data(image.ptr<float>(), image.ptr<float>() + image.total());

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    // 执行推理
    std::vector<const char*> input_node_names{input_name};
    std::vector<const char*> output_node_names{output_name};

    // 获取输出
    std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr},
        input_node_names.data(),
        &input_tensor,
        1,
        output_node_names.data(),
        1
    );

    // 获取输出结果
    float* output_data = output_tensors[0].GetTensorMutableData<float>();

	std::cout << "Output probabilities: ";
	for (int i = 0; i < 10; ++i) {
		std::cout << output_data[i] << " ";
	}
	std::cout << std::endl;

    // 输出预测结果
    int predicted_class = std::distance(output_data, std::max_element(output_data, output_data + 10));
    std::cout << "Predicted Class: " << predicted_class << std::endl;

    return 0;
}
