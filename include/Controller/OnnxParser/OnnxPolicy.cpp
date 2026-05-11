#include "Controller/OnnxParser/OnnxPolicy.hpp"

#include <iostream>
#include <vector>

namespace {
int flattenStaticShape(const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return 0;
    }

    int64_t prod = 1;
    for (std::size_t i = 1; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            return 0;
        }
        prod *= shape[i];
    }
    return static_cast<int>(prod);
}

}

OnnxPolicy::OnnxPolicy()
    : env_(ORT_LOGGING_LEVEL_WARNING, "sim2sim_policy")
{
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetInterOpNumThreads(1);
    session_options_.DisableMemPattern();
    session_options_.DisableCpuMemArena();
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    input_name_ptr_ = input_name_.c_str();
    output_name_ptr_ = output_name_.c_str();
}

void OnnxPolicy::setIoNames(const std::string& input_name, const std::string& output_name)
{
    input_name_ = input_name;
    output_name_ = output_name;
    input_name_ptr_ = input_name_.c_str();
    output_name_ptr_ = output_name_.c_str();
}

void OnnxPolicy::setExpectedDims(int input_dim, int output_dim)
{
    input_dim_ = input_dim;
    output_dim_ = output_dim;
    if (input_dim_ > 0) {
        input_shape_ = {1, static_cast<int64_t>(input_dim_)};
    } else {
        input_shape_.clear();
    }
    if (output_dim_ > 0) {
        output_shape_ = {1, static_cast<int64_t>(output_dim_)};
    } else {
        output_shape_.clear();
    }
}

bool OnnxPolicy::load(const std::string& model_path)
{
    loaded_ = false;
    session_.reset();

    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

        const std::size_t num_inputs = session_->GetInputCount();
        const std::size_t num_outputs = session_->GetOutputCount();
        if (num_inputs != 1 || num_outputs != 1) {
            std::cerr << "[ OnnxPolicy ] Expected 1 input / 1 output, got "
                      << num_inputs << " / " << num_outputs << std::endl;
            session_.reset();
            return false;
        }

        if (input_dim_ > 0 && output_dim_ > 0) {
            if (input_shape_.empty()) {
                input_shape_ = {1, static_cast<int64_t>(input_dim_)};
            }
            if (output_shape_.empty()) {
                output_shape_ = {1, static_cast<int64_t>(output_dim_)};
            }
        } else {
            auto input_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            input_shape_ = input_info.GetShape();
            if (!input_shape_.empty()) input_shape_[0] = 1;
            input_dim_ = flattenStaticShape(input_shape_);

            auto output_info = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
            output_shape_ = output_info.GetShape();
            if (!output_shape_.empty()) output_shape_[0] = 1;
            output_dim_ = flattenStaticShape(output_shape_);
        }

        loaded_ = (input_dim_ > 0 && output_dim_ > 0);
        if (!loaded_) {
            std::cerr << "[ OnnxPolicy ] Failed to infer static input/output dims from: "
                      << model_path << std::endl;
            session_.reset();
            return false;
        }

        std::cout << "[ OnnxPolicy ] " << model_path << " (" << input_dim_ << " -> " << output_dim_ << ")" << std::endl;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[ OnnxPolicy ] ONNX Runtime load failed: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ OnnxPolicy ] Standard exception during ONNX load: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[ OnnxPolicy ] Unknown exception during ONNX load" << std::endl;
    }
    return false;
}

Eigen::VectorXd OnnxPolicy::forward(const Eigen::VectorXd& input)
{
    if (!loaded_) {
        std::cerr << "[ OnnxPolicy ] forward() called before successful load()" << std::endl;
        return Eigen::VectorXd::Zero(output_dim_);
    }
    if (input.size() != input_dim_) {
        std::cerr << "[ OnnxPolicy ] Input size mismatch: got " << input.size()
                  << ", expected " << input_dim_ << std::endl;
        return Eigen::VectorXd::Zero(output_dim_);
    }
    if (!session_) {
        std::cerr << "[ OnnxPolicy ] Session is not available" << std::endl;
        return Eigen::VectorXd::Zero(output_dim_);
    }

    std::vector<float> input_buffer(static_cast<std::size_t>(input_dim_), 0.0f);
    for (int i = 0; i < input_dim_; ++i) {
        input_buffer[static_cast<std::size_t>(i)] = static_cast<float>(input(i));
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_buffer.data(),
        input_buffer.size(),
        input_shape_.data(),
        input_shape_.size());

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        &input_name_ptr_,
        &input_tensor,
        1,
        &output_name_ptr_,
        1);

    if (output_tensors.empty() || !output_tensors.front().IsTensor()) {
        std::cerr << "[ OnnxPolicy ] Inference returned no tensor output" << std::endl;
        return Eigen::VectorXd::Zero(output_dim_);
    }

    const float* output_data = output_tensors.front().GetTensorData<float>();
    Eigen::VectorXd output(output_dim_);
    for (int i = 0; i < output_dim_; ++i) {
        output(i) = static_cast<double>(output_data[static_cast<std::size_t>(i)]);
    }
    return output;
}
