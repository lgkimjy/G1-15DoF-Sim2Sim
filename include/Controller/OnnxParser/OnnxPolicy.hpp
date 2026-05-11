#pragma once

#include <Eigen/Dense>
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

class OnnxPolicy {
public:
    OnnxPolicy();

    bool load(const std::string& model_path);
    bool isLoaded() const { return loaded_; }

    int inputDim() const { return input_dim_; }
    int outputDim() const { return output_dim_; }

    Eigen::VectorXd forward(const Eigen::VectorXd& input);
    void setIoNames(const std::string& input_name, const std::string& output_name);
    void setExpectedDims(int input_dim, int output_dim);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    bool loaded_ = false;
    int input_dim_ = 0;
    int output_dim_ = 0;

    std::string input_name_ = "obs";
    std::string output_name_ = "actions";
    const char* input_name_ptr_ = nullptr;
    const char* output_name_ptr_ = nullptr;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
};
