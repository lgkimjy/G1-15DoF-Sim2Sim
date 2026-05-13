#include "FSM_ROM2FullOrder.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>

namespace {
template <typename Derived>
bool allFinite(const Eigen::MatrixBase<Derived>& value)
{
    return value.array().isFinite().all();
}
}

template <typename T>
FSM_ROM2FullOrderState<T>::FSM_ROM2FullOrderState(RobotData& robot) :
    robot_data_(&robot)
{
    std::cout << "[ FSM_ROM2FullOrderState ] Constructed" << std::endl;
}

template <typename T>
void FSM_ROM2FullOrderState<T>::onEnter()
{
    std::cout << "[ FSM_ROM2FullOrderState ] OnEnter" << std::endl;

    if (viz_) {
        viz_->clear();
    }

    readConfig(CMAKE_SOURCE_DIR "/config/fsm_RoM2FullOrder_config.yaml");

    this->state_time = 0.0;
}

template <typename T>
void FSM_ROM2FullOrderState<T>::runNominal()
{
    this->state_time += 0.001;
}

template <typename T>
void FSM_ROM2FullOrderState<T>::setVisualizer(mujoco::TrajVizUtil* visualizer)
{
    viz_ = visualizer;
}

template <typename T>
void FSM_ROM2FullOrderState<T>::readConfig(std::string config_file)
{
    std::cout << "[ FSM_ROM2FullOrderState ] readConfig: " << config_file << std::endl;
    const YAML::Node config = YAML::LoadFile(config_file);

    try {
        robot_data_->policy.onnx_path = config["policy"]["path"].as<std::string>();
        robot_data_->policy.input_name = config["policy"]["input_name"].as<std::string>();
        robot_data_->policy.output_name = config["policy"]["output_name"].as<std::string>();
        robot_data_->policy.action_scale = config["policy"]["action_scale"].as<double>();
        robot_data_->policy.action_clip = config["policy"]["action_clip"].as<double>();
        robot_data_->policy.obs_base_ang_vel_scale = config["policy"]["obs_base_ang_vel_scale"].as<double>();
        robot_data_->policy.obs_joint_vel_scale = config["policy"]["obs_joint_vel_scale"].as<double>();
        robot_data_->policy.policy_decimation = config["policy"]["policy_decimation"].as<int>();
        robot_data_->policy.history_length = config["policy"]["history_length"].as<int>();
        robot_data_->policy.policy_to_robot.clear();
        if (config["policy"]["policy_to_robot"]) {
            for (const auto& idx : config["policy"]["policy_to_robot"]) {
                robot_data_->policy.policy_to_robot.push_back(idx.as<int>());
            }
        }
        for(int i = 0; i < num_act_joint; i++) {
            robot_data_->param.Kp[i] = config["Kp"][i].as<T>();
            robot_data_->param.Kd[i] = config["Kd"][i].as<T>();
            if (config["tau_limit"] && i < static_cast<int>(config["tau_limit"].size())) {
                robot_data_->param.tau_limit[i] = config["tau_limit"][i].as<T>();
            }
            default_jpos_[i] = config["default_jpos"][i].as<T>();
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading from config file: " << e.what() << std::endl;
    }

}

template class FSM_ROM2FullOrderState<double>;