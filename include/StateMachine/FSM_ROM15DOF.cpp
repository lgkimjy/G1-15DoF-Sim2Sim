#include "FSM_ROM15DOF.hpp"

#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>

template <typename T>
FSM_ROM15DOFState<T>::FSM_ROM15DOFState(RobotData& robot) :
    robot_data_(&robot)
{
    std::cout << "[ FSM_ROM15DOFState ] Constructed" << std::endl;
}

template <typename T>
void FSM_ROM15DOFState<T>::onEnter()
{
    std::cout << "[ FSM_ROM15DOFState ] OnEnter" << std::endl;

    if (viz_) {
        viz_->clearPrefix("ROM15DOF/");
    }

    readConfig(CMAKE_SOURCE_DIR "/config/fsm_ROM15DOF_config.yaml");
    loop_count_ = 0;

    policy_.setIoNames(robot_data_->policy.input_name, robot_data_->policy.output_name);
    policy_.setExpectedDims(std::max(1, robot_data_->policy.history_length) * (9 + 3 * g1::num_act_joint), g1::num_act_joint);
    // policy_.load(resolvePolicyPath(robot_data_->policy.onnx_path, config_file));

    robot_data_->ctrl.jpos_d = default_jpos_;


    this->state_time = 0.0;
}

template <typename T>
void FSM_ROM15DOFState<T>::updateObservation()
{

}

template <typename T>
void FSM_ROM15DOFState<T>::updateAction()
{
    
}

template <typename T>
void FSM_ROM15DOFState<T>::applyPolicyAction(const Eigen::VectorXd& action)
{
    for(int i=0; i<g1::num_act_joint; i++) {
        robot_data_->ctrl.jpos_d[i] = default_jpos_[i] + 0.25 * action[i];
    }
}

template <typename T>
void FSM_ROM15DOFState<T>::updateCommand()
{
    if (loop_count_ % std::max(1, robot_data_->policy.policy_decimation) == 0) {
        // updateObservation();
        // updateAction();
    }
    // applyPolicyAction(clipped_action_);
       
    robot_data_->ctrl.torq_d = robot_data_->param.Kp.asDiagonal() * (robot_data_->ctrl.jpos_d - robot_data_->fbk.jpos)
                        - robot_data_->param.Kd.asDiagonal() * (robot_data_->fbk.jvel);
    ++loop_count_;
}


template <typename T>
void FSM_ROM15DOFState<T>::runNominal()
{
    updateModel();
    updateCommand();
    updateVisualization();

    this->state_time += 0.001;
}

template <typename T>
void FSM_ROM15DOFState<T>::setVisualizer(mujoco::TrajVizUtil* visualizer)
{
    viz_ = visualizer;
}

template <typename T>
void FSM_ROM15DOFState<T>::updateModel()
{
}

template <typename T>
void FSM_ROM15DOFState<T>::updateVisualization()
{
    if (!viz_) return;

    viz_->sphere("ROM15DOF/base",
        robot_data_->fbk.p_B,
        0.035, {0.3f, 0.0f, 0.3f, 0.8f}
    );
}

template <typename T>
void FSM_ROM15DOFState<T>::readConfig(std::string config_file)
{
    std::cout << "[ FSM_ROM15DOFState ] readConfig: " << config_file << std::endl;
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
        for(int i = 0; i < num_act_joint; i++) {
            robot_data_->param.Kp[i] = config["Kp"][i].as<T>();
            robot_data_->param.Kd[i] = config["Kd"][i].as<T>();
            default_jpos_[i] = config["default_jpos"][i].as<T>();
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading from config file: " << e.what() << std::endl;
    }

}

// template class FSM_ROM15DOFState<float>;
template class FSM_ROM15DOFState<double>;
