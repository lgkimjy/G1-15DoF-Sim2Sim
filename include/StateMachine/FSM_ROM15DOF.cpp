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
    this->state_time = 0.0;
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
void FSM_ROM15DOFState<T>::updateCommand()
{
    robot_data_->ctrl.torq_d.setZero();
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
    (void)config;
}

// template class FSM_ROM15DOFState<float>;
template class FSM_ROM15DOFState<double>;
