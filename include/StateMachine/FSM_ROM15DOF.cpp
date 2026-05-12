#include "FSM_ROM15DOF.hpp"

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
        // viz_->clearPrefix("ROM15DOF/");
        viz_->clear();
    }

    readConfig(CMAKE_SOURCE_DIR "/config/fsm_ROM15DOF_config.yaml");
    loop_count_ = 0;
    policy_to_robot_ = robot_data_->policy.policy_to_robot;
    if (policy_to_robot_.empty()) {
        policy_to_robot_.assign(g1::robot_joint_idx.begin(), g1::robot_joint_idx.end());
    }

    policy_.setIoNames(robot_data_->policy.input_name, robot_data_->policy.output_name);
    policy_.setExpectedDims(std::max(1, robot_data_->policy.history_length) * (9 + 3 * g1::num_act_joint), g1::num_act_joint);
    policy_.load(resolvePolicyPath(robot_data_->policy.onnx_path, CMAKE_SOURCE_DIR "/config/fsm_ROM15DOF_config.yaml"));

    action_dim_ = policy_.isLoaded() ? policy_.outputDim() : g1::num_act_joint;
    frame_obs_dim_ = 9 + 3 * action_dim_;
    policy_obs_dim_ = policy_.isLoaded() ? policy_.inputDim()
                                         : frame_obs_dim_ * std::max(1, robot_data_->policy.history_length);

    raw_action_ = Eigen::VectorXd::Zero(action_dim_);
    clipped_action_ = Eigen::VectorXd::Zero(action_dim_);
    last_good_raw_action_ = Eigen::VectorXd::Zero(action_dim_);
    frame_obs_ = Eigen::VectorXd::Zero(frame_obs_dim_);
    obs_ = Eigen::VectorXd::Zero(policy_obs_dim_);
    obs_history_.clear();

    robot_data_->ctrl.jpos_d = default_jpos_;
    robot_data_->ctrl.jvel_d.setZero();
    robot_data_->ctrl.torq_d.setZero();

    updateObservation();
    updateAction();
    applyPolicyAction(clipped_action_);

    this->state_time = 0.0;
}

template <typename T>
Eigen::VectorXd FSM_ROM15DOFState<T>::makeObservation() const
{
    Eigen::Quaterniond base_q = robot_data_->fbk.quat_B;
    if (!std::isfinite(base_q.w()) || !std::isfinite(base_q.x()) ||
        !std::isfinite(base_q.y()) || !std::isfinite(base_q.z()) ||
        base_q.norm() < 1e-9) {
        base_q = Eigen::Quaterniond::Identity();
    } else {
        base_q.normalize();
    }

    const Eigen::Vector3d gravity_b = base_q.inverse() * Eigen::Vector3d(0.0, 0.0, -1.0);
    Eigen::Vector3d cmd;
    cmd << robot_data_->ctrl.lin_vel_d.x(),
           robot_data_->ctrl.lin_vel_d.y(),
           robot_data_->ctrl.ang_vel_d.z();

    Eigen::VectorXd q_err = Eigen::VectorXd::Zero(action_dim_);
    Eigen::VectorXd dq_sel = Eigen::VectorXd::Zero(action_dim_);
    for (int i = 0; i < action_dim_; ++i) {
        const int j = (i < static_cast<int>(policy_to_robot_.size())) ? policy_to_robot_[i] : i;
        if (j >= 0 && j < static_cast<int>(g1::num_act_joint)) {
            q_err(i) = robot_data_->fbk.jpos(j) - default_jpos_(j);
            dq_sel(i) = robot_data_->policy.obs_joint_vel_scale * robot_data_->fbk.jvel(j);
        }
    }

    Eigen::VectorXd obs(3 + 3 + 3 + action_dim_ + action_dim_);
    obs << robot_data_->policy.obs_base_ang_vel_scale * robot_data_->fbk.varphi_B,
           gravity_b,
           cmd,
           q_err,
           dq_sel;
    return obs;
}

template <typename T>
void FSM_ROM15DOFState<T>::updateObservation()
{
    const Eigen::VectorXd obs_now = makeObservation();
    frame_obs_ = Eigen::VectorXd::Zero(frame_obs_dim_);
    frame_obs_.head(std::min<int>(frame_obs_dim_, obs_now.size())) =
        obs_now.head(std::min<int>(frame_obs_dim_, obs_now.size()));
    frame_obs_.tail(action_dim_) = raw_action_;

    if (obs_history_.empty()) {
        for (int i = 0; i < std::max(1, robot_data_->policy.history_length); ++i) {
            obs_history_.push_back(frame_obs_);
        }
    } else {
        obs_history_.push_back(frame_obs_);
        while (static_cast<int>(obs_history_.size()) > std::max(1, robot_data_->policy.history_length)) {
            obs_history_.pop_front();
        }
    }

    obs_ = Eigen::VectorXd::Zero(policy_obs_dim_);
    const int omega_offset = 0;
    const int gravity_offset = 3;
    const int cmd_offset = 6;
    const int q_offset = 9;
    const int dq_offset = 9 + action_dim_;
    const int last_action_offset = 9 + 2 * action_dim_;
    int out = 0;

    for (const Eigen::VectorXd& frame : obs_history_) {
        if (out + 3 <= policy_obs_dim_) obs_.segment(out, 3) = frame.segment(omega_offset, 3);
        out += 3;
    }
    for (const Eigen::VectorXd& frame : obs_history_) {
        if (out + 3 <= policy_obs_dim_) obs_.segment(out, 3) = frame.segment(gravity_offset, 3);
        out += 3;
    }
    for (const Eigen::VectorXd& frame : obs_history_) {
        if (out + 3 <= policy_obs_dim_) obs_.segment(out, 3) = frame.segment(cmd_offset, 3);
        out += 3;
    }
    for (const Eigen::VectorXd& frame : obs_history_) {
        if (out + action_dim_ <= policy_obs_dim_) {
            obs_.segment(out, action_dim_) = frame.segment(q_offset, action_dim_);
        }
        out += action_dim_;
    }
    for (const Eigen::VectorXd& frame : obs_history_) {
        if (out + action_dim_ <= policy_obs_dim_) {
            obs_.segment(out, action_dim_) = frame.segment(dq_offset, action_dim_);
        }
        out += action_dim_;
    }
    for (const Eigen::VectorXd& frame : obs_history_) {
        if (out + action_dim_ <= policy_obs_dim_) {
            obs_.segment(out, action_dim_) = frame.segment(last_action_offset, action_dim_);
        }
        out += action_dim_;
    }

    if (!allFinite(obs_)) {
        obs_.setZero();
    }
}

template <typename T>
void FSM_ROM15DOFState<T>::updateAction()
{
    clipped_action_.setZero();
    if (!policy_.isLoaded()) {
        raw_action_ = last_good_raw_action_;
        return;
    }

    const Eigen::VectorXd policy_out = policy_.forward(obs_);
    if (policy_out.size() <= 0 || !allFinite(policy_out)) {
        std::cerr << "[ FSM_ROM15DOFState ] Bad policy output; reusing last finite action." << std::endl;
        raw_action_ = last_good_raw_action_;
    } else {
        raw_action_.setZero();
        const int use_dim = std::min<int>(action_dim_, policy_out.size());
        raw_action_.head(use_dim) = policy_out.head(use_dim);
        last_good_raw_action_ = raw_action_;
    }

    clipped_action_ = raw_action_.unaryExpr(
        [clip = std::max(0.0, robot_data_->policy.action_clip)](double x) {
            return std::clamp(x, -clip, clip);
        });
}

template <typename T>
void FSM_ROM15DOFState<T>::applyPolicyAction(const Eigen::VectorXd& action)
{
    robot_data_->ctrl.jpos_d = default_jpos_;

    const int n = std::min<int>(action.size(), policy_to_robot_.size());
    for (int i = 0; i < n; ++i) {
        const int j = policy_to_robot_[i];
        if (j >= 0 && j < static_cast<int>(g1::num_act_joint)) {
            robot_data_->ctrl.jpos_d(j) = default_jpos_(j) + robot_data_->policy.action_scale * action(i);
        }
    }
}

template <typename T>
void FSM_ROM15DOFState<T>::updateCommand()
{
    if (loop_count_ % std::max(1, robot_data_->policy.policy_decimation) == 0) {
        updateObservation();
        updateAction();
    }
    applyPolicyAction(clipped_action_);
       
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

    const Eigen::Vector3d cmd_b(
        robot_data_->ctrl.lin_vel_d.x(),
        robot_data_->ctrl.lin_vel_d.y(),
        0.0
    );

    const double yaw = std::atan2(robot_data_->fbk.R_B(1, 0), robot_data_->fbk.R_B(0, 0));
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    const Eigen::Vector3d cmd_world(
        c * cmd_b.x() - s * cmd_b.y(),
        s * cmd_b.x() + c * cmd_b.y(),
        0.0
    );

    constexpr double kArrowScale = 0.6;
    const Eigen::Vector3d arrow_start = robot_data_->fbk.p_B + Eigen::Vector3d(0.0, 0.0, 0.35);

    if (cmd_b.head<2>().norm() < 1.0e-6) {
        viz_->remove("ROM15DOF/lin_vel_cmd");
    } else {
        viz_->arrowVector(
            "ROM15DOF/lin_vel_cmd",
            arrow_start,
            kArrowScale * cmd_world,
            0.012,
            {0.0f, 1.0f, 0.0f, 1.0f}
        );
    }

    const Eigen::Vector3d actual_world(
        robot_data_->fbk.pdot_B.x(),
        robot_data_->fbk.pdot_B.y(),
        0.0
    );
    if (actual_world.head<2>().norm() < 1.0e-6) {
        viz_->remove("ROM15DOF/lin_vel_actual");
    } else {
        viz_->arrowVector(
            "ROM15DOF/lin_vel_actual",
            arrow_start,
            kArrowScale * actual_world,
            0.010,
            {0.0f, 0.0f, 1.0f, 1.0f}
        );
    }
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

template <typename T>
std::string FSM_ROM15DOFState<T>::resolvePolicyPath(const std::string& raw_path, const std::string& config_file) const
{
    if (raw_path.empty()) {
        return raw_path;
    }

    const std::filesystem::path candidate(raw_path);
    if (candidate.is_absolute()) {
        return raw_path;
    }

    const std::filesystem::path cfg_path(config_file);
    return std::filesystem::weakly_canonical(cfg_path.parent_path() / candidate).string();
}

// template class FSM_ROM15DOFState<float>;
template class FSM_ROM15DOFState<double>;
