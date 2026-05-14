#include "FSM_ROM2FullOrder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <string>
#include <yaml-cpp/yaml.h>

namespace {
constexpr int kPolicyActionDim = static_cast<int>(g1_reduced_order::num_act_joint);
constexpr std::array<int, 12> kLegPolicySlots = {0, 1, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14};
constexpr std::array<int, 12> kLegJointIndices = {0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11};
constexpr std::array<int, 3> kTorsoOrientationSlots = {2, 5, 8};
constexpr std::array<const char*, kPolicyActionDim> kReferenceActionJointNames = {
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "torso_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "torso_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "torso_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint"};
constexpr std::array<int, kPolicyActionDim> kReferenceActionFullJointIndex = {
    0, 6, -1,
    1, 7, -1,
    2, 8, -1,
    3, 9,
    4, 10,
    5, 11};
constexpr std::array<int, 11> kIkJointIndices = {
    12, 13, 14,      // waist yaw/roll/pitch
    15, 16, 17, 18,  // left shoulder pitch/roll/yaw + elbow
    22, 23, 24, 25   // right shoulder pitch/roll/yaw + elbow
};
constexpr int kIkDof = static_cast<int>(kIkJointIndices.size());
const Eigen::Vector3d kFallbackTorsoComLocal(0.046665935, 0.003256103, 0.192284379);
const Eigen::Vector3d kVirtualTorsoOrientationLimit(0.52, 0.52, 2.618);
constexpr std::array<double, kIkDof> kIkJointLower = {
    -2.618, -0.52, -0.52,
    -0.75, -0.75, -1.00, -0.60,
    -0.75, -0.75, -1.00, -0.60};
constexpr std::array<double, kIkDof> kIkJointUpper = {
    2.618, 0.52, 0.52,
    1.20, 0.80, 1.00, 1.20,
    1.20, 0.75, 1.00, 1.20};
constexpr std::array<double, kIkDof> kIkJointMobility = {
    1.00, 1.00, 1.00,
    0.45, 0.35, 0.20, 0.15,
    0.45, 0.35, 0.20, 0.15};
constexpr std::array<double, kIkDof> kIkJointMaxStep = {
    0.006, 0.006, 0.006,
    0.006, 0.006, 0.005, 0.004,
    0.006, 0.006, 0.005, 0.004};
constexpr std::array<double, kIkDof> kIkJointMaxTargetError = {
    0.08, 0.08, 0.15,
    0.12, 0.12, 0.10, 0.10,
    0.12, 0.12, 0.10, 0.10};

template <typename Derived>
bool allFinite(const Eigen::MatrixBase<Derived>& value)
{
    return value.array().isFinite().all();
}

Eigen::Vector3d bodyRpyFromQuat(const Eigen::Quaterniond& q_raw)
{
    Eigen::Quaterniond q = q_raw;
    if (q.norm() < 1.0e-9) {
        return Eigen::Vector3d::Zero();
    }
    q.normalize();
    const Eigen::Vector3d zyx = q.toRotationMatrix().eulerAngles(2, 1, 0);
    return Eigen::Vector3d(zyx.z(), zyx.y(), zyx.x());
}

Eigen::Quaterniond yawOnlyQuaternion(const Eigen::Quaterniond& q_raw)
{
    Eigen::Quaterniond q = q_raw;
    if (q.norm() < 1.0e-9) {
        return Eigen::Quaterniond::Identity();
    }
    q.normalize();
    const Eigen::Matrix3d R = q.toRotationMatrix();
    const double yaw = std::atan2(R(1, 0), R(0, 0));
    return Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
}

Eigen::Vector3d orientationErrorWorld(const Eigen::Quaterniond& q_ref_raw, const Eigen::Quaterniond& q_meas_raw)
{
    Eigen::Quaterniond q_ref = q_ref_raw;
    Eigen::Quaterniond q_meas = q_meas_raw;
    if (q_ref.norm() < 1.0e-9 || q_meas.norm() < 1.0e-9) {
        return Eigen::Vector3d::Zero();
    }
    q_ref.normalize();
    q_meas.normalize();

    Eigen::Quaterniond q_err = q_ref * q_meas.conjugate();
    if (q_err.w() < 0.0) {
        q_err.coeffs() *= -1.0;
    }
    q_err.normalize();

    Eigen::AngleAxisd aa(q_err);
    return aa.axis() * aa.angle();
}

Eigen::VectorXd limitNorm(const Eigen::VectorXd& value, double max_norm)
{
    if (max_norm <= 0.0) {
        return Eigen::VectorXd::Zero(value.size());
    }
    const double norm = value.norm();
    if (norm <= max_norm || norm <= 1.0e-12) {
        return value;
    }
    return value * (max_norm / norm);
}

double smoothUnit(double value, double zero_at, double one_at)
{
    if (one_at <= zero_at) {
        return value >= one_at ? 1.0 : 0.0;
    }
    const double x = std::clamp((value - zero_at) / (one_at - zero_at), 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

Eigen::MatrixXd weightedDampedPseudoInverse(
    const Eigen::MatrixXd& J,
    const Eigen::MatrixXd& mobility,
    double damping)
{
    const int task_dim = static_cast<int>(J.rows());
    const Eigen::MatrixXd task_metric =
        J * mobility * J.transpose() + damping * Eigen::MatrixXd::Identity(task_dim, task_dim);
    return mobility * J.transpose() * task_metric.ldlt().solve(Eigen::MatrixXd::Identity(task_dim, task_dim));
}
}

template <typename T>
FSM_ROM2FullOrderState<T>::FSM_ROM2FullOrderState(RobotData& robot) :
    robot_data_(&robot)
{
    std::cout << "[ FSM_ROM2FullOrderState ] Constructed" << std::endl;
}

template <typename T>
FSM_ROM2FullOrderState<T>::~FSM_ROM2FullOrderState()
{
    if (ref_data_) {
        mj_deleteData(ref_data_);
        ref_data_ = nullptr;
    }
    if (ref_model_) {
        mj_deleteModel(ref_model_);
        ref_model_ = nullptr;
    }
}

template <typename T>
void FSM_ROM2FullOrderState<T>::onEnter()
{
    std::cout << "[ FSM_ROM2FullOrderState ] OnEnter" << std::endl;

    if (viz_) {
        viz_->clear();
    }

    const std::string config_file = CMAKE_SOURCE_DIR "/config/fsm_RoM2FullOrder_config.yaml";
    readConfig(config_file);
    loop_count_ = 0;
    policy_to_robot_ = robot_data_->policy.policy_to_robot;
    if (policy_to_robot_.empty()) {
        policy_to_robot_.assign(g1_reduced_order::policy_to_joint_idx.begin(),
                                g1_reduced_order::policy_to_joint_idx.end());
    }

    policy_.setIoNames(robot_data_->policy.input_name, robot_data_->policy.output_name);
    policy_.setExpectedDims(
        std::max(1, robot_data_->policy.history_length) * (9 + 3 * kPolicyActionDim),
        kPolicyActionDim);
    policy_.load(resolvePolicyPath(robot_data_->policy.onnx_path, config_file));

    action_dim_ = policy_.isLoaded() ? policy_.outputDim() : kPolicyActionDim;
    frame_obs_dim_ = 9 + 3 * action_dim_;
    policy_obs_dim_ = policy_.isLoaded() ? policy_.inputDim()
                                         : frame_obs_dim_ * std::max(1, robot_data_->policy.history_length);

    raw_action_ = Eigen::VectorXd::Zero(action_dim_);
    clipped_action_ = Eigen::VectorXd::Zero(action_dim_);
    last_good_raw_action_ = Eigen::VectorXd::Zero(action_dim_);
    last_action_obs_ = Eigen::VectorXd::Zero(action_dim_);
    frame_obs_ = Eigen::VectorXd::Zero(frame_obs_dim_);
    obs_ = Eigen::VectorXd::Zero(policy_obs_dim_);
    obs_history_.clear();
    last_virtual_torso_rpy_.setZero();
    last_pos_err_.setZero();
    last_ori_err_.setZero();
    last_dq_primary_ = Eigen::VectorXd::Zero(kIkDof);
    last_dq_secondary_ = Eigen::VectorXd::Zero(kIkDof);
    last_dq_total_ = Eigen::VectorXd::Zero(kIkDof);
    ik_joint_target_ = Eigen::VectorXd::Zero(kIkDof);

    robot_data_->ctrl.jpos_d = default_jpos_;
    robot_data_->ctrl.jvel_d.setZero();
    robot_data_->ctrl.torq_d.setZero();
    robot_data_->ctrl.upper_body_com_ref = robot_data_->fbk.upper_body_com;
    robot_data_->ctrl.upper_body_quat_ref = robot_data_->fbk.upper_body_quat;
    for (int i = 0; i < kIkDof; ++i) {
        const int joint = kIkJointIndices[static_cast<size_t>(i)];
        ik_joint_target_(i) = default_jpos_(joint);
    }

    initializeReferenceModel();
    updateObservation();
    updateAction();
    applyPolicyAction(clipped_action_);
    applyUpperBodyNullspaceIk(clipped_action_);
    updateAppliedActionObservation(clipped_action_);

    this->state_time = 0.0;
}

template <typename T>
Eigen::VectorXd FSM_ROM2FullOrderState<T>::makeObservation() const
{
    Eigen::Quaterniond base_q = robot_data_->fbk.quat_B;
    if (!std::isfinite(base_q.w()) || !std::isfinite(base_q.x()) ||
        !std::isfinite(base_q.y()) || !std::isfinite(base_q.z()) ||
        base_q.norm() < 1.0e-9) {
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

    for (int i = 0; i < static_cast<int>(kLegPolicySlots.size()); ++i) {
        const int slot = kLegPolicySlots[static_cast<size_t>(i)];
        const int joint = kLegJointIndices[static_cast<size_t>(i)];
        if (slot < action_dim_ && joint < robot_data_->fbk.jpos.size()) {
            q_err(slot) = robot_data_->fbk.jpos(joint) - default_jpos_(joint);
            dq_sel(slot) = robot_data_->policy.obs_joint_vel_scale * robot_data_->fbk.jvel(joint);
        }
    }

    for (int axis = 0; axis < 3; ++axis) {
        const int slot = kTorsoOrientationSlots[static_cast<size_t>(axis)];
        if (slot < action_dim_) {
            q_err(slot) = last_virtual_torso_rpy_(axis);
            dq_sel(slot) = 0.0;
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
void FSM_ROM2FullOrderState<T>::updateObservation()
{
    const Eigen::VectorXd obs_now = makeObservation();
    frame_obs_ = Eigen::VectorXd::Zero(frame_obs_dim_);
    frame_obs_.head(std::min<int>(frame_obs_dim_, obs_now.size())) =
        obs_now.head(std::min<int>(frame_obs_dim_, obs_now.size()));
    frame_obs_.tail(action_dim_) = last_action_obs_;

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
void FSM_ROM2FullOrderState<T>::updateAction()
{
    clipped_action_.setZero();
    if (!policy_.isLoaded()) {
        raw_action_ = last_good_raw_action_;
        return;
    }

    const Eigen::VectorXd policy_out = policy_.forward(obs_);
    if (policy_out.size() <= 0 || !allFinite(policy_out)) {
        std::cerr << "[ FSM_ROM2FullOrderState ] Bad policy output; reusing last finite action." << std::endl;
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
void FSM_ROM2FullOrderState<T>::applyPolicyAction(const Eigen::VectorXd& action)
{
    robot_data_->ctrl.jpos_d = default_jpos_;

    for (int i = 0; i < static_cast<int>(kLegPolicySlots.size()); ++i) {
        const int slot = kLegPolicySlots[static_cast<size_t>(i)];
        const int joint = kLegJointIndices[static_cast<size_t>(i)];
        if (slot < action.size() && joint < robot_data_->ctrl.jpos_d.size()) {
            robot_data_->ctrl.jpos_d(joint) =
                default_jpos_(joint) + robot_data_->policy.action_scale * action(slot);
        }
    }
}

template <typename T>
bool FSM_ROM2FullOrderState<T>::initializeReferenceModel()
{
    if (ref_data_) {
        mj_deleteData(ref_data_);
        ref_data_ = nullptr;
    }
    if (ref_model_) {
        mj_deleteModel(ref_model_);
        ref_model_ = nullptr;
    }

    ref_action_qpos_addr_.fill(-1);

    const std::string ref_xml =
        std::string(CMAKE_SOURCE_DIR) + "/" + std::string(g1_reduced_order::model_xml);
    char error[1024] = "";
    ref_model_ = mj_loadXML(ref_xml.c_str(), nullptr, error, sizeof(error));
    if (!ref_model_) {
        std::cerr << "[ FSM_ROM2FullOrderState ] Failed to load reduced reference model: "
                  << error << std::endl;
        return false;
    }

    ref_data_ = mj_makeData(ref_model_);
    if (!ref_data_) {
        std::cerr << "[ FSM_ROM2FullOrderState ] Failed to allocate reference mjData." << std::endl;
        mj_deleteModel(ref_model_);
        ref_model_ = nullptr;
        return false;
    }

    ref_torso_body_id_ = mj_name2id(
        ref_model_,
        mjOBJ_BODY,
        std::string(g1_reduced_order::reference_body).c_str());
    if (ref_torso_body_id_ < 0) {
        std::cerr << "[ FSM_ROM2FullOrderState ] Reference torso body not found: "
                  << g1_reduced_order::reference_body << std::endl;
        return false;
    }

    bool ok = true;
    for (int slot = 0; slot < kPolicyActionDim; ++slot) {
        const int joint_id = mj_name2id(
            ref_model_,
            mjOBJ_JOINT,
            kReferenceActionJointNames[static_cast<size_t>(slot)]);
        if (joint_id < 0) {
            std::cerr << "[ FSM_ROM2FullOrderState ] Reference joint not found: "
                      << kReferenceActionJointNames[static_cast<size_t>(slot)] << std::endl;
            ok = false;
            continue;
        }
        ref_action_qpos_addr_[static_cast<size_t>(slot)] = ref_model_->jnt_qposadr[joint_id];
    }

    return ok;
}

template <typename T>
bool FSM_ROM2FullOrderState<T>::updateReferenceFromAction(const Eigen::VectorXd& action)
{
    if (!ref_model_ || !ref_data_ || ref_torso_body_id_ < 0) {
        return false;
    }

    Eigen::Quaterniond base_q = robot_data_->fbk.quat_B;
    if (base_q.norm() < 1.0e-9) {
        base_q = Eigen::Quaterniond::Identity();
    } else {
        base_q.normalize();
    }

    ref_data_->qpos[0] = robot_data_->fbk.p_B.x();
    ref_data_->qpos[1] = robot_data_->fbk.p_B.y();
    ref_data_->qpos[2] = robot_data_->fbk.p_B.z();
    const Eigen::Quaterniond ref_base_q = reference_base_yaw_only_ ? yawOnlyQuaternion(base_q) : base_q;
    ref_data_->qpos[3] = ref_base_q.w();
    ref_data_->qpos[4] = ref_base_q.x();
    ref_data_->qpos[5] = ref_base_q.y();
    ref_data_->qpos[6] = ref_base_q.z();

    Eigen::Vector3d torso_rpy = Eigen::Vector3d::Zero();
    for (int slot = 0; slot < std::min<int>(kPolicyActionDim, action.size()); ++slot) {
        const int qpos_addr = ref_action_qpos_addr_[static_cast<size_t>(slot)];
        if (qpos_addr < 0) {
            continue;
        }

        double q_ref = 0.0;
        const auto ori_it = std::find(kTorsoOrientationSlots.begin(), kTorsoOrientationSlots.end(), slot);
        if (ori_it != kTorsoOrientationSlots.end()) {
            const int axis = static_cast<int>(std::distance(kTorsoOrientationSlots.begin(), ori_it));
            q_ref = std::clamp(
                virtual_torso_orientation_scale_ * robot_data_->policy.action_scale * action(slot),
                -kVirtualTorsoOrientationLimit(axis),
                kVirtualTorsoOrientationLimit(axis));
            torso_rpy(axis) = q_ref;
        } else {
            const int robot_joint = kReferenceActionFullJointIndex[static_cast<size_t>(slot)];
            if (robot_joint >= 0 && robot_joint < default_jpos_.size()) {
                q_ref = default_jpos_(robot_joint) + robot_data_->policy.action_scale * action(slot);
            }
        }

        ref_data_->qpos[qpos_addr] = q_ref;
    }

    for (int i = 0; i < ref_model_->nv; ++i) {
        ref_data_->qvel[i] = 0.0;
    }

    mj_forward(ref_model_, ref_data_);

    last_virtual_torso_rpy_ = torso_rpy;
    robot_data_->ctrl.upper_body_com_ref =
        Eigen::Map<const Eigen::Vector3d>(ref_data_->xipos + 3 * ref_torso_body_id_);

    robot_data_->ctrl.upper_body_quat_ref = ref_base_q;
    return true;
}

template <typename T>
void FSM_ROM2FullOrderState<T>::applyUpperBodyNullspaceIk(const Eigen::VectorXd& action)
{
    if (!upper_body_ik_enabled_ ||
        !robot_data_->fbk.upper_body_com_valid ||
        !robot_data_->fbk.upper_body_ori_valid) {
        robot_data_->ctrl.upper_body_com_ref = robot_data_->fbk.upper_body_com;
        robot_data_->ctrl.upper_body_quat_ref = robot_data_->fbk.upper_body_quat;
        return;
    }

    Eigen::Quaterniond base_q = robot_data_->fbk.quat_B;
    if (base_q.norm() < 1.0e-9) {
        base_q = Eigen::Quaterniond::Identity();
    } else {
        base_q.normalize();
    }

    if (!updateReferenceFromAction(action)) {
        robot_data_->ctrl.upper_body_com_ref = robot_data_->fbk.p_B + yawOnlyQuaternion(base_q) * kFallbackTorsoComLocal;
        robot_data_->ctrl.upper_body_quat_ref = yawOnlyQuaternion(base_q);
    }

    Eigen::MatrixXd J_com(3, kIkDof);
    Eigen::MatrixXd J_ori(3, kIkDof);
    Eigen::VectorXd q_ik = Eigen::VectorXd::Zero(kIkDof);
    Eigen::VectorXd q_default_ik = Eigen::VectorXd::Zero(kIkDof);
    for (int i = 0; i < kIkDof; ++i) {
        const int joint = kIkJointIndices[static_cast<size_t>(i)];
        J_com.col(i) = robot_data_->fbk.upper_body_com_jac.col(joint);
        J_ori.col(i) = robot_data_->fbk.upper_body_ori_jac.col(joint);
        q_ik(i) = robot_data_->fbk.jpos(joint);
        q_default_ik(i) = default_jpos_(joint);
    }

    const Eigen::Vector3d pos_err = upper_body_com_gain_ * (robot_data_->ctrl.upper_body_com_ref - robot_data_->fbk.upper_body_com);
    const Eigen::Vector3d ori_err = upper_body_orientation_gain_ * orientationErrorWorld(robot_data_->ctrl.upper_body_quat_ref, robot_data_->fbk.upper_body_quat);
    last_pos_err_ = pos_err;
    last_ori_err_ = ori_err;

    Eigen::Quaterniond q_ref = robot_data_->ctrl.upper_body_quat_ref;
    if (q_ref.norm() < 1.0e-9) {
        q_ref = Eigen::Quaterniond::Identity();
    } else {
        q_ref.normalize();
    }
    const Eigen::Vector3d roll_axis_world = q_ref * Eigen::Vector3d::UnitX();
    const Eigen::Vector3d pitch_axis_world = q_ref * Eigen::Vector3d::UnitY();
    const Eigen::Vector3d yaw_axis_world = q_ref * Eigen::Vector3d::UnitZ();

    Eigen::MatrixXd mobility = Eigen::MatrixXd::Zero(kIkDof, kIkDof);
    for (int i = 0; i < kIkDof; ++i) {
        mobility(i, i) = kIkJointMobility[static_cast<size_t>(i)];
    }

    auto solveTask = [&](const Eigen::MatrixXd& J_task,
                         const Eigen::VectorXd& err_task,
                         const Eigen::VectorXd& dq_current,
                         const Eigen::MatrixXd& N_current) {
        const Eigen::MatrixXd J_task_N = J_task * N_current;
        const Eigen::MatrixXd J_task_N_pinv =
            weightedDampedPseudoInverse(J_task_N, mobility, upper_body_ik_damping_);
        const Eigen::VectorXd residual = (err_task - J_task * dq_current).eval();
        const Eigen::MatrixXd projected_pinv = (N_current * J_task_N_pinv).eval();
        return (projected_pinv * residual).eval();
    };

    auto updateNullspace = [&](const Eigen::MatrixXd& J_task,
                               const Eigen::MatrixXd& N_current) {
        const Eigen::MatrixXd J_task_N = J_task * N_current;
        const Eigen::MatrixXd J_task_N_pinv =
            weightedDampedPseudoInverse(J_task_N, mobility, upper_body_ik_damping_);
        const Eigen::MatrixXd task_projector =
            (Eigen::MatrixXd::Identity(kIkDof, kIkDof) - (J_task_N_pinv * J_task_N).eval()).eval();
        return (N_current * task_projector).eval();
    };

    // Task priority: torso pitch -> upper-body CoM -> torso roll/yaw -> upper-body posture.
    // Project orientation onto the yaw-aligned torso axes so "pitch" is not the fixed world-Y axis.
    Eigen::MatrixXd J_pitch(1, kIkDof);
    J_pitch.row(0) = pitch_axis_world.transpose() * J_ori;
    Eigen::VectorXd pitch_err(1);
    pitch_err << pitch_axis_world.dot(ori_err);

    Eigen::MatrixXd J_roll_yaw(2, kIkDof);
    J_roll_yaw.row(0) = roll_axis_world.transpose() * J_ori;
    J_roll_yaw.row(1) = yaw_axis_world.transpose() * J_ori;
    Eigen::VectorXd roll_yaw_err(2);
    roll_yaw_err << roll_axis_world.dot(ori_err), yaw_axis_world.dot(ori_err);

    const Eigen::MatrixXd I_ik = Eigen::MatrixXd::Identity(kIkDof, kIkDof);
    const Eigen::VectorXd dq_zero = Eigen::VectorXd::Zero(kIkDof);

    const Eigen::VectorXd dq_pitch = solveTask(J_pitch, pitch_err, dq_zero, I_ik);
    const Eigen::MatrixXd N_pitch = updateNullspace(J_pitch, I_ik);

    const Eigen::VectorXd dq_com = solveTask(J_com, pos_err, dq_pitch, N_pitch);
    const Eigen::MatrixXd N_com = updateNullspace(J_com, N_pitch);

    const Eigen::VectorXd dq_pitch_com = dq_pitch + dq_com;
    const Eigen::VectorXd dq_roll_yaw = solveTask(J_roll_yaw, roll_yaw_err, dq_pitch_com, N_com);
    const Eigen::MatrixXd N_roll_yaw = updateNullspace(J_roll_yaw, N_com);

    Eigen::VectorXd dq_posture = Eigen::VectorXd::Zero(kIkDof);
    if (upper_body_posture_gain_ > 0.0) {
        dq_posture = N_roll_yaw * (upper_body_posture_gain_ * (q_default_ik - q_ik));
    }

    const Eigen::VectorXd dq_primary = dq_pitch + dq_com;
    const Eigen::VectorXd dq_secondary = dq_roll_yaw + dq_posture;

    const Eigen::VectorXd dq_primary_applied = limitNorm(dq_primary, 0.85 * upper_body_max_delta_norm_);
    const Eigen::VectorXd dq_secondary_applied = limitNorm(dq_secondary, 0.50 * upper_body_max_delta_norm_);
    Eigen::VectorXd dq_limited = limitNorm(dq_primary_applied + dq_secondary_applied, upper_body_max_delta_norm_);
    for (int i = 0; i < kIkDof; ++i) {
        dq_limited(i) = std::clamp(
            dq_limited(i),
            -kIkJointMaxStep[static_cast<size_t>(i)],
            kIkJointMaxStep[static_cast<size_t>(i)]);
    }

    const Eigen::Vector3d gravity_b = base_q.inverse() * Eigen::Vector3d(0.0, 0.0, -1.0);
    const double uprightness = std::clamp(-gravity_b.z(), -1.0, 1.0);
    const double base_tilt = std::acos(uprightness);
    const double height_scale = smoothUnit(robot_data_->fbk.p_B.z(), upper_body_min_base_height_, upper_body_full_base_height_);
    const double tilt_scale = 1.0 - smoothUnit(base_tilt, upper_body_safe_tilt_, upper_body_max_tilt_);
    const double action_scale = 1.0 - smoothUnit(raw_action_.norm(), upper_body_raw_action_soft_limit_, upper_body_raw_action_hard_limit_);
    const double stability_scale = std::clamp(height_scale * tilt_scale * action_scale, 0.0, 1.0);
    const Eigen::VectorXd dq_return = limitNorm(q_default_ik - q_ik, upper_body_max_delta_norm_);
    // const Eigen::VectorXd dq = stability_scale * dq_limited + (1.0 - stability_scale) * dq_return;
    // const Eigen::VectorXd dq = dq_limited;
    const Eigen::VectorXd dq = dq_primary + dq_secondary;

    Eigen::VectorXd q_target_ik = ik_joint_target_;
    if (q_target_ik.size() != kIkDof || !allFinite(q_target_ik)) {
        q_target_ik = q_ik;
    }
    q_target_ik += dq;

    last_dq_primary_ = dq_primary;
    last_dq_secondary_ = dq_secondary;
    last_dq_total_ = dq;
    ik_joint_target_ = q_target_ik;
    for (int i = 0; i < kIkDof; ++i) {
        ik_joint_target_(i) = std::clamp(
            q_target_ik(i),
            q_ik(i) - kIkJointMaxTargetError[static_cast<size_t>(i)],
            q_ik(i) + kIkJointMaxTargetError[static_cast<size_t>(i)]);
        ik_joint_target_(i) = std::clamp(
            ik_joint_target_(i),
            kIkJointLower[static_cast<size_t>(i)],
            kIkJointUpper[static_cast<size_t>(i)]);
        const int joint = kIkJointIndices[static_cast<size_t>(i)];
        robot_data_->ctrl.jpos_d(joint) = ik_joint_target_(i);
    }
}

template <typename T>
void FSM_ROM2FullOrderState<T>::updateAppliedActionObservation(const Eigen::VectorXd& action)
{
    last_action_obs_ = Eigen::VectorXd::Zero(action_dim_);
    const int use_dim = std::min<int>(action_dim_, action.size());
    if (use_dim > 0) {
        last_action_obs_.head(use_dim) = action.head(use_dim);
    }

    const double denom = virtual_torso_orientation_scale_ * robot_data_->policy.action_scale;
    for (int axis = 0; axis < 3; ++axis) {
        const int slot = kTorsoOrientationSlots[static_cast<size_t>(axis)];
        if (slot < use_dim && std::abs(denom) > 1.0e-12) {
            const double applied_delta = std::clamp(
                denom * action(slot),
                -kVirtualTorsoOrientationLimit(axis),
                kVirtualTorsoOrientationLimit(axis));
            last_action_obs_(slot) = applied_delta / denom;
        }
    }
}

template <typename T>
void FSM_ROM2FullOrderState<T>::updateCommand()
{
    if (loop_count_ % std::max(1, robot_data_->policy.policy_decimation) == 0) {
        updateObservation();
        updateAction();
    }

    applyPolicyAction(clipped_action_);
    applyUpperBodyNullspaceIk(clipped_action_);
    updateAppliedActionObservation(clipped_action_);
    maybePrintDebug();

    robot_data_->ctrl.torq_d =
        robot_data_->param.Kp.asDiagonal() * (robot_data_->ctrl.jpos_d - robot_data_->fbk.jpos)
        - robot_data_->param.Kd.asDiagonal() * robot_data_->fbk.jvel;
    ++loop_count_;
}

template <typename T>
void FSM_ROM2FullOrderState<T>::runNominal()
{
    updateModel();
    updateCommand();
    updateVisualization();

    this->state_time += 0.001;
}

template <typename T>
void FSM_ROM2FullOrderState<T>::setVisualizer(mujoco::TrajVizUtil* visualizer)
{
    viz_ = visualizer;
}

template <typename T>
void FSM_ROM2FullOrderState<T>::updateModel()
{
}

template <typename T>
void FSM_ROM2FullOrderState<T>::updateVisualization()
{
    if (!viz_) return;

    viz_->sphere("ROM2Full/base",
        robot_data_->fbk.p_B,
        0.035, {0.3f, 0.0f, 0.3f, 0.8f}
    );
    viz_->sphere("ROM2Full/upper_com",
        robot_data_->fbk.upper_body_com,
        0.025, {0.0f, 0.2f, 1.0f, 0.8f}
    );
    viz_->sphere("ROM2Full/upper_com_ref",
        robot_data_->ctrl.upper_body_com_ref,
        0.025, {1.0f, 0.1f, 0.0f, 0.8f}
    );
    viz_->line("ROM2Full/upper_com_error",
        robot_data_->fbk.upper_body_com,
        robot_data_->ctrl.upper_body_com_ref,
        0.006, {1.0f, 0.4f, 0.0f, 0.7f}
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
        viz_->remove("ROM2Full/lin_vel_cmd");
    } else {
        viz_->arrowVector(
            "ROM2Full/lin_vel_cmd",
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
        viz_->remove("ROM2Full/lin_vel_actual");
    } else {
        viz_->arrowVector(
            "ROM2Full/lin_vel_actual",
            arrow_start,
            kArrowScale * actual_world,
            0.010,
            {0.0f, 0.0f, 1.0f, 1.0f}
        );
    }
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
        if (config["policy"]["debug_print_every"]) {
            debug_print_every_ = config["policy"]["debug_print_every"].as<int>();
        }

        YAML::Node kp_node = config["Kp"];
        YAML::Node kd_node = config["Kd"];
        YAML::Node tau_node = config["tau_limit"];
        if (config["Robot"]) {
            if (config["Robot"]["Kp"]) kp_node = config["Robot"]["Kp"];
            if (config["Robot"]["Kd"]) kd_node = config["Robot"]["Kd"];
            if (config["Robot"]["tau_limit"]) tau_node = config["Robot"]["tau_limit"];
        }

        robot_data_->param.Kp.setZero();
        robot_data_->param.Kd.setZero();
        robot_data_->param.tau_limit.setConstant(1.0e9);
        default_jpos_.setZero();
        for (int i = 0; i < static_cast<int>(num_act_joint); i++) {
            if (kp_node && i < static_cast<int>(kp_node.size())) {
                robot_data_->param.Kp[i] = kp_node[i].as<T>();
            }
            if (kd_node && i < static_cast<int>(kd_node.size())) {
                robot_data_->param.Kd[i] = kd_node[i].as<T>();
            }
            if (tau_node && i < static_cast<int>(tau_node.size())) {
                robot_data_->param.tau_limit[i] = tau_node[i].as<T>();
            }
            if (config["default_jpos"] && i < static_cast<int>(config["default_jpos"].size())) {
                default_jpos_[i] = config["default_jpos"][i].as<T>();
            }
        }

        if (config["upper_body_ik"]) {
            const YAML::Node ik = config["upper_body_ik"];
            if (ik["enabled"]) upper_body_ik_enabled_ = ik["enabled"].as<bool>();
            if (ik["com_gain"]) upper_body_com_gain_ = ik["com_gain"].as<double>();
            if (ik["orientation_gain"]) upper_body_orientation_gain_ = ik["orientation_gain"].as<double>();
            if (ik["posture_gain"]) upper_body_posture_gain_ = ik["posture_gain"].as<double>();
            if (ik["damping"]) upper_body_ik_damping_ = ik["damping"].as<double>();
            if (ik["max_delta_norm"]) upper_body_max_delta_norm_ = ik["max_delta_norm"].as<double>();
            if (ik["min_base_height"]) upper_body_min_base_height_ = ik["min_base_height"].as<double>();
            if (ik["full_base_height"]) upper_body_full_base_height_ = ik["full_base_height"].as<double>();
            if (ik["safe_tilt"]) upper_body_safe_tilt_ = ik["safe_tilt"].as<double>();
            if (ik["max_tilt"]) upper_body_max_tilt_ = ik["max_tilt"].as<double>();
            if (ik["raw_action_soft_limit"]) {
                upper_body_raw_action_soft_limit_ = ik["raw_action_soft_limit"].as<double>();
            }
            if (ik["raw_action_hard_limit"]) {
                upper_body_raw_action_hard_limit_ = ik["raw_action_hard_limit"].as<double>();
            }
        }
        if (config["virtual_torso"]) {
            const YAML::Node torso = config["virtual_torso"];
            if (torso["orientation_scale"]) {
                virtual_torso_orientation_scale_ = torso["orientation_scale"].as<double>();
            }
            if (torso["reference_base_yaw_only"]) {
                reference_base_yaw_only_ = torso["reference_base_yaw_only"].as<bool>();
            }
        }

        if (config["command"]) {
            robot_data_->ctrl.lin_vel_d.x() = config["command"]["vx"].as<double>();
            robot_data_->ctrl.lin_vel_d.y() = config["command"]["vy"].as<double>();
            robot_data_->ctrl.ang_vel_d.z() = config["command"]["wz"].as<double>();
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading from config file: " << e.what() << std::endl;
    }
}

template <typename T>
void FSM_ROM2FullOrderState<T>::maybePrintDebug() const
{
    if (debug_print_every_ <= 0 || (loop_count_ % debug_print_every_) != 0) {
        return;
    }

    std::cout << "[ FSM_ROM2FullOrderState ] loop=" << loop_count_
              << " torso_rpy_ref=" << last_virtual_torso_rpy_.transpose()
              << " com_meas=" << robot_data_->fbk.upper_body_com.transpose()
              << " com_ref=" << robot_data_->ctrl.upper_body_com_ref.transpose()
              << " |e_com|=" << last_pos_err_.norm()
              << " rpy_meas=" << bodyRpyFromQuat(robot_data_->fbk.upper_body_quat).transpose()
              << " rpy_ref=" << bodyRpyFromQuat(robot_data_->ctrl.upper_body_quat_ref).transpose()
              << " |e_ori|=" << last_ori_err_.norm()
              << " |dq|=" << last_dq_total_.norm()
              << std::endl;
}

template <typename T>
std::string FSM_ROM2FullOrderState<T>::resolvePolicyPath(const std::string& raw_path, const std::string& config_file) const
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

template class FSM_ROM2FullOrderState<double>;
