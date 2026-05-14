#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>

#include "RobotDefinition.hpp"

// --- Feedback (sensors / sim truth): log under e.g. "/fbk/..." for HDF5 ---
struct RobotFeedback {
    Eigen::VectorXd     qpos; // generalized coordinates
    Eigen::VectorXd     qvel; // generalized velocities
    
    // Position of base
    Eigen::Vector3d     p_B = Eigen::Vector3d::Zero();
    Eigen::Vector3d     pdot_B = Eigen::Vector3d::Zero(); // linear velocity w.r.t world frame

    // Orientation of base
    Eigen::Quaterniond  quat_B = Eigen::Quaterniond::Identity();
    Eigen::Matrix3d     R_B = Eigen::Matrix3d::Identity();
    Eigen::Vector3d     varphi_B = Eigen::Vector3d::Zero();
    Eigen::Vector3d     omega_B = Eigen::Vector3d::Zero();

    // Position of CoM
    Eigen::Vector3d     p_CoM = Eigen::Vector3d::Zero();
    Eigen::Vector3d     pdot_CoM = Eigen::Vector3d::Zero();

    // Joint States
    Eigen::VectorXd     jpos;
    Eigen::VectorXd     jvel;

    // Full-body upper-body quantities used by ROM-to-full IK.
    Eigen::Vector3d     upper_body_com = Eigen::Vector3d::Zero();
    Eigen::Vector3d     upper_body_com_vel = Eigen::Vector3d::Zero();
    Eigen::MatrixXd     upper_body_com_jac;
    Eigen::Quaterniond  upper_body_quat = Eigen::Quaterniond::Identity();
    Eigen::MatrixXd     upper_body_ori_jac;
    bool                upper_body_com_valid = false;
    bool                upper_body_ori_valid = false;

    explicit RobotFeedback(int nv = g1::nDoF)
        : qpos(Eigen::VectorXd::Zero(nv + 1)),
          qvel(Eigen::VectorXd::Zero(nv)),
          jpos(Eigen::VectorXd::Zero(g1::num_act_joint)),
          jvel(Eigen::VectorXd::Zero(g1::num_act_joint)),
          upper_body_com_jac(Eigen::MatrixXd::Zero(3, g1::num_act_joint)),
          upper_body_ori_jac(Eigen::MatrixXd::Zero(3, g1::num_act_joint)) {}
};

// --- Commands & actuator outputs: motion targets + torque (not "motor struct" only) ---
struct RobotCtrl {
    Eigen::Vector3d lin_vel_d = Eigen::Vector3d::Zero(); // desired linear velocity w.r.t. base frame
    Eigen::Vector3d ang_vel_d = Eigen::Vector3d::Zero(); // desired angular velocity w.r.t. base frame
    Eigen::Vector3d upper_body_com_ref = Eigen::Vector3d::Zero();
    Eigen::Quaterniond upper_body_quat_ref = Eigen::Quaterniond::Identity();

    Eigen::VectorXd jpos_d;
    Eigen::VectorXd jvel_d;
    Eigen::VectorXd torq_d;

    explicit RobotCtrl(int = g1::nDoF)
        : jpos_d(Eigen::VectorXd::Zero(g1::num_act_joint)),
          jvel_d(Eigen::VectorXd::Zero(g1::num_act_joint)),
          torq_d(Eigen::VectorXd::Zero(g1::num_act_joint)) {}
};

// --- Parameters from config (PD, nominal posture); joint order = RobotDefinition::kJointNames / joint_names ---
struct RobotParam {
    Eigen::VectorXd Kp;
    Eigen::VectorXd Kd;
    Eigen::VectorXd tau_limit;

    explicit RobotParam(int = g1::nDoF)
        : Kp(Eigen::VectorXd::Zero(g1::num_act_joint)),
          Kd(Eigen::VectorXd::Zero(g1::num_act_joint)),
          tau_limit(Eigen::VectorXd::Constant(g1::num_act_joint, 1.0e9)) {}
};

struct PolicyConfig {
    std::string onnx_path;
    std::string input_name = "obs";
    std::string output_name = "actions";
    double action_scale = 0.25;
    double action_clip = 1.0;
    double obs_base_ang_vel_scale = 0.2;
    double obs_joint_vel_scale = 0.05;
    int policy_decimation = 1;
    int history_length = 1;
    std::vector<int> policy_to_robot;
};

struct RobotData {
    RobotFeedback fbk;
    RobotCtrl ctrl;
    RobotParam param;
    PolicyConfig policy;
    
    // n_q : number of generalized coordinates)
    // n_v : number of generalized velocities
    // n_j : number of joints
    explicit RobotData(int nv = g1::nDoF) : fbk(nv), ctrl(nv), param(nv) {}
};
