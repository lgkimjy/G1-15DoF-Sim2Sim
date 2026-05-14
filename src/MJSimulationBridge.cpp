#include "MJSimulationBridge.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

SimulationBridge::SimulationBridge(const std::string& scene_file)
    : SimulationInterface(scene_file),
      robot_(nDoF),
      state_machine_(robot_)
{
    const auto now = std::chrono::system_clock::now();
    const auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");

    log_dir = CMAKE_SOURCE_DIR "/logs/" + ss.str() + "/";
    std::filesystem::create_directories(log_dir);
    logger = std::make_unique<HDF5Logger>(log_dir + log_file_name);

    std::cout << "[SimulationBridge] Constructed" << std::endl;
}

void SimulationBridge::Initialize()
{
    state_machine_.setVisualizer(&mjSim_->traj_viz_util_);
    upper_body_com_root_id_ = mj_name2id(mjModel_, mjOBJ_BODY, "waist_yaw_link");
    if (upper_body_com_root_id_ < 0) {
        std::cerr << "[SimulationBridge] waist_yaw_link body not found; upper-body CoM IK disabled." << std::endl;
    }
    upper_body_orientation_body_id_ = mj_name2id(mjModel_, mjOBJ_BODY, "torso_link");
    if (upper_body_orientation_body_id_ < 0) {
        std::cerr << "[SimulationBridge] torso_link body not found; torso orientation IK disabled." << std::endl;
    }
    state_machine_.initialize();
    std::cout << "[SimulationBridge] Initialized" << std::endl;
}

void SimulationBridge::UpdateSystemObserver()
{
    if (!mjData_) return;
    
	/////	Position vector of floating-base body w.r.t {I}
    robot_.fbk.p_B(0) = mjData_->qpos[0];
    robot_.fbk.p_B(1) = mjData_->qpos[1];
    robot_.fbk.p_B(2) = mjData_->qpos[2];

    /////   Orientation of floating-base body expressed in {I} (quaternion)
    robot_.fbk.quat_B.w() = mjData_->qpos[3];
    robot_.fbk.quat_B.x() = mjData_->qpos[4];
    robot_.fbk.quat_B.y() = mjData_->qpos[5];
    robot_.fbk.quat_B.z() = mjData_->qpos[6];
    robot_.fbk.R_B = robot_.fbk.quat_B.normalized().toRotationMatrix();

    //////	Linear velocity of floating-base body expressed in {I}
    robot_.fbk.pdot_B(0) = mjData_->qvel[0];
    robot_.fbk.pdot_B(1) = mjData_->qvel[1];
    robot_.fbk.pdot_B(2) = mjData_->qvel[2];

    //////	Angular velocity of floating-base body expressed in {B}
    robot_.fbk.varphi_B(0) = mjData_->qvel[3];
    robot_.fbk.varphi_B(1) = mjData_->qvel[4];
    robot_.fbk.varphi_B(2) = mjData_->qvel[5];
    //////	Angular velocity of floating-base body expressed in {I}
    robot_.fbk.omega_B = robot_.fbk.R_B * robot_.fbk.varphi_B;

    //////  Joint positions and velocities
    for (int i = 0; i < num_act_joint; ++i) {
        robot_.fbk.jpos(i) = mjData_->qpos[i + 7];
        robot_.fbk.jvel(i) = mjData_->qvel[i + 6];
    }

    //////  Generalized coordinates
    robot_.fbk.qpos.segment<3>(0) = robot_.fbk.p_B;
    robot_.fbk.qpos(3) = robot_.fbk.quat_B.w();
    robot_.fbk.qpos(4) = robot_.fbk.quat_B.x();
    robot_.fbk.qpos(5) = robot_.fbk.quat_B.y();
    robot_.fbk.qpos(6) = robot_.fbk.quat_B.z();
    robot_.fbk.qpos.tail(num_act_joint) = robot_.fbk.jpos;

    //////  Generalized velocities
    robot_.fbk.qvel.segment<3>(0) = robot_.fbk.pdot_B;
    robot_.fbk.qvel.segment<3>(3) = robot_.fbk.omega_B; // or varphi_B
    robot_.fbk.qvel.tail(num_act_joint) = robot_.fbk.jvel;

    robot_.fbk.upper_body_com_valid = false;
    robot_.fbk.upper_body_ori_valid = false;
    robot_.fbk.upper_body_com_jac.setZero();
    robot_.fbk.upper_body_ori_jac.setZero();
    robot_.fbk.upper_body_com_vel.setZero();

    if (upper_body_com_root_id_ >= 0) {
        robot_.fbk.upper_body_com =
            Eigen::Map<const Eigen::Vector3d>(mjData_->subtree_com + 3 * upper_body_com_root_id_);

        std::vector<mjtNum> jacp(3 * mjModel_->nv, 0.0);
        mj_jacSubtreeCom(mjModel_, mjData_, jacp.data(), upper_body_com_root_id_);
        for (int i = 0; i < num_act_joint; ++i) {
            const int v = i + 6;
            for (int axis = 0; axis < 3; ++axis) {
                robot_.fbk.upper_body_com_jac(axis, i) =
                    static_cast<double>(jacp[axis * mjModel_->nv + v]);
            }
        }
        robot_.fbk.upper_body_com_vel = robot_.fbk.upper_body_com_jac * robot_.fbk.jvel;
        robot_.fbk.upper_body_com_valid = true;
    }

    if (upper_body_orientation_body_id_ >= 0) {
        const mjtNum* xquat = mjData_->xquat + 4 * upper_body_orientation_body_id_;
        robot_.fbk.upper_body_quat.w() = xquat[0];
        robot_.fbk.upper_body_quat.x() = xquat[1];
        robot_.fbk.upper_body_quat.y() = xquat[2];
        robot_.fbk.upper_body_quat.z() = xquat[3];
        if (robot_.fbk.upper_body_quat.norm() > 1e-9) {
            robot_.fbk.upper_body_quat.normalize();
        }

        std::vector<mjtNum> jacr(3 * mjModel_->nv, 0.0);
        mj_jacBody(mjModel_, mjData_, nullptr, jacr.data(), upper_body_orientation_body_id_);
        for (int i = 0; i < num_act_joint; ++i) {
            const int v = i + 6;
            for (int axis = 0; axis < 3; ++axis) {
                robot_.fbk.upper_body_ori_jac(axis, i) =
                    static_cast<double>(jacr[axis * mjModel_->nv + v]);
            }
        }
        robot_.fbk.upper_body_ori_valid = true;
    }
}

void SimulationBridge::UpdateUserInput()
{
    robot_.ctrl.lin_vel_d.x() = mjSim_->lin_vel_d.x();
    robot_.ctrl.lin_vel_d.y() = mjSim_->lin_vel_d.y();
    robot_.ctrl.lin_vel_d.z() = 0.0;
    robot_.ctrl.ang_vel_d.x() = 0.0;
    robot_.ctrl.ang_vel_d.y() = 0.0;
    robot_.ctrl.ang_vel_d.z() = mjSim_->ang_vel_d.z();
}

void SimulationBridge::UpdateControlCommand()
{
    state_machine_.runState();

    for (int i = 0; i < std::min<int>(num_act_joint, mjModel_->nu); ++i) {
        const double tau_limit = std::max(0.0, robot_.param.tau_limit(i));
        mjData_->ctrl[i] = std::clamp(robot_.ctrl.torq_d(i), -tau_limit, tau_limit);
    }
}

void SimulationBridge::UpdateSystemVisualInfo()
{
    LogStates();
}

void SimulationBridge::LogStates()
{
    if (logger && mjData_) {
        logger->log(mjData_->time, robot_);
    }
}
