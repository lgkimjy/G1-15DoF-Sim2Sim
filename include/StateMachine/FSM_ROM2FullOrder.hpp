#ifndef __FSM_FSM_ROM2FullOrderState_HPP__
#define __FSM_FSM_ROM2FullOrderState_HPP__

#include <array>
#include <deque>
#include <string>

#include <mujoco/mujoco.h>

#include "RobotDefinition.hpp"
#include "States.hpp"
#include "RobotStates.hpp"

#include "Interface/MuJoCo/traj_viz_util.hpp"

#include "Controller/OnnxParser/OnnxPolicy.hpp"

using namespace g1;

template <typename T>
class FSM_ROM2FullOrderState : public States {
public:
    explicit FSM_ROM2FullOrderState(RobotData& robot);
    ~FSM_ROM2FullOrderState() override;

    void onEnter() override;
    void runNominal() override;
    void checkTransition() override {};
    void runTransition() override {};
    void setVisualizer(mujoco::TrajVizUtil* visualizer) override;

private:
    RobotData* robot_data_;
    mujoco::TrajVizUtil* viz_ = nullptr;
    
    OnnxPolicy policy_;

    int loop_count_ = 0;
    int action_dim_ = 0;
    int frame_obs_dim_ = 0;
    int policy_obs_dim_ = 0;
    std::vector<int> policy_to_robot_;

    Eigen::VectorXd raw_action_;
    Eigen::VectorXd clipped_action_;
    Eigen::VectorXd last_good_raw_action_;
    Eigen::VectorXd last_action_obs_;
    Eigen::VectorXd frame_obs_;
    Eigen::VectorXd obs_;
    std::deque<Eigen::VectorXd> obs_history_;

    Eigen::Matrix<T, num_act_joint, 1>      default_jpos_;
    Eigen::Vector3d last_virtual_torso_rpy_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d last_pos_err_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d last_ori_err_ = Eigen::Vector3d::Zero();
    Eigen::VectorXd last_dq_primary_;
    Eigen::VectorXd last_dq_secondary_;
    Eigen::VectorXd last_dq_total_;
    Eigen::VectorXd ik_joint_target_;

    bool upper_body_ik_enabled_ = true;
    double upper_body_com_gain_ = 500.0;
    double upper_body_orientation_gain_ = 10.0;
    double upper_body_posture_gain_ = 0.5;
    double upper_body_ik_damping_ = 1.0e-4;
    double upper_body_max_delta_norm_ = 10;
    double upper_body_min_base_height_ = 0.70;
    double upper_body_full_base_height_ = 0.78;
    double upper_body_safe_tilt_ = 0.25;
    double upper_body_max_tilt_ = 0.65;
    double upper_body_raw_action_soft_limit_ = 20.0;
    double upper_body_raw_action_hard_limit_ = 80.0;
    double virtual_torso_orientation_scale_ = 1.0;
    bool reference_base_yaw_only_ = true;
    int debug_print_every_ = 0;

    mjModel* ref_model_ = nullptr;
    mjData* ref_data_ = nullptr;
    int ref_torso_body_id_ = -1;
    std::array<int, g1_reduced_order::num_act_joint> ref_action_qpos_addr_{};

    Eigen::VectorXd makeObservation() const;
    void updateObservation();
    void updateAction();
    void applyPolicyAction(const Eigen::VectorXd& action);
    void applyUpperBodyNullspaceIk(const Eigen::VectorXd& action);
    void updateAppliedActionObservation(const Eigen::VectorXd& action);
    std::string resolvePolicyPath(const std::string& raw_path, const std::string& config_file) const;
    bool initializeReferenceModel();
    bool updateReferenceFromAction(const Eigen::VectorXd& action);
    void maybePrintDebug() const;

    void updateModel();
    void updateCommand();
    void updateVisualization();
    void readConfig(std::string config_file);
};

#endif // __FSM_ROM15DOF_HPP__
