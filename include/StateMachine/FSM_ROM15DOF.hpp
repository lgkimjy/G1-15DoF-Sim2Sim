#ifndef __FSM_ROM15DOF_HPP__
#define __FSM_ROM15DOF_HPP__

#include <deque>
#include <string>

#include "RobotDefinition.hpp"
#include "States.hpp"
#include "RobotStates.hpp"

#include "Interface/MuJoCo/traj_viz_util.hpp"

#include "Controller/OnnxParser/OnnxPolicy.hpp"

using namespace g1;

template <typename T>
class FSM_ROM15DOFState : public States {
public:
    explicit FSM_ROM15DOFState(RobotData& robot);
    ~FSM_ROM15DOFState() {};

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
    Eigen::VectorXd frame_obs_;
    Eigen::VectorXd obs_;
    std::deque<Eigen::VectorXd> obs_history_;

    Eigen::Matrix<T, num_act_joint, 1>      default_jpos_;

    Eigen::VectorXd makeObservation() const;
    void updateObservation();
    void updateAction();
    void applyPolicyAction(const Eigen::VectorXd& action);
    std::string resolvePolicyPath(const std::string& raw_path, const std::string& config_file) const;

    void updateModel();
    void updateCommand();
    void updateVisualization();
    void readConfig(std::string config_file);
};

#endif // __FSM_ROM15DOF_HPP__
