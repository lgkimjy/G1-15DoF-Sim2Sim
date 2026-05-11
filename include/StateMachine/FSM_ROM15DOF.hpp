#ifndef __FSM_ROM15DOF_HPP__
#define __FSM_ROM15DOF_HPP__

#include <string>

#include "RobotDefinition.hpp"
#include "States.hpp"
#include "RobotStates.hpp"

#include "Interface/MuJoCo/traj_viz_util.hpp"

using namespace robot_name;

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

    void updateModel();
    void updateCommand();
    void updateVisualization();
    void readConfig(std::string config_file);
};

#endif // __FSM_ROM15DOF_HPP__
