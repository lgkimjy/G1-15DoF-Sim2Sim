#include "StateMachine/StateMachineCtrl.hpp"

#include "StateMachine/FSM_JPosCtrl.hpp"
#include "StateMachine/FSM_ROM15DOF.hpp"
#include "StateMachine/FSM_ROM2FullOrder.hpp"

#include <iostream>

StateMachineCtrl::StateMachineCtrl(RobotData& robot)
{
    state_list_.resize(StateList::NUM_STATE, nullptr);
    state_list_[StateList::FSM_JPosCtrl] = new FSM_JPosCtrlState<double>(robot);
    state_list_[StateList::FSM_ROM15DOF] = new FSM_ROM15DOFState<double>(robot);
    state_list_[StateList::FSM_ROM2FULL] = new FSM_ROM2FullOrderState<double>(robot);

    current_state_ = nullptr;
    next_state_ = nullptr;
    std::cout << "[ StateMachineCtrl ] constructed" << std::endl;
}

StateMachineCtrl::~StateMachineCtrl()
{
    for (States* s : state_list_) {
        delete s;
    }
}

void StateMachineCtrl::initialize()
{
    current_state_ = state_list_[StateList::FSM_JPosCtrl];
    next_state_ = current_state_;
    first_run_ = true;
}

void StateMachineCtrl::runState()
{
    static int cnt = 0;
    if (first_run_) {
        current_state_->onEnter();
        cnt = 0;
        first_run_ = false;
    }

    if (cnt == 1001) {
        current_state_ = state_list_[StateList::FSM_ROM15DOF];
        current_state_->onEnter();
    }
    current_state_->runNominal();
    cnt ++;
}

void StateMachineCtrl::setVisualizer(mujoco::TrajVizUtil* visualizer)
{
    for (States* state : state_list_) {
        if (state) {
            state->setVisualizer(visualizer);
        }
    }
}
