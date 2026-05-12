#pragma once

#include <array>
#include <string_view>

namespace g1
{
    constexpr size_t nDoF_base = 6;
    constexpr size_t nDoFQuat_base = 7;

    // Fill these values after cloning this template.
    constexpr size_t num_act_joint = 15;

    constexpr size_t nDoF = num_act_joint + nDoF_base;
    constexpr size_t nDoFQuat = num_act_joint + nDoFQuat_base;

    constexpr std::array<int, num_act_joint> robot_joint_idx = {
        0, 6, 12,
        1, 7, 13,
        2, 8, 14,
        3, 9,
        4, 10,
        5, 11
    };

    inline constexpr std::string_view scene = "model/G1/scene.xml";
    inline constexpr std::string_view model_xml = "model/G1/g1_15dof_fake_torso.xml";
    inline constexpr std::string_view inference_model_xml = "model/G1/g1_15dof_fake_torso.xml";
    inline constexpr std::string_view model_urdf = "model/G1/g1_15dof_fake_torso.urdf";

    inline constexpr std::array<std::string_view, num_act_joint> joint_names = {
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "torso_roll_joint",
        "torso_pitch_joint",
        "torso_yaw_joint"
    };
}

namespace g1_reduced_order
{
    constexpr size_t nDoF_base = 6;
    constexpr size_t nDoFQuat_base = 7;

    // Fill these values after cloning this template.
    constexpr size_t num_act_joint = 15;

    constexpr size_t nDoF = num_act_joint + nDoF_base;
    constexpr size_t nDoFQuat = num_act_joint + nDoFQuat_base;

    constexpr std::array<int, num_act_joint> robot_joint_idx = {
        0, 6, 12,
        1, 7, 13,
        2, 8, 14,
        3, 9,
        4, 10,
        5, 11
    };

    inline constexpr std::string_view scene = "model/G1/scene.xml";
    inline constexpr std::string_view model_xml = "model/G1/g1_15dof_fake_torso.xml";
    inline constexpr std::string_view inference_model_xml = "model/G1/g1_15dof_fake_torso.xml";
    inline constexpr std::string_view model_urdf = "model/G1/g1_15dof_fake_torso.urdf";

    inline constexpr std::array<std::string_view, num_act_joint> joint_names = {
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "torso_roll_joint",
        "torso_pitch_joint",
        "torso_yaw_joint"
    };
}

namespace g1_full_order
{
    constexpr size_t nDoF_base = 6;
    constexpr size_t nDoFQuat_base = 7;

    // Fill these values after cloning this template.
    constexpr size_t num_act_joint = 15;

    constexpr size_t nDoF = num_act_joint + nDoF_base;
    constexpr size_t nDoFQuat = num_act_joint + nDoFQuat_base;

    constexpr std::array<int, num_act_joint> robot_joint_idx = {
        0, 6, 12,
        1, 7, 13,
        2, 8, 14,
        3, 9,
        4, 10,
        5, 11
    };

    inline constexpr std::string_view scene = "model/G1/scene.xml";
    inline constexpr std::string_view model_xml = "model/G1/g1_15dof_fake_torso.xml";
    inline constexpr std::string_view inference_model_xml = "model/G1/g1_15dof_fake_torso.xml";
    inline constexpr std::string_view model_urdf = "model/G1/g1_15dof_fake_torso.urdf";

    inline constexpr std::array<std::string_view, num_act_joint> joint_names = {
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "torso_roll_joint",
        "torso_pitch_joint",
        "torso_yaw_joint"
    };
}
