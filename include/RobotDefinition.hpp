#pragma once

#include <array>
#include <string_view>

namespace g1_reduced_order
{
    constexpr size_t nDoF_base = 6;
    constexpr size_t nDoFQuat_base = 7;
    constexpr size_t num_act_joint = 15;

    constexpr size_t nDoF = num_act_joint + nDoF_base;
    constexpr size_t nDoFQuat = num_act_joint + nDoFQuat_base;

    constexpr std::array<int, num_act_joint> policy_to_joint_idx = {
        0, 6, 12,
        1, 7, 13,
        2, 8, 14,
        3, 9,
        4, 10,
        5, 11
    };

    inline constexpr std::string_view model_xml = "model/G1/g1_15dof_fake_torso.xml";
    inline constexpr std::string_view model_urdf = "model/G1/g1_15dof_fake_torso.urdf";
    inline constexpr std::string_view reference_body = "torso_cylinder_link";

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
    constexpr size_t num_act_joint = 29;

    constexpr size_t nDoF = num_act_joint + nDoF_base;
    constexpr size_t nDoFQuat = num_act_joint + nDoFQuat_base;

    constexpr std::array<int, num_act_joint> robot_joint_idx = {
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10, 11,
        12, 13, 14,
        15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28
    };

    inline constexpr std::string_view scene = "model/G1/scene.xml";
    inline constexpr std::string_view model_xml = "model/G1/g1_29dof_rev_1_0.xml";
    inline constexpr std::string_view inference_model_xml = g1_reduced_order::model_xml;
    inline constexpr std::string_view model_urdf = "model/G1/g1_29dof_rev_1_0.urdf";

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
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint"
    };
}

namespace g1 = g1_full_order;
