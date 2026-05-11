#pragma once

#include <array>
#include <string_view>

namespace robot_name
{
    constexpr size_t nDoF_base = 6;
    constexpr size_t nDoFQuat_base = 7;

    // Fill these values after cloning this template.
    constexpr size_t num_act_joint = 15;

    constexpr size_t nDoF = num_act_joint + nDoF_base;
    constexpr size_t nDoFQuat = num_act_joint + nDoFQuat_base;

    constexpr std::array<int, num_act_joint> robot_joint_idx = {};

    inline constexpr std::string_view scene = "model/G1/scene.xml";
    inline constexpr std::string_view model_xml = "model/G1/g1_15dof_fake_torso.xml";
    inline constexpr std::string_view inference_model_xml = "model/G1/g1_15dof_fake_torso.xml";
    inline constexpr std::string_view model_urdf = "model/G1/g1_15dof_fake_torso.urdf";

    inline constexpr std::array<std::string_view, num_act_joint> joint_names = {};
}
