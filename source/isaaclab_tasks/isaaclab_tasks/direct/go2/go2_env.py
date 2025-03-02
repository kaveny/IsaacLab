# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# This script spawns the robot.

from __future__ import annotations

import math
import torch
import numpy as np
from collections.abc import Sequence

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab_tasks.direct.go2 import read_motion as rm
import os
from pathlib import Path




@configclass
class Go2EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0      # max episode length
    observation_space = 48       # number of observations input into NN
    # action_scale = 100.0  # [N]
    action_space = 12            # number of actions output from NN
    decimation = 2               # number of simulation time steps between each round of observations and actions
    state_space = 0
    
    ## AnymalCFlatEnvCfg
    # episode_length_s = 20.0
    # decimation = 4
    # action_scale = 0.5
    # action_space = 12
    # observation_space = 48
    # state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # reset
    # max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    # initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    # rew_scale_alive = 1.0
    # rew_scale_terminated = -2.0
    # rew_scale_pole_pos = -1.0
    # rew_scale_cart_vel = -0.01
    # rew_scale_pole_vel = -0.005
    # reward function parameters
    
    # imitation rewards
    # pose_weight = 0.5
    # velocity_weight = 0.05
    # end_effector_weight = 0.2
    # root_pose_weight = 0.15
    # root_velocity_weight = 0.1


class Go2Env(DirectRLEnv):
    cfg: Go2EnvCfg

    def __init__(self, cfg: Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


        # self._cart_dof_idx, _ = self.go2.find_joints(self.cfg.cart_dof_name)
        # self._pole_dof_idx, _ = self.go2.find_joints(self.cfg.pole_dof_name)  
        # self.action_scale = self.cfg.action_scale

        # self.joint_pos = self.go2.data.joint_pos
        # self.joint_vel = self.go2.data.joint_vel

        # reference motion
        current_dir = Path(os.path.dirname(os.path.abspath(__file__))) # Get the absolute path of the current file
        MOTION_FILE_PATH = current_dir / "trot_A1.txt" # Define the path to the motion file relative to this file
        # read motion file
        a1_f = rm.read_frames_from_json(MOTION_FILE_PATH)
        # convert a1 to go2 format
        self.go2_f = rm.convert_a1_to_go2_format(a1_f)
        # index for the next joint position
        self.idx = 0

    def _setup_scene(self):
        self.go2 = Articulation(self.cfg.robot_cfg)   # creates tensor of num_envs
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg()) 
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # prevent environments from colliding with each other
        self.scene.filter_collisions(global_prim_paths = [])
        # add articulation to scene
        self.scene.articulations["go2"] = self.go2
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # self.actions = self.action_scale * actions.clone()
        self.actions = actions.clone()
        # self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self) -> None:
        # self.go2.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)
        
        # anymal c
        # self._robot.set_joint_position_target(self._processed_actions)
        

        joints, self.idx = rm.get_next_front_right_joint_position(self.go2_f, self.idx)
        # convert np.array to tensor
        fr_joints = torch.tensor(joints, dtype=torch.float32, device=self.device)
        # joint ids for front right leg of go2 robot
        fr_joint_ids = torch.tensor([1,5,9], dtype=torch.int32, device=self.device)
        self.go2.set_joint_position_target(fr_joints, joint_ids=fr_joint_ids)
        # joint ids for all legs of go2 robot except front right leg
        other_joints = np.array([0,2,3,4,6,7,8,10,11], dtype=np.int32)
        other_joint_ids = torch.tensor(other_joints, dtype=torch.int32, device=self.device)
        self.go2.set_joint_position_target(self.go2.data.default_joint_pos[:,other_joints], joint_ids=other_joint_ids)

        # print('********************************')
        # print('type(default_joint_pos)')
        # # print(type(self.go2.data.default_joint_pos))
        # # print(self.go2.data.default_joint_pos.shape)
        # # print(self.go2.data.joint_names[other_joints])
        # # print()
        # print('go2.data.default_joint_pos[:,other_joints]')
        # print(self.go2.data.default_joint_pos[:,other_joints])
        # print('********************************')
        # print()


    def _get_observations(self) -> dict:
        # anymal_c
        # obs = torch.cat(
        #     [
        #         tensor
        #         for tensor in (
        #             self._robot.data.root_lin_vel_b,
        #             self._robot.data.root_ang_vel_b,
        #             self._robot.data.projected_gravity_b,
        #             self._commands,
        #             self._robot.data.joint_pos - self._robot.data.default_joint_pos,
        #             self._robot.data.joint_vel,
        #             self._actions,
        #         )
        #         if tensor is not None
        #     ],
        #     dim=-1,
        # )

        obs = torch.zeros((self.num_envs,48), dtype=torch.float32, device=self.device)   
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # pose_reward = self._calc_reward_pose()
        # velocity_reward = self._calc_reward_velocity()
        # end_effector_reward = self._calc_reward_end_effector()
        # # root_pose_reward = self._calc_reward_root_pose()
        # # root_velocity_reward = self._calc_reward_root_velocity()

        # reward =  self.pose_weight * pose_reward \
        #         + self.velocity_weight * velocity_reward \
        #         + self.end_effector_weight * end_effector_reward \
        #         # + self.root_pose_weight * root_pose_reward \
        #         # + self.root_velocity_weight * root_velocity_reward

        # return reward * self.weight
    
        total_reward = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)   
        return total_reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # self.joint_pos = self.go2.data.joint_pos
        # self.joint_vel = self.go2.data.joint_vel

        # time_out = self.episode_length_buf >= self.max_episode_length - 1
        # out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        failure_termination = torch.zeros((self.num_envs), dtype=torch.bool, device=self.device)
        
        # anymal c
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return time_out, failure_termination
        

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.go2._ALL_INDICES
        super()._reset_idx(env_ids)

        # anymal
        self.go2.reset(env_ids)
        # self._robot.reset(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        # self._actions[env_ids] = 0.0
        # self._previous_actions[env_ids] = 0.0
        # Sample new commands
        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        # Reset robot state
        joint_pos = self.go2.data.default_joint_pos[env_ids]
        joint_vel = self.go2.data.default_joint_vel[env_ids]
        default_root_state = self.go2.data.default_root_state[env_ids]       # first three are pos, next 4 quats, next 3 vel, next 3 ang vel
        default_root_state[:, :3] += self.scene.env_origins[env_ids]         # adds center of each env position to robot position
        self.go2.write_root_pose_to_sim(default_root_state[:, :7], env_ids)  
        self.go2.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.go2.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


        # joint_pos = self.go2.data.default_joint_pos[env_ids]
        # joint_pos[:, self._pole_dof_idx] += sample_uniform(
        #     self.cfg.initial_pole_angle_range[0] * math.pi,
        #     self.cfg.initial_pole_angle_range[1] * math.pi,
        #     joint_pos[:, self._pole_dof_idx].shape,
        #     joint_pos.device,
        # )
        # joint_vel = self.go2.data.default_joint_vel[env_ids]

        # default_root_state = self.go2.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # self.joint_pos[env_ids] = joint_pos
        # self.joint_vel[env_ids] = joint_vel

        # self.go2.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self.go2.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self.go2.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


# @torch.jit.script
# def compute_rewards(
#     rew_scale_alive: float,
#     rew_scale_terminated: float,
#     rew_scale_pole_pos: float,
#     rew_scale_cart_vel: float,
#     rew_scale_pole_vel: float,
#     pole_pos: torch.Tensor,
#     pole_vel: torch.Tensor,
#     cart_pos: torch.Tensor,
#     cart_vel: torch.Tensor,
#     reset_terminated: torch.Tensor,
# ):
#     rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
#     rew_termination = rew_scale_terminated * reset_terminated.float()
#     rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
#     rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
#     rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
#     total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
#     return total_reward
