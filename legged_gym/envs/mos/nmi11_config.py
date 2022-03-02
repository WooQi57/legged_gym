# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.mos.nmi11 import reference_handler
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

exp_name='nmi11'
# import要改
# 速度要改
# 改类名

class Nmi11RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096 # zc : why 1024?
        num_observations = 72
        num_actions = 20

    
    class terrain( LeggedRobotCfg.terrain):
        mesh_type ='plane' #  'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = False

    class commands( LeggedRobotCfg.commands):
        curriculum = False
        heading_command = False 
        class ranges:
            lin_vel_x = [0.399, 0.401] # run! # zc  # min max [m/s] [-1.0, 1.0]
            lin_vel_y = [-0.01, 0.01] # zc # min max [m/s] [-1.0, 1.0]
            ang_vel_yaw = [-0.01, 0.01]    # min max [rad/s]
            heading = [-0.3, 0.3]


    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'L_leg_1': 0,
            'L_leg_2': -0.524,
            'L_leg_3': 0,
            'L_leg_4': 0.785,
            'L_leg_5':-0.524,
            'L_leg_6': 0,
           
            'R_leg_1': 0,
            'R_leg_2': 0.524,
            'R_leg_3': 0,
            'R_leg_4': -0.785,
            'R_leg_5': 0.524,
            'R_leg_6': 0,

            'L_arm_1':0,
            'L_arm_2':1.57,
            'L_arm_3':2.6,
            'R_arm_1':0,
            'R_arm_2':-1.57,
            'R_arm_3':-2.6,
            'neck':0,
            'head':0
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'P' #position control
        # PD Drive parameters:
        stiffness = {   'L_arm_1':100.0,'L_arm_2':100.0,'L_arm_3':100.0,'R_arm_1':100.0,'R_arm_2':100.0,'R_arm_3':100.0,'neck':100.0,'head':100.0,
			'L_leg_1': 100.0,'L_leg_2': 100.0, 'L_leg_3': 100.0, 'L_leg_4': 100.0, 'L_leg_5': 100.0, 'L_leg_6': 100.0, 
			'R_leg_1': 100.0, 'R_leg_2': 100.0, 'R_leg_3': 100.0, 'R_leg_4': 100.0, 'R_leg_5': 100.0,'R_leg_6': 100.0}
			#'toe_joint': 40.}  
            # [N*m/rad]
        damping = { 'L_arm_1':3.0,'L_arm_2':3.0,'L_arm_3':3.0,'R_arm_1':3.0,'R_arm_2':3.0,'R_arm_3':3.0,'neck':3.0,'head':3.0,
			'L_leg_1': 3.0,'L_leg_2': 3.0,'L_leg_3': 3.0,'L_leg_4': 3.0,'L_leg_5': 3.0,'L_leg_6': 3.0,
		 	'R_leg_1': 3.0,'R_leg_2': 3.0,'R_leg_3': 3.0,'R_leg_4': 3.0,'R_leg_5': 3.0,'R_leg_6':3.0}
			#'toe_joint': 1.}  
            # [N*m*s/rad]   
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.5 # zc 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/mos/urdf/thmos_mix.urdf'
        foot_name = 'leg_6_link'
        terminate_after_contacts_on = []
        terminate_after_contacts_on = ['body', 'L_arm_1_link','L_arm_2_link','L_arm_3_link','R_arm_1_link','R_arm_2_link','R_arm_3_link','neck_link','head_link'
                                                                        'L_leg_1_link','L_leg_2_link','L_leg_3_link','L_leg_4_link','L_leg_5_link','R_leg_1_link','R_leg_2_link','R_leg_3_link','R_leg_4_link','R_leg_5_link']
        flip_visual_attachments = False
        self_collisions = 0 # zc 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limmesh_type ='plane' #  'trimesh' # "heightfield" # none, plane, heightfield or trimeshit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        # class scales( LeggedRobotCfg.rewards.scales ):
        #     termination = -50. #zc edit #-200.
        #     tracking_lin_vel =0.0
        #     tracking_ang_vel = 0.0  # 1.0
        #     torques = -5.e-6
        #     dof_acc = -2.e-7
        #     lin_vel_z = -0.5 
        #     feet_air_time = 0.
        #     dof_pos_limits = -1.
        #     no_fly = 0.
        #     dof_vel = -0.0
        #     ang_vel_xy = -0.05
        #     feet_contact_forces = -0.
        #     action_rate = -0.001
        #     stand_still = -0.
        #     orientation=0.
        #     reference= 0. #-1 # zc #-10
        #     track_ref=0.
        #     my_reward = 1.0
        #     performance = 1e-8
        #     imitation = 1e-8
        #     regularization = 0.
        #     show_omega = 1e-8
           

class Nmi11RoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = exp_name
        num_steps_per_env = 96 # zc 48  # per iteration
        max_iterations = 2000 # number of policy updates

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

