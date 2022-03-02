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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .mos.mos import Mos
from .mos.mos_config import MosRoughCfg, MosRoughCfgPPO
# from .mos.ada11 import Ada11
from .mos.ada11_config import Ada11RoughCfg, Ada11RoughCfgPPO
# from .mos.ada12 import Ada12
from .mos.ada12_config import Ada12RoughCfg, Ada12RoughCfgPPO
# from .mos.ada13 import Ada13
from .mos.ada13_config import Ada13RoughCfg, Ada13RoughCfgPPO
# from .mos.ada14 import Ada14
from .mos.ada14_config import Ada14RoughCfg, Ada14RoughCfgPPO
# from .mos.ada21 import Ada21
from .mos.ada21_config import Ada21RoughCfg, Ada21RoughCfgPPO
# from .mos.ada23 import Ada23
from .mos.ada23_config import Ada23RoughCfg, Ada23RoughCfgPPO
# from .mos.ada24 import Ada24
from .mos.ada24_config import Ada24RoughCfg, Ada24RoughCfgPPO

# from .mos.nmi11 import Nmi11
from .mos.nmi11_config import Nmi11RoughCfg, Nmi11RoughCfgPPO
# from .mos.nmi12 import Nmi12
from .mos.nmi12_config import Nmi12RoughCfg, Nmi12RoughCfgPPO
# from .mos.nmi13 import Nmi13
from .mos.nmi13_config import Nmi13RoughCfg, Nmi13RoughCfgPPO
# from .mos.nmi14 import Nmi14
from .mos.nmi14_config import Nmi14RoughCfg, Nmi14RoughCfgPPO
# from .mos.nmi21 import Nmi21
from .mos.nmi21_config import Nmi21RoughCfg, Nmi21RoughCfgPPO
# from .mos.nmi23 import Nmi23
from .mos.nmi23_config import Nmi23RoughCfg, Nmi23RoughCfgPPO
# from .mos.nmi24 import Nmi24
from .mos.nmi24_config import Nmi24RoughCfg, Nmi24RoughCfgPPO

# from .mos.dpmt11 import Dpmt11
from .mos.dpmt11_config import Dpmt11RoughCfg, Dpmt11RoughCfgPPO
# from .mos.dpmt12 import Dpmt12
from .mos.dpmt12_config import Dpmt12RoughCfg, Dpmt12RoughCfgPPO
# from .mos.dpmt13 import Dpmt13
from .mos.dpmt13_config import Dpmt13RoughCfg, Dpmt13RoughCfgPPO
# from .mos.dpmt14 import Dpmt14
from .mos.dpmt14_config import Dpmt14RoughCfg, Dpmt14RoughCfgPPO
# from .mos.dpmt21 import Dpmt21
from .mos.dpmt21_config import Dpmt21RoughCfg, Dpmt21RoughCfgPPO
# from .mos.dpmt23 import Dpmt23
from .mos.dpmt23_config import Dpmt23RoughCfg, Dpmt23RoughCfgPPO
# from .mos.dpmt24 import Dpmt24
from .mos.dpmt24_config import Dpmt24RoughCfg, Dpmt24RoughCfgPPO

# from .mos.dpmf11 import Dpmf11
from .mos.dpmf11_config import Dpmf11RoughCfg, Dpmf11RoughCfgPPO
# from .mos.dpmf12 import Dpmf12
from .mos.dpmf12_config import Dpmf12RoughCfg, Dpmf12RoughCfgPPO
# from .mos.dpmf13 import Dpmf13
from .mos.dpmf13_config import Dpmf13RoughCfg, Dpmf13RoughCfgPPO
# from .mos.dpmf14 import Dpmf14
from .mos.dpmf14_config import Dpmf14RoughCfg, Dpmf14RoughCfgPPO
# from .mos.dpmf21 import Dpmf21
from .mos.dpmf21_config import Dpmf21RoughCfg, Dpmf21RoughCfgPPO
# from .mos.dpmf23 import Dpmf23
from .mos.dpmf23_config import Dpmf23RoughCfg, Dpmf23RoughCfgPPO
# from .mos.dpmf24 import Dpmf24
from .mos.dpmf24_config import Dpmf24RoughCfg, Dpmf24RoughCfgPPO

# from .mos.dpme11 import Dpme11
from .mos.dpme11_config import Dpme11RoughCfg, Dpme11RoughCfgPPO
# from .mos.dpme12 import Dpme12
from .mos.dpme12_config import Dpme12RoughCfg, Dpme12RoughCfgPPO
# from .mos.dpme13 import Dpme13
from .mos.dpme13_config import Dpme13RoughCfg, Dpme13RoughCfgPPO
# from .mos.dpme14 import Dpme14
from .mos.dpme14_config import Dpme14RoughCfg, Dpme14RoughCfgPPO
# from .mos.dpme21 import Dpme21
from .mos.dpme21_config import Dpme21RoughCfg, Dpme21RoughCfgPPO
# from .mos.dpme23 import Dpme23
from .mos.dpme23_config import Dpme23RoughCfg, Dpme23RoughCfgPPO
# from .mos.dpme24 import Dpme24
from .mos.dpme24_config import Dpme24RoughCfg, Dpme24RoughCfgPPO

from .mos.mozzy import Mozzy

from legged_gym.utils.task_registry import task_registry


class Ada11(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="ada11")
        
class Ada12(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="ada12")

class Ada13(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="ada13")

class Ada14(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__( cfg, sim_params, physics_engine, sim_device, headless,setting="ada14")

class Ada21(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="ada21")

class Ada23(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="ada23")

class Ada24(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="ada24")


class Nmi11(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="nmi11")
        
class Nmi12(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="nmi12")

class Nmi13(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="nmi13")

class Nmi14(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__( cfg, sim_params, physics_engine, sim_device, headless,setting="nmi14")

class Nmi21(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="nmi21")

class Nmi23(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="nmi23")

class Nmi24(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="nmi24")



class Dpmt11(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpmt11")
        
class Dpmt12(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpmt12")

class Dpmt13(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpmt13")

class Dpmt14(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__( cfg, sim_params, physics_engine, sim_device, headless,setting="dpmt14")

class Dpmt21(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpmt21")

class Dpmt23(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpmt23")

class Dpmt24(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpmt24")



class Dpmf11(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpmf11")
        
class Dpmf12(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpmf12")

class Dpmf13(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpmf13")

class Dpmf14(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__( cfg, sim_params, physics_engine, sim_device, headless,setting="dpmf14")

class Dpmf21(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpmf21")

class Dpmf23(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpmf23")

class Dpmf24(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpmf24")


class Dpme11(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpme11")
        
class Dpme12(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpme12")

class Dpme13(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpme13")

class Dpme14(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__( cfg, sim_params, physics_engine, sim_device, headless,setting="dpme14")

class Dpme21(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpme21")

class Dpme23(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpme23")

class Dpme24(Mozzy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,setting="dpme24")


import os



task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "mos", Mos, MosRoughCfg(),  MosRoughCfgPPO() )
task_registry.register( "ada11", Ada11, Ada11RoughCfg(),  Ada11RoughCfgPPO() )
task_registry.register( "ada12", Ada12, Ada12RoughCfg(),  Ada12RoughCfgPPO() )
task_registry.register( "ada13", Ada13, Ada13RoughCfg(),  Ada13RoughCfgPPO() )
task_registry.register( "ada14", Ada14, Ada14RoughCfg(),  Ada14RoughCfgPPO() )
task_registry.register( "ada21", Ada21, Ada21RoughCfg(),  Ada21RoughCfgPPO() )
task_registry.register( "ada23", Ada23, Ada23RoughCfg(),  Ada23RoughCfgPPO() )
task_registry.register( "ada24", Ada24, Ada24RoughCfg(),  Ada24RoughCfgPPO() )

task_registry.register( "nmi11", Nmi11, Nmi11RoughCfg(),  Nmi11RoughCfgPPO() )
task_registry.register( "nmi12", Nmi12, Nmi12RoughCfg(),  Nmi12RoughCfgPPO() )
task_registry.register( "nmi13", Nmi13, Nmi13RoughCfg(),  Nmi13RoughCfgPPO() )
task_registry.register( "nmi14", Nmi14, Nmi14RoughCfg(),  Nmi14RoughCfgPPO() )
task_registry.register( "nmi21", Nmi21, Nmi21RoughCfg(),  Nmi21RoughCfgPPO() )
task_registry.register( "nmi23", Nmi23, Nmi23RoughCfg(),  Nmi23RoughCfgPPO() )
task_registry.register( "nmi24", Nmi24, Nmi24RoughCfg(),  Nmi24RoughCfgPPO() )

task_registry.register( "dpmt11", Dpmt11, Dpmt11RoughCfg(),  Dpmt11RoughCfgPPO() )
task_registry.register( "dpmt12", Dpmt12, Dpmt12RoughCfg(),  Dpmt12RoughCfgPPO() )
task_registry.register( "dpmt13", Dpmt13, Dpmt13RoughCfg(),  Dpmt13RoughCfgPPO() )
task_registry.register( "dpmt14", Dpmt14, Dpmt14RoughCfg(),  Dpmt14RoughCfgPPO() )
task_registry.register( "dpmt21", Dpmt21, Dpmt21RoughCfg(),  Dpmt21RoughCfgPPO() )
task_registry.register( "dpmt23", Dpmt23, Dpmt23RoughCfg(),  Dpmt23RoughCfgPPO() )
task_registry.register( "dpmt24", Dpmt24, Dpmt24RoughCfg(),  Dpmt24RoughCfgPPO() )

task_registry.register( "dpmf11", Dpmf11, Dpmf11RoughCfg(),  Dpmf11RoughCfgPPO() )
task_registry.register( "dpmf12", Dpmf12, Dpmf12RoughCfg(),  Dpmf12RoughCfgPPO() )
task_registry.register( "dpmf13", Dpmf13, Dpmf13RoughCfg(),  Dpmf13RoughCfgPPO() )
task_registry.register( "dpmf14", Dpmf14, Dpmf14RoughCfg(),  Dpmf14RoughCfgPPO() )
task_registry.register( "dpmf21", Dpmf21, Dpmf21RoughCfg(),  Dpmf21RoughCfgPPO() )
task_registry.register( "dpmf23", Dpmf23, Dpmf23RoughCfg(),  Dpmf23RoughCfgPPO() )
task_registry.register( "dpmf24", Dpmf24, Dpmf24RoughCfg(),  Dpmf24RoughCfgPPO() )

task_registry.register( "dpme11", Dpme11, Dpme11RoughCfg(),  Dpme11RoughCfgPPO() )
task_registry.register( "dpme12", Dpme12, Dpme12RoughCfg(),  Dpme12RoughCfgPPO() )
task_registry.register( "dpme13", Dpme13, Dpme13RoughCfg(),  Dpme13RoughCfgPPO() )
task_registry.register( "dpme14", Dpme14, Dpme14RoughCfg(),  Dpme14RoughCfgPPO() )
task_registry.register( "dpme21", Dpme21, Dpme21RoughCfg(),  Dpme21RoughCfgPPO() )
task_registry.register( "dpme23", Dpme23, Dpme23RoughCfg(),  Dpme23RoughCfgPPO() )
task_registry.register( "dpme24", Dpme24, Dpme24RoughCfg(),  Dpme24RoughCfgPPO() )

