# VisFly 中文入门

[English README](readme.md)

VisFly 是一个四旋翼飞行仿真器。你可以让模拟无人机用机载相机观察场景，训练它飞向目标、穿过交叉航线或降落。

第一次使用，建议从 **`cluttered_flight` 单机避障导航**开始。这一页会带你完成一次短训练，得到一个模型文件，再用它做一次评估并保存飞行曲线。

[安装](#安装) · [快速开始](#快速开始) · [修改任务配置](#修改任务配置) · [进阶用法](#进阶用法) · [论文与相关项目](#论文与相关项目)

## 安装

下面以 Ubuntu 为例。准备好 Conda 和 Git。仓库自带的 PPO 示例使用 NVIDIA GPU 训练，视觉仿真还需要能通过 EGL 创建渲染上下文。

命令以 `~/projects` 为工作目录，其中放着 `VisFly/` 和单独构建的 `habitat-sim/`。如果你已经把仓库克隆到其他地方，把命令中的 `~/projects` 换成 **`VisFly/` 的实际父目录**。

### 1. 克隆仓库并创建 Python 环境

```bash
sudo apt-get install git-lfs libcgal-dev
mkdir -p ~/projects
cd ~/projects
git clone --recurse-submodules https://github.com/SJTU-ViSYS-team/VisFly
cd VisFly
```

`environment.yml` 末尾的 `prefix:` 记录的是原作者机器上的 Conda 路径。删掉本地文件里的这一行，再在 `VisFly/` 下创建并激活环境。[environment.yml](environment.yml)

```bash
conda env create -f environment.yml
conda activate visfly
```

### 2. 下载场景数据

`visfly-beta` 提供示例用的室内模型和场景布局。它是公开数据集，可以匿名下载。如果本机的 `datasets/visfly-beta/` 已经有场景数据，可直接使用。

在 `VisFly/` 下运行以下命令。最后一条会进入数据集仓库执行 Git LFS 下载，你所在的目录仍是 `VisFly/`。[示例数据集](https://huggingface.co/datasets/LiFanxing/visfly-beta)

```bash
git lfs install
git clone https://huggingface.co/datasets/LiFanxing/visfly-beta.git datasets/visfly-beta
git -C datasets/visfly-beta lfs pull
```

### 3. 构建修改版 Habitat-Sim

Habitat-Sim 负责渲染无人机看到的画面。VisFly 使用 `Fanxing-LI/habitat-sim` 的修改版。在已激活的 `visfly` 环境中克隆它，并按构建指南的 **Build from Source** 一节准备依赖。[Habitat-Sim 构建指南](https://github.com/Fanxing-LI/habitat-sim/blob/main/BUILD_FROM_SOURCE.md#build-from-source)

```bash
cd ~/projects
git clone https://github.com/Fanxing-LI/habitat-sim
cd habitat-sim
```

依赖准备好后，在 `habitat-sim/` 目录中运行下面的命令，构建带窗口支持的版本。

```bash
HEADLESS=False python setup.py build_ext --parallel 2 install --bullet --cmake
```

## 快速开始

**从 `VisFly/` 的父目录运行示例。** 配置中的场景路径以这里为起点。先确认脚本可以启动。

```bash
cd ~/projects
conda activate visfly
python VisFly/exps/examples/run.py --help
```

### 1. 先把训练缩短

`cluttered_flight` 的训练步数默认设为 1000 万。首次运行用 **1 架无人机、32 步**就能检查训练和评估流程。

打开下面两个 YAML 文件，在现有层级修改列出的字段，保留其余配置。

[环境配置 cluttered_flight.yaml](exps/examples/env_cfgs/cluttered_flight.yaml)

```yaml
env:
  num_agent_per_scene: 1  # 每个场景只用一架无人机训练
```

[训练配置 PPO.yaml](exps/examples/alg_cfgs/cluttered_flight/PPO.yaml)

```yaml
algorithm:
  n_steps: 32      # 每架无人机在一次更新前采集的步数
  batch_size: 32  # 每个训练批次的样本数
  n_epochs: 1     # 每批采集的数据只训练一轮
learn:
  total_timesteps: 32
test:
  is_fig: false
  is_fig_save: true    # 保存飞行曲线
  is_video: false
  is_video_save: false # 先只保存图表
```

### 2. 训练并找到模型

```bash
python VisFly/exps/examples/run.py -t 1 -e cluttered_flight -c quickstart
```

这条命令用 PPO（一种强化学习算法）训练飞行策略。策略根据状态、目标位置和深度图决定下一步的控制动作。训练时，环境用奖励衡量飞行表现，PPO 用收集到的数据更新策略。

结束时，终端会打印 `Saving model to ...`。在 `VisFly/exps/examples/saved/cluttered_flight/` 下会看到 `PPO_quickstart_1.zip` 这样的模型文件，也叫**检查点**。

### 3. 加载模型并评估

```bash
python VisFly/exps/examples/run.py -t 0 -e cluttered_flight -w PPO_quickstart_1.zip
```

将 `-w` 后面的值换成上一步实际生成的文件名。评估用该模型飞行一个回合，保存两张图，分别记录飞行状态和最近障碍距离。以 `PPO_quickstart_1.zip` 为例，文件位置如下。

```text
VisFly/exps/examples/saved/cluttered_flight/
├── PPO_quickstart_1.zip              # 训练得到的模型
└── test/PPO_quickstart_1/
    ├── 0.png                        # 位置、姿态、速度和角速度
    └── 1.png                        # 最近障碍距离
```

看到模型文件和图表，就跑通了流程。正式实验可以恢复原配置，用新的 `-c` 备注再训练。

## 修改任务配置

### 换一个任务

用同一个 `run.py` 入口，把 `-e` 后的任务名换成所需任务。每个任务有各自的环境和 PPO 配置。

| 任务名 | 无人机要做什么 | 环境实现 |
| --- | --- | --- |
| `cluttered_flight` | 单机在有障碍物的室内场景飞向目标 | [NavigationEnv.py](envs/NavigationEnv.py) |
| `crossing` | 多架无人机沿交叉航线飞往各自的目标 | [MultiNavigationEnv.py](envs/MultiNavigationEnv.py) |
| `landing` | 控制无人机在目标位置降落 | [LandingEnv.py](envs/LandingEnv.py) |

| 参数 | 用法 |
| --- | --- |
| `-t` / `--train` | `1` 为训练，`0` 为评估。 |
| `-e` / `--env` | 选择上表中的任务名。 |
| `-w` / `--weight` | 要加载的模型文件名。评估时必填，训练时用于续训。 |
| `-c` / `--comment` | 给新训练的模型和日志加备注，默认是 `std`。比较不同实验时可换一个备注。 |

### 根据目标找配置

下表中的文件位置相对于 `VisFly/`。把 `<任务名>` 换成命令中 `-e` 的值。

| 想改什么 | 在哪里改 |
| --- | --- |
| 并行训练的无人机数量 | `exps/examples/env_cfgs/<任务名>.yaml` 中的 `env.num_scene` 和 `env.num_agent_per_scene` |
| 障碍场景、无人机动作和机载相机 | 同一文件的 `env.scene_kwargs`、`env.dynamics_kwargs` 和 `env.sensor_kwargs` |
| 训练步数、学习率和网络结构 | `exps/examples/alg_cfgs/<任务名>/PPO.yaml` 中的 `learn.total_timesteps` 和 `algorithm` |
| 评估时是否保存图表和视频 | PPO 配置里的 `test.is_fig_save` 和 `test.is_video_save` |
| 评估录像的取景和分辨率 | 环境配置里的 `eval_env.scene_kwargs.render_settings` |

`eval_env` 会继承训练配置 `env`，并覆盖其中单独指定的字段。例如 `cluttered_flight` 的评估默认使用一架无人机，仍沿用训练场景和机载深度相机。机载相机的画面供策略输入，`render_settings` 配置的是评估录像用的外部相机。[示例入口](exps/examples/run.py) · [配置加载实现](utils/common.py)

训练日志可用 TensorBoard 查看，和模型文件一起保存在 `exps/examples/saved/<任务名>/`。

可以查看 32 步验证、正式 PPO 训练和固定起点评估的结果。[cluttered_flight 实验记录](docs/experiments/cluttered_flight/2026-11-18/README.md)

## 进阶用法

需要改任务行为或网络结构时，再从相应的示例入手。VisFly 提供 Gym 风格的接口，环境负责给出观测、推进仿真和计算奖励，训练代码负责更新策略。可微动力学也可用于基于梯度的策略优化和规划。

| 想实现什么 | 从哪里开始 |
| --- | --- |
| 改飞行目标、成功条件或奖励 | 继承 `NavigationEnv`，重写对应方法。 |
| 换初始位置、相机或控制模式 | 修改环境的初始化参数。 |
| 加入新的策略输入 | 在环境中声明并返回观测，再为策略增加特征提取分支。 |
| 创建新的室内场景 | 查看示例数据的 `configs/scenes/` 和 Habitat-Sim 的场景配置格式。 |

<details>
<summary>用 Python 创建一个单机导航环境</summary>

`NavigationEnv` 将场景、动力学、机载传感器和随机化配置传给仿真环境。下面从 `VisFly/` 的父目录使用示例数据集，创建一个单场景、单无人机的导航任务。

```python
import habitat_sim
import torch as th

from VisFly.envs.NavigationEnv import NavigationEnv

env = NavigationEnv(
    num_scene=1,
    num_agent_per_scene=1,
    visual=True,
    device="cpu",
    max_episode_steps=256,
    seed=42,
    dynamics_kwargs={
        "action_type": "bodyrate",
        "ori_output_type": "quaternion",
        "dt": 0.0025,  # 仿真步长，单位为秒。
        "ctrl_dt": 0.02,  # 控制步长，单位为秒。
        "integrator": "euler",
        "ctrl_delay": True,
    },
    scene_kwargs={
        "path": "VisFly/datasets/visfly-beta/configs/scenes/garage_simple_l_medium",
        "scene_type": "json",
        # 导出图像或视频时配置外部相机。
        "render_settings": {
            "mode": "fix",
            "view": "custom",
            "resolution": [1080, 1920],
            "sensor_type": "color",
            "position": [[7.0, 6.8, 5.5], [7.0, 4.8, 4.5]],  # 相机位置和注视点。
            "line_width": 6.0,
            "axes": False,
            "trajectory": True,
        },
    },
    sensor_kwargs=[
        {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "uuid": "depth",
            "resolution": [64, 64],
            "position": [0, 0, 0],
            "orientation": [0, 0, 0],
        },
    ],
    random_kwargs={
        "noise_kwargs": {
            "depth": {"model": "RedwoodDepthNoiseModel", "kwargs": {}},
        },
        "state_generator": {
            "class": "Uniform",
            "kwargs": [
                {
                    "position": {"mean": [1.0, 0.0, 1.5], "half": [0.0, 2.0, 1.0]},
                    "orientation": {"mean": [0.0, 0.0, 0.0], "half": [0.0, 0.0, 0.0]},
                    "velocity": {"mean": [0.0, 0.0, 0.0], "half": [0.0, 0.0, 0.0]},
                    "angular_velocity": {"mean": [0.0, 0.0, 0.0], "half": [0.0, 0.0, 0.0]},
                },
            ],
        },
    },
    target=th.tensor([[9.0, 0.0, 1.0]]),
)
```

- 动作类型可选 `bodyrate`、`thrust`、`velocity` 或 `position`，姿态可用四元数 `quaternion` 或欧拉角 `euler` 表示，积分器可选 `euler` 或 `rk4`。
- 机载相机可在 `sensor_kwargs` 中配置多个。传感器的 `uuid` 应包含 `depth`、`color` 或 `semantic`，例如 `color_0`。图像噪声模型可在 `random_kwargs.noise_kwargs` 中设置。[Habitat-Sim 传感器噪声模型](https://aihabitat.org/docs/habitat-sim/habitat_sim.sensors.noise_models.html)
- 初始状态生成器支持 `Uniform`、`Normal` 和 `Union`。可以提供一份共享配置，也可以按场景或智能体配置，具体取决于生成器类型。[状态生成器实现](envs/base/droneEnv.py)

</details>

<details>
<summary>自定义成功条件和奖励</summary>

可以继承 `DroneGymEnvsBase` 创建新任务，也可以继承已有任务复用其配置和观测逻辑。以 `NavigationEnv` 为例，构造环境时在 `observation_space` 中注册三维 `target` 观测，`get_observation()` 返回飞行状态、目标位置以及开启视觉时的深度图。下面通过继承导航环境改写成功条件和奖励。

```python
import torch as th

from VisFly.envs.NavigationEnv import NavigationEnv


class MyNavigationEnv(NavigationEnv):
    def get_success(self) -> th.Tensor:
        distance = (self.position - self.target).norm(dim=1)
        return distance <= self.success_radius

    def get_reward(self, predicted_obs=None) -> th.Tensor:
        distance = (self.position - self.target).norm(dim=1)
        speed = self.velocity.norm(dim=1)
        angular_speed = self.angular_velocity.norm(dim=1)
        upright = self.orientation.new_tensor([1, 0, 0, 0])
        attitude_error = (self.orientation - upright).norm(dim=1)
        remaining_steps = self.max_episode_steps - self._step_count
        return (
            0.1
            - (0.1 / 9) * distance
            - 0.00001 * attitude_error
            - 0.002 * speed
            - 0.002 * angular_speed
            + 0.1 * self._success * remaining_steps
        )
```

这段奖励使用四元数姿态。新增观测时，同步更新 `observation_space` 和 `get_observation()`。如果任务维护每架无人机独有的状态，可以扩展 `reset_agent_by_id()`、`reset_env_by_id()` 或 `reset()`，按相应范围重置。障碍规避可利用最近障碍点 `self.collision_point`、指向障碍的向量 `self.collision_vector` 和距离 `self.collision_dis`。[环境基类](envs/base/droneGymEnv.py) · [导航任务实现](envs/NavigationEnv.py)

</details>

<details>
<summary>配置处理深度图、状态和目标的策略</summary>

VisFly 的 PPO 继承自 Stable-Baselines3，并通过 `CustomMultiInputPolicy` 支持配置多输入策略。下面让 actor 和 critic 共享由深度图、飞行状态和目标位置组成的特征，再分别使用各自的 MLP。

```python
import torch

from VisFly.utils.algorithms.PPO import PPO
from VisFly.utils.policies.extractors import StateTargetImageExtractor

model = PPO(
    policy="CustomMultiInputPolicy",
    env=env,
    policy_kwargs=dict(
        features_extractor_class=StateTargetImageExtractor,
        features_extractor_kwargs={
            "net_arch": {
                "depth": {
                    "channel": [6, 12, 18],
                    "kernel_size": [5, 3, 3],
                    "padding": [0, 0, 0],
                    "stride": [1, 1, 1],
                    "layer": [128],
                },
                "state": {"layer": [128, 64], "bn": False, "ln": False},
                "target": {"layer": [128, 64], "bn": False, "ln": False},
            },
            "activation_fn": torch.nn.ReLU,
        },
        share_features_extractor=True,
        net_arch={
            "pi": [64, 64],
            "vf": [64, 64],
            "pi_bn": False,
            "vf_bn": False,
            "pi_ln": False,
            "vf_ln": False,
        },
        activation_fn=torch.nn.ReLU,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"weight_decay": 1e-5},
    ),
)
```

`depth` 需要与机载相机的 `uuid` 一致。图像分支可以像上面一样配置 CNN，也可以用 `"backbone": "resnet18"` 等参数选用预训练的 ResNet、EfficientNet 或 MobileNet。`layer` 定义 MLP 层宽，`bn` 和 `ln` 控制归一化。若使用独立的 actor 和 critic 特征提取器，将 `share_features_extractor` 设为 `False`、通用的 `features_extractor_class` 和 `features_extractor_kwargs` 设为 `None`，再分别提供 `pi_features_extractor_*` 和 `vf_features_extractor_*`。[特征提取器实现](utils/policies/extractors.py)

</details>

<details>
<summary>让策略读取最近 5 步的动作</summary>

如果要把最近 5 步的动作纳入策略输入，可以为 `StateTargetImageExtractor` 增加 `action` 分支。下面将每架无人机的 `(5, 4)` 动作历史展平，再交给 MLP。父类会把各分支的输出拼接起来。

```python
from gymnasium.spaces.utils import flatten_space

from VisFly.utils.policies.extractors import (
    StateTargetImageExtractor,
    set_mlp_feature_extractor,
)


class ActionStateTargetImageExtractor(StateTargetImageExtractor):
    def _build(self, observation_space, net_arch, activation_fn):
        super()._build(observation_space, net_arch, activation_fn)
        self._features_dim += set_mlp_feature_extractor(
            self,
            "action",
            flatten_space(observation_space["action"]),
            net_arch["action"],
            activation_fn,
        )

    def extract(self, observations):
        observations = dict(observations)
        observations["action"] = observations["action"].flatten(start_dim=-2)
        return super().extract(observations)
```

环境侧要在 `observation_space` 中声明 `action`，并在 `get_observation()` 中返回动作历史。下面演示初始化和读取历史。要实际使用，还需在生成新观测前写入当前动作，并在回合重置时清空对应智能体的历史。

```python
import numpy as np
import torch as th
from gymnasium import spaces

from VisFly.envs.NavigationEnv import NavigationEnv


class ObserveActionNavigationEnv(NavigationEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space["action"] = spaces.Box(
            low=-1, high=1, shape=(5, 4), dtype=np.float32
        )
        self._past_actions = th.zeros((self.num_envs, 5, 4), device=self.device)

    def get_observation(self, indices=None, predicted_obs=None):
        observations = super().get_observation(indices, predicted_obs)
        observations["action"] = self._past_actions
        return observations
```

实例化 PPO 时，在 `features_extractor_class` 中使用这个类，并在特征提取器的 `net_arch` 中加入 `"action": {"layer": [64]}`。如果通过 YAML 配置自定义类，参照示例入口的 `EXTRACTOR_ALIAS` 注册类名。[配置加载示例](exps/examples/run.py)

</details>

<details>
<summary>创建场景</summary>

示例场景使用 ReplicaCAD 的物体搭建。要创建自己的场景，可以参考 Habitat-Sim 的 JSON 属性配置格式，以及示例数据集 `configs/scenes/` 下的场景实例。[Habitat-Sim 场景配置文档](https://aihabitat.org/docs/habitat-sim/attributesJSON.html)

</details>

<details>
<summary>分阶段训练</summary>

1. **先练基本飞行。** 在简单场景里让无人机接近目标并保持姿态稳定，可以从上面的自定义奖励开始。
2. **再练避障。** 将 `env.scene_kwargs.path` 切换到有障碍的场景，例如 `VisFly/datasets/visfly-beta/configs/scenes/garage_simple_l_medium`。比较两阶段的观测和动作设置。奖励可加入朝向目标的速度和靠近障碍物的惩罚。[导航任务奖励](envs/NavigationEnv.py)

两阶段的模型结构、观测和动作空间兼容时，可以在训练命令中用 `-w` 加载第一阶段的模型。当前入口会复用检查点里的策略和保存路径。第二阶段需要单独保存时，可以在 Python 中用 `PPO.load()` 加载、`learn()` 续训、`model.save(path=...)` 输出到新路径。若新增视觉输入导致网络结构改变，需要在自定义训练代码中迁移可复用的权重。[PPO 实现](utils/algorithms/PPO.py)

</details>

## 论文与相关项目

VisFly 基于 Habitat-Sim 渲染 RGB 和深度图像。论文报告在 NVIDIA RTX 4090 的基准设置下，`64×64` RGB 观测的渲染吞吐量约为 `10⁴ Hz`。[VisFly 论文](https://arxiv.org/abs/2407.14783)

- 四旋翼控制的一阶强化学习可微框架。[VisFly-Lab](https://github.com/Fanxing-LI/APG)
- 通过可微仿真学习稳定的目标跟踪策略。[StableTracker](https://github.com/Fanxing-LI/obj_track)
- 通过高保真可微仿真实现稳定、快速、安全的端到端控制。[Simple but Stable, Fast and Safe](https://github.com/Fanxing-LI/avoidance)
- 面向随机障碍环境的四旋翼竞速课程强化学习。[Curriculum Reinforcement Learning for Quadrotor Racing with Random Obstacles](https://github.com/SJTU-ViSYS-team/CRL-Drone-Racing)

如果你在研究中使用了 VisFly，欢迎引用下面的论文。

```bibtex
@misc{li2024visflyefficientversatilesimulator,
    title={VisFly: An Efficient and Versatile Simulator for Training Vision-based Flight},
    author={Fanxing Li and Fangyu Sun and Tianbao Zhang and Danping Zou},
    year={2024},
    eprint={2407.14783},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2407.14783},
}
```

VisFly 仍在开发，公开接口和示例可能调整。遇到问题可以通过 GitHub Issues 反馈，也可以通过邮件联系维护者。[GitHub Issues](https://github.com/SJTU-ViSYS-team/VisFly/issues)
