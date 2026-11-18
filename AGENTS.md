# Repository Guidelines

## 项目结构与模块组织

- `envs/` 定义飞行任务及 Gym/VecEnv 封装；`envs/base/` 包含动力学模型和控制器。
- `utils/SceneManager.py` 对接 Habitat-Sim；`utils/algorithms/` 存放学习算法，`utils/policies/` 存放网络，`utils/evaluate.py` 负责评估和视频导出。
- `exps/examples/run.py` 是示例入口。同目录下的 `env_cfgs/` 存放环境 YAML，`alg_cfgs/<task>/` 存放算法配置。
- `tests/` 存放回归测试，`debug/` 存放探索性调试脚本。场景资产位于 `datasets/visfly-beta/`；`utils/FigFashion/` 是 Git 子模块。

## 构建、测试与开发命令

使用 `visfly` Conda 环境及 `environment.yml` 中固定版本的依赖；在本地副本中调整与机器相关的 `prefix`。按照 `readme.md` 单独构建所需的修改版 `Fanxing-LI/habitat-sim`。在该项目源码目录中，带桌面窗口的构建命令为：

```bash
HEADLESS=False python setup.py build_ext --parallel 2 install --bullet --cmake
```

以下命令从 `VisFly/` 的父目录执行，确保包导入和资产相对路径正确：

```bash
conda activate visfly
python -m unittest discover -s VisFly/tests -p 'test_*.py' -v
python VisFly/exps/examples/run.py -t 1 -e crossing
python VisFly/exps/examples/run.py -t 0 -e crossing -w CHECKPOINT.zip
```

训练步数由任务 YAML 决定，运行可能较长；调试时先限制训练步数。模型检查点和日志保存在 `exps/examples/saved/<task>/`。

## 代码风格与命名约定

使用四个空格缩进；函数和变量采用 `snake_case`，类名采用 `CamelCase`。保留现有文件名，遵循邻近代码的风格。明确张量形状、数据类型、设备和坐标转换。仓库尚未统一配置格式化或静态检查工具；使用 `git diff --check` 检查补丁中的空白问题。

## 测试规范

使用标准库 `unittest`，测试文件命名为 `test_*.py`。修复行为问题时，添加针对性的回归用例。完整集成测试需要下载场景资产、安装修改版渲染器，并具备 GPU EGL 渲染环境；特征提取器和日志测试也提供 CPU 检查。当前未设置最低覆盖率门槛。多机相关改动须验证终止观测（terminal observation）、回合同步结束、场景间复位隔离和渲染一致性。

## 提交与拉取请求规范

历史提交采用简短的祈使句标题，例如 `fix relative path`、`add collision refinement`。每次提交聚焦一个主题。拉取请求（PR）应说明触发条件、修改后的行为、验证命令及关联问题。渲染改动附截图或视频，训练结果注明随机种子和配置。凭据、数据集、模型检查点及生成的媒体文件不要纳入提交。
