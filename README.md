# SWM

SWM（Symbolic World Model）用于从机器人操作视频或场景图像中生成可验证的 PDDL 规划数据，并导出视觉语言模型训练样本。

```text
视频/图像 + 任务指令 → Temporal-Gradient Keyframe Extraction → Keyframe-based Action Sequence Extraction → PDDL 生成 → Fast Downward 求解 → 一致性评估
```

## 安装

```bash
conda create -n swm python=3.11 -y
conda activate swm
pip install -e .
```

关键帧提取需要 FFmpeg：

```bash
sudo apt install ffmpeg
```

PDDL 求解需要将 Fast Downward 安装到仓库根目录：

```bash
git clone https://github.com/aibasel/downward.git downward
python downward/build.py
```

## API 配置

在根目录创建 `.env`：

```dotenv
A6_API_KEY=...
BOYUE_API_KEY=...
QWEN_API_KEY=...
```

## 数据目录

```text
dataset/videos/<task_domain>/<episode_id>.mp4
tasks/images/<task_domain>/<task_id>/<episode_id>.png
tasks/instructions/instructions_<task_domain>.json
tasks/steps/steps_<task_domain>.json
```

如已获得数据压缩包，可执行：

```bash
tar -xzf dataset/data/swm_data_json.tar.gz -C dataset/data/
```

数据、模型权重、缓存和实验结果均已通过 `.gitignore` 排除，不会提交到仓库。

## 运行

先修改 `scripts/domain_generation.py` 文件顶部的配置。`STEP_SOURCE = "video"` 会依次执行 Temporal-Gradient Keyframe Extraction 和 Keyframe-based Action Sequence Extraction；`STEP_SOURCE = "steps_json"` 会直接读取 `tasks/steps` 中的步骤和 `tasks/images` 中的场景图。然后运行：

```bash
python scripts/domain_generation.py
```

结果保存在：

```text
eval_results/<model>/<task_domain>/<task_id>/<episode_id>/
```

视频分支把动作序列保存为 `kf_actions.txt`；其中 `[G#]` 仅记录动作来源的关键帧组，PDDL 生成、检索和语义评估在读取时会去掉该索引。

常用脚本：

- `scripts/steps_generation.py`：生成原子动作序列
- `scripts/instruction_aug_generation.py`：生成增强指令和元数据
- `scripts/sharegpt.py`：导出多模态 ShareGPT 数据
- `scripts/keyframe/eval_keyframe.py`、`scripts/eval_planning.py`：评测关键帧和 PDDL 规划

部分脚本包含本地路径和模型配置，运行前请按实际环境修改。

## License

[MIT License](LICENSE)
