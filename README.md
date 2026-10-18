# SWM

SWM（Symbolic World Model）致力于构建连接真实世界感知与机器人可验证行动的符号智能底座：从大规模机器人操作演示中提炼视觉状态、动作与因果关系，构建 PDDL 规划训练数据，支撑视觉语言模型将场景图像和自然语言任务编译为可求解的符号世界模型，最终由规划器生成可验证的高层机器人行动方案。

```text
视频 / 场景图像 + 任务指令
          -> 动作序列
          -> PDDL domain/problem
          -> Fast Downward plan
          -> 计划回放与 Judge 验证
```

## 安装

要求 Python `>=3.9`，推荐 Python 3.11：

```bash
conda create -n swm python=3.11 -y
conda activate swm
pip install -e .
```

视频处理需要 FFmpeg：

```bash
sudo apt install ffmpeg
```

PDDL 求解器需要安装到仓库根目录：

```bash
git clone https://github.com/aibasel/downward.git downward
python downward/build.py
```

## 配置

在仓库根目录创建 `.env`：

```dotenv
A6_API_KEY=...
BOYUE_API_KEY=...
SII_API_KEY_1=...
```

模型服务由模型名前缀自动选择：

- `gemini*`、`gpt*`：使用 `A6_API_KEY`
- `kimi-k3*`、`qwen3.7-*`、`glm-5.2*`：使用 `BOYUE_API_KEY`
- `Qwen3.8-27B*`：使用 `SII_API_KEY_1`
- 其他模型：使用本地服务 `http://127.0.0.1:8001/v1`

## 数据

```text
dataset/videos/<domain>/<episode>.mp4
tasks/images/<domain>/<task>/<episode>.png
tasks/instructions/instructions_<domain>.json
tasks/steps/steps_<domain>.json
```

视频运行时会自动生成：

```text
dataset/frames/<domain>/<episode>/
dataset/keyframes/<domain>/<task>/<episode>/seg_*/
```

## 主流程

编辑 `scripts/domain_generation.py` 顶部配置：

```python
STEP_SOURCE = "video"          # "video" 或 "steps_json"
TASK_DOMAIN = "droid"
PDDL_MODEL = "gpt-5.6-sol"
ACTION_EXTRACTION_MODEL = "gemini-3.8-flash"
JUDGE_MODEL = "gemini-3.8-flash"
ROBOT_CONFIGURATION = "single-arm"
```

运行：

```bash
python scripts/domain_generation.py
```

- `video`：抽帧、提取 temporal-gradient 关键帧、生成动作序列，再生成 PDDL。
- `steps_json`：直接读取 `tasks/steps` 和场景图像，跳过视频处理。

视频动作提取使用逐组 Generator 和一个全段 Compiler：G320 对照图加单帧用于观察，
C320 横/竖屏固定 3×3 拼图用于全段整理。结果每行包含动作及 Before/After，默认保存
`action_traces/` 和实际附图。调用沿用非流式 `call_gpt`，使用 medium、temperature=0；
传输、JSON 或引用契约失败按同一完整范围重试。PDDL 生成接收完整动作和状态；
PFV、PSV 的参考输入则去掉 `[G#]`、Before/After，仅保留从 1 开始编号的动作。

结果保存在：

```text
eval_results/<model>/<domain>/<task>/<episode>/
  kf_actions.txt
  round1/
    domain.pddl
    problem.pddl
    plan.txt
    judge.json
  round2/
  round3/
```

每个 episode 最多尝试 3 轮。solver 或 Judge 失败时会将反馈传给下一轮；已通过验证的 episode 会自动跳过。PDDL 生成后会进行类型检查、Fast Downward 求解、grounded plan replay 和 goal 验证。

## 常用脚本

| 脚本 | 用途 |
| --- | --- |
| `scripts/steps_generation.py` | 根据单张图像和指令生成原子动作步骤 |
| `scripts/instruction_aug_generation.py` | 从中间关键帧生成增强指令、步骤和元数据 |
| `scripts/merge.py` | 合并各 episode 的 operator，生成 unified domain |
| `scripts/sharegpt.py` | 导出通过验证的多模态 ShareGPT 数据 |
| `scripts/eval_planning.py` | 评测模型生成的 PDDL 或自然语言 plan |
| `scripts/keyframe/eval_keyframe.py` | 评测关键帧动作序列 |
| `scripts/clean_operator/validate_operator_cleanup.py` | 验证 operator 清洗结果 |
| `scripts/clean_operator/audit_cleaned_operator_contracts.py` | 审计 operator contract 一致性 |

`scripts/eval_planning.py` 的 PDDL 模式默认先保存原始评测报告，再静态补全
缺失的谓词、对象和动作参数，对有改动的任务重新求解、Judge 和归因。
修复文件、改动记录 `postprocess.json` 和第二份报告保存到评测目录的
`postprocessed/` 下；未改动任务复用原结果，两份报告的任务总数一致。
将脚本顶部的 `rule_postprocess` 设为 `False` 可关闭此阶段。

PDDL 评测还会通过 `swm.simulator.verify_cross_domain_files` 使用 GT PDDL
回放候选计划，结果单独写入 `simulator.json`；`judge.json` 保持原样。
模拟器保留 `PASS`、`FAIL`、`UNKNOWN` 三态，缺少 GT 或无效输入记为
`INPUT_ERROR`。PDDL 评测默认使用任务图像和 VLM 映射不同名称的物体，
需要配置 `BOYUE_API_KEY`；映射服务或图像不可用时会明确报告输入错误，
不会退回无映射模式。

脚本大多采用“修改文件顶部配置后直接运行”的方式。例如：

```bash
python scripts/steps_generation.py
python scripts/instruction_aug_generation.py
python scripts/sharegpt.py
```

ShareGPT 默认输出到：

```text
eval_results/<model>/data/<domain1>_<domain2>.json
```

## 测试

```bash
pytest
```

## 详细文档

- [项目规格](docs/PROJECT_SPEC.md)
- [关键帧方法说明](docs/KEYFRAME_EXTRACTION_THEORY.md)
- [Operator 清洗手册](docs/OPERATOR_CLEANUP_PLAYBOOK.md)
- [主实验测试集设计](docs/experience.md)

## License

[MIT License](LICENSE)
