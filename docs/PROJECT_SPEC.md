# Project Specification

## 1. 项目概述

本项目面向机器人任务规划，目标是从机器人操作视频中自动构建多模态 PDDL 规划训练数据，并使用这些数据对视觉语言模型进行监督微调。

最终希望训练一个视觉语言模型，使其能够根据当前场景图像和自然语言任务指令，直接生成：

- PDDL domain
- PDDL problem

生成的 PDDL domain 和 problem 应当具有以下特点：

- 与输入图像中的场景状态一致；
- 与自然语言任务指令一致；
- 逻辑完整；
- 能够被 PDDL solver 正确求解；
- 求解得到的 plan 能够合理完成机器人操作任务。

模型输出的 PDDL 将交给 PDDL solver 求解，最终得到机器人可以执行的高层任务规划。

---

## 2. 模型输入与输出

### 2.1 输入

模型输入包括：

- 当前机器人操作场景的图像；
- 描述目标任务的自然语言指令。

### 2.2 输出

模型输出包括：

- PDDL domain；
- PDDL problem。

### 2.3 使用流程

```text
场景图像 + 任务指令
          ↓
         VLM
          ↓
PDDL domain + PDDL problem
          ↓
     PDDL solver
          ↓
机器人高层任务规划 plan
```

---

## 3. 完整数据生成流程

项目的数据生成流程主要包括以下阶段：

```text
机器人操作视频
        ↓
Temporal-Gradient Keyframe Extraction
        ↓
Keyframe-based Action Sequence Extraction
        ↓
PDDL domain 和 problem 生成
        ↓
PDDL solver 求解
        ↓
grounded symbolic trace 构建
        ↓
reference-backed 程序检查与多模态 Judge
        ↓
中间关键帧数据增强
        ↓
任务指令增强
        ↓
ShareGPT 格式训练数据
        ↓
视觉语言模型 SFT
```

---

## 4. Temporal-Gradient Keyframe Extraction

输入机器人操作视频后，首先提取视频中的所有帧。

随后依次执行：

1. 将图像转换为灰度图；
2. 用中心帧两侧图像计算对称时序梯度 $D_t=(Y_{t+1}-Y_{t-1})/2$，首尾复制最近端点；
3. 对每帧的梯度平方和形成能量曲线，并进行 5 帧移动平均；
4. 在随视频长度调整的局部窗口内同时提取峰值和谷值；
5. 对幅值相近的相邻极值只保留偏离全局中位数更远者；
6. 将保留的极值组织成峰—谷—峰证据组，并确保首尾帧被覆盖。

相邻证据组共享边界峰值，以便后续模型连续观察动作前后状态。

这一方法主要基于以下直觉：

1. 机械臂的运动速度和图像内容的变化速度通常具有较强相关性；
2. 高能量峰更可能包含明显的视觉转换；
3. 低能量谷更可能提供清晰的状态证据；
4. 峰—谷—峰结构能够同时保留动态变化与稳定状态。

通过这种方法，一个完整视频可以被划分为多个关键帧组。

关键帧组是动作理解的视觉证据窗，不预先宣称每组严格等于一个原子动作。

---

## 5. Keyframe-based Action Sequence Extraction

完成关键帧分组后，系统按照时间顺序处理每个关键帧组，并提取整个视频的任务级自然语言动作序列。

### 5.1 局部证据生成与状态校验

Generator 对每个关键帧组读取：

- 当前组中的全部关键帧；
- 原始任务指令；
- 已接受的历史动作；
- 当前组失败重试时的校验反馈。

Generator 先描述每对相邻关键帧的可见变化，再输出当前组内已经完成的原子任务动作及其手部状态变化。程序只维护 `holding(hand,obj)` 和 `hand_free(hand)`，拒绝不一致的获取、释放或换手，并最多重试当前组两次。

### 5.2 全局动作序列编译

全部关键帧组通过局部校验后，Compiler 读取完整 evidence ledger 和视频首帧，检查动作覆盖、重复、对象、来源、目标、空间关系与时间顺序，并仅通过必要的插入、替换或删除生成最终动作序列。

最终，一个完整的机器人操作视频会被转换成自然语言动作序列，例如：

```text
[G0] Pick up the yellow block from the blue block.
[G1] Place the yellow block into the red bowl.
```

该产物统一保存为 `kf_actions.txt`；`[G#]` 只记录动作来源的关键帧组，读取后会先去除该索引再传给 PDDL 生成和评估模块。

---

## 6. PDDL Domain 和 Problem 生成

得到关键帧动作序列后，将关键帧动作序列与原始任务指令组合，形成增强后的任务指令。

随后，将以下内容输入视觉语言模型：

- 视频的初始状态图像；
- 增强后的任务指令。

模型需要生成完成任务所需的：

- PDDL domain；
- PDDL problem。

### 6.1 PDDL Domain

PDDL domain 用于描述任务规划中的通用规则，包括：

- 对象类型；
- 谓词；
- 动作 operator；
- 动作参数；
- 动作前置条件；
- 动作执行效果。

Domain 使用经典 typed PDDL：在 `:requirements` 中声明 `:typing`，通过
`:types` 声明浅层类型体系，并为 predicate 参数和 action 参数标注类型。
稳定对象类别由类型表达，不再同时作为一元类别 predicate 写入状态。

`open`、`closed`、`upright`、`clear`、`hand_free` 等状态，以及不能由稳定
类别完整替代的能力或位置角色，仍保留为 predicate。共享关系可使用内置
`object` 或有依据的共同父类型，避免把原本合法的对象绑定人为收窄。

类型声明遵循 PDDL 分组语义，例如 `?a ?b - block`；未显式标注的符号默认为
`object`。仅在任务语义确有需要时使用浅层继承。

### 6.2 PDDL Problem

PDDL problem 用于描述当前具体任务，包括：

- 当前任务中的对象实例；
- 场景初始状态；
- 任务目标状态。

`:objects` 为每个实例声明兼容类型。`:init` 只包含状态、角色和关系事实，
不重复写入已由对象类型完整表达的类别事实。

---

## 7. PDDL Solver 求解

生成 PDDL domain 和 problem 后，使用 PDDL solver 进行求解。

如果 domain 和 problem 的语法正确、类型引用及继承一致、动作实参与参数类型
兼容，并且任务可解，solver 会输出一个 PDDL 格式的动作计划。

例如：

```text
(move-to yellow-block)
(pick yellow-block blue-block)
(move-to red-bowl)
(place yellow-block red-bowl)
```

该动作计划表示机器人为完成当前任务需要依次执行的高层动作。

---

## 8. Grounded Symbolic Trace 构建

Solver 成功生成 `plan.txt` 后，系统结合当前 PDDL domain 和 problem 的类型声明，
将计划中的动作实例化为 grounded symbolic trace。实例化会检查对象是否已声明、
实际类型是否为参数类型的合法子类型，并拒绝跨类别绑定。Trace 主要保留每个动作
的对象参数、参数类型、关键前置条件和状态变化，使 Judge 能够直接检查计划的因果
过程，而不再依赖 operator 注释将 plan 转换成自然语言。

---

## 9. Hybrid Contract Judge 与闭环修正

Judge 用于判断 solver plan 是否可执行、是否完整满足原始任务指令，以及是否与视频提取出的动作证据保持合理一致。它不要求 Candidate plan 逐动作复制演示，而允许因果上等价的动作表达和顺序。

### 9.1 输入与判定流程

Judge 的主要输入包括：

- 当前任务的起始场景图像；
- 原始任务指令；
- Candidate grounded symbolic trace；
- 从关键帧中提取的动作序列；
- 当前轮 PDDL，以及可用时的 GT/reference PDDL。

系统优先执行程序化符号检查，包括验证计划能否达到目标、检查对象和动作参数的
类型兼容、检查明确的初始状态或任务契约冲突，以及判断 Candidate 是否可由
reference 计划等价解释。对象角色同时读取 declared type 和仍保留的一元角色
predicate；typed Candidate 与旧式一元类别 reference 可以通过统一的内部表示比较，
但声明类型不会被当成动作效果或独立视觉证据。能够获得明确结论时直接通过或拒绝；
无法确定时，再调用多模态 Judge 综合图像、指令、Candidate trace 和关键帧动作证据
进行判断。

多模态 Judge 重点检查：

1. 指令要求的对象、动作、来源、目标和最终状态是否完整覆盖；
2. 每个动作的前置条件是否由初始场景或之前的状态变化支持；
3. 是否存在缺失动作、多余动作、错误顺序或错误状态转移；
4. 计划是否在任务完成后留下不应持续运行的设备或过程。

关键帧动作序列只作为观测证据，不会被直接拼接到 Candidate plan 中。Judge 也不会仅根据不确定的视觉位置、抓取方式或手部偏好拒绝计划。

### 9.2 输出与反馈闭环

Judge 统一输出：

```json
{
  "reasoning": "判定理由",
  "pass": true,
  "feedback": ""
}
```

通过时，当前轮结果保存到 `judge.json` 并进入后续训练数据候选集；拒绝时，`feedback` 与上一轮 domain、problem 和 plan 一起反馈给 PDDL 生成模型。

每个 episode 最多进行 3 轮 PDDL 生成：

- solver 解析或求解失败时，根据 solver feedback 修正；
- solver 成功但 Judge 拒绝时，根据 Judge feedback 修正；
- 任意一轮 Judge 通过即停止，后续运行也会跳过已有通过记录的 episode。

```text
生成 PDDL
    ↓
PDDL solver 求解
    ↓
构建 grounded symbolic trace
    ↓
程序化检查 → 必要时多模态 Judge
    ↓
是否通过？
 ┌───────┴───────┐
通过             不通过
 ↓                  ↓
保存候选样本     生成 feedback
                    ↓
                重新生成 PDDL
```

Judge 提供的是面向数据筛选的任务一致性检查。其中符号检查可以验证生成模型内部的逻辑关系，多模态判定仍属于经验性判断，不代表对真实物理可执行性的形式证明。

---

## 10. 训练样本格式

每个通过验证的训练样本包含以下内容。

### 10.1 输入

- 当前场景图像；
- 自然语言任务指令。

### 10.2 输出

- PDDL domain；
- PDDL problem。

模型通过这些数据学习以下映射：

```text
当前场景图像 + 自然语言任务指令
                  ↓
                 VLM
                  ↓
       PDDL domain + PDDL problem
```

---

## 11. 中间状态数据增强

项目不仅使用视频首帧构建训练样本，还会利用视频执行过程中的中间关键帧构建新的训练样本。

对于某个关键帧组 `Gk`，执行以下操作：

1. 取该关键帧组的第一张图像作为新的初始状态图像；
2. 获取从该状态开始仍需执行的动作序列；
3. 根据剩余动作重新生成新的自然语言任务指令；
4. 使用新的初始图像、新任务指令和剩余动作生成对应的 PDDL domain 和 problem；
5. 对新生成的 PDDL 再次执行 solver 求解和一致性评估。

例如，原始关键帧动作序列为：

```text
[G3] Pick up the yellow block from the blue block.
[G4] Place the yellow block into the red bowl.
[G5] Move the gripper away from the red bowl.
```

可以使用 `G3` 的第一张图像作为新的初始状态图像，并根据后续任务重新生成任务指令：

```text
Pick up the yellow block and place it into the red bowl.
```

这样可以从同一个视频中的不同中间状态出发，构建多个训练样本。

该方法能够更加充分地利用机器人操作视频中的中间状态，而不仅仅使用视频首帧。

---

## 12. 指令增强

为了提高训练数据中的语言多样性，项目会对任务指令进行增强。

在保持原始任务语义不变的前提下，将一条任务指令改写为多种不同表达形式，例如：

- 简洁的命令式表达；
- 更详细的任务描述；
- 不同句式；
- 不同语言风格；
- 不同动作描述粒度；
- 不同词汇表达。

通过指令增强，可以减少模型对固定指令模板的依赖，提高模型对不同自然语言表达的理解能力。

---

## 13. ShareGPT 格式训练数据构建

经过验证和增强的数据最终会被转换成适合监督微调的 ShareGPT 格式。

每个 ShareGPT 样本通常包含：

- 图像路径或图像信息；
- 用户输入的自然语言任务指令；
- 模型需要生成的 PDDL domain；
- 模型需要生成的 PDDL problem；
- 对话角色；
- 对话消息顺序。

典型的对话结构包括：

- system 消息；
- human 或 user 消息；
- assistant 消息。

其中：

- human 消息包含场景图像和任务指令；
- assistant 消息包含对应的 PDDL domain 和 PDDL problem。

这些 ShareGPT 格式数据将用于后续视觉语言模型的监督微调。

训练导出只读取已经完成类型一致性、重新求解、计划回放和任务等价验证的
source round。导出过程保留 domain 的 `:types`、predicate/action 参数类型和
problem 的 typed `:objects`，不会回写或删除源 PDDL；action、对象和类型符号保持
源计划接口不变，并在导出后再次执行类型兼容、计划回放和目标验证。

---

## 14. 当前数据规模

### 14.1 单臂数据

单臂机器人操作视频的主要来源包括：

- DROID：约 9,000 个视频；
- BridgeData V2：约 10,000 个视频；
- 人类自拍视频：约 240 个视频。

经过关键帧图像裁剪和中间状态增强后，当前获得约：

- 3,000 个样本。

经过指令风格增强后，数据规模扩展至约：

- 56,000 个样本。

### 14.2 双臂数据

当前双臂机器人数据主要来自 AgiBot：

- AgiBot：约 1,000 个视频。

经过关键帧图像裁剪和中间状态增强后，当前获得约：

- 25,000 个样本。

---

## 15. 模型训练

项目计划使用上述训练数据对 Qwen3-VL-8B 进行监督微调。

训练输入包括：

- 当前机器人场景图像；
- 自然语言任务指令。

训练输出包括：

- PDDL domain；
- PDDL problem。

模型需要学习以下能力：

1. 从图像中识别物体；
2. 理解物体的位置、状态和空间关系；
3. 理解自然语言任务目标；
4. 构造合理的 PDDL 对象类型；
5. 构造合理的谓词；
6. 生成动作 operator；
7. 生成正确的动作前置条件；
8. 生成正确的动作执行效果；
9. 构造与图像一致的初始状态；
10. 构造能够表达任务目标的 goal；
11. 保证生成的 domain 和 problem 可以被 solver 求解；
12. 保证 solver 求解结果与机器人真实任务过程保持一致。

---

## 16. 最终目标

本项目最终希望建立一个面向机器人 task planning 的视觉语言模型。

该模型能够完成以下过程：

```text
观察当前机器人场景
          ↓
理解自然语言任务指令
          ↓
生成 PDDL domain 和 problem
          ↓
使用 PDDL solver 求解
          ↓
获得机器人高层任务规划
```

最终系统应能够根据机器人当前看到的场景和用户给出的自然语言指令，自动生成逻辑一致、可以求解并且能够执行的高层任务规划。
