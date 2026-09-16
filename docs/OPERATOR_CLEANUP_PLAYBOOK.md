# PDDL Operator 跨数据集清洗手册

## 0. 手册的职责

本手册定义 PDDL 数据清洗的**判定、实施和验收协议**。它服务于 Human、DROID、
AgiBot、BridgeData 及同类数据集，但不要求它们使用相同的对象类别、目录布局或动作粒度。

清洗的目标是得到一个最小、可执行、可追溯的世界模型：

1. operator 的参数、前置条件和效果只表达直接因果；
2. predicate 的名称、arity、参数方向和生命周期一致；
3. 新 plan 完成 instruction 与参考轨迹所要求的同一任务，而不是机械复刻其偶然顺序；
4. domain、problem、plan、plan_nl 和聚合产物始终来自同一版本的合同；
5. 每项变更可 staging、可回放、可重求解、可复审，第二遍没有新增变更。

本手册不是：

- 把所有数据强制改成 `unified_domian.pddl` 的本体；
- 用更短 plan 替代任务要求的方法和安全边界；
- 按 action/predicate 名称做仓库级字符串替换的许可；
- 用统一域、统计频次或模型猜测代替 episode 证据的规则集。

`unified_domian.pddl`、各数据集的 `unified_domain.pddl` 和 source mapping 只用于检索、
聚类、审计与反查。episode 的 `domain.pddl`、`problem.pddl`、instruction、`kf_actions`、
关键帧和相邻动作链才是局部合同的事实来源。

---

## 1. 不可违反的总原则

### 1.1 最小因果，不是最小文本

每个 literal 都必须能回答一个问题：它是否影响该动作的可执行性，或是否由该动作直接造成？

- 不是使能条件、直接结果、资源状态或后续消费者所需的 scene snapshot 不应进入合同；
- 不能为复刻演示顺序引入 `ready_for_*`、`passed_once`、`phase_*` 等进度标记；
- 但当最终物理状态回到初始状态，而 GT 明确要求一次真实过程时，带有明确
  producer/consumer 的历史状态可以保留。例如 `spoon_extracted` 可防止“未抽匙便直接放回”的空计划；
- 参数减少只有在动作可适用范围和直接状态转换均不变时才合法；不能为“简洁”删掉真实载荷、
  control、接收器或双手角色。

### 1.2 任务等价，不是逐行相等

新 plan 可以与 `kf_actions` 使用不同的独立步骤顺序、不同的同义 action 名，或包含被修复后
必要的完成边界；但必须保持：

- instruction 要求的对象、数量、工具、目标关系和最终结果；
- 有限物体、工具和内容物的真实流转；
- GT 明示的必要过程、安全条件和不可交换的因果边界；
- 所有由新 domain 声明的资源、互斥和关系生命周期。

若交换两个动作不破坏任何事实，planner 应能交换它们。不要把 `kf_actions` 的时间相邻误建模为
世界因果。反之，若交换会破坏安全、资源、载荷或结果，必须由 precondition/effect 阻止。

### 1.3 证据有分工，不存在万能优先级

| 证据 | 主要回答的问题 |
|---|---|
| instruction | 最终任务意图、数量、工具、结果和明确顺序要求 |
| `kf_actions` / `kf_actions` | 已标注动作边界、关键过程和必要里程碑 |
| 关键帧/视频 | 可观察的物体、手、位置、门/开关状态和实际操纵方式 |
| domain/problem | 当前 symbolic 声称、可表达的 ontology、producer/consumer |
| 同数据集已验证 episode | 目标 ontology 内稳定、可复用的合同 |
| unified domain/source mapping | 候选合同及其真实 source 的索引 |

任何结论都要把证据和它支持的命题对应起来。名称相似、频率高、统一域已有同名 action，
都只能召回候选，不能证明合同正确。

---

## 2. 先建立清洗档案，再碰 PDDL

每个数据集或子集必须有一份可复现的清洗档案。档案未完成时，禁止进入批量 apply。

| 字段 | 必须明确的内容 |
|---|---|
| source of truth | PDDL 是 episode 文件、JSON 字段还是生成产物；哪一份允许修改 |
| identity | task、episode、round、JSON row 如何唯一定位 |
| scope | 当前选中哪些数据根、任务、episode 与 round；哪些明确排除 |
| round policy | 最大有效 round 的定义；缺 `domain/problem/plan/kf_actions` 时的处理 |
| artifacts | domain、problem、plan、plan_nl、kf_actions、group、图像、日志的位置 |
| PDDL profile | `:requirements`、typed/unary 类型风格、equality 支持、使用的 solver |
| equivalence oracle | 任务等价的 instruction、kf、图像与人工复核准则 |
| aggregate outputs | 需要重建的 unified domain、source mapping、catalog、JSON 字段 |
| exclusion policy | 缺文件、解析失败、证据冲突、不可唯一绑定时怎样隔离 |

### 2.1 选择 round 的纪律

默认只处理每个 episode 的最大**完整且有效** round。有效性至少要求所需的
`domain.pddl`、`problem.pddl`、`plan.txt` 和任务等价参考存在；`plan_nl.txt` 是可再生
产物，缺失不能悄悄回退到旧 round。

- 更大的 round 缺关键文件时，记录 `LATEST_ROUND_INCOMPLETE`；不要私自改旧 round；
- 历史 round、关键帧、日志、元数据和用户未选中的子集一律保持不动；
- round 选择与 reference 真伪是两件事：仅当第 3 节裁决确认错误，才可修正**本次选中有效
  round** 所属的 reference；必须保留原文/原 hash、修正后 hash、证据和受影响范围。历史或
  未选中 round 的 reference 仍保持不动，不能把修 reference 当作让 PDDL 通过的快捷方式；
- 是否新建 round 由本次 source manifest 和上游工作流决定。不得既覆盖旧 round 又声称保留历史，
  也不得为了形式新建 round 而遗留失效的最新计划。

### 2.2 冻结基线

Dry-run 前保存 manifest、源文件 hash 和至少以下统计：可用/排除 episode、解析数、
可 replay 数、缺失产物、predicate `name+arity`、同名异合同、异名同合同、action 数、
unified/source mapping 摘要。清洗后必须解释这些值的变化。

大规模统计不可用“递归扫所有文件”的偶然结果代替 manifest。只枚举目标结构，例如
`task_*/episode_*/round*/domain.pddl`，并把缺失项显式写入报告。

### 2.3 Reference 权威性与有效计划表示

在比较计划或修改 reference 前，manifest 必须为当前数据集显式写明 reference 的权威性：
`HUMAN_AUTHORITATIVE`、`EVIDENCE_REVISABLE` 或 `UNKNOWN`。同一仓库的不同数据集可以不同；
不能从文件名、过去一次修复或 judge 的 `pass` 值推断。

- 统计 `plan.txt` 时只计有效 grounded action：忽略空行、`#`/`;` 注释和 Fast Downward 的
  `; cost = ...`；PDDL action 行以 `(` 开始。计数只用于召回，不能裁决语义正确性；
- count mismatch 必须分别记录为 `FORMAT_ONLY`、`SEMANTIC_EQUIVALENCE_REVIEW`、
  `REFERENCE_DEFECT`、`MODEL_DEFECT` 或 `MODEL_REVIEW_REQUIRED`，不能直接以数量覆盖 reference；
- reference 可修订时，只有 instruction、关键帧和可重放因果链共同指向唯一结论，才可更新
  本次选中 scope 内的 `kf_actions`/`kf_actions`。reference 权威时，PDDL 必须适配它；
- 任何 reference 改动都保存旧 hash、新 hash、裁决证据和重解后的等价说明。单次 solver 成功、
  更短 plan 或统一域中的相似动作都不足以构成 reference 改动证据。
- judge/VLM 输出是可复审证据，不是 reference 权威来源。reference、goal 或 plan 语义变化后，
  与旧语义绑定的 judge 不得继续充当当前结论；要么重新生成，要么写入带独立
  `judge_version` 的人工 supersession，并保留旧 judge 的 hash、模型、verdict 和裁决理由。
  不得把人工结论伪装成旧模型的重新判定。

---

## 3. 证据冲突与 reference 裁决

### 3.1 标准裁决流程

发现 instruction、`kf_actions`、图像和 PDDL 不一致时，按以下顺序处理：

1. 先读取完整 instruction，区分“启动设备”和“完成处理”；
2. 检查 `kf_actions`、`kf_actions` 的动作边界与末尾状态；
3. 检查首末关键帧及必要中间帧，确认可观察事实，不从不可见区域臆造状态；
4. 从 init 开始重放当前 plan，写出每个关键 predicate 的 producer、consumer 和 invalidator；
5. 判断错误属于 reference、PDDL 还是证据不足；
6. 仅在结论唯一时修复；否则记录 `MODEL_REVIEW_REQUIRED`，不进入自动 batch。

### 3.1.1 对象身份、姿态与端点误判

端点帧的“看不见”或计数变化只能召回审查，不能单独证明目标丢失、放错或任务失败。对同一
有限物体至少沿 `初始 source -> gripper/hand -> target relation -> release/retract` 追踪；只有
中间关键帧、手的状态和最终关系共同支持时，才可确认放置成功或失败。被遮挡、画面边缘、相似
物体和计数误读都必须保留为不确定性，不能用一个 endpoint VLM verdict 覆盖该因果链。

`upright`、`flat`、`vertical` 等 pose literal 只有在 instruction 明示，或经标定的重力/容器参考
和连续图像共同证明时才能进入 reference、goal 或 effect。相机 roll、裁剪和图像坐标轴的“上下”
不是物理竖直证据；只有第 3 节裁决已确认它是无根据的额外 literal 时才能删除，姿态不能唯一
确定时应标为 `MODEL_REVIEW_REQUIRED`，而不是把它解释成执行失败。

### 3.2 三类结果

| 结论 | 处理 |
|---|---|
| `REFERENCE_DEFECT` | instruction、图像与可重放因果链共同证明 kf/goal 漏标或误标；仅在本次选中 scope 修正对应 reference，保留修正前证据后重解 PDDL |
| `MODEL_DEFECT` | reference 语义正确，PDDL 不能表达或允许绕过它；修 domain/problem/plan，保留 reference |
| `MODEL_REVIEW_REQUIRED` | 载荷、角色、完成边界或对象身份无法唯一确定；不猜测，不 apply |

不要把当前 PDDL 当作证明自身正确的证据，也不要假定 `kf_actions` 永远正确。reference 的每次改动
必须是有证据的独立事务。

### 3.3 已验证的微波炉缺失完成边界

一般情况下，`kf_actions` 以 start 结束且 goal 同时要求 `is_on` 与加热结果，必须先隔离，
不能自动拆分。但是下列所有条件同时成立时，可以认定 reference/goal 漏掉了完成边界：

1. 完整 instruction 的意图是“heat/warm/microwave 某物”，而非“press start”；
2. `kf_actions` 在 start 结束，最终帧只证明关门或手离开，不能证明设备持续运行；
3. start action 已把同一载荷的 `heated/warm/warmed` 错误作为直接 effect；
4. problem goal 同时要求这个结果和同一 microwave `is_on`；
5. 载荷可由 start action 与 plan 唯一绑定，且能构造同一载荷的 completion；
6. 每个 episode 都已人工审阅 instruction、group、关键帧与 PDDL。

满足时：goal 改为 `is_off`；`turn_on_microwave` 只做 off-to-on；
`turn_off_microwave` 以相同载荷和 `is_on` 为前提并提交真实结果；参考文本补入
“等待完成并关闭”；随后重求解、重放和第二遍审计。已人工确认的精确路径必须保存为 allowlist，
不得按 task 名、action 名或字符串 `heat` 扩张。

结果 predicate 由证据决定：水可为 `boiled`，食物通常为 `heated`，`warm`/`warmed` 只在本体
确有此区分时保留。不能把所有微波炉完成都改成 `boiled`。

---

## 4. Predicate 与 ontology 先于 operator

### 4.1 分类与一致性

先区分 type/object、role、configuration、availability、result、spatial relation 和其他 relation。
同一概念在 typed 和 unary type predicate 中可以跨数据集映射，但不得为“统一外观”改写目标本体。

所有批次必须满足：

1. predicate 的 `name+arity` 唯一；
2. 二元/多元 relation 的参数方向稳定；
3. alias 合并同时满足语义、arity、方向和生命周期等价；
4. 删除 declaration 前，domain/problem/goal/action 中没有任何残余引用；
5. 若使用 `(not (= ?x ?y))`，声明 `:equality` 与所需的 negative preconditions；
6. equality 在 parser、grounder 和 replay 中按内建相等关系求值，不能把 `=` 当普通 predicate；
7. 任何动态 predicate 都能找到 producer、consumer 与必要 invalidator；
8. 合法的零 action domain 不是解析失败：只有 `:init` 已满足 goal 时才接受零步计划，否则应报告
   无解，而不是为使工具通过凭空添加 action。

### 4.2 已确认的归一化边界

- `different`、`distinct_hands` 等仅表达参数不同时，统一为 `(not (= ?x ?y))`；必须同步
  declaration、init、goal、precondition/effect 和 requirements；
- `empty_hand`、`handempty`、`hand_empty` 只有作为单手 unary resource、并拥有完整
  acquire/release 生命周期时，才能归为 `hand_free(?hand)`；
- `both_hands_free(?robot)`、`hands_free(?robot)`、`separate_hands` 和 robot 级 holding
  不是 `hand_free(?hand)` 的词法别名。能恢复两只具体 hand 时使用两条 `hand_free` 加不等式；
  不能恢复时保留聚合 resource 或隔离；
- `boiled`、`boiling`、`heated`、`warm`、`warmed` 表达的温度/过程边界不同，不能因都与加热有关
  而合并；
- 单复数 object type、`toward/towards`、相近空间词和低频结果 predicate 都先做角色与生命周期审计，
  不能仅按拼写合并。

### 4.3 状态、历史与进度

不要按词缀删除 `ready_*`、`phase_*`、`passed_*`。先问：它是否在 goal 中、是否防止空计划、
是否有真实 producer/consumer，是否表达不可由物理关系替代的重复次数？

- 仅强迫演示顺序、且不在 goal 中的 annotation-only 进度标记可移除；
- 真实完成状态如 `wiped`、`heated`、`folded_over` 不能被当作历史噪声删除；
- 必要历史状态必须随着真实动作建立、被后续动作消费，并在重新开始时正确失效；
- 若物理关系（例如 `in drink glass`）已经证明过程结果，不再增加同义 `drink_poured`。

### 4.4 全局命名冲突与编号后缀

先做目标 latest-round manifest 的流式 `predicate name -> arity set` 审计，再改 operator。出现同名
不同 arity 时，不得删除其中一项或把所有出现点做词法替换；要为每个 `name+arity` 写出角色、方向、
producer/consumer 和生命周期，再给出可读的语义名，并同步 declaration、problem、goal、action 与
derived plan。若修复引入 `(not (= ?x ?y))`，同时验证 `:equality` 和 negative precondition profile。

source action 的数字后缀也不是可以直接剥离的噪声：它可能表示 first/second/third stroke、spread、
pass 或其他不可交换里程碑。仅在完整合同相同的情况下才能合并；合同不同则以该里程碑改为语义名，
并重解。反之，unified catalog 为区分同名异合同行为生成的 `_1`、`_2` 是**索引名**，不是 episode
source 的待清洗证据，不能反向写回。

---

## 5. Operator 合同的提取与匹配

### 5.1 先写目标合同

为每个候选 action 记录：actor/hand、单/双手模式、affected entity、source、target、tool/control、
required old state、enabling relation、add/delete effect、过程边界和每个非显然 literal 的证据。

参数名、literal 顺序和注释不属于合同；对象角色、predicate 极性、资源阶段、关系方向、
动作边界和直接 effect 属于合同。

### 5.2 参考检索顺序

1. 同一 episode 已验证 action；
2. 同数据集的 unified domain，并回查 provenance source；
3. 外部 unified reference；
4. 没有等价模板时，按本 episode 证据新建最小合同。

比较顺序必须是：持久 add/delete effect、hand/resource 阶段、affected/source/target/tool 角色、
关系、机械/载荷 guard、单/双手、类型、边界，最后才是名称。

### 5.3 五种结论

| 结论 | 含义 | 处理 |
|---|---|---|
| `KEEP` | 当前名称和完整合同已正确 | 不改 |
| `EXACT_REUSE` | 忽略变量和 literal 顺序后完全相同 | 复用合同与名称 |
| `ADAPTED_REUSE` | 转换骨架相同，仅有本 episode 证据支持的最小差异 | 记录 delta 后适配 |
| `NEW_OPERATOR_REQUIRED` | 没有等价局部转换 | 逐 episode 新建并验证 |
| `MODEL_REVIEW_REQUIRED` | 证据、绑定或 ontology 不足 | 隔离，不修改 |

编号后缀是聚合 catalog 内的局部签名编号，不代表全局身份、质量或优先级。不同合同不能共用
一个 action 名；不同数据集也不能照搬对方的编号。

### 5.4 Unified domain 的正确用途

unified domain 与 source mapping 是 provenance-preserving 的候选索引，不是待执行 episode 的真相。
它可用于按合同召回 source、发现 name/arity/资源异常和证明某合同并非孤例；每个命中仍必须回到
同一 episode 的 domain/problem、instruction、reference 和关键帧裁决。

重建聚合前后均要检查：每个 unified action 恰有一个 mapping entry、每个 mapping 指向存在的 source、
final action 名唯一、source predicate arity 和 source numeric action 名的审计结果可解释。聚合由单一
writer 执行，不能在同一输出路径并发 merge；报告须记录输入 manifest、输出 hash 和 source-mapping hash。
mapping 的 source 路径可以是绝对路径或相对路径；反查时必须先按清洗根规范化，再以完整路径匹配，
不能假定某一种字符串前缀。一次修复若使 source 合同与现有合同精确合并，unified action 数减少是
正常结果；报告必须说明数量 delta，并确认被改 episode 已进入对应 mapping entry。

---

## 6. 所有 operator 必须满足的因果不变量

### 6.1 Hand 资源守恒

| 边界 | 最小合同 |
|---|---|
| pick/acquire | `hand_free` -> 删除 free、加入 `holding`，并删除有限 source relation |
| place/release | `holding` -> 删除 holding、恢复 free、建立 target relation |
| press-and-hold | 持续占用 hand；release 后才恢复资源和提交有证据的结果 |
| ordinary toggle | 只要求可用 hand；不伪造 acquire/release 循环 |
| tool in container | 手不再持有 tool 时删除 holding、恢复 free；取出时反向维护 |

同一 hand 不能同时 free 与 holding/pressing，也不能无证据持有多个对象。两个 hand 参数只有在
同时承担独立物理角色时才要求不等式；若第二只 hand 只是不产生效果的冗余 free guard，经 GT
完整回放证明后可删除。不要把显式双手 lift、fold、hold 或 robot-level resource 错删为单手。

#### 6.1.1 同一夹爪的成组抓取与死合同

同一 hand 真正同时操纵两个独立物体时，不能写两个独立的 `holding(?h, ?o)`。在 instruction、
keyframe 和 action boundary 共同证明这个原子事件后，使用显式聚合资源（例如
`holding_pair(?h, ?o1, ?o2)`）：

- 从 free 获取一对时，要求 `hand_free`、两个不同对象及各自真实 source relation；删除 free 和有限
  source relation，加入一个 pair resource；
- 从已持一个对象升级为 pair 时，明确消费旧 `holding`、排除另一独立占用、要求对象不等；不得留下
  旧 holding 与 pair 并存；
- release 必须消费同一 pair resource、恢复 free，并建立每个对象的真实 target relation；嵌套物体
  的已证实 `on` 关系可保留，但不能伪造第二个 release；
- 若图像固定了特定一对，可在 problem 中加入最小的 pair binding 以阻止 solver 换成未观察组合；
  该 binding 不是所有 pair action 的通用进度 predicate；
- 不得用 `?o1 = ?o2` 伪造双件动作。plan 中的 grounded pair 必须绑定两个不同对象，且 domain 声明
  支持 equality/negative preconditions。

反过来，一个 action 若让同一 hand 加入两个独立 `holding`，但没有兼容的 release consumer、现有
plan 不使用它、reference/图像也未要求成组抓取，同时单件生命周期已能完成 instruction，则它是
`MODEL_DEFECT` 候选而非 pair-normalization 候选。删除前必须验证该 action 的局部 producer/consumer、
重新求解后保留所需对象数量和动作边界，并只删除经结构定位的 action/comment，绝不以 plan 改名掩盖。

### 6.2 有限转移、无限供给与关系生命周期

- 有限 contents 从 source 到 target 时删除 source relation、加入 target relation；
- faucet/dispenser 的持续能力使用 `dispenses(source, material)`，一次 fill 不得消耗它；
- `in`、`on`、`under`、`connected`、`inserted`、`blocks_*` 等关系要有建立者、消费者和离开时的
  invalidator；
- `open/closed`、`is_on/is_off`、互斥 pose 和有限位置不能并存；建立新状态时删除真实旧状态。

### 6.3 物理使能闭包

设备/过程/流动的 start 必须要求目标 ontology **已经表达且由该过程实际消费**的载荷、工件、接收器、槽位、连接或对位。
典型证据是 `(in item device)`、`(in carrier device)`、`(on item plate)+(in plate microwave)`、
`(under receiver outlet)`。

从实际 `is_off -> is_on` 的变量出发，检查 start、placement、completion 和 `:init`；不要仅按名字猜测。
例如 `(under ?receiver ?outlet)` 只在本 episode 已建模、receiver 与 flow 可唯一绑定，并由
start/completion 或后续消费者使用时才是硬 guard；它不是所有 faucet/dispenser action 的通用前提。
若 action 没有可绑定工件，instruction 只是“启动设备”，或本体没有可表达的 relation，则保留明确的
无载荷抽象，不能凭空添加参数或 predicate。无法区分时隔离。

### 6.4 开始、进行与完成

- start 只建立 `is_on/running/started/pressed`；
- completion、turn-off 或 release 在同一输入/接收器/工作位置仍成立时提交 `heated/washed/filled` 等结果；
- 只有数据明确把整个过程标为一个原子 macro，且证据支持该粒度时，才允许 start 直接带结果；
- 不要因为需要拆分而虚构未观察的抓取、盖子、传手或时间步骤；
- 也不要因为某个名字是 `turn_off_*` 就强行让它产出某一种结果。结果由处理对象和证据决定。

---

## 7. 高频 family 的已验证建模

### 7.1 开闭与 lid/cap

`open_*` / `close_*` 只转换 `closed <-> open`。它们不获得、持有、放置 lid/cap，也不隐含第二只手。

只有真实 `pick/remove/lift_lid`、`pick/remove_cap`、`place/replace_lid` 等 action 维护 lid/cap 的
holding、`on` 或位置。若 GT 把“放盖并闭合”定义为一个不可拆的事件，名称必须显式表达组合边界，
不能伪装为 pure `close_*`。没有可证明 detachable 机制时，不新增 lid predicate。

### 7.2 水壶、水龙头与分配器

`place_kettle_under_faucet` 表示对位，effect 只应建立 `(under kettle faucet)`；
它不是接水，不能改名为 `fill_kettle_from_faucet`。

`fill_kettle_from_faucet` 是 transfer：只要求本 episode 已建模且被该 transfer 消费的
holding/open/on/under/`dispenses` 等真实 guard，并把水或 `filled` 结果加入 kettle。水龙头作为
无限供给不能被当成 `(in water faucet)` 的有限库存；不能因为一个 family 有 `under`，就把它加到
所有接水合同。
receiver 移开时应删除 `under`。

同理，清洗时把 bowl/cup/plate 放在 faucet 或 spout 下与“完成冲洗/接水”是不同边界：
对位由 place 建立，flow start 消费对位，completion 在证据支持时提交水/冲洗结果。不要把每个
`under` 都删掉，也不要让 toggle 偷偷完成填充。

### 7.3 微波炉、烧水与其他设备

`turn_on_microwave` 的名称统一表达 start，但具体参数保留 source 已证明的 button、载荷和 carrier。
它只切换状态；完成 action 使用同一 microwave、同一载荷和真实载荷链提交结果。直接 item、
item-in-cup、item-on-plate/plate-in-microwave 是不同关系形态，不要为了大域“统一”丢失 carrier。

水壶、oven、washing machine、rice cooker、ice maker、capsule machine 和 dispenser 使用同一
开始/使能/完成分析框架，但不能相互套用具体 guard。特别是：

- kettle start 只在 water、closed、base/power 等已建模条件成立时启动；
- ice maker 的 water 是有限转移，不能套 faucet 的无限 `dispenses`；
- capsule/slot、inner pot/cooker、tray/oven 等需要检查完整载荷链，而非只检查外层设备；
- 无载荷的纯开机任务保持无载荷，不被“补 guard”。

### 7.4 搅拌：内容物是结果主体

命名和结果遵循 `stir_<actual_contents>_in_<container>_with_<tool>`。容器是工作位置，
`stirred` 必须落在被处理的 drink/water/soup/oatmeal/coffee 或已证明的 product 上，不能落在
shaker、cup、glass、bowl 或 pot。

`stir_drink_in_shaker_with_bar_spoon` 只在以下闭环成立时使用：唯一 product identity、真实配料在
shaker、GT 有 stir/extract/pour/return spoon 边界、同一 hand/tool/container binding 可重放，且
product 被后续 pour 或 goal 消费。此时 spoon 的 `holding/in` 生命周期要完整维护。

多份配料确实形成新物质时，可以在 `EPISODE_RECONSTRUCTION` 中新建 product object；它必须由
instruction/goal/后续转移证明，并有 producer/consumer。未知内容物不是 `mixture`、container 或
模糊 predicate 的许可；无法确定时隔离。

---

## 8. 变更分级与允许的自动化

| 级别 | 典型变更 | 自动化边界 |
|---|---|---|
| `CONTRACT_PRESERVING` | 格式、稳定排序、完整等价的词法 alias、纯名称对齐 | 可批量候选，但仍需 AST、replay 与幂等验证 |
| `CONTRACT_REPAIRING` | hand/关系生命周期、缺失 guard、start/completion、错误 result | 每个 detector 必须有完整结构 gate 和求解验证 |
| `EPISODE_RECONSTRUCTION` | 新 object/predicate、重建 goal/reference、改变动作边界 | 逐 episode 人工证据审阅，显式批准后才 apply |

以下做法永远不构成自动 apply 条件：相似名字、相同 task id、相同图像背景、相同编号后缀、
单次 solver 成功、或一条 regex 命中。

所有 PDDL 改写必须使用 balanced S-expression/AST。禁止在 PDDL 上做不理解作用域、注释、
变量绑定和嵌套结构的全局文本替换。

---

## 9. 事务式执行流程

### 9.1 Dry-run

每个候选 action 输出至少：

| 字段 | 内容 |
|---|---|
| source | dataset/task/episode/round 与输入 hash |
| old contract | 名称、参数、precondition、add/delete effect |
| proposed contract | 修改后的完整合同，不只给名称 |
| class/decision | 变更级别与五种 operator 结论 |
| evidence | instruction、kf、图像、ontology、producer/consumer、reference source |
| detector gate | 完整触发条件、唯一绑定方式和排除条件 |
| delta | 与旧合同/参考合同的最小差异 |
| expected artifacts | 哪些 domain/problem/plan/reference/aggregate 文件会变 |
| verification | replay、solver、等价和不变量的结果 |

`MODEL_REVIEW_REQUIRED` 是明确输出，不是可被静默跳过或“尽量修复”的失败容忍。

### 9.2 Staging

在独立 staging tree 生成完整候选，不写 source。每个被改 episode 必须同步：

- `domain.pddl` 的 action、predicate 和 requirements；
- `problem.pddl` 的 objects、init 和 goal；
- 新求解的 `plan.txt`；
- 由最终 domain + plan 翻译的 `plan_nl.txt`；
- 当 reference 经过第 3 节裁决确认错误时，受影响的 `kf_actions` 和 `kf_actions`；
- 适用时的内嵌 JSON 字段、judge supersession 与 provenance。

不得手工把旧 plan action 名改成新名，也不得只改 unified domain 或仅改 domain。

staging 的目录层级必须与消费它的 resolver 根完全一致。验证前逐个解析预期覆盖的
`domain.pddl`、`problem.pddl`、`plan.txt` 实际路径，并断言它们都来自 staging；如果本应有候选却
显示零个 override，或 override 数与 manifest 的稀疏覆盖清单不符，即使 replay 通过也属于
`STAGING_LAYOUT_ERROR`，必须重建 staging 后再验证。

同一有限修复表出现“已修复样本 + 待修复样本”是正常的增量状态，不是自动失败理由。事务必须：

1. 对已修复样本检查完整目标合同、reference 更新和 replay，而非再次改写；
2. 只把仍满足完整 detector gate 的待修复样本写入 staging；
3. 把已修复、待修复、source drift 和 review-required 四类分别写入报告；
4. 若所谓“已修复”样本缺 literal、binding、reference 或 plan 同步，停止并标为
   `MODEL_REVIEW_REQUIRED`，不能把它默认为正确。

### 9.3 验证顺序

1. AST/requirements/predicate arity/变量/grounded object/action 参数静态检查；
2. 从 `:init` 逐步 replay 新 plan，检查每步 precondition/effect、hand、互斥和 goal；
3. 用项目批准的独立 solver 重求解。受限内部 STRIPS 搜索可用于预检与回放，不能单独作为
   大批量完成证明；
4. 比较 instruction、kf 和图像要求的任务等价，并按第 3.1.1 节追踪目标身份与姿态证据；
5. 检查每个 grounded action 对应一条 `plan_nl`，所有 judge/provenance 均对应当前合同或明确
   supersede 旧合同；
6. 运行 family-specific 审计：载荷闭包、有限/无限资源、lid/cap、result 主体、关系生命周期；
7. 用同一输入对 staging 再跑一次。第二遍必须为零变更、零解析/验证错误；
8. apply 后在真实 source 再次运行独立 solver，且用其输出覆盖真实 `plan.txt`，然后重复 replay
   与审计，再重建 aggregate。

若 solver 找到更短但绕过 GT 必要边界的 plan，不能直接发布。只有这个捷径被证明确实满足
instruction 的全部方法约束，或 keyframe-constrained witness 也可执行且报告明确该模式时，
才能接受。

### 9.4 Apply 与聚合闭环

apply 只复制 manifest 中 `VALIDATED` 且无 review 的 artifacts。copy 前验证 source hash，避免覆盖
并行修改。每个 batch 完成后：

1. 在真实路径重放与审计；
2. 重建目标数据集的 `unified_domain.pddl` 与 source mapping；
3. 检查统一域的 predicate name/arity、重复 action 名和同名异合同；
4. 断言 unified action 数与 mapping entry 数相等、final action 名唯一、每个 mapping source 存在，
   并反查本次改动的 source 已映射到预期合同；
5. 对全量最新 round 重跑 contract lint 与 predicate arity 流式审计；
6. 再审视统一域，确认没有新出现的可证明问题；无问题才结束闭环。

### 9.5 聚合驱动的下一轮闭环

在第 9.4 节重建后的 unified domain 上运行只读合同审计，并把每项结果分成 `CLEAN`、
`MODEL_REVIEW_REQUIRED` 和可验证修复候选。最低限度检查包括：每个 final action 到 mapping 再到现存
source 的完整性、重复 final action、source predicate arity、source numeric action 名、同一 hand 在一个 effect 中写入多个独立
`holding`、以及 source domain 中无任何 release consumer 的 `holding_pair` producer。

这些扫描只负责**发现**，不能直接 apply。每个候选必须沿 source mapping 回到局部证据，重新走第 3、
5、6 和 9.1--9.3 节。只有全量 replay 与 unified 审计都返回 `CLEAN`，或剩余项都带有可追溯的
`MODEL_REVIEW_REQUIRED` 裁决，当前迭代才可结束；否则以新的、有限 manifest 进入下一轮。

---

## 10. 大规模执行的工程纪律

- 先在每个 family 的代表 episode 验证 detector，再扩大到同一完整 gate；
- 受限环境下按 offset/limit 或 manifest shard 分批，保留每个 shard 的输入范围、报告和汇总；
- 不对 keyframe、日志、图像树做无目的递归扫描；
- 解析失败、非标准 PDDL、缺文件和 solver timeout 必须成为独立记录，不能被计作未修改；
- 第二遍审计是幂等性测试：已有正确合同必须返回 `UNCHANGED`，不应制造 review 噪声；
- full report 过大时，使用流式聚合保存 `predicate -> arity set`、合同签名和错误样本，避免因
  审计器自身内存失败而错误宣称零问题。
- 受控环境会中断长任务时，用确定性 manifest 分片、可组合汇总和明确的 shard hash；不能以完成的
  子样本代表全量完成。聚合重建属于单写者步骤，必须等待前一次 writer 结束并复核输出 hash。
- CLI 参数错误、候选路径未被 resolver 读取、输出文件未更新或辅助 parser 假设错误都属于工具失败，
  不能因 source baseline 恰好通过而记作 staging 通过；修正工具调用后必须重跑受影响验证。
- 最终 source 与 aggregate 均验收后，保留 manifest、最终验证报告和 hash；删除仅含候选副本的
  staging tree，避免后续任务把过期候选误当作 source。

工具与规则都必须可验证其自身：regex、parser、参数重写、group index、计划重写都需要代表性
回归样本。工具错误不应被误记为数据语义错误。

---

## 11. 已被实践否定的做法

下列方案不得重新引入：

1. 把 `place_kettle_under_faucet` 改名成 `fill_kettle_from_faucet`；前者只建立 `under`，后者才转移水；
2. 规定所有微波炉关机都产生 `boiled`，或把所有 start-only reference 一律拆成 turn-off；
3. 把 `both_hands_free`、robot-level `holding` 等聚合资源机械降成 `hand_free`；
4. 因机器人有两只手就给普通 action 加双手、传手或固定左右手顺序；
5. 用 `ready_for_*` 等虚构进度强迫 planner 复刻擦拭、放置或递手顺序；
6. 让 `open_*` 暗中获得 lid/cap，或让 pure `close_*` 暗中移动 lid/cap；
7. 把搅拌结果写在容器上，或用未知 `mixture` 掩盖内容物身份；
8. 把无限 faucet/dispenser 的能力当成一次使用即耗尽的有限 water object；
9. 只改 plan、只改统一域、或按 action 名对所有 PDDL 做文本替换；
10. 以统一域的名字、频次或更短 solver plan 覆盖 episode 证据。
11. 因 action 有数字后缀就直接删号或合并，或把 unified catalog 的 `_1`、`_2` 当作 source defect；
12. 用同一 grounded object 绑定 pair 参数，或把无 release 的双 `holding` action 机械改成 pair；
13. 因 plan/kf 的原始行数不同就改 reference，忽略注释、粒度和 reference authority；
14. 因 batch 中同时存在已修复与待修复样本而跳过全部验证或重复改写已修复样本；
15. 在同一 unified 输出路径并发 merge，或在没有 source mapping 完整性检查时宣称聚合完成。
16. 根据 endpoint 的物体计数、被遮挡的终帧或图像坐标轴，臆造“目标消失”或 `upright/flat` 等
    pose 约束；
17. 把旧 VLM/judge 的 verdict 当作修正 reference 后的当前结论，或以旧模型名伪造人工复核；
18. staging resolver 没有实际读取候选文件时，仍把 baseline 的 replay pass 记为 staged pass；
19. 假定 source mapping 一定使用相对路径，或在 source 精确合并后把合理的 unified action 数量下降
    误判为丢失合同。

---

## 12. 完成标准

一次清洗只有同时满足全部条件才算完成：

- 清洗档案、manifest、排除项和源 hash 完整；
- 每项修改有合同 delta、证据、变更级别和决策；
- 所有改动的 PDDL 已重新求解，plan 与 plan_nl 来自最终 PDDL；
- 新 plan 与正确的 reference 在任务结果、对象流转和必要边界上逻辑等价；
- predicate arity、变量、hand、互斥状态、有限/无限资源、关系生命周期、start/completion 全部通过；
- staging 和真实 source 均完成验证，第二遍零变更；
- staging resolver 已确认读取预期候选文件；reference 改动后的 judge/provenance 已重生成或可追溯地
  supersede，未留下语义过期的当前判定；
- 只修改批准范围，未决 episode 明确隔离；
- unified domain/provenance 已从最终 source 重建并再次审计；
- reference authority、有效 action 计数规则、reference hash 变化与每个 count mismatch 的裁决类别已记录；
- 同一夹爪的多对象资源、pair release consumer、source numeric 名和 aggregate source mapping 均已通过
  对应的结构审计；
- aggregate 的 action/mapping 数量 delta 可解释，本次改动 source 可从规范化路径反查到其最终合同；
- 最终报告和 hash 已保留，临时 staging 候选已在真实 source 验证后清理；
- 全量最新 round 的结构/合同审计没有可证明的新问题，或所有剩余项都有可追溯的 review 结论。

当上述条件不满足时，正确结果是“继续审计”或 `MODEL_REVIEW_REQUIRED`，不是用更宽泛的规则
把不确定样本改到看起来一致。
