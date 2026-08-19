# PDDL Operator 跨数据集同步清理手册

本手册用于让 Codex 参考当前 Human-Aug 统一 operator 库为基准，清理其他
PDDL 数据集中的 operator 名称与建模合同。

## 1. 基准与适用范围

### 1.1 参考建模基准

以此文件中的实际 action block 为 operator 名称与建模的参考基准：

```text
/home/xyx/下载/swm/eval_results/gpt-5.6-sol/human_aug/unified_domain.pddl
```

同目录的 `unified_operator_sources.json` 用于把合并后的 operator 反查到实际
episode；统一域是审计索引，episode 的 `domain.pddl` 与 `problem.pddl` 才是修改和
求解的事实来源。

### 1.2 清理范围

除非用户另有指定：

- 只处理每个 task/episode 中数字最大的有效 `roundN`
- 同步处理 `domain.pddl`、`problem.pddl`、grounded plan 和派生的
  `plan_nl`；
- 不修改旧 round、关键帧、日志和无关元数据；
- 先生成 dry-run 映射与 staging diff，确认后再 apply；
- 保留工作区中用户已有的无关修改。

## 2. 不可违反的总规则

1. **按合同匹配，不按旧名字匹配。** 名称、动词或视频动作相近，不能单独
   证明两个 operator 相同。
2. **统一库优先。** 存在相同或真正相似的 operator 时，优先复用其名称、
   参数角色、因果 precondition、add/delete effect 和动作边界。
3. **局部因果。** Operator 只描述一次动作直接需要和直接改变的状态，不是
   场景快照，也不是整个任务脚本。
4. **双臂需求进入 precondition。** 只有确实必须双臂同时执行的 operator
   才升级为双臂合同；数据来自双臂机器人不等于所有 action 都需要双臂。
5. **语义修改必须成套同步。** 不能只改 action header，而不改参数、problem、
   grounded plan 和派生文本。
6. **不为求解成功伪造语义。** 一个 plan 能求解，不代表 operator 局部正确。
7. **物理使能闭包。** 启动设备、开始过程、开启流动或提交加工结果的 action，
   必须在同一 action 的 precondition 中约束该过程已建模的输入、工件、接收
   容器或工作位置。不能让 planner 先启动、之后才装载或对位。
8. **关系必须有生命周期。** `under`、`in`、`on`、`connected` 等事实不是注释；
   每个持续关系都要有真实 producer、consumer 和使其失效的 action。对象离开
   该位置、槽位或连接时，effect 必须删除旧 relation。
9. **用因果依赖表达必要顺序。** 关键帧计划中的安全顺序不能只靠 planner 的偶然
   排序。若先后顺序改变会让过程失真，就用 precondition/effect 让错误顺序不可
   执行；若两步独立，则允许重排。

## 3. 每个 Operator 的标准决策流程

### 3.1 先提取目标合同

在搜索统一库前，先从目标 PDDL 和数据证据中写清：

- `affected entity`：哪个对象或设备的持久状态被改变；
- `source / target`：对象或有限资源从哪里离开、进入哪里；
- `hand mode`：不占手、单臂、双臂、持物、按住或释放；
- `required old state`：动作成立前真正必要的状态；
- `enabling relation`：过程开始时已必须成立的输入、工件、接收容器或工作
  位置关系；
- `add effects / delete effects`：动作立即造成的状态增量；
- `boundary`：这是原子转换、过程开始、过程结束，还是错误的多步 macro。

参数名和 conjunction 中 literal 的书写顺序不属于语义。

### 3.1.1 物理使能闭包检查

对任何使 `is_off -> is_on`、`idle -> running`、未按下 -> 已按下，或开始
`dispensing / brewing / heating / washing` 的 action，依次检查：

1. 该过程是否消费、加工、承载或指向某个任务对象；
2. 目标 ontology 是否已经表达对象到工作位置的关系，例如 `in`、`on`、
   `under`、`connected`、`loaded` 或槽位关系；
3. 若两者都成立，启动 action 必须显式拥有对应对象/位置参数，并把该关系放入
   precondition；
4. 若配对的停止、完成或放出 action 已要求该关系，启动 action 也必须要求它，
   除非证据表明过程允许在运行中才建立该关系；
5. 若领域没有对象、接收位置或关系 predicate，不能凭空添加一个不可达 guard；
   这类纯通电/纯放出模型标记为 `MODEL_REVIEW_REQUIRED`，或保持其明确的
   无载荷合同。

通用模式是让启动 action 对已有物理状态负责，而不是让后续 action 补救。
以下为 schema notation，不是要创建固定的 `workpiece`、`receiver` 或 `outlet`
predicate：

```lisp
; 加工设备：工件已装入设备，才能启动。
(:action start_process
  :parameters (... ?workpiece ?device)
  :precondition (and ... (is_off ?device) (in ?workpiece ?device)))

; 出水/出料：接收容器已在实际出料位置，才能开始。
(:action start_flow
  :parameters (... ?receiver ?outlet)
  :precondition (and ... (is_off ?control) (under ?receiver ?outlet)))
```

这里的 `?workpiece`、`?receiver` 与 `?outlet` 是因果角色，不是固定类别。
它们可以是衣物、餐具、水、杯子、玻璃杯、托盘、槽位或其他由目标 ontology
支持的实体。不要为了复用规则加入任务中不存在的类别、物体或位置关系。

### 3.1.2 对位关系、流动合同与类型角色

把 `under` 视为“接收容器当前对准某个实际出口”的动态关系，而不是一个静态
场景标签。这个原则同样适用于装入槽位、连接电源和固定在支架等关系。

1. **建立。** 放置到出口下的 action，或过程已在初始状态中运行的 problem，必须
   显式建立 `(under ?receiver ?outlet)`；predicate 也必须在 domain 中声明。
2. **消费。** 所有会使流动开始或提交该次流动结果的 action，都要以同一个
   receiver/outlet 参数要求该 `under`。不能只让 `turn_off_*_after_filling` 检查
   对位，而让 `turn_on_*` 无条件执行。
3. **失效。** 从托盘、支架或槽位拿走接收容器的 action 必须删除对应 `under`。
   否则 planner 可以先把杯/壶移走，再利用遗留事实接水或结束流动。
4. **角色匹配。** ontology 已经区分冷热出口、不同喷嘴或不同流体时，控制 action
   和 completion action 都必须约束匹配的 outlet 与内容类型。`dispenses` 只能说明
   能提供某物，不能替代已存在的温度或设备类别 guard。
5. **不虚构 guard。** 若一个普通托盘转移任务没有出口对象、`under` predicate 或
   任何流动 action，就不能仅因名称含 `drip_tray` 强行加入 `under`。应标记为
   普通取放，或要求模型复核。

例 1：杯子离开热水出口时，对位关系必须随取杯失效：

```lisp
(:action pick_cup_from_drip_tray_under_hot_water_outlet
  :parameters (?h ?c ?t ?o)
  :precondition (and ... (on ?c ?t) (hot_water_outlet ?o) (under ?c ?o))
  :effect (and ... (holding ?h ?c) (not (on ?c ?t)) (not (under ?c ?o))))
```

例 2：若水泵初始已开，problem 不能只写 `(is_on ?p)`；还必须写壶的当前对位，
并让结束和取壶动作共享它：

```lisp
; problem :init
(is_on ?p)
(under ?k ?p)

; 结束流动：水只会进入仍在出口下的壶。
(:action turn_off_water_pump_after_filling
  :parameters (?h ?p ?w ?k)
  :precondition (and ... (is_on ?p) (open ?k) (dispenses ?p ?w) (under ?k ?p))
  :effect (and ... (in ?w ?k) (is_off ?p)))

; 拿走壶：先满足任务要求的 closed，再删除对位。
(:action pick_kettle_from_box_top_under_water_pump
  :parameters (?h ?k ?b ?p)
  :precondition (and ... (closed ?k) (under ?k ?p))
  :effect (and ... (holding ?h ?k) (not (under ?k ?p))))
```

例 3：只有任务语义要求混合水或“先热后冷”时，冷水启动才应见证杯中已有热水；
这不是所有冷水任务的通用前提：

```lisp
(:action turn_on_cold_water_button
  :parameters (?h ?b ?hw ?m ?cold_outlet)
  :precondition (and ... (cold_water_outlet ?cold_outlet)
                     (under ?m ?cold_outlet)
                     (hot_water ?hw) (in ?hw ?m)))
```

若该任务只要求一杯冷水，加入 `?hw` 会改变任务语义，必须禁止。

### 3.1.3 访问阻挡的动态二元建模

开门、开抽屉和关抽屉等访问 guard 必须表达**当前真实障碍**，不能用
`clear_to_open`、`clear_to_close` 一类与障碍物脱节的 unary 补集事实。

- 用 `(blocks_opening ?obstacle ?target)` 表示 `?obstacle` 当前阻挡
  `?target` 打开；用 `(blocks_closing ?content ?drawer)` 表示内容物当前阻挡
  抽屉关闭。predicate 的参数顺序应始终是“障碍物、被阻挡对象”。
- `:init` 必须写入初始场景中实际存在的阻挡关系；使障碍出现的动作加入该
  relation，使障碍消失的动作删除它。例如上层抽屉打开后阻挡下层抽屉，关闭
  上层抽屉后删除对应 `blocks_opening`；大盒子进入抽屉后阻挡关闭，取出后删除
  对应 `blocks_closing`。
- 被阻挡的动作以否定 relation 作为 precondition，并在 domain 的
  `:requirements` 中声明 `:negative-preconditions`。若存在多个独立障碍物，
  必须分别检查每一个 relation。
- guard 中的障碍物必须是该 action 明确声明、且由类别 predicate 约束的因果
  参数；不能增加一个自由变量来伪装“没有任意障碍物”。PDDL 的单个
  `(not (blocks_* ?x ?target))` 只能检查已绑定的 `?x`，不能表达对所有对象的
  全称否定。
- 删除 `clear_to_open` / `clear_to_close` 的 predicate 声明、`:init`、effect
  和 precondition。若任务中没有可识别的真实障碍物或障碍关系，则只删除这个
  恒为真的 clear guard，不能为了形式统一虚构 blocker。

以下片段只说明关系方向与 guard；类别、hand 和状态 literal 应按目标 domain
补齐：

```lisp
; 上层抽屉打开时，阻挡下层抽屉打开。
(blocks_opening ?upper ?lower)
(not (blocks_opening ?upper ?lower))

; 抽屉入口处的物体阻挡打开。
(blocks_opening ?bin ?drawer)
(not (blocks_opening ?bin ?drawer))

; 抽屉内过大的盒子阻挡关闭。
(blocks_closing ?box ?drawer)
(not (blocks_closing ?box ?drawer))
```

### 3.2 再按以下顺序匹配

1. add/delete effect 的状态转移；
2. `hand_free`、`holding`、`pressing` 等资源变化；
3. `in`、`on`、`under` 等 source/target 关系；
4. 机械、访问、过程和有限资源 precondition；
5. 对象类别；
6. 最后才看 action name。

不得仅用字符串相似度、共同动词或名称前缀选择参考 action。

### 3.3 只允许五种结论

| 结论 | 条件 | 处理 |
|---|---|---|
| `KEEP` | 目标名称和 canonical contract 已与基准一致 | 不改 |
| `EXACT_REUSE` | 忽略变量名和 literal 顺序后，合同与某个基准 action 相同 | 复用其完整名称和合同 |
| `ADAPTED_REUSE` | 满足第 4 节的相似条件，仅类别、受证据支持的 guard 或单双臂接口不同 | 复用建模骨架并做最小适配 |
| `NEW_OPERATOR_REQUIRED` | 统一库没有同一转换模板 | 按第 8 节新建 |
| `MODEL_REVIEW_REQUIRED` | 多个候选无法由证据消歧，或当前 operator 自相矛盾 | 不改数据，报告冲突 |

不得用重命名掩盖参数、precondition、effect 或动作粒度错误。

## 4. 何时可以直接复用“相似 Operator”的建模逻辑

### 4.1 相似的必要条件

只有同时满足下列条件，才可判定为可复用的相似 operator：

1. 改变的是同一种持久状态或关系；
2. source、target、affected entity 和 tool 的因果角色一一对应；
3. 对于启动、流动或过程 action，输入、工件、接收位置和工作槽位等物理使能
   关系一致，或有证据支持最小适配；
4. 有限资源消耗或无限供给语义一致；
5. `hand_free / holding / pressing` 的资源阶段一致，双臂升级除外；
6. 动作边界一致，例如都是普通 toggle、都是过程开始，或都是提交结果的
   过程结束。

类别不同但角色和转换相同，通常可以复用。例如一个具体容器上的普通
`pick` 可以复用统一库中另一具体容器 `pick` 的取放骨架，再替换真实类别和
关系。

### 4.2 不能视为相似的情况

出现任一情况都不能整块复用：

- 一个 action 消耗有限内容，另一个来自无限 `dispenses` 供给；
- 一个只切换设备状态，另一个同时提交 `heated / rinsed / washed / filled`；
- 一个获取或释放持物资源，另一个不改变 hand occupancy；
- 一个是普通开关，另一个是 `press_and_hold / release`；
- 一个是单一原子转换，另一个把 pick、move、place 或 start、wait、finish
  合成一个 macro；
- 共同点只是名字、动词、指令顺序或视觉动作。

这类 action 只能借鉴对应 family 的一般原则，不能复制完整合同。

### 4.3 “复用建模逻辑”具体复用什么

应复用：

- 参数的因果角色及其依赖关系；
- 必要的 old-state、source/target 和机械 guard；
- add/delete 成对状态转换；
- 手、工具、按钮等资源的占用与释放；
- operator 的开始/结束边界。

不得机械复制：

- 变量名、注释和 literal 排序；
- 目标场景中不存在的物体或类别；
- 只属于参考任务的 contents、邻近物体、历史状态或目标事实；
- 为了匹配参考名字而伪造的 predicate；
- 未被目标数据支持的单臂或双臂假设。

### 4.4 编号变体的选择

当前统一库中的 `_1`、`_2` 等后缀表示同一 base name 下不同的 canonical
contract，不表示质量排序，也不表示同义词。

- `EXACT_REUSE` 必须复用匹配 action 的完整名称，包括已有数字后缀；
- 不能因为 `_1` 看起来更“标准”就优先选 `_1`；
- 不能按来源数量多数投票替代目标证据；
- `ADAPTED_REUSE` 不得猜造数字后缀；先使用由参考结构得到的语义 base
  name，后续 merge 再根据实际 signature 分配 catalog ID；
- 同一 target domain 中，两个不同合同不能使用同一个 action name。

例如，`pick_bottle_from_counter_1` 与 `_2` 是否要求并更新 `clear` 不同，属于
不同合同。`place` 的 held object 仅因**输入** pose 不同，不能据此直接判定为
不同编号变体；须先按 4.4.1 判断该 pose 是否有因果消费者。

### 4.4.1 `place` 姿态的因果最小化

`place` 的输入姿态只是物体被拿起时的历史状态，不自动构成 action 的合同、
precondition 或名称后缀。先逐一判断放置后的姿态是否有真实的因果消费者：

1. 同一物体的后续保留 action 是否以该姿态为 precondition；
2. 该姿态是否是不可删除的 goal。对仅含该 pose 的 goal，临时删除该 literal 后
   重新求解；若新 plan 仍实现相同的非姿态目标、对象流转和任务结果，则该 goal
   pose 是冗余事实；
3. 将随本次清理一并删除的 source-pose precondition，不是下游消费者。

若同一放置动作的差别仅为 held object 的输入 pose，且其余非姿态合同（hand
资源、对象/target 角色、guard、目标 relation）相同，必须合并为一个 action：

- 名称不得含 `_from_<pose>_pose`；
- precondition 只保留真实的 `holding`、类别和 target/资源 guard，不保留 held
  object 的来源 pose；
- 若目标 pose 有消费者，名称使用
  `place_OBJECT_TARGETPOSE_on/in_TARGET`，effect 写入该目标 pose，并删除真实的
  非目标 pose；
- 若目标 pose 没有消费者，名称使用 `place_OBJECT_on/in_TARGET`，effect 只保留
  target relation 和真实资源变化；同时从 `:init`、`:goal` 删除冗余 pose fact，
  并且仅在 domain 中再无任何使用时删除该 predicate 声明。

只清理 held object 的来源 pose，不能删除 target object、container 或设备为自身
转换所需的 pose guard。简化后若出现同名 action，只有完整的非姿态合同也相同时
才能合并；否则保留不同名称并标记 `MODEL_REVIEW_REQUIRED`。

例子：盘子起初可以是 `(vertical ?plate)`，但若后续放置食物要求 `(flat
?plate)`，则保留初始事实，并合并为 `place_plate_flat_on_counter`；动作不要求
盘子放置前已 flat/vertical，而是在 effect 中建立 flat。bowl 或 cup 若后续
`pour`、或放置其内容的 action 确实要求 upright，也同样保留 upright 结果。反之，
若 bottle 的 upright 既不被后续 action 使用，删除其 goal literal 后仍能得到
等价任务计划，则使用 `place_bottle_on_desk`，并删除该 bottle 的冗余 upright
建模。

### 4.5 同一合同对应多个基准名称

当前统一库中存在少量 canonical contract 完全相同、但 action name 不同的
条目。这时依次处理：

1. 若目标当前名称已是其中一个基准名称，且名称没有误述 effect 或角色，标记
   `KEEP`，避免无意义改名；
2. 否则排除与真实机制不符的动词，以及遗漏或误述对象类别、source/target
   角色的名称；
3. 在剩余候选中优先使用符合第 6 节公式、语义明确且更简洁的名称；
4. 仍有多个同等合理候选时标记 `MODEL_REVIEW_REQUIRED`，不能按来源数量、
   字典序或数字大小自动决定。

例如 `close_bottom_drawer` 与 `close_drawer_1` 的合同相同。面对真正的
`bottom_drawer`，前者比无语义解释的数字变体更直接；但已有且正确使用后者
的数据不需要仅为风格再次改名。

## 5. 双臂 Operator 的强制规则

### 5.1 何时升级为双臂

只有机械上必须同时使用两臂，或数据明确显示双臂共同承载、共同保持、共同
展开/折叠的 action，才使用双臂 precondition。普通 pick、place、open、
close、press、pour 若单臂即可完成，继续复用单臂合同。

当前 Human-Aug 基准快照没有双 hand 参数的 action。因此，从其中一个单臂
action 迁移到确实只能双臂完成的目标 action 时，结论必须是
`ADAPTED_REUSE`，并在报告中显式记录 `single -> dual` 接口差异；除 hand
resource 外的因果逻辑仍从匹配 action 复用。

### 5.2 双臂 precondition 的最小接口

双臂必须是两个可互换、单位容量且不同的 hand resource，采用类似于(not (= ?hr ?hl))作变量区分

若动作从双手空闲开始，再加入：

```lisp
(hand_free ?h1)
(hand_free ?h2)
```

若动作要求双手已经共同持有同一对象，则使用：

```lisp
(holding ?h1 ?object)
(holding ?h2 ?object)
```

### 5.3 Effect 必须与双臂资源一致

用户要求的核心是把双臂必要性写进 precondition；但当 action 会获取或释放
持物资源时，只改 precondition 仍会产生错误状态。因此还必须同步 effect：

- 双臂获取：删除两只手的 `hand_free`，为两只手加入正确的 `holding`；
- 双臂释放：删除两只手对应的 `holding`，恢复两只手的 `hand_free`；
- 只瞬时使用双臂、动作前后都空闲：保留双 `hand_free` precondition，不必
  在 effect 中无意义地删后再加；
- 两臂分别操作不同对象或部件：每个 `holding` 必须绑定真实对象，不能把
  两只手都默认绑定到同一个变量。

不得只增加一个未使用的 `?h2` 参数。Grounded plan 也必须传入两个不同的
hand object。

### 5.4 双臂不改变语义名称

默认仍按动作的物理状态变化命名，不在名称中添加 `left/right`、机器人型号
或 `dual_arm`。单双臂差异由参数和 precondition 表达。只有同一个 domain
确实同时保留两个不可互换的执行机制时，才增加最短的机制区别并进入人工
复核。

## 6. 统一命名规则

### 6.1 优先级

1. `EXACT_REUSE`：完整复用基准 action name；
2. `ADAPTED_REUSE`：沿用最相似基准的 verb、role 顺序和命名结构；
3. 无可复用模板时才使用默认公式：

```text
VERB_OBJECT[_from_SOURCE][_PREP_TARGET][_with_TOOL]
```

### 6.2 命名约束

- `OBJECT` 是被直接操作或持久改变状态的具体功能类别；
- source/target 只表示真实的转移角色；
- 名称中的 `in / on / under` 必须与 effect 中的目标关系一致；
- hand 不是 tool；
- pose、访问机制、过程结果只在它们确实区分合同的时候进入名称；对 `place`，
  只有被后续因果消费的**目标** pose 可以进入名称，输入 pose 不得进入名称，
  尤其不得使用 `_from_<pose>_pose`；
- 名称描述对象角色和直接状态转换，不描述已被 relation 表达的历史机制；例如
  `open_bottom_drawer`，而不是 `open_interlocked_bottom_drawer`；对层级明确的
  target，使用 `place_block_in_bottom_drawer`，而不是泛化的
  `place_block_in_drawer`；
- `_after_PROCESS` 只用于该 action 真正提交过程结果的结束边界；
- `press_and_hold`、`release`、普通 `press` 和 latched `turn_on/turn_off`
  是不同交互合同，不得混名；
- 不加入任务 ID、hash、颜色等非操作属性、步骤序号或任意数字后缀；
- 不用 `_when_`、`and`、`then` 或目的性 `to` 把多个动作写进一个名字；
- 不根据自然语言步骤改写已经匹配成功的统一库名称。

## 7. 从当前统一库提炼的核心 Family 合同

下表用于判断和适配；如果存在 exact match，仍以对应 action block 为准。

| Family | 必要 precondition | 直接 effect |
|---|---|---|
| `pick_OBJECT_from_SOURCE` | hand free、对象类别、source 类别、真实 source relation；容器访问 guard 仅在必要时保留 | 删除 `hand_free` 和 source relation，加入 `holding` |
| `place_OBJECT[_TARGETPOSE]_on/in_TARGET` | 正确的 `holding`、对象与 target 类别，以及 target 自身真正需要的 guard；不包含 held object 的输入 pose | 删除 `holding`，恢复 `hand_free`，加入 target relation；仅在目标 pose 有因果消费者时建立它并删除互斥来源 pose |
| `open/close_OBJECT` | hand free、对象类别、旧的 `closed/open`，以及真实 lock/interlock/clearance guard | 删除旧状态，加入互斥新状态 |
| stack/unstack | 在普通 pick/place 基础上保留该 family 真正使用的 `clear` 与 support 关系 | 同步 moving object、support 的 `clear` 和 `on/in` |
| detachable lid / threaded cap | 分别使用 `remove/place_lid` 或 `unscrew/screw_cap` 的真实机制 | 同步 `holding`、cover relation 和 `open/closed` |
| finite pour/scoop/transfer | 有限内容位于 source，所需容器/工具和持物状态成立 | 删除内容的 source relation，再加入 target relation |
| faucet/dispenser fill | running/pressed source、持有或正确放置的 target、持久 `dispenses` 能力；若 ontology 区分类型，source 与内容类型匹配；开始与结束都要求同一 `under` | 加入 `filled/has_water` 等直接结果；不删除 `dispenses` |
| pump/dispenser flow with movable receiver | `is_on/is_off`、open receiver、`dispenses`，以及 receiver 与 outlet 的 `under`；若任务要求先封盖再搬运，pick 还要求 `closed` | completion 提交 `in/filled` 并停止流动；pick 删除 `under`，使移动后不能继续或伪造结束流动 |
| device toggle | hand free、设备类别、旧的 `is_off/is_on`；若设备处理已建模输入，输入已在设备或其真实工作槽位中 | 只切换设备或 `started` 状态 |
| process start / activation | 旧的 idle/off 状态，以及已建模的工件、容器、工作位置或连接关系 | 只建立 `running/started/on/pressed`，不提前提交最终加工结果 |
| process completion | 过程正在运行且完成结果所需状态成立 | 在结束边界提交 `heated/rinsed/washed/filled` 等结果，并同步结束状态 |
| `press_and_hold/release` | control 的 pressed 状态与 `pressing` hand resource 一致 | hold 占用手；release 删除 `pressing/pressed`、恢复 hand free，必要时提交结果 |
| insert/remove/plug/unplug | 正确 `holding` 或已插入关系及真实 lock/power guard | 同步连接关系和 hand occupancy |
| wipe/wash | 持有真实工具或满足真实处理条件 | 只加入直接的 `wiped/clean/washed/wet` 等结果 |

特别注意：

- `clear` 不是所有 pick/place 的通用事实。只有目标 ontology 确实用它维护
  stack/access invariant 时，才选择带 `clear` 的编号变体；
- 所有加工型启动动作都适用物理使能闭包，不限于某个名字或设备：例如加热、
  烘烤、洗涤、烹饪、煮水、出水、出料。设备内的工件、槽位中的载荷或出水口
  下的接收容器必须在启动前已成立；但 start action 本身不应直接产生最终
  `heated/washed/filled` 结果；
- 不能只在 `turn_off_*_after_*`、`release_*_after_*`、fill 或 completion
  action 中约束工件位置，而让对应的 start/on/press action 无条件执行；
  除非过程明确允许运行中装载或移动接收容器；
- 有结果的 `turn_off_*_after_*` 或 `release_*_after_*` 与普通关闭/释放不是
  同一合同。
- 对位关系应在 producer、start、completion 和 pick/move 四处成对审计：缺失
  declaration、初始事实、guard 或 delete effect 中任一项，都会留下可利用的
  PDDL 路径。

## 8. 没有参考 Operator 时如何新建

只有 `NEW_OPERATOR_REQUIRED` 才允许新建。新合同必须满足：

1. 一个 operator 只表示一个稳定、可复用的局部转换；
2. 参数只保留 actor/resource、affected entity、source、target 和真实 tool；
3. 每个 precondition 都必须影响可执行性或直接结果；
4. 每个 effect 都必须由该动作立即造成；
5. 新增互斥状态时删除旧状态，转移有限对象/内容时删除 source relation；
6. `holding` 是可复用接口时，获取与释放不能被吞进 macro；
7. 时间相邻不等于因果，不能把前序步骤中的物体和状态塞进当前 operator；
8. 先完成合同，再按第 6 节命名，然后重新搜索统一库，防止漏掉可复用模板。

若删除某个参数或 literal 后，动作的适用范围和直接状态转移都没有变化，它
通常不应保留。

## 9. Domain、Problem 与 Plan 必须事务式同步

任何 action 合同变化都必须一次性同步：

- Domain 中 action name、parameters、precondition、effect；
- Domain 中所用 predicate 的唯一 name+arity 声明；
- Problem 中 objects、`:init`、`:goal`；
- 必须重新求解修改后的 PDDL，并用求得的 symbolic plan 覆盖对应 `plan.txt`；
  不得手工重命名或沿用旧 plan；
- 必须由新的 symbolic plan 对照修改后的 domain 重新生成并覆盖 `plan_nl.txt`；
- 新 plan 必须与该 episode 的 `kf_plan.txt` 逻辑等价：不仅最终 goal 成立，还要
  包含参考任务的关键过程结果和安全里程碑。不得靠手工改 action 名或保留旧 plan
  伪造一致性；
- 目标数据集自己的 retrieval/catalog/source mapping。

参数必须按语义角色映射，不能因为统一库和目标 action 的变量顺序不同就按
位置盲拷贝。不得用 repository-wide 文本替换处理 PDDL 结构。

若修改的范围覆盖统一库的来源，完成所有 episode 验证后必须从**最大 round**重建
该数据集的 `unified_domain.pddl` 与 provenance/source mapping；不要手工修补聚合
域中的编号变体。

## 10. Dry-run 报告格式

Apply 前为每个目标 action 输出一行：

| 字段 | 内容 |
|---|---|
| source | task/episode/round 与文件路径 |
| old action | 原名称与 canonical contract |
| decision | `KEEP / EXACT_REUSE / ADAPTED_REUSE / NEW_OPERATOR_REQUIRED / MODEL_REVIEW_REQUIRED` |
| reference | 匹配的 unified action；没有则为 `null` |
| arm mode | `none / single / dual`，以及判定证据 |
| proposed action | 新名称、参数角色、precondition、add/delete effect |
| enablement evidence | 启动/过程 action 的输入、工件或接收位置 guard；若没有，说明 ontology 证据 |
| delta | 相对 reference 唯一允许的适配差异 |
| reason | 简短因果理由及未决假设 |

汇总必须给出：目标 round 数、action 数、每类 decision 数、双臂升级数、未决
冲突数和预计修改文件数。`MODEL_REVIEW_REQUIRED > 0` 时不得 apply。

## 11. Apply 前后的绝对验证

必须全部通过，不能只报告“错误数没有增加”：

- [ ] 所有选中 domain/problem 均可解析；
- [ ] 每个变量已声明并被使用，每个 predicate 的 name+arity 唯一一致；
- [ ] 每个 grounded action 存在、参数数量正确、对象类别与角色匹配；
- [ ] `EXACT_REUSE` 与基准 canonical contract 完全一致；
- [ ] `ADAPTED_REUSE` 的差异仅限报告批准的类别、必要 guard 或双臂接口；
- [ ] grounded plan 从 init 逐步可执行，最终状态满足 goal；
- [ ] 每一步后 hand resource 一致：每只手单位容量，不能同时 free 与
      holding/pressing；
- [ ] 同一对象只在明确的双臂共同抓持合同中允许同时出现两个 holding facts；
- [ ] `open/closed`、`is_on/is_off`、pose 和有限位置等互斥状态不共存；
- [ ] `place` action 不含 `_from_<pose>_pose`，也不以 held object 的来源 pose
      作为 precondition；仅保留已证明有消费者的目标 pose；
- [ ] 删除 pose goal 后可满足相同非姿态目标的，已删除该冗余 goal/init literal，
      并在 predicate 无剩余用途时删除其声明；
- [ ] 有限转移删除旧 source，获取/释放正确维护 hand state；
- [ ] 每个 process start、device start 和 flow start 都包含已建模的物理使能
      guard；不存在先 `start/on/press`、后装载或对位的可执行路径；
- [ ] 停止/完成 action 使用的工件或接收位置 guard，已在对应启动 action 中
      复核；若未同步，报告过程允许运行中改变位置的证据；
- [ ] 每个 `under`、`in`、`on` 或连接关系都可追溯到 producer；对象被拿走、移出
      槽位或断开时，旧 relation 已被删除；
- [ ] 初始已运行的过程同时具备其真实输入/接收位置事实，不能只初始化
      `is_on/pressed`；
- [ ] 区分冷热/出口类型时，start 与 completion 使用同一温度、内容和出口角色；
- [ ] 对 `kf_plan.txt` 中的关键安全顺序，已验证错误重排在修改后不可执行，而独立
      步骤的可交换顺序不会被误判为错误；
- [ ] action name 与 effect、source/target 和 process boundary 一致；
- [ ] `plan_nl` 与最终 grounded plan 一致；
- [ ] 第二次运行清理器产生 `changed_files=0`、`changed_actions=0`、
      `errors=0`。

先在 staging tree 验证；任一 gate 失败，不覆盖目标原文件。Apply 后在真实路径
重复同一组验证。

## 12. 推荐执行顺序

1. 枚举目标数据集的有效最大 round 和现有 action；
2. 解析统一库 action block，并建立 canonical signature 索引与 provenance 索引；
3. 为每个目标 action 提取合同，先审计 relation 生命周期、物理使能和类型角色，
   再给出五类 decision；
4. 人工检查所有 `ADAPTED_REUSE`、双臂升级和未决冲突；
5. 在 staging 中事务式修改 domain/problem/plan/plan_nl；
6. 重新求解并验证与 `kf_plan.txt` 的逻辑等价，再生成 `plan_nl.txt`；
7. 运行全部绝对验证和第二遍零变化验证；
8. 用户确认后 apply，并重新验证；
9. 只重建目标数据集自己的 unified output，检查同名不同合同和 provenance；
10. 输出最终 mapping、计数、验证命令和剩余假设。
