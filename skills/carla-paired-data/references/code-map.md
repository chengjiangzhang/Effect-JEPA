# 代码索引

路径相对本 skill 根目录。运行命令见 [工作流](workflow.md)，参数与数据语义见 [数据契约](data-contract.md)。生产入口使用包内代码，无需其他项目源码。

## 编排与场景配置

| 环节 | 文件及入口 | 输入与输出 |
| --- | --- | --- |
| 命令编排 | `scripts/pipeline.py`：`prepare`、`execute`、`delivery` | 冻结配置、恢复状态、执行各阶段、生成批次验收清单 |
| 模拟器启动 | `scripts/server.py` | CARLA 安装路径、非 root 账户、端口 → Epic/offscreen 服务及日志 |
| 本机服务回收 | `scripts/managed_server.py` | 候选/分支同步边界回收自有服务；版本、地图、端口和配置恢复核验；可选固定引擎随机种子 |
| 网格配额 | `scripts/plan_grid.py`：`build_plan` | 启用网格、数量、seed、预算 → 均衡场景任务 |
| 地点与天气分配 | `scripts/sampling.py`：`assign_weather`、`location_shortlist`、`coverage_summary` | 模板内均衡天气、优先少用的真实地点、分别报告计划/编译/验收覆盖 |
| 模型通信 | `scripts/model_api.py`：`ModelClient.request_json` | 地图选择及完整 spec 编写的动态 schema、失败报告 → 可校验提案及调用溯源 |
| 地图目录 | `scripts/scene_compiler.py`：`build_catalog`、`Catalog.public` | OpenDRIVE 和模板集合 → 有实际地图依据的候选 ID |
| 联合随机生成 | `scripts/scene_randomization.py` | 独立命名随机流、连续联合参数、参与者数量、逐对象固定偏好及最终生效记录 |
| 采样偏差审计 | `scripts/randomization_report.py` | 计划/提案/编译/拒绝/物理/视觉分母、分布相关提示及可比场景整段轨迹摘要；不按后果筛景 |
| 模板内交互分层 | `scripts/interaction_design.py` | 十二模板的预先分层、联合位置/速度/队列及动作时序；不读取未来后果 |
| 行人原生生成 | `scripts/walker_spawn.py` | 逐地图原生生成与销毁、方向和模型可行性筛选、冻结规格的位姿核验 |
| 专用交互几何 | `scripts/interaction_geometry.py` | 信号标识、过街路径、双向窄车道及其静态配置 |
| 场景编写 | `scripts/scene_spec.py`：`authoring_payload`、`authoring_schema`、`authoring_context`、`apply_authoring` | 每角色初态、NPC 响应参数、分叉、五分支计划与无关角色 → 校验后的完整 spec |
| 场景编译 | `scripts/scene_compiler.py`：`compile_one`、`write_plan` | 单个任务、地图选择、完整编写提案 → 冻结 `spec.json`、manifest 与地图 |
| 独立离线编译 | `scripts/scene_compiler.py`：`compile_plan` | 整份配额计划 → 批量静态配置和失败记录，不运行服务端 |
| 预检决策工具 | `scripts/preflight_decision.py`：`decide` | 显式检查证据 → 有限修正或进入 RGB 阶段；独立辅助接口 |
| 整景去重与划分 | `scripts/curate_dataset.py`：`curate_dataset` | 已验收整景、独立训练校准或冻结 policy → 保留/重复整景、保护对照、train/validation/test 与泄漏检查 |

## 实测规划与最终发布

| 文件 | 职责 |
| --- | --- |
| `scripts/planning_stage.py` | 冻结初始配置、按有界校准重跑公共历史、确认动作计划、维护恢复绑定及两阶段前缀门禁 |
| `scripts/encounter_calibration.py` | 编译冻结相遇目标，仅由实测共同历史修订布局并绑定每次历史证据；不使用未来响应 |
| `scripts/measured_t0.py` | 从实测 t0 全演员状态计算路线/冲突距离和四秒运动、停车、等待恢复预算；不读取未来结果 |
| `scripts/runtime/preflight_physics.py --history-only` | 只采集 -2..0 秒共同历史，到 t0 停止，无未来动作/相机 |
| `scripts/runtime/audit_history_prefix.py` | 全物理与正式采集的每个分支逐字段匹配首次实测公共历史 |
| `scripts/publish_dataset.py` | 原封存验证、整景整理选择、干净场景目录、相对查看链接、校验和与幂等发布 |

## 采集与检查

| 环节 | 包内文件 | 职责 |
| --- | --- | --- |
| 地图几何 | `scripts/runtime/scenarios_v2.py` | Driving 车道、连接、多边形、真实邻道路线、车辆布局辅助函数 |
| 动作与周车 | `scripts/runtime/controllers_v2.py` | 模板策略、正常策略继续、真实邻道跟踪、实时 IDM/冲突让行、物理车辆控制 |
| 周车动态汇入 | `scripts/runtime/npc_merge.py` | 当前车身前后间隙、源/目标地图走廊和内部状态 → 等待/开始/早期取消/继续；可复算预热 |
| 行人与信号灯 | `scripts/runtime/auxiliary_actors.py` | 绑定实际信号灯、按场景时间执行相位、生成并控制行人、记录和清理演员 |
| 行人实时反馈 | `scripts/runtime/pedestrian_policy.py` | 同帧实测车辆状态与车身几何 → 等待／接受间隙／沿固定路线过街；不读取分支或未来 |
| 规格冻结及单分支采集 | `scripts/runtime/collect_multicam.py` | `finalize_scene / finalize_manifest / collect`；演员与传感器创建、同帧屏障保存 |
| 无图物理预检 | `scripts/runtime/preflight_physics.py` | 无 RGB 的完整历史/未来、状态与碰撞传感器采集 |
| 物理检查 | `scripts/runtime/audit_preflight_progress.py` | 初始化/历史/重复性、接触前动力学、接触记录和变道过程 |
| 执行与后果 | `scripts/runtime/audit_execution.py` | 冻结策略重算、控制/运动验收、后果与 NPC 响应、无关角色检查 |
| 批量 RGB | `scripts/runtime/collect_multicam_batch.py` | 每景先参考与重复分支，随后剩余动作；已完成分支恢复 |
| 独立传感器协议 | `scripts/runtime/audit_protocol.py` | 实际相机属性/天气、内外参及渲染预热证据，不生成后果标签 |
| 行人与信号证据 | `scripts/runtime/audit_auxiliary.py` | 行人状态/框/控制/身份、相位逐灯读回、公共历史与重复性、物理与 RGB 对应 |
| 原生天气随机流与贴花排序 | `scripts/runtime/native/weather_rng/` | 精确绑定 Shipping 和天气资产的 C/汇编/共享库及清单；独立天气随机流、保留显式优先级的稳定对象路径排序 |
| 原生加载与取样证据 | `scripts/native_weather_proof.py` | 只读验证进程、库、天气上下文和贴花证明；绑定实际 spec/分支/状态，在恢复及封存时复核 |
| 原生贴花证明 | `scripts/native_decal_proof.py` | 实际PID和入口核验、排序调用连续性、绑定快照及哈希；明确零次调用的证明范围 |
| RGB 重复性 | `scripts/runtime/audit_render_consistency.py` | 参考重复全程与各分支公共历史的原图比较 |
| 原生语义点云 | `scripts/runtime/semantic_lidar.py` | 点记录解析、同步保存及协议/标定/角色映射检查 |
| 数值等价 | `scripts/runtime/audit_capture_equivalence.py` | 无图预检与正式 RGB 采集的逐字段核对 |
| 结构与完整性 | `scripts/runtime/audit_multicam.py` | 原图、字段、时序、相机标定与数据完整性 |
| 实测行为覆盖 | `scripts/runtime/interaction_coverage.py` | t0实际阶段、事件、阻挡原因、全部对象10对响应、2Hz时间支持；不筛选强响应 |
| 动作细节 | `scripts/runtime/audit_action_diversity.py`、`audit_lane_change.py` | 纵向差异、变道切入/完成/保持、接触前后分开描述 |
| 共同坐标与查看 | `scripts/runtime/export_common.py` | `ego(t0)` 轨迹、完整 3D 框角点、稳定 ID、六相机 HTML 查看器 |

## 动作和响应

| 文件 | 入口或职责 |
| --- | --- |
| `scripts/action_design.py` | 每模板五条件、冲突/停车位置、运动预算及设计证据 |
| `scripts/action_opportunity.py` | 固定2秒历史/4秒未来内，从实测t0检查速度、行人响应时间及背景交通间隙 |
| `scripts/runtime/action_separation.py` | 五动作全部10对的持续控制/运动区分门槛；碰撞前证据、repeat噪声与整景失败 |
| `scripts/runtime/ego_decisions.py` | 动作 schema、共享纵向目标速度、停车保持、指定对象清空与当前通行条件同时成立后的等待释放 |
| `scripts/runtime/audit_realized_actions.py` | `audit_motion_similarity`：完整 ego 轨迹及重复噪声诊断；`normalize_motion_report` / `select_branches`：旧报告/入口兼容 |

## 资源与验证

| 路径 | 内容 |
| --- | --- |
| `assets/grid.example.json` | 模板/NPC 风格网格、七种天气和配额配置 |
| `assets/model-config.example.json` | 模型通信配置模板 |
| `scripts/runtime/resources/daylight.json` | 天气、曝光、相机和引擎请求参数基线 |
| `scripts/runtime/resources/engine.ini` | 启动时请求的 Epic 配置 |
| `scripts/runtime/resources/physics_6mps.json`、`physics_10mps.json` | 四车型起停参数的历史标定证据 |
| `scripts/runtime/SOURCE_PROVENANCE.json` | 运行库来源和打包身份 |
| `requirements.txt` | Python 依赖 |
| `scripts/test_*.py`、`scripts/runtime/test_*.py` | 离线测试；完整运行入口见 [离线测试](workflow.md#离线测试) |

`resource://` 解析到包内资源。资源证据不能替代当前环境的物理或渲染验收。

## 格式来源

- [CARLA 0.9.16 传感器](https://carla.readthedocs.io/en/0.9.16/ref_sensors/)：RGB、碰撞事件和坐标定义。
- [Bench2Drive 标注](https://raw.githubusercontent.com/Thinklab-SJTU/Bench2Drive/main/docs/anno.md)：六相机、标定和车辆框格式参考；本管线使用自定义地图场景，并非官方场景重放。

过街中持续反馈：`scripts/runtime/pedestrian_crossing.py`（有界运动、连续扫掠风险、恢复滞回），由 `pedestrian_policy.py` 按冻结版本调度；`audit_auxiliary.py` 从逐帧实测数据复算全部决策和控制。
