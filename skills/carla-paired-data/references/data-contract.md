# 数据契约

本页是场景语义、数据字段和验收规则的定义来源；执行命令见 [工作流](workflow.md)。

- [样本与时间](#样本单位与时间)、[模板与地图](#模板与地图边界)、[天气覆盖](#地点天气与时段覆盖)、[场景编写](#场景编写契约)
- [输出组织](#输出组织)、[状态](#状态与可用范围)、[原始字段](#原始字段)、[语义 LiDAR](#语义-lidar)、[共同坐标](#共同参考坐标)
- [动作与执行](#动作策略执行与观测结果)、[NPC 响应](#npc-响应与无关车辆)、[整景去重与划分](#整景去重与划分文件)

## 样本单位与时间

客户端与服务端固定为 CARLA 0.9.16；同步物理 20 Hz，六相机原生 RGB 1600×900 和语义 LiDAR 均保存 2 Hz，默认 ego + 6 辆周车。不创建俯视 RGB 传感器或 preview 图。

一个样本组由同一场景的公共历史和多个 ego 动作分支组成。场景内车辆、地图、天气、相机、NPC 策略与初始化配置相同；动作从 `t0` 状态之后的第一个物理区间开始改变。

- 物理：`−2 .. 4` 秒，间隔 `0.05` 秒，含端点共 **121** 帧。
- RGB：相同时间范围，间隔 `0.5` 秒，共 **13** 个时刻。
- 历史 RGB：索引 `0..4`；`t0` 物理索引 `40`，图像索引 `4`。
- `control[40]` 是作用于 `t0` 之后区间的命令，允许随分支不同；共同历史控制检查仅使用 `t<0`。
- 末帧没有后续区间，不能把终端零控制解释成发生过的动作。

公共历史几何门禁对记录区间 `−2≤t≤0`（含端点的 41 个物理帧）中所有车辆对使用三维 OBB 分离轴检查。输入为 `transform_world_vehicle` 及原生 `bbox_extent/bbox_offset/bbox_rotation_deg`，无框膨胀；沿分离轴的最大间隔小于 `−1e-6 m` 为内部相交，边界数值误差不超过 `1e-6 m` 时不据此拒绝。该间隔不是欧氏距离，也不代表原生碰撞网格已经接触。`reports/t0_planning.json` 的共同初态失败原因保存 `common_history_vehicle_bbox_interior_overlap`、车辆对、相对时间、物理帧及 SAT 间隔；原生碰撞事件另列。首次实测及各重放前缀都须通过，缺少几何或历史覆盖时不可验收。该门禁不检查 `t>0`，有效动作引起的未来风险后果仍可保留。

`spec.fork.time_s` 使用“共同受控运动开始”的时钟，Agent 可在 2.5–5 秒内按 0.05 秒间隔配置。对应共同运动预热为 `fork_time_s−2` 秒；保存的公共历史始终为分叉前最后 2 秒，记录的相对 `time_s` 仍为 `−2..4`。不要把可配置的预热/分叉时刻解释为改变保存长度。

跨分支使用 `(scene_id, role, relative_time)` 匹配车辆。绝对 `frame_id`、`simulation_time_s` 和 CARLA `actor_id` 用于本次 episode 的审计，不能作为跨分支恒定身份。

## 模板与地图边界

默认网格 `assets/grid.example.json` 的 12 类模板与 `yielding / normal / assertive` 三套 NPC 参数组合，共 36 格。200 个目标场景分配为每格 5 或 6 景；各格实际完成数由结果清单报告。

| 模板 | Ego 与相关 NPC 的关系 |
| --- | --- |
| `follower_response_straight` | Ego 改变速度，后车使用实时跟驰策略作出响应 |
| `ego_lane_change_straight` | Ego 驶向实际相邻车道，目标车道车辆参与交互 |
| `merge_straight` | 邻道 NPC 沿地图确定的平滑变道路线汇入 ego 所在车道 |
| `merge_ramp` | Ego 在主路，较窄独立入口的 NPC 汇入同一接收车道 |
| `ego_merge_ramp` | Ego 从较窄独立入口主动汇入，观察接收车道 NPC |
| `cross_unsignalized` | 不同进口的道路轨迹相交，采用固定无信号优先规则 |
| `left_unsignalized` | Ego 沿真实左转连接跨越对向直行车辆路线 |
| `left_signalized` | 真实信号化左转连接与对向直行路线，使用固定信号相位协议 |
| `cross_signalized` | 真实信号化交叉路线，冻结各进口信号时序并控制 NPC 遵灯 |
| `pedestrian_straight` | 普通道路上有地图几何依据的行人过街，记录行人控制和轨迹 |
| `pedestrian_unsignalized` | 无信号控制路口附近的行人过街，与车辆实时交互 |
| `narrow_meeting` | 地图真实双向窄路上，对向车辆依据实时距离与道路余量响应 |

两个汇入模板的几何约束为“独立较窄入口汇入较宽连续道路的同一车道”；不单凭名称推定高速公路道路等级。两个无信号模板使用固定关闭交通灯的受控交互协议；不意味着地图上没有信号灯设施。

12 类均参加默认配额分配。具体候选必须满足各自的道路、信号灯或行人路径约束；所选地图没有候选时保留该类缺额，不改名用另一类填充。`narrow_ramp` 仍作为语义不兼容组合排除：同向入口汇入不是双向会车。

信号化模板绑定 OpenDRIVE 中真实信号的标识，并在每个分支执行相同的相位时序。NPC 按实际灯色和停止位置调整速度；ego 保留指定的动作干预，因此可能产生闯灯或接触后果。左转模板采用对向同时绿灯的非保护左转协议。交叉模板采用清空阶段初态：ego 的车头在模拟开始时已越过自身停止线，交叉车辆的保守车框前沿排在其停止线；固定三秒黄灯和 0.5 秒全红后，基础版本在−0.10秒转绿；`interaction-design-v3`按请求阶段冻结为−0.10/−0.80/−1.40/+0.45秒，乐观进入时间随之重算，最终仍由实测t0检查。初始配置检查完整共同历史后的制动、控制响应与停车余量；ego 设计速度额外预留一个物理步制动的速度余量，目标车辆的乐观车头进入估计在未来3秒门槛前再留两个物理步。四车型的2.3米车头前界和2.8米保守半长仅用于规划，仍须由原生实测框、t0速度及机会门禁复核，不承诺实际进入或反应。过去2秒、未来4秒及所有验收门槛不变。ego 先前进入路口的过程没有被观测，不能声称其进入合法；路口内停车或继续进入属于明确的受控条件。配置保存完整时序，采集保存逐灯实际状态。

新编译的行人场景采用实时反馈过街策略：从真实人行道位置等待，根据车辆实测位置、速度、减速度、车身范围与声明的冲突区决定是否开始过街。过街路径固定，实际起步时间不固定；不读取 ego 分支名、未来控制或未来轨迹。车辆同时根据行人的实时状态响应。不同场景可使用不同模型、过街方向、最早准备时间和速度，同一场景所有分支冻结这些参数。当前过街路径不宣称一定是地图上已有斑马线。新编译规格使用 `reactive_crossing_v4`，起步后继续按当前观测加快、减速、停步、有限退让并恢复，见 [行人反馈](pedestrian-feedback.md)；旧 `gap_acceptance_v1` 仍只在起步前反馈。

窄路模板限定每方向一条 Driving 车道、双向总宽度不超过 6.4 米，记录沿程实际宽度；当前默认地图中由 Town07 提供候选。新 `narrow-current-occupancy-v2`先核对沿路线的车身横向空间，允许减速错身；空间不足时根据通道实际占用、冻结优先权和连续清空确认交替通行。它是原生双向窄路和显式通行策略，不等同于单车宽瓶颈，也不人为添加未记录的障碍。Town07 曝光参数为待实际采集检查的配置基线，不是已经验收的画质结论。

新版 NPC 在既定跟驰/路权策略之外，还对当前车身和速度所示的前方短时冲突作出制动响应，不读取 ego 分支或未来轨迹；此启发式不能保证无碰撞。零目标速度使用制动保持，避免油门偏置造成爬行。NPC 风格对应 IDM 加速度、制动、时距、最小间距、期望速度倍率及冲突让行参数，是明确控制配置，不是 Traffic Manager 的隐含风格名。

## 地点、天气与时段覆盖

分配顺序为**交互模板/NPC 风格配额 → 实际地图地点 → 模板内天气分散**。小批量优先给不同交互模板分配名额；地点调度优先选择尚未使用或使用较少的可行地点，再选择路线候选。一个地图路口的多条 connector、同一道路的多条车道不算多个地点。地图本身没有可行候选时保留缺额，不用天气变化冒充新的地理覆盖。

`assets/grid.example.json` 当前包含以下七个项目自定义候选；未列出的参数继承 `scripts/runtime/resources/daylight.json`。

| ID | 条件 | 主要参数 |
| --- | --- | --- |
| `daylight` | 晴朗白天 | 云量 5，太阳高度 45° |
| `soft_daylight` | 部分多云的白天 | 云量 25，太阳高度 50° |
| `late_afternoon` | 午后光照 | 云量 20，太阳高度 35° |
| `overcast` | 阴天 | 云量 90，太阳高度 45° |
| `light_rain` | 小雨、湿路 | 云量 80，降雨强度 20，积水 10，路面湿度 45 |
| `light_fog` | 薄雾 | 雾密度 12，起雾距离 30 m，太阳高度 30° |
| `night` | 夜间 | 太阳高度 −25°；ISO 200、f/4、快门 1/200 s；车辆位置灯和近光灯 |

这些数值是待采样验证的工程配置，名称不代表真实天气量级标定；太阳角度也不是当地钟表时刻。夜间曝光覆盖在地图基础参数之后应用，所有相机和动作分支使用同一组设置；采集核对实际相机属性与车灯状态。新条件仍必须通过原生六相机曝光、可见性及公共历史/repeat 检查，不能因为配置已写入就称为画质合格。

规划器在每个模板内部优先选择当前分配最少的天气，NPC 风格内分布作为次级均衡条件。天气不增加网格维度，也不要求为每个地点复制七份。默认 200 景时每个模板覆盖全部七种天气；预算较小时尽量分散并报告缺项，不牺牲交互类型和地点名额来凑全天气。天气与交互模板没有固定绑定关系。同景五个动作和 QA repeat 始终复用冻结天气。

每个 slot 的 `weather_id` 写入 `plan.json`，修正及替换候选继续使用同一天气。模型只能选择调度器给出的候选地点及已分配天气，不能为通过验收反复改回晴天。`dataset_manifest.json.coverage` 分别报告计划分布、编译分布和最终验收分布；失败造成的天气缺额如实保留，不把计划均衡当成实际均衡。

天气参数控制视觉条件，本管线不随小雨配置更改轮胎摩擦或 NPC 策略，不能由湿路画面推断已模拟湿滑动力学；CARLA 将降雨、积水、湿度作为独立参数。[CARLA 0.9.16 WeatherParameters](https://carla.readthedocs.io/en/0.9.16/python_api/#carlaweatherparameters)

## 场景编写契约

可选模型先选择候选，再填写完整规格；未接模型时模板生成默认规格。模型不生成 RGB、点云或物理真值。第一阶段返回：

```json
{
  "candidate_id": "输入候选列表中的 ID",
  "layout_seed": 123456,
  "speed_scale": 1.0,
  "weather_id": "daylight"
}
```

第二阶段收到 `base_spec`、每位演员的真实地图路线、路线长度、世界位姿、冲突区、信号程序和前次失败报告，按动态 schema 返回完整的 `actors / fork_time_s / profiles / unrelated_roles`：

| 可编写内容 | 具体字段和约束 |
| --- | --- |
| Ego/NPC 初态 | 每个角色的 `spawn_route_s`、`initial_speed_mps`、`desired_speed_mps`、`blueprint_id`；世界位置和航向从真实路线站点解析，不接受任意离路坐标 |
| NPC 响应策略 | 每个 NPC 的 `npc_policy`：IDM 加速度/舒适制动/时距/最小间距、让行时间与距离、碰撞前视与制动参数；全分支冻结 |
| 分叉控制 | `fork_time_s`，时钟和保存窗口见 [样本单位与时间](#样本单位与时间) |
| 五分支计划 | 每条 `profiles` 的启动时刻、加减速强度、速度下限/上限、停止位置、等待/释放/超时、变道时长；类型、方向和角色须来自当前模板的可执行集合 |
| 无关角色声明 | `unrelated_roles`，随后执行跨分支轨迹一致性检查；修正不得通过取消既有声明隐藏失败 |

`scripts/scene_spec.py` 验证 schema、数值范围、出生重叠、停止位置、路线覆盖与变道时间预算。Agent 可以改变间距、初速、车型、NPC 参数和动作时机，不能执行模型生成的代码，不能改验收阈值或分支内 NPC 策略。路线、角色和信号时序属于模板几何契约；需要不同道路时重新选择候选。

第一阶段 `layout_seed` 范围 `[0,2^31)`、`speed_scale` 范围 `[0.85,1.15]`，候选来自地点覆盖优先的输入列表；weather_id 必须使用输入中已经分配的唯一值。第二阶段初速 0–15 m/s、期望速度 0.1–20 m/s 等边界以动态 schema 为准。模型名称不固定；服务需响应选定协议并产出可校验 JSON。

## 输出组织

`weather_variant_id` 标识视觉条件，`weather` 保存继承覆盖后的完整参数；`camera_override`、`render.camera_attributes` 和 `vehicle_light_mode` 保存曝光/车灯要求，夜间另在分支 `meta.json.vehicle_lights` 保存实际读回。跨天气基础身份 `spec.json.base_scene_id` 的定义见 [整景去重与划分](#整景去重与划分文件)。

```text
<dataset-output>/
  dataset_manifest.json
  manifest.json
  index.html
  checksums.json
  maps/<Town>.xodr
  splits/evaluation.json                # 独立评测模式；其他模式按对应split读取
  scene_XXXX/
    spec.json                         # 最终接受的唯一规格
    branches/<branch>/
      meta.json
      states.npz
      controller_diagnostics.json
      pedestrians.npz                 # 配置行人时
      signal_states.json              # 配置信号时
      samples.npz
      camera/CAM_FRONT/00000.png ...   # 六相机原图
      lidar/LIDAR_TOP/00000.npz ...
    common/                           # ego(t0)共同坐标、稳定跟踪ID
    qa/                               # 对应的执行、传感器、重复性和来源证据
```

最终数据没有 `candidate` 或 `attempt` 目录，每个场景只出现一个验收并经整景整理保留的版本。`maintain_repeat` 保留在原分支结构以维护原始规格/身份对应，但在所有模型清单中明确排除，仅用于 QA。

候选与修正过程独立保存在 `--output` 工作目录：`attempts/<scene>/candidate_XX/attempt_XX/`。其中 `initial_plan/` 为共同初始化规格，`common_history_XXX/` 为只到实测 t0 的观测，`data/` 为确认后的动作计划及正式数据，`physics_XXX/` 为完整无图预检，`reports/` 为验收，`export/` 为共同坐标导出。它们不是正式数据集的样本层级。

发布先验证原始封存，保留原图、状态和规格字节，更新交付清单/查看器的相对链接，并生成可独立核对的 `checksums.json`。默认同盘硬链接，跨盘复制；按只读数据使用。原始文件及派生文件分开，不把历史来源路径当成当前读取入口。

逐景只写 `spec.json`，包含地图、演员、路线、天气、相机、时间和身份，以及 `spec_schema_version=3`、`authoring_contract`、`fork`、`execution_contract`、`response_contract` 和逐 NPC 的 `npc_policy`。不另写逐景 `config.json`；`pipeline_config.json` 是批次恢复身份。`spec_sha256` 在分支元数据、清单和报告之间对应。


新批次的正式动作计划在公共历史实测后冻结。`spec.measured_t0_plan` 绑定初始规格、历史状态/元数据及规划证据；Ego/NPC 初态、路线、策略和共同历史配置不得随动作分支改变。规划读取所有实际 t0 位姿、速度、朝向及沿各自路线的冲突距离；必要时整景修正并重新运行历史。默认只检查并确认原 profiles；有明确时机边界才可调整 onset。可达性估算、等待/恢复预算与实际执行结果分开保存，不能把机会检查当成保证 NPC 通行。完整无图分支预检通过后才进入 RGB 采集。

## 状态与可用范围

| 状态 | 含义 |
| --- | --- |
| `planned` | 分布、seed、预算与运行身份已冻结 |
| `compiled` | 真实地图配置和静态几何契约有效 |
| `physics_checked` | 所有分支已通过该阶段物理检查 |
| `accepted` | 物理、RGB/语义 LiDAR、重复性、执行/无关角色检查及导出按管线完成 |
| `exhausted` | 当前配额的有限尝试耗尽，保持缺额 |
| `invalid_delivery` | 已交付文件缺失或 SHA 改变，撤销验收并停止 |

批次还会报告 `complete / incomplete` 及连接中断状态。查看器中可见图像不等于样本已验收；编译数量也不等于采集数量。

## 原始字段

设物理帧数 `T=121`，图像时刻数 `S=13`，车辆数 `N`；当前生产模板 `N=7`。读取数组时使用实际角色表和 shape。

| 内容 | 字段 / 位置 | 单位和意义 |
| --- | --- | --- |
| 六路原图 | `camera/CAM_*/00000.png` | 原生 1600×900、无损 RGB；索引对应 `samples.time_s` |
| 世界位置 | `states.position [T,N,3]` | 米 |
| 姿态 | `rotation_deg [T,N,3]` | `[roll,pitch,yaw]`，度 |
| 运动 | `velocity, acceleration [T,N,3]` | m/s、m/s² |
| 角速度 | `angular_velocity [T,N,3]` | 度/秒 |
| 车辆位姿 | `transform_world_vehicle [T,N,4,4]` | vehicle→world 齐次变换 |
| 控制 | `control, observed_control [T,N,3]` | 发送和读回的 `[throttle,steer,brake]` |
| 控制辅助 | `target_speed`、requested/observed gear 和 manual_gear_shift | 期望速度及变速控制溯源 |
| 3D 框 | `bbox_extent, bbox_offset, bbox_rotation_deg [N,3]` | actor 局部半长宽高、偏移、旋转；extent 不是完整尺寸 |
| 稳定角色 | spec/meta 的 `vehicle_roles` | 跨分支身份由 `scene_id:role` 定义 |
| episode 句柄 | meta 的 `actor_ids` | 仅本次分支 episode 有效 |
| 时间 | `time_s, frame_id, simulation_time_s [T]` | 相对干预时间、绝对仿真帧、仿真时间 |
| 图像对齐 | `samples.time_s, physics_indices, frame_id [S]` | 图像时刻在状态序列中的位置 |
| 传感器帧 | `sensor_frame_id [S,7]` | 六相机加 `LIDAR_TOP`；按 `meta.sensor_names` 解释，已无俯视预览 |
| 相机内参 | `intrinsics [6,3,3]` | 原图分辨率下的内参 |
| 相机外参 | `camera_to_ego [6,4,4]`、`world_to_camera [S,6,4,4]` | 传感器安装变换和 world→CARLA 相机传感器坐标变换 |
| 图像时刻自车位姿 | `ego_to_world [S,4,4]` | 图像采样时刻的 ego→world 变换 |
| 接触事件 | meta 的 `collisions` | 时间、帧、角色/对象和 `normal_impulse`；保留原始事件序列 |
| 重放身份 | spec、manifest、pipeline_config、源码身份和日志 | seed、地图 hash、路线、车型、NPC 参数、动作、天气、传感器、物理和渲染配置 |

CARLA 碰撞传感器事件是物理接触记录；初始包围框重叠检测是另一项几何证据。连续接触可能产生多个事件，事件数不能直接解释成独立事故数。初始化及干预前接触需修正，干预后有效动作引起的接触保留为结果。

## 语义 LiDAR

每条分支的 `lidar/LIDAR_TOP/00000.npz..00012.npz` 与 RGB 的 13 个保存时刻逐帧对应。传感器每个 0.05 秒物理 tick 产生完整 360° 扫描，默认 64 线、80 m、1,280,000 点/秒请求、20 Hz 旋转频率、上/下视场 10°/−30°、`sensor_tick=0`；未命中射线不产生返回点，不能用请求射线数作为每帧实际点数。

| 每帧 NPZ 字段 | 含义 |
| --- | --- |
| `points` | NumPy structured array；`x,y,z,CosAngle` 为 little-endian float32，`ObjIdx,ObjTag` 为 uint32；每条记录 24 字节 |
| `frame_id`、`timestamp_s`、`time_s` | 绝对仿真帧、仿真时钟、相对干预时间，与同帧 RGB/物理状态核对 |
| `lidar_to_world [4,4]` | 从实际 measurement 读取的传感器世界位姿 |
| `lidar_to_ego [4,4]` | 冻结安装标定 |
| `channels`、`horizontal_angle`、`point_count_by_channel` | 实际通道数、扫描角和通道返回点数 |

`points` 保留原生 LiDAR 局部坐标：x 前、y 右、z 上，单位米。转换到公共参考系为 `T_ego0_from_world @ lidar_to_world @ point_local`。`CosAngle` 是入射相关余弦，不是反射强度；语义 LiDAR 未提供的 intensity/noise 不伪造。

`samples.npz` 另存 `lidar_to_world [S,4,4]`、`lidar_to_ego [4,4]`、`lidar_frame_id [S]` 和 `lidar_timestamp_s [S]`。六相机与 LiDAR 同一帧全部收齐后才进入下一个世界 tick。采集后检查原始字段、有限值、通道点数总和、帧、时间、实际属性和刚性安装标定。

`meta.semantic_lidar.actor_to_role` 仅把本次生成的车辆/行人 `ObjIdx` 映射为稳定 `scene_id:role`。未匹配的静态对象或其他引擎 ID 保留原值，不能跨分支据此假设同一 track。原始 ObjTag 保存 CARLA 语义编号；后续类别映射必须显式版本化。LiDAR 点云和 3D 框已具备，但尚未声称生成完整 BEV 语义或 OccWorld occupancy 格式。

## 共同参考坐标

每景以 `maintain` 的实测 `t0` ego 位姿定义唯一参考坐标，所有分支、所有未来时刻共用同一个矩阵：

```python
T_world_from_ego0 = reference['transform_world_vehicle'][t0_index, ego_index]
T_ego0_from_world = np.linalg.inv(T_world_from_ego0)
p_ego0 = (T_ego0_from_world @ np.r_[p_world, 1.0])[:3]
v_ego0 = T_ego0_from_world[:3, :3] @ v_world
```

`ego_index` 从角色表读取。每个分支随自身未来 ego 更新参考系会把自车运动混入周车响应，因此不能用于跨分支后果差异。

`export/<scene>/frame.json` 保存公共变换、单位、轴向和稳定 `track_ids`。每条分支的导出包含：

- `position_ego0_m`、`velocity_ego0_m_s`、`acceleration_ego0_m_s2`。
- `transform_ego0_vehicle` 与 `bbox_corners_ego0_m`。
- `world_to_ego0`、原始时间和帧号。
- 原始相机内参、图像时刻及 `ego0_to_camera`。
- `tracks.json` 中稳定轨迹 ID 到本分支 actor_id 的映射。

框角点由 `±extent` 构造，依次应用框局部旋转/offset、车辆世界位姿和共同参考变换。不能省略 bbox 局部偏移或把半尺寸当成完整尺寸。

CARLA 坐标轴为 x 前、y 右、z 上。`world_to_camera` 保存 world→CARLA 传感器坐标矩阵，还没有转换成针孔坐标；应用外参之后，用 `camera_pinhole=[sensor_y,-sensor_z,sensor_x]` 转换，再乘内参并按深度归一化。导出的 `ego0_to_camera` 也采用这个传感器坐标约定。原始 world 坐标始终保留。

## 模型输入和派生评价

模型输入使用六相机公共历史，并按实验提供分支动作条件。真实未来、碰撞、受影响车辆和区域标注是目标或评价信息。`maintain_repeat` 仅用于质量检查。

完整数值状态和语义点云支持后续生成共同坐标 BEV、reactive 区域、OCC、TTC、最小间距等派生量。当前执行报告已计算车辆跨分支位移差、两档 reactive 数量和弱中强标签；区域栅格、动态补偿、多帧点云融合与语义 occupancy 尚需独立派生。不要把语义 LiDAR 原始点云当作完整 occupancy 真值。

如果使用轨迹差阈值标 reactive，必须结合相同动作重复误差定义噪声水平。未判 reactive 的车辆仍可能正常移动，不能全部标为静态背景。

采集后的缩放或有损下载副本应保存转换参数及对应新内参；不得覆盖原始图像或把低分辨率图片与原分辨率标定混用。

## 行人与交通灯记录

行人不占用车辆数组中的槽位。场景配置 `pedestrians` 保存角色、blueprint、初始变换、地图过街路径、最早准备时间、速度和 `response_policy`；`meta.pedestrians` 保存实际 actor ID、blueprint 与属性。`pedestrians.npz` 与 `states.npz` 使用完全相同的时间、帧号和仿真时间。

反馈策略的 `start_time_s` 表示最早允许决策的时间，不是预定的实际起步时间。车辆快速接近且间隙不足时等待；实测停车让行、低速减速且有足够停车余量、车辆已经清空或当前间隙足够完成过街时可以启动。还要检查声明冲突路线上的其他车辆，不能仅因 ego 停车便忽略邻车。v1启动后沿冻结路线继续；v2/v3/v4起步后持续反馈，v3/v4有有界加快通过，v4另比较当前入口侧的可达停步/退让位置。它们是可重放的规则模型，不是经过实车标定的人类行为模型或无碰撞保证。

`pedestrian_decisions.json` 记录每个物理时刻的阶段及判断依据。v2记录四种、v3/v4记录包含加快的五种运动选择的短时风险、选择原因、前后策略状态及有符号速度；summary 区分命令模式变化与实测停车、反向位移。summary 区分 `command_release_time_s`（首次允许行走）和 `observed_first_moving_time_s`（实测水平速度首次超过 0.05 m/s）；发出命令不等于已经起步。验收从 `states.npz` 和 `pedestrians.npz` 逐帧重算控制、诊断及这两个时间；缺少响应证据或控制不一致不能通过。各分支公共历史必须一致，同动作重复及物理／RGB重放须一致；不同 ego 动作之后允许行人起步和轨迹不同。没有 `response_policy` 的历史记录仍按原固定时间策略解释，不能追认为 reactive 数据。

设行人数为 `P`：位置、姿态、速度、加速度、角速度的 shape 为 `[T,P,3]`；`transform_world_actor` 为 `[T,P,4,4]`；局部框字段为 `[P,3]`。`control` 和 `observed_control` 为 `[T,P,5]`，顺序是世界坐标方向 `[dx,dy,dz]`、速度 m/s、jump。发送控制属于下一物理区间，读回控制属于已经执行的区间；末帧发送控制为未执行的零占位。公共历史检查包含 `t0` 状态，排除 `t0` 的后续区间控制。

`traffic_signal_program` 冻结按相对场景时间排列的相位、OpenDRIVE 信号标识、受控角色及其路线停止位置。`signal_states.json` 保存冻结程序、实际灯体身份/位置和每个物理采样时刻的逐灯读回状态。各动作分支复用整段信号程序，不能依据 ego 分支改变灯色；记录检查会核对实际灯色与计划相位。信号灯不使用跨 episode 的 CARLA actor ID 做共同身份匹配。

`geometry.traffic_design` 及信号程序另存 `initially_committed`/清空阶段证据。NPC 恒速外推的短时框冲突诊断不是 CARLA 碰撞记录；实际接触从碰撞传感器读取。

行人导出使用相同的 `maintain` 实测 `ego(t0)` 变换，另存 `pedestrians_ego0.npz` 和 `pedestrian_tracks.json`，稳定 ID 为 `scene_id:role`，类型标为 `pedestrian`。原始行人、车辆与信号记录仍保留；后续生成占用、受影响集合或其他评价时按演员类型读取。

## 动作策略、执行与观测结果

新版配置 `action.profile_version=2` / `schema_version=2` 保存五个具名 `profiles`、固定 `profile_order`、`nominal_policy`、`conflicts`、真实邻道和设计证据。每个 profile 保存类型、启动时刻、强度/上限，或停车位置、等待释放/超时、变道方向/时长等字段。相同场景所有分支在干预前执行同一 nominal 历史；`nominal` 分支继续该策略。旧版本配置仍可读取。

读取每景 `profile_order`，不按全批动作词汇的并集做笛卡尔积。变道方向由 profile 的 `direction` 决定，不能仅凭分支名称推断；延迟变道也检查完整轨迹和车身框。

| 模板 | 默认五个条件 |
| --- | --- |
| `follower_response_straight` | 保持、轻制动、急制动、加速、延迟急制动 |
| `ego_lane_change_straight`，单侧可行 | 保持、制动、加速、立即向可行侧变道、延迟同方向变道 |
| `ego_lane_change_straight`，双侧可行 | 保持、左变道、右变道、其中一侧延迟变道、加速 |
| `merge_straight`、`merge_ramp` | 保持、减速、加速、让行等待、正常策略继续 |
| `ego_merge_ramp` | 保持、主动进入、延迟进入、停车保持、让行等待 |
| `left_unsignalized`、`left_signalized` | 保持、减速接近、停车保持、让行等待、主动进入 |
| `cross_unsignalized`、`cross_signalized` | 保持、减速接近、停车保持、让行等待、主动进入 |
| `pedestrian_straight`、`pedestrian_unsignalized` | 保持、正常策略继续、减速、停车保持、让行等待 |
| `narrow_meeting` | 保持、低速接近、停车保持、让行等待、主动进入 |

`nominal` 使用显式 IDM 参数、实时前车、冲突区和实际灯色；它是本包的启发式正常策略，尚非经过驾驶质量验证的专家。`maintain` 沿声明路线跟踪实测 t0 速度，不自动让行。轻/急制动保持目标斜率 −1.2/−4.5 m/s²，加速保持 +1.5 m/s²、增速上限 +5 m/s；`decelerate` 默认下限为 1 m/s，不能称为停车。低初速下须查看完整轨迹，不能只比较末速。

基础动作目录的停车/等待规格声明 `stop_deadline_s=3.5` 和 `stop_margin_m=0.75`，分别是尚需停车时的实测停车期限和最大停车位置之前的控制余量；最新五动作目录及信号化交叉的停止期限例外见下文。控制使用实测 t0 速度和空间制动约束，允许提前停住，原有 0.25 m/s 速度和 0.5 m 越界验收阈值不变。等待位置必须让 ego 的整个车身留在本次冲突区之外，不阻挡指定对象的路线。

`stop_hold` 制动并保持到声明窗口结束，不因对方通过而自动恢复。新 `yield_wait` 明确指定让行对象；恢复必须同时满足：指定对象已经通过并清空冲突区，以及 ego 当前可以进入。车辆按实际姿态、框偏移和尺寸确认车尾清空；行人需整个身体离开 ego 将经过的区域。区外停车、没有 ETA、已等待 0.25 秒都不是“已通过”的证据。指定对象清空之后，其他车辆、行人或观测到的阻挡及适用灯色仍可禁止恢复，恢复后也持续检查阻挡。

新规格使用 `release_condition=specified_actors_cleared_and_entry_clear_v1`，`wait_for_roles` 固定本次冲突的全部指定参与者；Agent 修正不能删掉目标或退回旧释放规则。当前通行检查覆盖已观测车辆、行人包围框及信号；不宣称能检测未纳入观测的任意静态障碍。完整车身等待位置先用保守尺寸检查，再用公共历史结束时的实测尺寸与框偏移复核，默认净空 0.5 m。

执行验收另外输出 `checks.ego_physical_execution.waiting_footprint`：在干预后、接触前且尚未释放的静止时刻，检查实际 ego 车身是否已侵入声明冲突区。它使用实际位置、朝向和包围框，不以规划停车点正确代替实际执行正确。`target_passage_required_for_release=true` 与 `target_passage_required_for_scene_acceptance=false` 分别表示释放前必须通过、整景验收不强迫对方通过。

若对方在 ego 减速过程中已经清空且当前可进入，可以滚动恢复，不要求为 `wait` 名称先完全停车。四秒内对象没有清空而 ego 正确等待，属于有效等待结果；不强迫 NPC 通过，不将等待超时当作控制失败。恢复速度单独声明，不用零初速作为恢复目标。新旧释放协议有显式版本区别，旧数据仍按原冻结策略审计。

`commit_go` 不执行上述冲突让行，沿固定路线提高目标速度；`commit_delayed` 在一秒启动前按声明停车位置接近和制动。它们都不能保证先通过。未遵灯、危险切入及接触应按风险条件解释；NPC 和信号时序不随分支改变。停车位置与信号停止线分别保存，不能默认所有干预都遵灯。

真实变道跟踪同向邻道中心线，默认控制过渡2.2秒，另留0.9秒收敛及0.5秒保持。新五动作目录不再依赖延迟0.4秒的同向变道构成第五条：双侧可用时采用 maintain/left/right/brake/accelerate，单侧可用时采用 maintain/brake_light/brake/accelerate/该侧。旧规格仍按其冻结时长重放。地图许可、实际包围框/中心/航向检查保持原标准。路口左转或匝道汇入是固定路线，不等于 `left/right` 变道分支。

交互机会由初始几何、速度、道路冲突位置和观察窗口构造；配置保存估计到达时刻、停车距离及目标速度曲线。窄路优先条件可以在不同场景间变化，同景固定。反馈行人何时实际起步只能由运行时状态决定，事前可达性或间隙估计不保证四秒内完成过街。弱反应、一直等待及有效控制后的碰撞如实保留；当前旧数据不因代码更新自动获得新策略标签。

`controller_diagnostics.json` 按物理区间保存策略阶段、停车/释放/超时和当前冲突依据；不要把目标速度当作实际速度。实际速度在 `states.velocity`，实际控制在 `observed_control`，命令在 `control`。诊断中风险条件不等于已经发生事故，碰撞仍由传感器记录。

原始 manifest、批次 `dataset_manifest.json` 和公共坐标 `frame.json` 的 `model_branches` 都保留该景五条非 QA 控制条件。旧协议仅诊断完整未来运动相似度，不能宣称五条轨迹不同。新编译场景增加下述整景门槛；失败候选也保留全部分支，不移除相似分支。

### 五种有效 ego 动作的新纳入标准

规格冻结 `action_separation_contract.policy_id=five-effective-ego-actions-v1`。时间协议仍为过去2秒、未来4秒；共同 t0 实测速度至少2.5m/s。五条正式分支的10对必须全部满足：

- 实际送出的油门/制动差至少0.1，或转向差至少0.05，连续至少0.25秒；末帧未执行的控制不计。
- 实际水平位置差至少1m，或水平速度差至少0.5m/s，同一种指标连续至少0.5秒；不只比较末点或单帧峰值。
- 每项门槛还须达到独立 maintain repeat 噪声的5倍；原有 repeat、公共历史、控制执行、物理和图像门槛仍须通过。
- 以上差异须在两条分支中首次 ego 接触之前建立，碰撞后的位移不能单独构成第五种动作。已经有效区分后的接触继续保存。

这些是采集前声明的工程纳入标准，不是从评测集估计的显著性阈值。任一对失败，整个候选不进入新标准数据集；失败原始证据保留。正确等待仍可通过执行验收，但不足以区分五动作的整景不能通过此新增门槛。

新行人目录以 accelerate 替代可能等同于 maintain 的 nominal；合流目录用 brake 替代 nominal。stop_hold 通常使用2.25秒停止期限和6m/s²设计制动能力；cross_signalized 的紧停止空间例外按初态固定为 `onset_s + response_time_s + initial_speed_mps / comfortable_deceleration_m_s2 + 2*dt_s`：该模板先将名义响应0.75秒加上20Hz下两tick跟踪/离散余量0.10秒，冻结 `response_time_s=0.85`；零目标持续时间仍为初速除以设计能力6m/s²再加两tick速度漂移余量0.10秒，初速2.8m/s时完整停止期限约1.4167秒。增加的响应余量同时收紧空间速度包络并预留物理完成时间，不改变零目标时间斜率或QA响应宽限，使其时间目标早于温和让行；yield_wait 仍使用3.5秒期限和4.5m/s²设计能力，全部指定对象与实时入口释放规则不变；实际速度、控制和停止位置必须实测验收，参数不等于实际加速度。二者即使都未恢复，也须通过完整的两两运动门槛。

新行人从真实人行道起步；默认近侧，几何与时间允许时使用另一侧。路缘偏移至多0.75m且保留0.35m。`reactive_crossing_v4` 可在共同预热中开始决策，保存完整预热观测和策略状态并复算到 −2 秒；不会直接把人摆到路中。v1/v2保留其原时序语义。实测t0仍检查至少1秒估计观察余量；v3/v4的背景恒速间隙作为诊断，实际启动由带车身几何的逐时刻扫掠决定。只使用当前/过去状态和冻结参数，不保证实际起步、清空或4秒内恢复。十二模板的分层、动作延迟与覆盖说明见[交互多样性](interaction-diversity.md)。

`action_execution.json` 与 `action_selection.json` 新增 `action_separation`，含每对持续时间、接触截断时刻、重复噪声和失败原因。`execution.json` 的 `response_metrics.relevant_actor_response` 逐个相关车辆/行人记录10对水平位移、速度差、起步时刻及repeat噪声。NPC响应差异只报告，不按分支名称生成，也不要求每个对象对所有动作都产生不同响应。旧200景的原始数据及资格结论不变，不能追认为满足本标准。

`execution_contract` 在采集前冻结，模型修正不能放宽它。现有默认包括速度跟踪容差 2 m/s、0.75 秒响应宽限、持续超差 0.75 秒检查、可观测目标速度变化的至少 50%、停车速度不超过 0.25 m/s、停车位置越界不超过 0.5 m，以及真实变道末段连续 0.5 秒车身进入目标道、中心误差不超过 0.5 m、航向误差不超过 10°。这些是工程验收参数，须用后续实际控制数据验证适用性，不能称为已验证的物理性能。

`audit_execution.py` 根据记录的当前状态重算冻结策略，核对发送命令、下一帧读回控制、目标与实际运动。规定减速未实现、停车越界、控制不符或无接触却未完成规定变道等进入有限修正；接触前已发生的错误仍判失败。

`reports/physics_execution.json` 与 `reports/execution.json` 的 `scenes[scene_id]` 包含场景级 `execution_pass`、失败列表、各分支检查、`response_metrics`、`unrelated_invariance` 和接触截断列表。各分支字段为：

| 字段 | 解释 |
| --- | --- |
| `execution_pass` | true：被检查范围执行通过；false：控制/运动未达规格；null：有效早期接触导致可观测运动不足，不宣称已经充分验证 |
| `failure_reason` | 具体错误列表，如策略/实际控制不符、未减速、停车越界、声明无关车辆变化；为空不等于末端完成 |
| `quality_status` | passed / failed / censored |
| `execution_scope` | 全窗口或严格接触前运动与策略命令 |
| `observed_outcome.ego` | 实际保持、加减速、等待、恢复、超时、变道完成/未完成、接触等事件 |
| `observed_outcome.npcs[role]` | NPC 实际速度、路线进度、约束原因及事件；不强制响应 |
| `observed_outcome.collision`、`first_contact_s` | 是否有接触及首次接触时间 |
| `observed_outcome.censored_after_contact` | 接触后运动不用于证明未受干扰的执行完成 |

正常等待不要求最终通行；NPC 沿冻结策略停住或未完成汇入可以是有效后果。当前固定路线 NPC 没有独立“取消路线”状态，`cancellation_state_supported=false`，不把等待擅自解释为取消。以后实现明确取消时可记录为后果，不能混淆为控制失效。

`action_execution.json` / `action_selection.json` 保留既有文件路径，分别保存物理和正式阶段的 ego 运动相似性诊断。规范入口为 `audit_motion_similarity()`，每景字段如下；它们不代替上述执行验收，也不删除分支。

| 字段 | 含义 |
| --- | --- |
| `model_branches` | 完整五个非 QA 控制条件 |
| `motion_audit_valid` | 轨迹相似性诊断的数据与重复噪声检查是否有效 |
| `distinct_motion_count` | 按完整未来运动及重复噪声得到的不同代表轨迹数；诊断无效时为 null |
| `equivalent_motion_count` | 条件数减去代表轨迹数，即与代表相似的额外分支数，不是相似分支对数；诊断无效时为 null |

旧 `select_branches()` 作为兼容包装器保留；`normalize_motion_report()` 统一读取旧 JSON 字段。新报告不再传递 planned/selected 双列表或恒为空的 `excluded_duplicate_branches`。

## NPC 响应与无关车辆

对每个非 ego 车辆 `j`，在五条非 QA 分支所有无序分支对和相同 `t>0` 时刻中取水平位移差最大值 `d_j`，使用同一公共坐标。`response_metrics` 包含：

| 字段 | 精确定义 |
| --- | --- |
| `max_npc_displacement_diff_m` | `max_j d_j`，含有效接触后的实际结果 |
| `reactive_npc_count_05` | 满足 `d_j > 0.5 m` 的 NPC 数量 |
| `reactive_npc_count_10` | 满足 `d_j > 1 m` 的 NPC 数量 |
| `effect_strength` | weak：最大值 ≤0.5；medium：0.5 < 最大值 ≤1；strong：最大值 >1 |
| `max_npc_repeat_xy_noise_m` | maintain 与 maintain_repeat 的 NPC 未来水平位移差最大值 |
| `threshold_05_above_repeat_noise` / `threshold_10_above_repeat_noise` | 重复误差是否严格低于对应标签阈值 |
| `per_npc` | 每车所有分支对最大差、峰值时刻和 repeat 误差 |

这是五分支全配对统计，不只计算相对 maintain 的差。`used_as_acceptance_gate=false`；弱反应和零反应不会使生产失败。标签是固定工程分桶，不是因果显著性检验。反应标签仅用于分析/数据划分，不作为视觉模型的输入。

`unrelated_invariance` 对 spec 明确声明的无关角色，检查全部六分支（包括 repeat）、全部未来时刻的 3D 位置、速度和环绕航向。默认三项容差均为 0；角色变动或重复误差超出契约即失败。没有声明角色时报告空集合通过，不等于声称所有 NPC 都无关。修正不得删除既有无关声明、改变分支内 NPC 规则或改写轨迹以通过检查；未声明角色允许直接或间接受到影响。

## 整景去重与划分文件

`dataset_purpose=evaluation` 表示整个批次用于评测：所有保留整景在工作目录记入 `curation/splits/evaluation.json`，发布后使用最终数据根的 `splits/evaluation.json`，不产生训练划分，不在评测记录上拟合相似阈值或尺度。它只处理确定的初态重复，继续保护不同后果、动作计划和视觉条件，并单独报告独立基础场景数。近重复及外部训练集重叠尚未核验，不能把批内身份检查解释为已经通过外部训练泄漏检查。下述训练校准与 80/10/10 划分适用于默认 `dataset_purpose=train_eval`。

去重单位是**完整配对场景组**，不会删去同景 `maintain/brake/accelerate` 等分支。它首先在模板 × NPC 风格内比较实测 `maintain(t0)` 的相对位置、相对速度、航向、包围框尺度、恒速包围圆 TTC 代理及沿声明路线到冲突位置的时间。无穷/无事件时间用 20 秒上界和独立有效掩码保存，不能把缺失时间默认为真实 20 秒碰撞。

批次 `accepted_scene_count` 是生产质量通过的原始整景数；`curation_manifest.retained_scene_count` 是去除整景近重复后的使用数量，两者应分别报告。网格原始配额完成不保证去重后各格仍等量；整景排除与弱中强覆盖不足都保留明细，不能把 retained 数量冒充实际采集产量。

阈值和尺度只用独立训练校准集，或按初态指纹确定、预留给训练的 35% 校准候选拟合；不使用验证/测试候选拟合。每个模板/车辆数分组使用 IQR 尺度和明确的数值下限；近邻距离的第 10 百分位乘 0.5 作为保守候选阈值，并限制单坐标差。少于四个校准场景或阈值为零时，只处理严格初态重复并报告校准不足。这是可审计的分布推导规则，仍需审查后冻结，不宣称阈值已经由真实数据验证。

高度相似、五分支动作规格和观测事件签名相同的整景才可合并，先排除质量失败，再按显式 `quality_score` 及稳定顺序选代表。相似初态但 NPC 等待/进入/碰撞等事件、动作计划或响应强度不同的对照被保护。相近组无论后果是否相同，都必须放在同一个 train/validation/test 中；完全相同的实际初态或声明出生布局跨风格、跨模板也不能泄漏到不同集合。接近决策边界的差异不会因为“长得像”而被删掉。

划分目标为 80/10/10，按模板/风格/弱中强响应平衡；校准场景及其近邻只能在训练集。满足独立组数量时尽量保证各强度覆盖，否则 `coverage_shortfalls` 报告不足，不能拆散一组反事实或跨集合复制相同初态来凑数。已有输入 split 标签中的近重复也会单列冲突并给出新的整组划分。

`base_scene_id` 由地图、ego/NPC 出生位姿及初速确定，忽略天气、seed、文件名和路线补点；若显式 ID 与实际规格不符则拒绝。不同天气、曝光和车灯版本可以具有不同 `scene_id`，但始终归入同一训练/验证/测试组。同一基础场景每个特征组只取按 scene_id 排序的首个版本拟合尺度，避免天气副本重复增加校准权重，不按未来效果选择版本。清单以 `retained_scene_count` / `retained_base_scene_count` 分别报告版本数与基础场景数，划分覆盖按基础场景统计；近重复分组仍适用，不能更换 seed 绕过检查。

| 文件 | 内容 |
| --- | --- |
| `curation_manifest.json` | `selection_unit=whole paired scene`、保留/排除整景及代表映射、保护的不同后果对照、划分和覆盖缺额 |
| `feature_table.json` | maintain(t0) 实测特征及初态/出生指纹 |
| `curation_policy.json` | 训练校准来源与 SHA、尺度、阈值、特征定义及校准实例 |
| `leakage_audit.json` | 近重复连接组、跨风格/模板同初态联系、输入 split 冲突、校准重叠及跨集合检查 |
| `splits/train.json`、`validation.json`、`test.json` | 每景完整五分支清单 |

排除仅改变使用清单，原始数据不删除。

普通生成的行人 `entering/crossing` job可冻结 `pedestrian_pace`（`unhurried/ordinary/brisk`）。新任务按阶段交叉均衡档位，候选修正保留job字段；该字段只影响初始名义步速及联合布局，不传给反馈模式选择，也不是期望结果。实际采集仍报告真实t0阶段和步速。

## 联合随机生成与分母记录

当前 `interaction-design-v5` / `joint-scene-randomization-v1` 在模板内改变参与者数量、连续交互进度及逐对象偏好。信号绿灯/黄灯起始时间也在相应阶段的有界区间内连续采样；上文离散数值描述对应历史版本，不是新分布唯一取值。`sampling_proposal.json` 在编译前落盘，最终规格保存 `randomization.effective` 和真实地图道路/车道链；原生实测阶段另报。完整规则、边界及 `randomization_report.json` 的阶段分母见[联合随机生成](scene-randomization.md)。历史2秒、未来4秒、每景公共初态/策略、全部10对ego动作及原QA门槛不变。


### 新增相遇校准和 NPC 汇入证据

`spec.encounter_calibration` 冻结目标、容差、最大修订数与布局改动依据；`reports/encounter_calibration_*.json` 绑定每次实测公共历史。最终公共历史仍为−2..0秒，未来0..4秒，全部五模型动作及QA重复重放同一个最终初态。

带 `route_maneuver.policy_version=observed-gap-merge-v1` 的车辆冻结映射的源/目标路线、最早意图时刻、间隙参数和有限取消规则；`vehicle.route` 保留名义交互几何，实际跟踪路线由当前状态选择。分支 `npc_merge_state.json` 保存预热观测、决策和−2秒/t0内部状态；公共历史门禁逐值比较，策略验收从预热复算，源码身份包括 `npc_merge.py`。全部分支的同一车辆参数一致，控制器不能读取分支名或未来。


### 行人步行执行核验

在公共历史和六条无图及传感器分支，以20 Hz对齐本帧原生步行指令、下一帧控制回读和同一区间的地面位移。连续0.50秒方向稳定、水平步行指令至少0.40 m/s，而实际水平路径长度不超过0.05 m，判为步行受阻并拒收。与该行人有关的动作后车辆接触单独记录，保留接触后果。停步指令下的静止可正常验收。实测 t=0 速度低于0.05 m/s 时，同时核对上一实际施加的指令，区分等待、启动过渡及执行异常。检查只要求原生运动执行，允许所有分支有相同的合理等待。

混合渲染版本的正式发布在集合 manifest.json 的 scene_render_presets 中按 scene_id 记录各景覆盖文件及其 SHA256。未列入映射的景读取根目录 render_preset.json。每景的原生 spec、分支记录和源文件 SHA 保留；读取配置时使用 check_frozen_render_preset 核对实际采集版本。
