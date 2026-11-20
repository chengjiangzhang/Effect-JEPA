# 生产工作流

含待办行人配额的原生采集先使用空地图检测参与者实际能否生成。查询 `<output>/walker_spawn_checks/` 可查看两端与各模型的结果。每景使用的实测位置与记录哈希保存在 `reports/walker_spawn_checks.json`。纯编译支持离线运行。详见[候选与相遇校准](candidate-calibration.md)。


场景、动作、传感器和验收定义统一见 [数据契约](data-contract.md)；本页说明运行、模型连接、恢复与调试。

## 1. 环境检查

本包包含生产代码，不包含 CARLA 二进制、地图资产或 GPU 驱动。需要安装 CARLA **0.9.16**，Python 环境支持同版本客户端，并保证运行账户可使用渲染 GPU。服务端由非 root 账户运行。

解压后进入包含 `SKILL.md` 的 `carla-paired-data/` 目录。下列路径为部署示例，替换为当前机器的安装目录和数据盘目录：

```bash
cd /path/to/carla-paired-data
python -m pip install -r requirements.txt
export CARLA_ROOT=/opt/carla-0.9.16
python scripts/pipeline.py doctor --carla-root "$CARLA_ROOT"
```

默认读取 `${CARLA_ROOT}/CarlaUE4/Content/Carla/Maps/OpenDrive`；也可以用 `--opendrive /path/to/OpenDrive` 指定地图源。默认地图是 Town04、Town05、Town07、Town10HD，Town07 提供本包窄路会车所需的双向窄车道候选。只提供 OpenDRIVE 足以编译配置，实际采集还需要服务端安装相应渲染地图。

工作记录和最终数据都放在 skill 之外，彼此分离。`--output /data/carla-runs/batch_001` 保存规划、候选、修正和日志；`--dataset-output /data/carla-paired/batch_001` 保存可直接读取的最终场景。省略后者时默认为工作目录同级的 `<output>_dataset`。两者均可放数据盘，无固定项目路径依赖。

## 2. 托管服务与渲染复现

正式成对 RGB 采集使用独立托管服务，无需先手动启动 CARLA。创建独立配置，例如：

```json
{
  "carla_root": "/opt/carla-0.9.16",
  "user": "carla",
  "log_root": "/data/carla-server/batch_001",
  "fixed_seed": true,
  "startup_benchmark": true
}
```

`carla` 为已有 GPU 权限的非 root 运行账户。在 `run` 命令中传入 `--managed-server-config /path/to/server.json`。管线在每个候选物理预检前、每个 RGB 分支前重启自己拥有的服务，核对进程退出、Engine.ini 恢复、端口释放、版本和实际渲染地图；不接管其他服务。每个分支完整重放公共历史与动作，原始传感器和验收要求保持不变。

在精确匹配的 Linux Vulkan SM5 构建上，托管入口在服务器日志根目录自动准备一次独立的修正版运行目录，同一日志根的后续分支复用。手动启动时用本包的 scripts/server.py。每个服务的 dfao_shader_read.json 记录实际缓存读取，场景预设摘要记录期望版本。[渲染复现](render-reproducibility.md)


包内天气随机流仅在新原生进程中初始化一次，地图重载不会重置它。托管入口需要校验实际加载的库、资源身份、天气调用日志及同步设置读回；完成检查后才可将采集纳入验收。详细范围、固定构建限制、资格证据及 RGB 阈值见 [渲染重复性](render-reproducibility.md)。长期运行的手动服务可用于环境诊断，不能据此声称符合独立分支的随机状态复现协议。

2026-11-04天气修正版本的Town04小雨、Town05无雨薄雾与普通入口资格保留为历史证据。2026-11-10新增的贴花稳定排序有单独的源码身份、原生证明和验收，当前状态及适用范围见[渲染重复性](render-reproducibility.md)。这些固定案例不代表任意场景或新生产批次已经通过验收。已撤销的RGB重试限额及候选排除保护不属于当前策略。

`startup_benchmark: true` 在启动命令中添加 `-benchmark FPS=20`，并保存实际参数；CARLA 后续应用世界设置可能改变引擎计时状态，它不保证全生命周期固定时钟。首次 RPC 就绪后立即设为同步 0.05 秒步长；切图保留设置并读回核对。实际图像仍须逐景通过验收。更改已经冻结批次的启动或原生库配置时，使用新的批次和来源身份。

同一 CARLA 世界由一个采集进程推进同步 tick；多任务并行必须使用独立服务、端口及不冲突的引擎配置账户。

单分支采集有600秒总执行预算，并输出初始化、渲染预热和采样进度。采样入队日志不表示文件已完成，完整性以最终检查为准。结构化采集超时或服务生命周期错误中断批次，保留同一场景供恢复，不把基础设施故障当成修改布局/天气的理由或耗尽修正预算。

## 3. 一次运行完整流程

```bash
python scripts/pipeline.py run \
  --carla-root "$CARLA_ROOT" \
  --output /data/carla-runs/batch_001 \
  --dataset-output /data/carla-paired/batch_001 \
  --count 200 --seed 20261028 \
  --managed-server-config /path/to/server.json \
  --max-repairs 3 --candidates-per-slot 2 --max-candidates 400 \
  --stage all
```

`run` 自动冻结网格配额、seed、预算及源码身份，然后逐景完成初始规格编译、公共历史实测、分支计划确认、全部分支无 RGB 物理预检、正式采集、独立验收和共同坐标导出，最后整景去重、生成划分并发布干净的场景目录。默认网格为 `assets/grid.example.json`；修改网格、地图或协议后使用新输出目录。

仅生产独立评测集时，在首次运行命令中增加 `--dataset-purpose evaluation`，恢复时保持该参数。全部保留整景写入 `curation/splits/evaluation.json`；不划出训练集，不用这批数据拟合距离尺度或相似阈值。此模式只按明确的初态身份处理整景重复，保留不同动作计划、后果和天气条件；尚未检查与外部训练数据的重叠。默认 `--dataset-purpose train_eval` 保持原有训练／验证／测试划分。

`max-repairs=3` 是初次尝试之外最多三轮修正；`candidates-per-slot=2` 是每个配额最多两个候选，整批还受 `max-candidates` 限制。失败尝试保留配置、原因和尝试号；修正与候选替换不改变网格归属或 slot 天气，耗尽后报告缺额。

可选运行参数 `--defer-weather light_rain` 暂缓指定天气的未完成场景，只改变本次调度，不改冻结的场景 ID、天气配额、seed 或验收要求，也不消耗被暂缓场景的候选／修正预算。状态、事件和交付报告记录本次过滤条件及具体缺额；存在未完成配额时不报告整批完成。使用相同源码和科学配置加 `--resume`、省略 `--defer-weather` 即释放暂缓任务；已验收数据保留，已耗尽的候选不会因此自动复活。默认不暂缓任何天气。

正式采集先运行 `maintain` 和 `maintain_repeat`，通过参考重复检查后再采其他分支。完成后执行全分支原始数据、执行与传感器检查，并核对物理预检和正式采集的数值等价。只有完成全部门槛和导出的场景进入 `accepted`。渲染请求为 Epic，曝光、纹理、阴影和反射仍需浏览实际六路原图检查；新增地图或天气同样需要运行证据。

新编译场景另须通过 `five-effective-ego-actions-v1`：无图阶段先检查10对 ego 动作，全部通过才采 RGB；正式阶段重算同一门槛并核对数值。公共历史2秒、未来4秒、20Hz物理与2Hz六相机/语义LiDAR保持不变。阈值是预先声明的工程标准，不使用旧200景或新评测数据拟合。新门槛改变场景纳入标准，必须建立新来源/批次，不能将旧200景追认为新标准数据。

## 公共历史实测后确定动作计划

1. 静态编译结果保存为 `initial_plan/`，作为不可覆盖的共同配置版本。
2. `preflight_physics.py --history-only` 只运行共同初始化和最后两秒历史，到真实 `t0` 观察后停止；不执行未来动作，不创建 RGB 传感器。
3. 读取 Ego、NPC 及已配置行人的实测位置、速度、朝向、车辆尺寸、沿各自路线的站点与冲突区距离，并读取实际信号状态。测量文件及初始规格的 SHA 随报告保存。
   在确定动作计划前，使用记录的车辆刚体变换和原生局部包围框尺寸、偏移、旋转，逐帧检查公共历史 `−2≤t≤0` 内所有车辆对的三维 OBB；包括 `t0`，不读取未来。无膨胀 SAT 间隔小于 `−1e-6 m` 视为包围框内部相交，作为共同初态质量失败返回整景修正，并保存车辆对、帧、时间与间隔。`1e-6 m` 仅为数值边界误差，不是安全距离。缺少完整历史或几何记录时停止验收，不补造几何。包围框相交与 CARLA 碰撞传感器接触分别记录，不互相冒充。
4. 从这些实际值检查四秒接近机会、制动距离、停车期限、变道时长、等待对象的通行机会和等待/恢复时间预算。常速 ETA、乐观可达上界、不能保证恢复的警告分别记录；估算不是实际执行证明。普通入口默认确认原动作计划；不可行时进入整景修正。规划函数另支持显式传入 `onset_s` 允许区间做受限时机调整，不改加减速度、停止期限或超时门槛。
5. 通过后，在新的 `data/<scene>/spec.json` 冻结正式计划，并绑定 `measured_t0_plan` 证据。之后先重放所有完整无图分支并验收，再进入 RGB/LiDAR 阶段。
6. 全部分支的公共历史必须与首次实测前缀逐字段一致：状态检查包含 `t0`，发出的控制仅比较 `t<0`；`t0` 控制作用于下一段未来，允许因动作而异。正式采集还要通过原有 RGB 前缀和重复性检查。
   前缀审计对首次实测和实际重放两侧分别执行同一公共历史 OBB 检查；逐字段一致不代表共同初态本身有效。未来 `t>0` 的相交或传感器接触不由此门禁剔除，仍按动作执行与风险后果协议记录。

如位置或时机不满足必要条件，失败报告返回整景修正流程：在新配置版本统一调整出生点、间距、历史策略或合法动作计划，再重新执行公共历史。不得对单个分支修改初态或 NPC 策略。实际等待超时、NPC 放弃汇入、弱反应及有效控制后的碰撞仍属于观察结果，不按反应幅度筛除。

`reports/t0_planning.json` 保存实测规划与时间预算；`physics_window.json` 单独保存完整无图预检观察到的停止、等待释放和剩余观察时长，不回灌到事前规划，不作为强制NPC通行的筛选门槛；`history_physics.json`、`history_capture.json` 保存独立实测前缀与全分支物理/传感器采集的对应检查。本协议代码有离线验证，新批次仍需真实 CARLA 运行验证，既有数据不追认成采用了此流程。

新编译的行人模板保存实时反馈规则；从冻结的最早决策时刻起（v3可位于共同预热内），行人是否起步取决于当帧车辆位置、速度、减速度和车身占用。各分支使用完全相同的行人初态、路线及规则，不按 ego 动作名切换行人行为。验收保存并重算 `pedestrian_decisions.json`，同时检查实际控制和运动。最早准备时间、释放行走命令和实际开始运动是不同事件。

新 `yield_wait` 在规格中冻结指定让行对象和版本化释放条件。只有对象整个身体已清空冲突区且当前可进入才恢复，不能用 ETA 缺失或等待时长代替实际通过。停车点按整个 ego 车身及目标路线检查；其余观测车辆、行人和适用灯色仍参与恢复判断。可在减速途中满足条件而继续行驶；正确一直等待不判为“未完成通行”。`stop_hold` 不随对象通过自动恢复。详见[动作语义](data-contract.md#动作策略执行与观测结果)。

## 可选：接入模型

不指定 `--model-config` 时使用按 seed 确定的默认规格。模型参与时先选真实地图候选，再填写完整场景并根据失败报告有限修正；可编辑字段和不可放宽的约束见 [场景编写契约](data-contract.md#场景编写契约)。

复制 `assets/model-config.example.json`，填写模型 ID、API 根地址 `base_url` 和密钥环境变量名 `key_env`。密钥通过进程环境注入（例如 `SCENE_MODEL_API_KEY`），不写进配置正文。在上面的首次 `run` 命令添加 `--model-config /path/to/model-config.json`，恢复时继续传相同配置。

| `provider` | 可连接的接口 |
| --- | --- |
| `openai_chat` | Chat Completions 及兼容接口 |
| `openai_responses` | Responses 接口 |
| `anthropic` | Messages 接口 |
| `gemini` | GenerateContent 接口 |
| `command` | 自定义本地程序适配其他 API 或本地模型 |

自定义适配器配置示例：

```json
{
  "provider": "command",
  "model": "custom-model",
  "command": ["python3", "/path/to/provider_adapter.py"],
  "timeout_s": 60,
  "max_calls": 1000
}
```

适配程序从 stdin 读取一个 JSON 对象，包含 `protocol`、`model`、`system`、`request`、`schema`；stdout 返回一个满足 schema 的 JSON 对象。程序可封装其他供应商 SDK。管线使用 argv 执行适配器，不执行模型输出的代码。

API 的尝试次数和连接重试有上限。连接故障会记录连接中断状态，恢复连接后使用同一命令加 `--resume`；不会把 API 故障冒充有效场景结果。

## 读取交付与恢复

```bash
python scripts/pipeline.py inspect --output /data/carla-runs/batch_001
```

正式数据从 `--dataset-output` 读取：根目录直接包含 `scene_XXXX/`、清单、地图及查看页，不包含候选/修正目录。`scene_XXXX/spec.json` 是最终规格；`branches/` 为五个模型动作及一个明确标记 QA-only 的 repeat，`common/` 为共同坐标导出，`qa/` 保存对应证据。`dataset_manifest.json` 与 `splits/` 使用此根目录下的路径。部分交付明确记录数量与未完成目标。工作目录的 `dataset_manifest.json` 则仅是生产溯源入口。文件树见 [输出组织](data-contract.md#输出组织)。

训练、验证和测试按最终数据根目录 `splits/` 的对应清单读取去重后保留的整景；验收通过数量与最终使用数量分别统计。

中断或断电后，先确认 CARLA 服务仍在运行；否则按第 2 步重启，并运行 `doctor --connect` 检查连接，再用原 `run` 命令加 `--resume`。它要求配置、地图、网格、模型配置及源码身份不变，允许从编译推进到预检和全流程；改变这些内容须新建批次。输出目录有独占进程锁，避免同时推进同一任务。

成功交付的 `reports/delivery_files.json` 对原图、数值、规格、导出和关键报告逐文件记录 SHA-256；恢复时重新核验。文件缺失或被修改会撤销验收并停止，不能仅凭已有图片继续计入产量。API 连接中断使用同一恢复路径。

已暂停或历史批次可单独发布，不启动模拟器：

```bash
python scripts/publish_dataset.py \
  --dataset-manifest /data/carla-runs/batch_001/dataset_manifest.json \
  --curation-manifest /data/carla-runs/batch_001/curation/curation_manifest.json \
  --output /data/carla-paired/batch_001
```

发布要求已有明确的整景整理清单，逐文件验证原封存后再导出。默认同盘硬链接、跨盘复制，不使用依赖外部目录的软链接；数据按只读使用，若需要独立修改的副本请指定 `--link-mode copy`。未完成配额不标为全量完成。不会删除原工作记录、把历史失败换成通过，或把旧数据标记成新规划协议采集。

## 调试与独立操作

默认全流程不需要逐阶段命令。排查问题时，保留同一组输入参数，用下表命令替换入口或阶段；对已有输出均加 `--resume`。

| 操作 | 命令选择 | 用途 |
| --- | --- | --- |
| 仅规划 | 将 `run --stage all` 换为 `plan` | 只冻结配额、seed、预算和身份，不调用模型或模拟器 |
| 静态编译 | `run --stage compile` | 查询 OpenDRIVE、生成并校验完整 spec，无需服务端 |
| 物理预检 | `run --stage physics` | 编译后运行全部动作和重复分支，无 RGB；需要服务端 |
| 继续生产 | `run --stage all --resume` | 从已有状态继续采集、验收及导出 |

`compiled` 只代表静态契约有效，不能证明实际生成或采集成功。物理与正式阶段各自运行执行验收，两份报告保留为不同证据。

全流程会自动整理数据。需要独立重算去重/划分时运行：

```bash
python scripts/curate_dataset.py \
  --dataset-manifest /data/carla-runs/batch_001/dataset_manifest.json \
  --out /data/carla-runs/batch_001/curation \
  --seed 20261028
```

可添加 `--calibration-manifest /path/to/training_calibration/dataset_manifest.json` 指定独立训练校准，或 `--policy /path/to/frozen/curation_policy.json` 复用冻结策略；两者互斥。目标几千景时提高 `--count` 和候选预算；目前未宣称完成几千景原生采集压测。

独立重算评测批次时同样添加 `--dataset-purpose evaluation`；此模式不接受 `--calibration-manifest` 或 `--policy`。验收整景数、去重后版本数和独立基础场景数分别报告，不能用同一初态的天气版本凑独立场景数量。

普通管线还保存并封存每个实际分支的原生加载和取样证明，见 [原生天气证据](native-weather-proof.md)。

## 离线测试

```bash
python -B scripts/test_pipeline.py
python -B scripts/test_sampling.py
python -B scripts/test_scene_compiler.py
python -B scripts/test_scene_compiler.py --opendrive "$CARLA_ROOT/CarlaUE4/Content/Carla/Maps/OpenDrive"
python -B scripts/test_action_design.py
python -B scripts/test_action_opportunity.py
python -B scripts/test_interaction_diversity.py
python -B scripts/test_scene_randomization.py
python -B scripts/runtime/test_action_separation.py
python -B scripts/test_interaction_geometry.py
python -B scripts/test_scene_spec.py
python -B scripts/test_curation.py
python -B scripts/test_model_api.py
python -B scripts/test_orchestrator.py
python -B scripts/test_encounter_calibration.py
python -B scripts/runtime/test_npc_merge.py
python -B scripts/test_measured_t0.py
python -B scripts/test_planning_stage.py
python -B scripts/test_publish_dataset.py
python -B scripts/runtime/test_history_only.py
python -B scripts/test_native_weather_proof.py
python -B scripts/test_native_decal_proof.py
python -B scripts/test_ready_sync.py
python -B scripts/test_server.py
python -B scripts/test_managed_server.py
python -B scripts/test_infrastructure.py
python -B scripts/runtime/test_runtime.py
python -B scripts/runtime/test_protocol.py
python -B scripts/runtime/test_auxiliary.py
python -B scripts/runtime/test_auxiliary_audit.py
python -B scripts/runtime/test_pedestrian_policy.py
python -B scripts/runtime/test_pedestrian_crossing.py
python -B scripts/runtime/test_pedestrian_refuge.py
python -B scripts/runtime/test_decisions.py
python -B scripts/runtime/test_strict_yield.py
python -B scripts/runtime/test_realized_actions.py
python -B scripts/runtime/test_execution.py
python -B scripts/runtime/test_semantic_lidar.py
python -B scripts/runtime/test_capture_prefix.py
python -B scripts/runtime/test_collection_liveness.py
python -B scripts/runtime/test_weather.py
```

离线测试须在独立开发副本运行，再冻结来源。不要在正在采集的冻结源码树内执行测试；部分测试会启动子Python，父进程的 `-B` 不会自动传给子进程，生成的字节码缓存同样会触发严格文件树身份检查。不得因此放宽运行中源码检查。

离线测试覆盖配额、完整场景编写 schema、修正预算、真实地图静态编译、执行/等待/接触截断、无关角色、语义点云原始字节及同步、整景去重与划分。`test_server.py` 使用伪服务测试启动器，不启动真实 CARLA；部分历史原图检查在未提供可选数据时跳过。本版本另有两种固定天气案例的完整原生资格及一次普通入口真实交付验证，见 [渲染重复性](render-reproducibility.md)。新增批次仍须逐景检查真实GPU渲染、控制执行、点云命中/语义质量及重复性；离线测试不能替代这些检查。

新交互阶段生成、预热决策证据和实测覆盖报告见[交互多样性](interaction-diversity.md)。新的科学源码使用新的来源身份和输出目录，不能在旧批次中恢复或追认旧200景。

联合随机采样的提案、最终生效参数及批次分布/拒绝审计见[联合随机生成](scene-randomization.md)。阶段结束只读取已有数值证据生成一次报告，不重复采图或发布；中断运行可显式调用报告工具。


带 `encounter_calibration` 的新规格在未来预检前，先运行 `common_history_calibration_000`。未达冻结目标时另写 `calibrated_plan_001/002` 并重新采共同历史；旧规格/历史不可覆盖，最终 `t0_planning` 绑定实际采用的规格及全部修订。实测停止、已清空等合理无响应不作为校准失败。校准耗尽是明确的候选几何/时机失败，基础设施/证据损坏仍为运行错误；两者不能混用。不得用新版恢复旧冻结来源的采集。
