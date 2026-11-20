# 原生天气随机流、贴花排序与采集证据

本版本将已验证的 `Rain / BP_Weather_C` 两个原生调用点置于独立随机流。每次仍消耗一次真实 libc `rand()`，其他调用原样返回。算法、seed=0、128-byte 状态、源码、汇编、共享库、CARLA Shipping 和天气蓝图资产的 SHA256 都固定在随包 manifest 中。支持范围限定为该 manifest 指定的 Linux x86_64、glibc 2.35 与精确资产构建；不匹配时停止，不能仅凭版本号继续。

`native_qualified:false` 是原始构建时的记录，保持不变。后续原生资格报告单独封存并绑定该构建；不要为表示资格通过而修改已经测试的 native manifest。构建通过、实际加载与每景图像验收是三个独立结论。

新增 `stable_decal_paths_v2` 在同一精确构建中保留贴花显式优先级，以稳定材质/组件路径替代内存地址决胜。`native_decal_proof.py` 验证本次实际PID的函数入口安装记录、排序调用序列和快照。运行期间只读取构造时缓存的对象身份，不在渲染线程追踪已卸载UObject；不改变可见列表、原始绘制载荷或质量设置。根因、资格范围见[渲染复现](render-reproducibility.md)。

## 普通生产入口

使用 `--managed-server-config`，让普通管线为物理预检及各 RGB 分支管理实际服务身份。`server.py` 启动前检查真实二进制及天气资产；`ManagedServer` 在服务就绪时检查自己拥有的 Shipping PID/start_ticks、启动环境及该 PID 写出的原生 header。共享库身份来自固定的启动环境、库 SHA 和实际原生 header；不把无法读取的 `/proc/maps` 当成已经验证的证据。

每个 world worker 成功返回后，只读复制该 epoch 的原生 trace，并绑定实际 spec、分支 meta 和 states 的 SHA。物理预检允许没有 Rain 调用；RGB 中 spec 的 `weather.precipitation > 0` 必须实际出现已隔离的 Rain 调用，干燥 RGB 允许零次。校验包括 context、主线程、字节码位置、事件和私有 ordinal 连续性、非天气返回值原样、合法值域、完整 JSON 行及日志未截断。进程尚存时可以没有析构 footer；保存的是 worker 返回后的完整只读快照，事件数必须严格低于日志上限，已有 footer 必须计数一致。

证据保存在 `reports/native_weather/physics.json` 和各 `reports/native_weather/<branch>.json`。六分支场景共七份。正式渲染审计前及 delivery seal 前重验，并连同原数据纳入交付 SHA 清单；缺证据、证据变化、库不匹配不能接受。已有完整分支如果中断时未能封存原 epoch 证据，恢复会明确停止，不会归因到新进程或覆盖它。外部未管理的服务不能产出本版本要求的原生证据，故不能以旧无证据数据完成正式验收。

同一证据中的 `decal_order` 还保存 `weather_rng/decal-order-PID.jsonl` 的只读快照、SHA-256和检查结果。排序事件必须连续、无错误且低于原生日志上限；出现footer时核对实际调用数。没有至少两个可见贴花的列表时允许零次排序，结果明确标记为仅证明安装，不能由零调用推断已修复该次渲染。原生排序摘要是定位工具，不能替代原始RGB像素比较。

这些检查不调用 CARLA API，不推进 tick、不改变图像或质量参数、不改变 C/S/SO 或天气数值。原有物理、执行、198 对公共历史/重复 RGB、结构化标签等验收保持独立且不放宽；原生随机流证据通过并不自动证明所有场景渲染一致。

## 离线检查

```bash
cd scripts
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest test_native_decal_proof test_native_weather_proof test_ready_sync test_managed_server test_server test_infrastructure test_orchestrator test_pipeline
```

这些测试使用临时目录、模拟世界与自有假进程，验证缺证据拒绝、身份及资产篡改、恢复约束和封存。它们不是新的原生 CARLA 采集结果。
