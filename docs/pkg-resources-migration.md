# pkg_resources 依赖迁移与验证

训练/预处理环境为 Python 3.10；librosa 0.9.2 → 0.11.0、Lightning 2.3.3 → 2.6.6、pyworld 0.3.4 → 0.3.5，setuptools 81.0.0。保留匹配的 CUDA PyTorch 2.11.0+cu130、torchvision 0.26.0+cu130、torchaudio 2.11.0+cu130；没有替换为 CPU 版本。setuptools 保持 <82 以满足当前 Torch 依赖。

新版 Lightning 删除了项目复制的私有初始化方法；get_strategy 改为调用父类初始化，同时保留策略参数及由 Trainer 完成最终绑定的行为。没有更改训练算法或批大小。

已实际通过：pip check；禁止导入 pkg_resources 的新进程导入检查；2 进程、2 条训练来源和 5 条验证来源的三个预处理包；CUDA/bf16 两步 Muon 训练及两路验证；旧 Lightning checkpoint 恢复后继续到第 3 步。日志位于 artifacts/pkg_resources_migration/。这些只验证流程跑通；收敛与音质训练 SKIPPED（用户决定），人工听音 MANUAL/PENDING。

测试入口 tests/run_without_pkg_resources.py 通过环境变量 DSRX_MIGRATION_ENTRYPOINT 选择 binarize/train，接受原 CLI 参数，并在 Windows 子进程内继续拦截旧接口导入；仅用于验证，不是生产兼容层。

requirements-onnx.txt 保留旧 Torch 1.13 声学/唱法导出栈，不在此次升级范围。现代训练、预处理和 all-in-one 导出使用 requirements.txt。setuptools 自带及可选旧工具中仍可能保留 pkg_resources 文本；不宣称整个环境已删除该模块。

## 待排查：云端声学预处理速度

2026-11-14 用户报告：云端声学预处理由此前 it/s 降为 s/it。用户明确要求后续再修，本轮仅记录，不实施性能修改。

云端根因尚未定位；目前没有云端逐阶段计时、硬件/进程数和同数据对照证据。本机小样本日志不能作为云端根因结论：旧版两进程的 2 条训练来源约 34 秒，新版约 42 秒；新版验证首条约 22.9 秒，其余 4 条合计不足 1 秒，存在首条启动开销。旧版单进程约 5 秒，不能与新版两进程直接比较。

后续最小排查：收集云端配置、硬件、依赖版本与完整日志；固定同一批数据及进程数，分别计时启动、音频读取、mel、F0、VR 分离与增强，区分冷启动和稳定吞吐，再定位回归。当前不将初始化、缓存或 Numba 编译认定为已证实根因。

最终自动回归：35 项通过，覆盖导入门禁、CPU/CUDA Trainer 构造、策略参数、Mini-DDSP 预处理/训练/推理/ONNX 及 all-in-one 控制接口；pip check 与 git diff --check 通过。
