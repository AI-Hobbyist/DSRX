# pkg_resources 迁移及 cu132 / 最新 setuptools 升级

## 当前环境（2026-11-14）

实际已安装并验证：Python 3.10、Torch 2.14.1+cu132、torchvision 0.29.2+cu132、torchaudio 2.11.1+cu132、setuptools 84.2.1；librosa 0.11.0、Lightning 2.6.6、pyworld 0.3.5。CUDA runtime 13.2，RTX 5060 Laptop GPU。没有替换为 CPU 版本，没有保留 setuptools<82 上限。

此前 Torch 2.11 的 setuptools<82 元数据约束只适用于旧版本。本次实际查询并安装指定 cu132 源的三个包，依赖解析、实际 CUDA 运算和 CUDA 音频重采样均通过；没有仅凭 torchaudio 与 Torch 版本号不同判为不兼容。

## 更新命令

进入项目、激活训练环境后执行；此次升级不再使用此前保留旧 Torch 的 constraints 文件：

```bash
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132
python -m pip install --upgrade setuptools -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

使用 requirements.txt；requirements-onnx.txt 保留独立旧 Torch 1.13 acoustic/variance 导出栈，不用于本次现代训练/预处理/all-in-one 环境。现代环境的回归包含 ONNX 执行检查，但不声称旧导出环境已升级。

## 实际验证

40 项自动回归 PASS（含主流程导入、策略构造、CUDA Trainer、Mini-DDSP 预处理/训练/推理/ONNX 和 all-in-one 控制接口）。pip check 和 requirements.txt 安装解析通过。tests/test_dependency_imports.py 现在同时覆盖 setuptools、torchvision、torchaudio 与独立声码器预处理导入。

tests/run_without_pkg_resources.py 可选择 binarize、binarize_vocoder、train；真实 CLI 的主进程及 Windows spawn 子进程均安装导入拦截器，遇到 pkg_resources 会立即失败，不是生产兼容层。最新环境下三个入口实际完成，均未触发该模块。

以用户 data/all_in_one_mini_ddsp_cloud.yaml 为基准，在隔离目录进行 2 条训练来源+5 条 val 的实际试跑；仅覆盖本机字典/DS/数据/输出路径、预处理 worker2、DataLoader0、步数及 Mini-DDSP 启用，不改网络、Muon、BS64/50000 或正式 warmup/ramp：

- 声学/唱法两进程、独立声码器两进程预处理 PASS。train acoustic8/variance2/vocoder2，valid各5；源 WAV/CSV SHA256 不变，ID 关联及train/val隔离、有限值、波形长度通过。
- 完整内置声码器 all-in-one CUDA bf16/Muon 两步训练、两路各5条val、TensorBoard图/试听、step2完整checkpoint保存 PASS。checkpoint含 acoustic/variance/vocoder，模型参数均有限。
- 另一步仅测试覆盖 warmup/ramp=0，主后级/AUX 联合损失 7.25694/7.24069，正常反传更新并退出 PASS。

原来的 DS 字典不匹配段仍按已有行为警告跳过，未计为通过；两路5条数据集 val 实际执行。收敛和音质训练 SKIPPED（用户决定），人工听音 MANUAL/PENDING。实际训练只验证跑通，不证明质量。没有为了通过而屏蔽错误、删除功能或换回旧依赖。

产物 artifacts/cu132_upgrade/：environment.json、probe.log、regressions.log、acoustic-variance-preprocessing.log、vocoder-preprocessing.log、training.log、joint-training.log、verification.json、pip-check.log、requirements-plan.log、after-freeze.txt。完整测试checkpoint：ckpt/cu132_all_in_one_smoke/model_ckpt_steps_2.ckpt。

无需新增生产兼容代码：此前移除旧依赖直接 pkg_resources 导入的版本升级，以及 Lightning 策略构造修复，在新 Torch 上通过实跑。本次提交更新依赖清单、验证入口和说明。

## 阶段进度

1. cu132 / 最新 setuptools：实现和自动验证已通过，待本阶段提交推送检查点。
2. 其他后级的 all-in-one 模板：尚未实施。下一阶段仅补充内置声码器联合训练相关设置及独立预处理说明，不修改各后级网络、优化器或批量配置。

声学预处理性能拆分已在上一项完成，见 docs/mini-ddsp-preprocessing.md；云端4090吞吐仍为 MANUAL/PENDING，不影响本次依赖实际跑通结论。
