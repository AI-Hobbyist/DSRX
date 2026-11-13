# All-in-one Mini-DDSP 交付与验证结果

日期：2026-11-13。实现按 M1～M6 交付；M5 提交 `ef0ffa332daa45f2eff6ac7d1a9fbd9f790978fa`，本报告随 M6 提交。运行环境：Windows、Python 3.10、PyTorch 2.11.0+cu130、RTX 5060 Laptop GPU 8 GiB。

## 验收范围

用户要求：“需要真正实际训练的测试一律视作人工/跳过并注明”，并进一步明确“小数据量预处理训练测试只测试是否跑通”。因此本次保留已执行的小数据流程证据；不新增收敛训练，不把小步模型、假模型或有限输出当作音质结论。

| 项目 | 状态与证据 |
|---|---|
| M1 默认关闭、可微模块、内置资源 | PASS；三采样率、B=2、奇偶/零有效长度、全部控制器梯度、分组/直接/独立OLA对照及状态回载 |
| M2 三包、增强、VR复用、配对 | PASS；真实唱法开/关各11个唯一波形/11次VR；源数据哈希不变；新配置另有2进程、2训练+5 val真实三包验收 |
| M3 训练/梯度/恢复 | PASS（流程）；既有12步及恢复第6步实验验证主骨干/编码器/vocoder非零波形梯度、padding不变和优化器恢复；新云端配置另完成2次Muon更新及两轮5条val |
| M4 原推理、内置默认、显式NSF、分段、精简 | PASS；正式CLI真实乐谱、多段offset/gap/crossfade、外部路径缺失、save-mel、旧模型、全部37项vocoder状态保留 |
| M5 完整单ONNX与手动控制 | PASS；正式CLI、独立默认ORT、18种动态长度/自动/手动/混合、F0=0精确保留；真实checkpoint声学/波形eager对照与显式NSF |
| M6 交付、可复现配置、推理成本、包大小 | PASS；下述相同5条val输入测量及云端尺寸假模型完整导出功能检查 |
| 长期收敛训练、训练后质量对照 | SKIPPED（用户决定，未执行） |
| 收敛后频谱/F0质量收益 | SKIPPED（需要相应训练权重；未用随机模型制造质量指标） |
| 人工音质验收 | MANUAL/PENDING；已有1000步TensorBoard原图/试听已发送，无主观通过结论 |

各阶段详细报告、命令和提交记录见 [计划](all-in-one-mini-ddsp-plan.md)。不涉及SVF兼容、GUI、NSF训练、判别器或新数据库。

## 云端训练配置与数据

正式配置：[all_in_one_mini_ddsp_cloud.yaml](../configs/experiments/all_in_one_mini_ddsp_cloud.yaml)。仅继承 acoustic/variance 基础配置，单 opencpop、中文、energy关闭；声学/AUX/音高/唱法网络使用DSLight 50MB模板尺寸。Muon_AdamW、lr=0.0006、BS64/50000frames、累积1、增强及StepLR5000/0.8沿原压缩包配置。预处理2进程，云端DataLoader workers=4；本机训练冒烟因主机内存限制仅覆盖workers=0，未改batch。

正式原始目录 `data/opencpop_zh`，使用已制作的CSV和WAV；正式输出 `data/pre/binary/all_in_one_mini_ddsp_cloud`。试跑输出 `artifacts/dsrx_cfg_import/preprocessing_smoke/binary`：train acoustic8/variance2/vocoder8，valid各5。5条val为2033001224、2070002621、2012000457、2003000103、2082003039；同一val loader同时用于原声学验证和直接声码器重建，三段mel图复用原plot_mel格式。联动乐谱为 `samples/02_不老梦_variance.ds`。

缓存修复：spawn worker重读配置时不包含临时cache key，现由父进程设置run-scoped继承环境，finally恢复；真实2进程通过。训练配置补齐必需 `lr_scheduler_args.scheduler_cls`。源CSV/WAV、压缩包词典及原checkpoint哈希均验证未变。

## 单文件部署与手动控制

详见 [部署接口](all-in-one-mini-ddsp-deployment.md)。完整图包含variance→时长归一化/选择→音高/唱法→手动覆盖→声学→内置声码器。输出预测及实际使用的时长、F0/唱法、mel、waveform和有效采样长度；完整mel-only图不包含声码器及其噪声资源。

负padding的ORT错误已用等价concat/slice消除，无需关闭优化或放宽误差阈值。M5真实小步checkpoint的波形对照最大绝对误差8.72e-7；确定性声学对照最大绝对误差1.34e-5，保持rtol=2e-4/atol=2e-5。

M6额外修正：手动时长不再要求未使用的自动预测先成功展开。全零自动时长＋有效手动时长的Torch/ORT回归通过。`predicted_durations`暴露原RhythmRegulator归一化/取整的预测，`durations`再按实际乐谱总帧数对齐；不为未收敛的全零自动预测虚构音素时长。

用户允许用假模型验收导出。云端2步模型确有测试音素自动时长全零的情况，所以独立假模型只将 `model.variance.fs2.dur_predictor.linear.weight` 置0、bias置`log(1+log_offset)`，使原始时长预测为1；其余权重保持。原2步checkpoint SHA256：`1cac10f7c6d0720a10b5da1d6fce779594ee893e82d527fa7aff2d5c007eeb5f`，制作副本前后相同。假模型的18组完整图测试通过；实际la乐谱12音素/6音符、三路采样各20步，完整单图输出362帧/185344采样及全预测F0/三条唱法曲线，全部有限。它是功能夹具，不能作为训练模型使用或证明音质。

云端尺寸模型的ONNX大小（十进制MB，资源均内嵌）：

| 视图 | 字节 | MB |
|---|---:|---:|
| 完整variance/声学/Mini-DDSP | 61,343,381 | 61.34 |
| 完整mel-only | 46,879,555 | 46.88 |
| 声学/Mini-DDSP | 34,490,200 | 34.49 |

“50MB”是所选网络模板名称；完整波形图还包含声码器和噪声资源，实际不是50MB以内。假模型仅改变参数数值，不改变模型尺寸。

## 相同输入的声码器CUDA测量

固定seed42，使用未修改的真实2步内部声码器、现有NSF权重以及同一5条val真实mel/F0（总23.6147秒音频），每条预热2次、测3次取中位数。FP32，DDSP相位累计FP64，NSF selective FP16关闭，均同步CUDA计时。

| 后端 | 5条合计中位时延 | 总体RTF | 最大PyTorch CUDA allocated | 最大额外allocated |
|---|---:|---:|---:|---:|
| Mini-DDSP | 43.45 ms | 0.001840 | 247,624,192 B | 223,413,760 B |
| NSF-HiFiGAN | 201.16 ms | 0.008518 | 224,293,376 B | 143,254,016 B |

测量范围仅声码器，不含声学/variance采样、预处理、文件写入或质量指标；显存为PyTorch allocated计数，不是整机显存或reserved池。NSF模式中内部声码器仍随模型驻留，与实际显式后端切换一致。单次笔记本结果不作为跨硬件实时性能承诺。

原始记录：`artifacts/mini_ddsp_m6/benchmark.json`（含逐样本ID、时间、内存、配置/权重SHA256、代码基线），`fixture-controls-report.json`、`la-report.json`、`la-inputs.npz`及`la-outputs.npz`；数据、权重、ONNX和日志不提交Git。

复测命令（仓库根目录，使用已存在的资源）：

```powershell
python scripts/benchmark_mini_ddsp.py --runtime ckpt/all_in_one_cloud_smoke --checkpoint ckpt/all_in_one_cloud_smoke/model_ckpt_steps_2.ckpt --binary artifacts/dsrx_cfg_import/preprocessing_smoke/binary --nsf ckpt/kouon_nsf-hifigan_1031/kouon_nsf-hifigan_1031_44100hz_512hop_128bin.ckpt --deploy artifacts/mini_ddsp_m6/deploy_fixture --output artifacts/mini_ddsp_m6/benchmark.json
```

目前没有剩余必需实施项。用户后续云端训练后的收敛和音质评价属于已明确跳过的验证，不记为PASS。

最终回归：31项单元/接口回归通过，18项完整ONNX动态控制用例通过；Black与git diff --check通过。
