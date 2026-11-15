# All-in-one 内置 Mini-DDSP 联合训练计划

日期：2026-11-12。最终实施更新：2026-11-13，M1～M6实现与自动验收完成；长期收敛训练及训练后音质评估依用户决定SKIPPED。最终结果见all-in-one-mini-ddsp-results.md。

## 1. 最终需求与范围

本计划以本次对话最后确认的边界为准：

1. Mini-DDSP 是 **仅 all-in-one 生效、默认关闭** 的可选功能。关闭时维持现有预处理、训练、推理和导出行为；普通 acoustic/variance 模式不新增该功能。
2. 开启后，Mini-DDSP 是 all-in-one 模型内部的可训练子模块，随模型保存。训练完成的模型默认直接输出波形，不依赖外部声码器权重或参考仓库文件。
3. 推理可显式切换 NSF-HiFiGAN：使用同一份声学 mel/F0 生成波形。不是把 DDSP 波形输入 NSF-HiFiGAN，也不默认混合两路波形。后级分段拼接、offset、静音、交叉淡化和保存流程共用。
4. all-in-one 按现有方式**依次调度 acoustic → variance → vocoder 预处理**。三个处理器职责独立，分别写入 `acoustic/`、`variance/`、`vocoder/` 三个同款二进制包；关闭功能时仅执行前两者。
5. 复用公共音频、F0、唱法、VR 和增强配置；声码器独有配置直接放在 all-in-one 模板中。
6. 优先复用唱法预处理的完整 VR 谐波结果。没有启用唱法参数或没有可复用结果时，vocoder 预处理自动用同一 VR 模型生成，不要求手工准备谐波录音。
7. 仅正常预处理生成新的包；**不支持外部 `.npy` mel 导入，不实现既有二进制包升级/补齐工具**。启用新功能需重新运行预处理。
8. ONNX 必须包含内置声码器计算和资源，并直接输出波形；只输出 mel、依赖外部声码器的导出不算完成。

不包含：SVF 声库兼容、训练 NSF-HiFiGAN、GAN 判别器、独立声码器产品、GUI、无依据的实时流式承诺，以及不相关的架构重构。

## 2. 可行性分析与源码依据

原始设计结论：**可行，需要扩展现有模型、配对数据流和导出图，不能只挂接现有 DDSP 推理适配器。** 以下保留设计时的静态依据；实施及验证结果见第8～9节。

| 现状证据 | 判断及所需改动 |
|---|---|
| `modules/toplevel.py:32` 的 DiffSingerAllInOne 仅持有 acoustic、variance | 增加受开关控制的内部 vocoder；保持原两个子模块和参数命名 |
| `preprocessing/all_in_one_binarizer.py` 依次运行 AcousticBinarizer、VarianceBinarizer | 在最后追加独立 VocoderBinarizer，写第三个包，符合原有调度方式 |
| `training/all_in_one_task.py:163` 分别计算 acoustic 与 variance 损失 | vocoder 必须使用与 acoustic 相同样本/增强/裁剪的监督；不能用三个互不关联 batch 假装端到端联合 |
| `modules/core/reflow.py:46` 训练返回速度预测，采样入口带 no_grad | 新增训练期可微的干净 mel 估计出口；不能直接调用现有无梯度推理采样 |
| `modules/aux_decoder/__init__.py` 训练输出归一化 mel，推理才反归一化 | 喂入声码器前必须转换为统一自然对数 mel，不能直接用训练归一化张量 |
| `utils/decomposed_waveform.py:251` VR 同时缓存 harmonic/aperiodic，harmonic() 返回完整谐波 | 同一处理过程可直接复用；跨顺序预处理器需要小型共享中间缓存，现有内存对象不会自动跨包保留 |
| `augmentation/spec_stretch.py:28` 当前变换 mel、F0、时长等，未生成对应音频 | 原增强结果不能直接配原始 wav 计算波形损失；必须生成同一增强对应的波形监督 |
| `inference/ds_acoustic.py:351` 声码器调用后接现有拼接/输出 | 在波形生成处选择后端，其余后级共用 |
| `deployment/exporters/acoustic_exporter.py:207` 当前图输出 mel | 新 all-in-one 导出视图合并 Mini-DDSP，不能把原 mel 导出简单改名为波形导出 |
| `modules/vocoders/ddsp.py:38` 是外部 JIT、no_grad、旧接口适配器 | 不作为联合训练实现；保留原行为，新增可微内部模块 |

参考 `ref_only/Mini-DDSP` 提交 `305f069c07f0214bf65a28f519e2149e864885a2`：三层纯卷积控制器预测谐波幅度/相位及噪声幅度，DSP 合成波形，算法不含扩散迭代。约 1.38M 网络参数，但原实现四维合成张量显存较大。采用其算法时修正已确认的增强标签、设备硬编码、外部噪声依赖和接口问题，不整体搬入训练脚本。

### 2.1 采样率可配置的边界

内部合成器按目标 `audio_sample_rate`、`hop_size` 和 Nyquist 生成，噪声基底随目标采样率构造，不写死 44100 Hz。可以支持在不同目标采样率下重新预处理、重新训练。

这不等于已有 checkpoint 可以任意改采样率：声学 mel 契约、F0、FFT/window、频率上限、噪声资源都必须一致。验证先覆盖 24 kHz、44.1 kHz、48 kHz 的数据/前向/反向/导出；完整质量训练优先使用现有 44.1 kHz opencpop，其他采样率不能据冒烟测试宣称同等音质。

当前 VR loader 主要读取 FFT/hop/网络配置，未建立目标采样率转换契约。实现时明确 VR 权重工作采样率：目标波形必要时转到 VR 采样率，分离后转回并对齐样本长度，记录转换信息。不得把 44.1 kHz VR 权重直接运行在其他采样率后视为已正确分离。

可选 NSF-HiFiGAN 必须匹配采样率、hop、mel bins、频率范围和 log 定义；不匹配时明确拒绝该选项。它不限制内置 DDSP 的目标采样率，不以输出重采样冒充原生多采样率训练。

## 3. 配置与模型接口

### 3.1 配置归属

在已有 `configs/templates/all_in_one_*.yaml` 中加入相同的可选区块。以下是拟定接口，实施阶段可调整名称，但不可改变默认关闭和公共参数复用的含义：

```yaml
all_in_one:
  enabled: true
  mini_ddsp:
    enabled: false
    harmonics: 128
    noise_bands: 64
    hidden_size: 256
    num_layers: 3
    kernel_size: 31
    harmonic_analysis_cycles: 4
    segment_frames: 128
    warmup_steps: 2000
    joint_ramp_steps: 2000
    loss:
      spectral_weight: 1.0
      amplitude_weight: 0.3
      uv_weight: 0.0
      joint_weight: 1.0
```

这些数字是首轮实验起点，不是已经验证的最佳超参数。相位显式监督暂不启用；相位头通过波形损失学习。不新增另一份声码器音频 YAML。

直接复用：`audio_sample_rate`、`hop_size`、`fft_size`、`win_size`、`audio_num_mel_bins`、`fmin/fmax`、`mel_base`、F0 提取器及范围、数据集/说话人/语言划分、VR 权重路径、增强策略和训练精度。VR 工作采样率若缺少可靠元数据，只新增一个共享分离配置字段，不在 acoustic/vocoder 各写一份。

Mini-DDSP 独有：网络尺寸、谐波/噪声通道、分析周期、波形训练片段长度、损失和 warmup/ramp。训练频谱损失窗口尽量从公共 window/hop 派生固定多尺度，用固定尺度验证，避免随机尺度导致验证结果抖动。

非 all-in-one 配置误开此选项时给出清晰配置错误，不暗中改变普通任务。关闭时不构造内部声码器、不增加预处理成本、不新增 checkpoint 参数。

### 3.2 模型与波形生成

`DiffSingerAllInOne` 持有 `acoustic`、`variance`，以及开启时的 `vocoder`。声码器接口统一为 mel `[B,T,M]`、F0 `[B,T]`、有效长度 → waveform `[B,N]`，训练额外返回幅度/谐波等损失所需张量。对外不沿用参考仓库的另一套转置/四元组约定。

基础相位、窗口、谐波索引及必要噪声基底作为内部状态管理；推理不能依赖当前工作目录里的 `u_noise.ckpt`、`v_noise.ckpt`。优先生成可复现的分频噪声基底并持久保存，记录算法与版本。若引用上游代码，保留许可和来源声明；不默认复制来源未核对的噪声资源。

训练使用短片段与谐波分组控制内存；分组求和保持相同相位和 overlap-add 公式。采样相位累积和频谱损失优先 FP32。不为降低显存擅自改成另一个合成算法。

## 4. 三包顺序预处理

```text
现有原始数据/标注 + 公共配置
  → AcousticBinarizer → binary_data_dir/acoustic
  → VarianceBinarizer → binary_data_dir/variance
  → VocoderBinarizer  → binary_data_dir/vocoder（仅开启时）
```

三者使用项目现有 indexed binary writer、metadata、train/valid 划分方式。VocoderBinarizer 从原始录音和同一数据集配置工作，不读取 acoustic 二进制包来补丁升级；复用的是提取函数、处理过程中的明确缓存以及样本清单。

### 4.1 包字段和配对

vocoder 包包含：稳定 `sample_id`、增强 ID、采样率/有效样本长度、waveform、自然对数 mel、F0、UV、harmonic amplitude、对应有效帧数。相位标签在显式相位损失关闭时不要求存储。完整谐波波形可只作为预处理中间缓存，用于生成幅度标签，不必永久重复写入每个训练条目。

mel/F0 在 vocoder 包中保留，使其样本可独立做 warmup 和验证；有匹配提取缓存时直接复用。三个包共享数据来源、划分和增强清单标识，不能用“第 i 条”隐含对齐。acoustic 的增强样本与 vocoder 的对应样本 ID 必须一致，随机增强参数只生成一次并供两个处理器消费；variance 保持自身受支持的样本流。

联合 Dataset 将 acoustic/vocoder 按 ID 配对，Collater 保持有效长度和裁剪边界；缺失配对在训练前报清晰错误，不能静默随机换样本。波形损失不计入 padding。

### 4.2 VR 复用规则

- 唱法提取已经调用 VR 时，把完整 `harmonic()` 结果写入本轮共享中间缓存；不是只保存 `harmonic(0)` 或唱法曲线。
- vocoder 优先复用输入音频、VR 权重/配置、采样率和声道处理均匹配的结果。同一个原始样本不应因为三包调度重复运行同一 VR 分离。
- 唱法参数未启用或尚未执行实际分离时，由 vocoder 自行调用 VR 并写缓存。不能仅凭唱法开关推断结果存在。
- 既有唱法分离若使用 WORLD，不把它当作同款 VR 结果；vocoder 使用指定的 VR 路径，原唱法选择保持不变。
- 缓存失配重新计算，源 wav 永不删除。失败条目记录名称和真实异常。

### 4.3 数据增强一致性

现有频谱增强没有对应 waveform，必须在 all-in-one Mini-DDSP 开启路径补齐这一契约：

1. 对统一增强清单中的变调、变速生成配对波形；完整谐波采用相同变换和精确长度约定，无法保证配对时对变换后的音频重新分离，而不是拿原标签凑数。
2. 从变换后的音频提取标准 mel/F0/UV 和幅度监督，供该模式下 acoustic/vocoder 对应样本共同使用；时长、mel2ph、速度和 key shift 与同一变换保持一致。
3. 不能只缩放 F0、只改 mel 或只重采样标签却仍使用原 waveform。若采用的变调/变速实现改变了旧频谱增强的数值分布，限于本开关路径，记录差异并做真实样本比较。
4. 有音量增强时，waveform、谐波幅度同步乘 gain，log mel 加 ln(gain)。没有现成音量增强则不为本任务新添产品选项。
5. 关闭 Mini-DDSP 时，原频谱增强保持原行为。不得以跳过所有增强或屏蔽增强样本的波形损失来宣称增强接入完成。

增强音频渲染方案在 M2 用已有依赖优先实现，核对变调后 F0、变速后时长和帧对齐；新依赖只有在现有实现无法满足时才提出具体取舍。

## 5. 联合训练与梯度

保留 acoustic/variance 原损失和现有两路训练调度。新增 vocoder 数据来自第三个包，但与 acoustic 流配对读取，不需要新增一个与声学 batch 无关的 max_size_cycle 流。预热阶段可只更新内部 vocoder，其数据仍来自相同包。

### 5.0 Batch size 分配

保留 `training_stream_count=2`：`max_batch_size` 与 `max_batch_frames` 各按原规则整除 2 分给 acoustic、variance。vocoder 不再占第三份独立样本预算，而是跟随实际 acoustic batch，一条 acoustic 对应一条 vocoder 监督。总 max_batch_size=8 时，两路各最多 4 条，vocoder 对应 acoustic 的最多 4 条；实际数量还受帧预算限制。现有总 max_batch_size=2 的模板仍是 acoustic 1 + variance 1。

声学分支保留其既有有效序列，波形损失从每条配对样本选同一位置的 `segment_frames` 片段；mel、F0、waveform、幅度标签和 mask 一起裁剪。额外上下文按控制器感受野及合成窗处理，只在中心有效区计算损失。不能独立随机裁剪 mel 和波形。

同样 BS 开启后会增加声码器激活和梯度显存，帧预算不能视作不变的显存保证。先通过短片段和谐波分组限制峰值，记录真实峰值；必要时明确调整本功能的模板 BS/累积步数，不静默改变全局采样规则。原 `accumulate_grad_batches` 语义保持，梯度累积不等于增加同时驻留显存的样本数。

### 5.1 分阶段训练日程

- **warmup**：内部 vocoder 使用真实 mel/F0 和波形/幅度标签训练；原 acoustic/variance 正常执行原损失。
- **joint ramp**：逐步加入预测 mel 路径的波形损失，梯度回到声学条件编码器、mel 预测分支以及内部 vocoder；保留一部分真实 mel 路径稳定声码器。
- **joint**：继续原损失并保持预测 mel 波形监督。禁止永久 detach 预测 mel 后仍称为端到端声学—声码器联合训练。

Reflow 训练期利用已有 `x_t,t,v_pred` 得到归一化干净谱估计 `x1_hat=x_t+(1-t)*v_pred`，反归一化后送入 vocoder，避免每个 batch 额外展开完整 20 步推理。DDPM 若当前配置使用噪声预测，则按其实际 alpha/noise 参数化重建干净谱，不能照搬 Reflow 公式。新增出口以可选返回方式保留原 API 默认行为。

辅助 mel 分支存在且参与训练时，可同样加入声码器监督，先正确反归一化，尊重原有梯度缩放。主分支的波形梯度也必须有覆盖，不能仅优化辅助头而宣称主采样骨干获得波形监督。

单步干净谱估计包含训练样本信息，**不等于完整推理采样分布**；验证必须用真正的预测 mel 和现有采样步数重建波形。若质量不足，记录实验结果并评估后续调整，不在本计划偷偷添加多步蒸馏体系。

初始联合目标：原 acoustic loss + 原 variance loss + vocoder 频谱损失 + 谐波幅度损失 + ramp 后的预测 mel 波形损失。各项独立记录，权重不能混淆真实 mel 与预测 mel 路径。相位采用可微波形监督，不启用参考实现已注释的相位 loss。

本计划不声称波形梯度穿过离散时长展开回到整个 variance 网络；variance 保持已有监督训练。完整推理仍串接预测时长/音高/唱法，作为端到端行为验证。

### 5.2 训练状态与恢复

新 checkpoint 包含 `model.vocoder.*` 和资源/结构版本；推理精简 checkpoint 保留它们。已有模型初始化新训练时仅允许新增 vocoder 参数初始化，明确列出 missing keys；不能用全局 strict=False 掩盖其他权重缺失。断点恢复需恢复三分支优化器状态和 warmup/ramp 进度。

开启后找不到新 vocoder 包或必要字段时提示重新预处理，不自动改写旧包。关闭开关的旧 checkpoint 推理/导出继续通过原回归。

## 6. 推理与公共后级

| 配置/选择 | 波形来源 | 外部声码器 |
|---|---|---|
| Mini-DDSP 关闭 | 既有行为 | 按原配置 |
| Mini-DDSP 开启，默认 | 内部 vocoder(final acoustic mel,F0) | 不加载、不要求存在 |
| Mini-DDSP 开启，显式 NSF-HiFiGAN | NSF-HiFiGAN(同一 final mel,F0) | 仅此时加载并校验 |
| 保存 mel | 保存同一声学 mel/F0 | 不运行任何声码器 |

统一波形结果包含有效长度与采样率，接入已有后级。避免把 B 维 flatten 到时间轴。训练验证音频、正式 CLI、all-in-one 的唱法联动验证都使用相同的后端选择逻辑，不各自保留一套默认外部声码器调用。

“高质量”是可选路线名称，不预先承诺 NSF 必然优于联合训练后的 DDSP；报告各自音质和耗时。默认 DDSP 不额外计算 NSF；NSF 路线也不要求先生成一遍 DDSP 波形。

## 7. ONNX 和部署包

2026-11-13 用户修订：增加完整 all-in-one 单 ONNX 模式，包含 variance（时长、音高、唱法）→ acoustic → 内置 Mini-DDSP，默认波形推理仅需加载一个模型。原分开导出方式保留。完整单图须保留可编辑的预测结果与逐参数自动/手动覆盖接口，包括时长、F0、当前启用的唱法曲线及既有音色/速度/采样控制。输出波形、有效长度、mel、实际采用的F0/时长/唱法曲线，便于调节及检查。

默认波形图输出 `waveform` 及有效长度，并保留 `mel`（F0 为已知输入或对应输出）供调试。需要外部 NSF 时，使用同一次导出附带的 mel-only 视图，避免仅取 mel 也执行 DDSP 合成。两种视图来自同一权重与特征元数据，不训练第二份声学模型。

内置图的 noise/window/索引等资源随 initializer 或图算子保存；部署不引用训练机路径、外部 Mini-DDSP JIT 或噪声 ckpt。不把 Python 拼装另一个外部声码器当作“ONNX 已内置”。NSF 权重仍为显式可选外部资产。

优先保持现有 opset15 和导出工具链；针对 cos/cumsum/repeat、奇偶帧 overlap-add、动态 reshape 实际验证。若出现具体不支持算子，先选择等价可导出实现；改变最低运行时版本须记录证据和影响。

验收包含：检查图及资源、ORT 独立运行、与 eager 数值比较、奇偶长度及非跟踪长度、长句、不同目标采样率。测试在仅包含部署产物的新目录启动，证明默认波形图不查找外部声码器。数值阈值在 M1 建立确定性基准后锁定，不能通过放宽阈值掩盖波形偏移或长度错误。

## 8. 实施阶段与验收

阶段必须顺序完成。每阶段自动验证通过后更新本计划状态、检查 git status/diff/diff --check，单独提交、推送并确认成功，再进入下一阶段。人工听音记录 MANUAL/PENDING，不作为 blocked 条件。最终 M1～M5 **PASS**；M6 **DONE**（流程/性能PASS；长期训练/收敛后音质及频谱/F0质量收益按用户决定SKIPPED）。

| 阶段 | 实施范围 | 必需验收 |
|---|---|---|
| M1 配置与内部模块 | all-in-one 模板开关、可微 Mini-DDSP、统一形状、内置噪声、公共采样参数 | 默认关闭原行为；开启前向/反向有限；批量/奇偶帧/静音；无参考目录依赖；24k/44.1k/48k 冒烟；分组与直接合成等价 |
| M2 三包预处理 | VocoderBinarizer、顺序调度、共享 VR 缓存、统一增强清单和配对波形 | 开关关闭无 vocoder 包；开启三个包可加载；唱法开/关两种 VR 路径；重复分离计数；变调/变速单独及组合配对；源音频不变；train/valid 不交叉 |
| M3 联合训练 | paired Dataset/Collater、clean-mel 训练出口、损失/日程/恢复 | 真实小样本训练完成；独立隔离波形损失证明 vocoder、声学编码器及主预测分支获得非零有限梯度；恢复日程正确；padding 不影响有效损失；原两流预算规则不被破坏 |
| M4 推理与验证统一 | 内置默认、显式 NSF、save_mel、公共后级、checkpoint 精简 | 不提供外部声码器也能正式 CLI 生成 WAV；切换 NSF 用同一 mel/F0；多段 offset/静音/crossfade 正确；旧模型回归；真实 variance→acoustic→vocoder 全链路 |
| M5 ONNX | 声学+内置声码器波形图、mel-only 视图、元数据/资源 | ORT 独立执行输出波形；动态奇偶长度及长句；eager 对照；部署目录脱离源码资源；可选 NSF 使用导出的同契约 mel；旧导出不受影响 |
| M6 真实评估与交付 | 固定配置/样本清单的训练与对照、文档、交接 | 真实 opencpop 留出重建与完整预测；DDSP/NSF 时延、峰值内存、包大小、频谱/F0 指标；记录训练预算/seed/提交；听音状态如实记录；结论不以随机权重冒烟代替质量 |

真实数据使用已提供 `data/opencpop`、现有 all-in-one 权重和 VR/NSF 资源，不修改原始资产。正式跑 M6 前固定训练预算与评估清单；已有配置只有一个留出样本，应明确报告覆盖范围，不能靠这一条断言普遍音质收益。无需扩展成长期无限训练任务。

主要涉及：`modules/toplevel.py`、新增内部 vocoder 模块和损失、`modules/core/reflow.py`/`ddpm.py` 的可选训练输出、`preprocessing/all_in_one_binarizer.py` 及新增 vocoder binarizer、all-in-one 范围内的增强/缓存接点、`training/all_in_one_task.py`、推理后端选择、现有导出入口、all-in-one 模板及对应测试。普通模式的共享文件只做开关隔离所需的最小修改。

所有训练/构建/测试前台 PowerShell 运行，输出 Tee 至 build.log，立即检查退出码并读失败日志。不启动 GUI，不修改 ref_only 源码，不顺带修复无关功能。

## 9. 当前交付状态

以下按实施时间保留阶段记录；后面的最终验收覆盖此前“未开始/待解决”状态。最终交付矩阵、资源大小、测量范围和SKIPPED项目见[交付结果](all-in-one-mini-ddsp-results.md)。

M1 **PASS**（2026-11-13）：五份 all-in-one 模板默认关闭；内部 MiniDDSP 持有控制器、分组合成、Hann/window/索引和版本化噪声基底。配置加载拒绝非 all-in-one 误开，关闭时不构造模块、不增加 checkpoint 参数。噪声由固定 seed=1729 的互斥 FFT 频带生成，持久保存，不读取参考目录或外部噪声文件。Mini-DDSP MIT 许可保存在 docs/licenses/mini-ddsp.txt。

自动验收：`tests.test_mini_ddsp` 四项与 `tests.test_all_in_one_infer_cli` 两项共六项通过。覆盖 24k/44.1k/48k、B=2、T=1/4/5、无声/零有效长度、mel 与全部控制器参数有限梯度、严格资源回载逐值一致、分组/直接波形和梯度、独立逐帧重叠相加对照、原 CLI 回归。采样率测试是模块冒烟，不代表真实训练音质。FP32 分组对照锁定 rtol=2e-5、波形 atol=2e-6、梯度 atol=2e-5；后续 ORT/eager 波形对照锁定 rtol=2e-4、atol=2e-5，不能为通过而放宽。

新 Python 文件已用系统 Python 的 Black 26.10.0 格式化；diffs 环境没有 Black，未安装新依赖。测试前台 PowerShell Tee 日志、退出码 0。首次沙箱进程在导入阶段停滞，核实进程后 Ctrl-C 结束，正常 Windows 权限重跑通过，不记录该次为 PASS。

M1 已提交推送：117f05a104f629b36bb70c768e8728babb837017，远端确认一致。

M2 **PASS**（2026-11-13）：新增独立 VocoderBinarizer，顺序 acoustic→variance→vocoder，三者使用原 indexed writer 和 metadata。acoustic 与 vocoder 按稳定 sample_id/augmentation_id 配对；使用已有随机增强清单，波形由 librosa 变调/变速后重新提取 mel/F0/UV、唱法和完整 VR 谐波幅度。关闭时保留既有频谱增强。ID 采用定长字节标量，符合现有读取器，不改公共二进制格式、不升级旧包、不导入外部 npy。

VR 完整 harmonic 缓存位于本轮临时目录，身份包含波形、模型路径/文件身份/配置、输入和工作采样率及 mono 约定，顺序处理器复用。没有唱法分离时 vocoder 自动生成；WORLD 不作为 VR 结果复用。共享 hnsep_sample_rate 和权重 sr 元数据校验，非工作采样率先转换、分离后转回并精确对齐。运行结束清理中间缓存，源 wav 不动。

真实自动验收：scripts/validate_mini_ddsp_data.py --singing on/off 分别使用2001000001、2001000002训练及2033001224留出，seed42。每次 acoustic/vocoder train=10/10、valid=1/1，variance train=2、valid=1，唯一配对11、VR调用11（无重复分离）。覆盖单独变调、单独变速及组合，配对 mel/F0 逐值一致，幅度有限且形状正确，源 wav/标注 SHA256 不变，train/valid ID 无交叉。两条路径完整报告分别在 artifacts/mini_ddsp_m2/singing_on/report.json 和 singing_off/report.json（不提交数据）。增强音频与旧频谱增强的数值分布不同，仅本开关路径采用真实波形变换。

四项预处理测试和 M1/既有 CLI 回归共10项通过；三采样率重采样/缓存单测使用可计数 separator 替身，真实 VR 三包验收为44.1k，不能混称三采样率真实 VR 音质验证。修改范围 Black 格式化及 git diff --check 通过。人工听音 MANUAL/PENDING。M2 已提交推送16b1d6a17b9c86edae7a81a9ea854f14ba5aac66，远端确认一致。

M3 **PASS**（2026-11-13）：新增 PairedAcousticDataset，初始化按 sample_id 建索引并在缺失时提示重新预处理，读取时校验同一 mel/F0，配对 Collater 保留帧/采样长度。原 training_stream_count=2 和整除2预算保持不变。MiniDDSPLoss 采用同位置短片段、控制器感受野及合成窗上下文，padding 不进入统计；固定窗口来自公共 win_size 的半倍/一倍/二倍，线性幅度及 log 幅度 L1 加谐波幅度监督，warmup/ramp/joint 分开记录。真实静音片段曾将相对频谱分母损失放大到5.2e7，改为绝对频谱 L1 后损失正常（最终训练6.28～20.03）。

Reflow 的可选干净谱出口使用 x_t+(1-t)*v_pred；DDPM 按其实际 alpha/noise 还原 x0。默认返回及既有损失不变。预测谱反归一化后进入 vocoder，辅助分支尊重原梯度缩放。只允许完整新增 vocoder 初始化，其他缺失/多余或部分 vocoder 均拒绝，严格加载保留；原 checkpoint 特征开关不得随测试资料多开。恢复使用正常 Lightning 全模型/优化器状态和 global_step。

真实自动验收：scripts/validate_mini_ddsp_training.py，seed42，原100000步声学/variance权重＋新 vocoder，使用 M2 唱法开启包，FP32，acoustic1+variance1，segment32，warmup2/ramp4，12步（第6步保存、恢复）。单独 backward ddsp_joint_loss，未加入原声学/variance损失：vocoder32个梯度张量L1=1073.37、声学encoder43个=40.97、token embedding1个=0.35188、主预测骨干92个=485.95，全部非零有限。额外非零padding不改变有效波形/幅度损失（逐值相等）。完整优化器动量恢复逐值相等，恢复后的joint factor正确；两路真实留出验证loss=3.03185，CUDA峰值855335424字节。报告/完整checkpoint位于artifacts/mini_ddsp_m3/（不提交）。12步仅验证联合训练路径，不作为音质训练完成。

22项单元/回归通过，覆盖参数化公式、原声学训练、默认关闭、预处理及all-in-one CLI/唱法开关/空验证。既有AST隔离测试加入真实配置helper的AST，保持其轻量隔离目的。Black仅作用新文件和修改范围，diff --check通过。测试脚本的中途保存使用标准训练回调；没有修改既有BaseTask保存机制。num_valid_plots=0使原绘图绑定的曲线指标未更新并产生warning，不作为这些指标PASS；本轮验证的是实际loss。

M3已提交推送5af8f90c6d8b3881405e5db352f6fcd70cbd0bc6，远端确认一致。

M4 **PASS**（2026-11-13）：AllInOneVocoder 与 build_all_in_one_vocoder 统一正式CLI、原训练验证音频及唱法联动验证的默认后端；模型内 Mini-DDSP 默认不读取外部声码器。`scripts/infer.py all-in-one --vocoder nsf-hifigan` 显式使用同一声学mel/F0；`--save-mel` 不运行任何声码器，即使显式选择NSF。训练验证可由 all_in_one_vocoder_backend 显式选择；默认内置。NSF在加载前严格检查采样率/hop/mel bins/FFT/window/频率范围/自然对数定义，不对不兼容资产重采样充数。外部适配器逐条处理batch再stack，内部保持[B,N]；公共run_vocoder接受历史1D或规范2D，不flatten批维。

真实验收：scripts/validate_mini_ddsp_inference.py 使用2033001224乐谱（从真实标注导出note_slur，不使用真实F0/时长作为预测输入），seed42、variance/acoustic均20步、三段含前置静音/间隔/重叠。正式CLI在配置外部路径不存在时输出internal.wav；同环境save-mel通过；显式NSF输出external.wav；原dslight100000步checkpoint关闭新开关输出legacy.wav。44.1k单声道425765采样，有限且非零，offset/gap/crossfade及最终长度通过。真实训练验证任务在外部路径不存在时构造内置后端，严格回载精简状态并生成有限波形。

现有export_inference_ckpt.py已逐值保留全部state，不需要修改；37项model.vocoder状态及内置资源逐值一致，优化器被移除。该真实checkpoint训练文件178723106字节→推理文件67112193字节。产物/报告/真实CLI日志在artifacts/mini_ddsp_m4/（不提交）；checkpoint来自M3的12步，不能作为音质验收。四项新推理测试加旧CLI/空验证和M3接口回归共13项通过，覆盖相同mel/F0传入、B=2及契约逐项拒绝；Black修改范围与diff --check通过。

M5～M6尚未实施：内置波形ONNX及真实质量/性能评估未验收。听音MANUAL/PENDING。M4提交推送成功后开始M5。
### 2026-11-13 用户追加的试听与日志要求（实施中）

随机seed42取150条opencpop录音：140训练、10验证，2033001224保留；其余验证仅保证未用于本轮DDSP训练，不宣称原声学预训练未见过。样本/源哈希清单在artifacts/mini_ddsp_preview/data150/selection.json。真实三包预处理完成：train acoustic/vocoder210条（含配对增强）、variance140条；valid三包各10条；源音频哈希不变。

用户指定合成测试为la，要求包含预测唱法曲线、CUDA训练、配置级联模板并局部覆盖、TensorBoard loss/mel/试听。另新增数据集直接声码器重建试听和mel，必须复用现有val dataloader的同一样本/索引；mel复用既有上中下预测/真实/误差对比格式。通过TensorBoard event直接提取Images/Audio原记录供用户查看，不另画替代图。

预览1000步CUDA实验在运行：RTX5060 Laptop GPU8151MiB，配置configs/experiments/all_in_one_mini_ddsp_preview.yaml级联原all-in-one模板和原checkpoint结构配置，warmup200/ramp200，每200步验证。记录不是M6正式完成声明。第200步TensorBoard原图/试听已提取至artifacts/mini_ddsp_preview/tensorboard_exports/step_200并发送。

M5待解决：完整单图调度/手动控制尚未实现。已实现的声学+DDSP合并图在ORT默认优化下与mel-only图depth0结果不同；最小单帧重现1.4秒，声学节点和权重字节完全相同、无新增重名，关闭ORT优化后差异为0。需定位并修复具体优化触发，不能以放宽阈值交付。内部声码器独立图长帧FP32相位累积跨mod边界的问题已修正为FP64累积，三采样率B2/奇偶/长帧通过M1原阈值；更改尚未提交。新增val直接重建与UV幅度学习修正有10项训练/推理回归通过，尚未提交。

### 2026-11-13 云端配置迁移与验收范围更正

用户要求停止长时间模拟训练，后续云端训练。原150条预览已自然完成1000步，每200步验证；step1000的TensorBoard原图/音频已发送，不再将该预览描述为运行中。新的小数据预处理/训练仅检查能否跑通；需要长期实际训练才能完成的收敛、音质与训练后对照项目，记为 **SKIPPED（用户决定，未执行）**，不得写PASS。既有已实际执行的历史证据保留，不因本次跳过而改写。允许用小型随机权重模型验证导出功能，不以此证明音质。

下载目录dsrx_cfg.7z与dsrx_cfg (1).7z SHA256一致：982e3af316eae680dcbc7342ba861b9898d303fe00c717e5f2ab5c7b77ef1120。原包解压至artifacts/dsrx_cfg_import；正式迁移配置为configs/experiments/all_in_one_mini_ddsp_cloud.yaml，中文词典原字节复制至dictionaries/opencpop-dsrx_cfg.txt。只继承configs/original/acoustic.yaml与variance.yaml，不依赖本机checkpoint或preview配置。仅保留opencpop，spk_id=0、num_spk=1、zh单语言；数据目录沿原包data/opencpop_zh。5条val为2033001224、2070002621、2012000457、2003000103、2082003039。val_with_variance直接选samples/02_不老梦_variance.ds；正式加载得到35条有效片段，原样本另9条词条不匹配片段按既有逻辑跳过。

声学hidden_size=192、DSLight128通道8层，AUX为DSLight128通道4层，pitch/唱法后级为DSLight96通道6层，时长ResNet128通道3层，与50MB模板网络参数一致。energy预测/嵌入均关闭，其余时长/音高/唱法与Mini-DDSP启用。Muon_AdamW、lr=0.0006、BS64/50000frames、增强参数、StepLR的step_size=5000/gamma=0.8沿压缩包aco配置；补齐当前训练入口必需的scheduler_cls，未以preview的小batch替换。两流预算仍各取一半。

真实试跑使用现成CSV/WAV的2条训练+5条val隔离副本，仅覆盖raw/binary路径，不重标注、不修改源文件。单进程三包预处理PASS。真实spawn多进程首次失败，定位为worker重读配置丢失运行期_mini_ddsp_cache_dir；已添加run-scoped继承环境变量，finally恢复旧环境，正式2进程三包预处理PASS。5项预处理回归（含真实spawn子进程）通过。日志artifacts/dsrx_cfg_import/cache-tests.log及preprocessing_smoke/preprocessing-two-workers.log。7条小样本不覆盖全词典的warning属于样本覆盖提示，不能称全量音素覆盖PASS。

训练冒烟首次完成一次Muon更新后，4个DataLoader worker导致本机主内存耗尽；本机复跑仅覆盖ds_workers=0，保留BS、Muon、bf16及网络。正式云端配置保留ds_workers=4，预处理num_workers=2。正式train.py完成2次Muon更新、每次两路各5条val、samples联动推理、mel/音频记录及checkpoint保存，max_steps=2正常退出；val_loss=65.20仅为流程数据，不代表音质。checkpoint全参数有限、global_step=2、优化器和scheduler状态存在。三包读取检查通过：train acoustic8/variance2/vocoder8，valid均5；配对mel/F0逐值一致、波形长度正确、train/valid ID无交叉、临时缓存已清理、源CSV/WAV SHA256保持不变。报告artifacts/dsrx_cfg_import/preprocessing_smoke/report.json，训练日志同目录training.log。不得把warmup内冒烟称为收敛或完整joint日程验收。

M5诊断新增证据：禁用ORT内存复用也使原图mel差异为0；离线BASIC优化序列化后重新载入，独立默认ORT执行动态waveform/mel/variance图通过，但与native acoustic eager仍有约4.18偏差，因此离线优化本身不构成修复，不得直接采用为PASS。诊断产物artifacts/mini_ddsp_m5/offline-validation.log。完整variance→acoustic→Mini-DDSP单文件与manual override仍未完成。

### M5 最终验收（2026-11-13，覆盖前述诊断中状态）

M5 **PASS**：新增正式`export.py all-in-one`入口，默认生成完整variance→时长归一化/手动时长→pitch→手动F0→singing→逐帧手动唱法→acoustic→Mini-DDSP单文件模型；同步提供完整mel-only、声学+声码器和原分离视图，`--split-only`显式保留分离方式。模型输出预测值和实际使用的时长/F0/唱法，手动控制语义与输入形状见docs/all-in-one-mini-ddsp-deployment.md。所有网络权重及版本化噪声内嵌；未关闭ORT优化，未放宽M1误差阈值。

真实根因定位到长度调节器的整数`Pad(1,-1)`：单token在ORT常量折叠和内存复用下，前置边界可变成无效大整数，污染mel2ph/F0掩码。仅将该操作替换为等价zero-concat/slice；同时替换导出词起始标记的负padding，以及all-in-one variance内部LR。原始正式验证脚本RED、修改源码后重新正式导出GREEN；不采用离线优化掩盖偏差。

正式CLI产物artifacts/mini_ddsp_m5/deploy_complete，checkpoint来自M3已训练12步实例（非质量模型，未新增训练）。scripts/validate_all_in_one_onnx_controls.py将两个完整视图复制到无源码/分离模型/权重的临时目录，以python -I和默认ORT执行18例：1/9/10/11/257/1025帧，单/多音素及词、自动/手动/混合、depth0及0.6；全部有限、实际时长总帧数/波形采样长度正确、手动F0及各唱法逐值保留，手动depth0完整mel-only与完整波形图mel一致。报告controls-report.json。

scripts/validate_mini_ddsp_onnx.py在独立ORT执行6例（含真实272帧乐谱），波形对同mel/F0的eager DDSP最大绝对误差8.72e-7；depth0声学对native eager误差4.53e-6/1.34e-5，均满足既定rtol2e-4/atol2e-5。独立variance各图执行通过，显式NSF消费同契约导出mel/F0通过。报告complete_validation/report.json；日志export-complete.log、complete-validation.log。随机采样分支不跨不同运行时比较随机数逐值一致，只对相同导出mel执行声码器数值对照。

22项DDSP/ONNX/控制器/训练梯度/推理/预处理回归通过；覆盖三采样率B2与T1/9/10/11/257/1025、嵌入噪声、UV监督、同val重建、spawn缓存继承。全模型接口验证使用真实小步checkpoint，不作为训练收敛或音质指标PASS。当前测试实例完整ONNX为77,454,501字节，完整mel-only为62,990,675字节，来源较宽旧权重，不能据此代表192宽云端模板体积。阶段提交/推送检查点完成后开始M6。

M5已提交推送ef0ffa332daa45f2eff6ac7d1a9fbd9f790978fa，远端同名分支核对一致。M6现开始：固定使用已存在cloud_smoke第2步checkpoint及既有5条留出包，不再训练；时延/显存/大小为工程测量，收敛后频谱/F0质量收益与长期训练SKIPPED（用户决定）。

### M6 最终交付验收（2026-11-13）

M6 DONE：云端尺寸假模型18例独立完整ORT控制、la乐谱20步全预测（362帧/185344采样）、资源内嵌/无外部依赖审计通过；同5条val的真实未改动2步声码器完成CUDA DDSP/NSF测量，完整模型61,343,381字节。零自动时长不再阻止手动override，Torch/ORT回归通过。最终31项单元/回归通过，Black与diff --check通过。结果、复测入口、配置/权重来源、具体成本和逐项验收矩阵见all-in-one-mini-ddsp-results.md。长期收敛训练、收敛后音质与频谱/F0收益按用户决定SKIPPED；听音MANUAL/PENDING，不冒充PASS。无剩余必需实施项。

### 2026-11-14 用户追加：独立声码器数据准备（模型保持内置）
声学恢复原增强路径；声码器从原音频独立预处理，仍为完整 all-in-one 内置模型。公共配置、同款二进制包、原始样本联合监督、训练时裁片/音量增强及实际验证见 docs/mini-ddsp-preprocessing.md。40项回归、两进程预处理、两步完整CUDA训练、一步非零joint分支通过；云端4090吞吐MANUAL/PENDING。最新版依赖升级按用户要求暂停，不混入本项。
