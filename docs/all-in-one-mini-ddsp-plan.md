# All-in-one 内置 Mini-DDSP 联合训练计划

日期：2026-11-12。实施更新：2026-11-13，M1、M2 自动验收通过；M3～M6 尚未开始。

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

结论：**可行，需要扩展现有模型、配对数据流和导出图，不能只挂接现有 DDSP 推理适配器。** 以下为静态分析，尚未运行新实现。

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

开启功能时导出 all-in-one 的声学—声码器视图：已有声学输入 → acoustic mel → 内置 Mini-DDSP → waveform。variance 保留现有独立导出与调度方式，整个交付包包含其必要文件；本任务不把离散时长处理强行合并成一个超大图。

默认波形图输出 `waveform` 及有效长度，并保留 `mel`（F0 为已知输入或对应输出）供调试。需要外部 NSF 时，使用同一次导出附带的 mel-only 视图，避免仅取 mel 也执行 DDSP 合成。两种视图来自同一权重与特征元数据，不训练第二份声学模型。

内置图的 noise/window/索引等资源随 initializer 或图算子保存；部署不引用训练机路径、外部 Mini-DDSP JIT 或噪声 ckpt。不把 Python 拼装另一个外部声码器当作“ONNX 已内置”。NSF 权重仍为显式可选外部资产。

优先保持现有 opset15 和导出工具链；针对 cos/cumsum/repeat、奇偶帧 overlap-add、动态 reshape 实际验证。若出现具体不支持算子，先选择等价可导出实现；改变最低运行时版本须记录证据和影响。

验收包含：检查图及资源、ORT 独立运行、与 eager 数值比较、奇偶长度及非跟踪长度、长句、不同目标采样率。测试在仅包含部署产物的新目录启动，证明默认波形图不查找外部声码器。数值阈值在 M1 建立确定性基准后锁定，不能通过放宽阈值掩盖波形偏移或长度错误。

## 8. 实施阶段与验收

阶段必须顺序完成。每阶段自动验证通过后更新本计划状态、检查 git status/diff/diff --check，单独提交、推送并确认成功，再进入下一阶段。人工听音记录 MANUAL/PENDING，不作为 blocked 条件。当前 M1、M2 **PASS**；M3～M6 **NOT STARTED**。

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

M1 **PASS**（2026-11-13）：五份 all-in-one 模板默认关闭；内部 MiniDDSP 持有控制器、分组合成、Hann/window/索引和版本化噪声基底。配置加载拒绝非 all-in-one 误开，关闭时不构造模块、不增加 checkpoint 参数。噪声由固定 seed=1729 的互斥 FFT 频带生成，持久保存，不读取参考目录或外部噪声文件。Mini-DDSP MIT 许可保存在 docs/licenses/mini-ddsp.txt。

自动验收：`tests.test_mini_ddsp` 四项与 `tests.test_all_in_one_infer_cli` 两项共六项通过。覆盖 24k/44.1k/48k、B=2、T=1/4/5、无声/零有效长度、mel 与全部控制器参数有限梯度、严格资源回载逐值一致、分组/直接波形和梯度、独立逐帧重叠相加对照、原 CLI 回归。采样率测试是模块冒烟，不代表真实训练音质。FP32 分组对照锁定 rtol=2e-5、波形 atol=2e-6、梯度 atol=2e-5；后续 ORT/eager 波形对照锁定 rtol=2e-4、atol=2e-5，不能为通过而放宽。

新 Python 文件已用系统 Python 的 Black 26.10.0 格式化；diffs 环境没有 Black，未安装新依赖。测试前台 PowerShell Tee 日志、退出码 0。首次沙箱进程在导入阶段停滞，核实进程后 Ctrl-C 结束，正常 Windows 权限重跑通过，不记录该次为 PASS。

M1 已提交推送：117f05a104f629b36bb70c768e8728babb837017，远端确认一致。

M2 **PASS**（2026-11-13）：新增独立 VocoderBinarizer，顺序 acoustic→variance→vocoder，三者使用原 indexed writer 和 metadata。acoustic 与 vocoder 按稳定 sample_id/augmentation_id 配对；使用已有随机增强清单，波形由 librosa 变调/变速后重新提取 mel/F0/UV、唱法和完整 VR 谐波幅度。关闭时保留既有频谱增强。ID 采用定长字节标量，符合现有读取器，不改公共二进制格式、不升级旧包、不导入外部 npy。

VR 完整 harmonic 缓存位于本轮临时目录，身份包含波形、模型路径/文件身份/配置、输入和工作采样率及 mono 约定，顺序处理器复用。没有唱法分离时 vocoder 自动生成；WORLD 不作为 VR 结果复用。共享 hnsep_sample_rate 和权重 sr 元数据校验，非工作采样率先转换、分离后转回并精确对齐。运行结束清理中间缓存，源 wav 不动。

真实自动验收：scripts/validate_mini_ddsp_data.py --singing on/off 分别使用2001000001、2001000002训练及2033001224留出，seed42。每次 acoustic/vocoder train=10/10、valid=1/1，variance train=2、valid=1，唯一配对11、VR调用11（无重复分离）。覆盖单独变调、单独变速及组合，配对 mel/F0 逐值一致，幅度有限且形状正确，源 wav/标注 SHA256 不变，train/valid ID 无交叉。两条路径完整报告分别在 artifacts/mini_ddsp_m2/singing_on/report.json 和 singing_off/report.json（不提交数据）。增强音频与旧频谱增强的数值分布不同，仅本开关路径采用真实波形变换。

四项预处理测试和 M1/既有 CLI 回归共10项通过；三采样率重采样/缓存单测使用可计数 separator 替身，真实 VR 三包验收为44.1k，不能混称三采样率真实 VR 音质验证。修改范围 Black 格式化及 git diff --check 通过。人工听音 MANUAL/PENDING。M3～M6 尚未实施：联合训练、正式内置推理、波形 ONNX、真实质量/性能评估未验收。M2 提交推送成功后开始 M3。