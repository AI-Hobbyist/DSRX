# DSRX — GGML 量化与推理计划

## 项目概览

**DSRX** 是 OpenVPI/DiffSinger 的一个分支，一个歌声合成系统。该项目采用三阶段流水线：

1. **方差模型** — 从歌词 + MIDI 音符预测音素时长、音高以及可选的表现力方差（气息感、张力、发声力、能量）
2. **声学模型** — 使用高斯扩散（DDPM）或整流流（Rectified Flow）骨干网络，从方差模型的输出生成梅尔频谱图
3. **NSF-HiFiGAN 声码器** — 将梅尔频谱图 + F0 转换为波形音频

每个模型从 PyTorch 检查点（.ckpt 文件）训练，并在 `deployment/exporters/` 中已有完整的 ONNX 导出流水线。

---

## 1. 当前模型架构分析

### 1.1 方差模型（DiffSingerVariance / DiffSingerVarianceONNX）

| 子模块 | ONNX 文件 | 输入 | 输出 |
|---|---|---|---|
| 语言编码器 | *.linguistic.onnx | tokens, [word_div,word_dur] 或 ph_dur, [languages] | encoder_out, x_masks |
| 时长预测器 | *.dur.onnx | encoder_out, x_masks, ph_midi, [spk_embed] | ph_dur_pred |
| 音高预处理 | (合并入 *.pitch.onnx) | encoder_out, ph_dur, note_midi, note_rest, note_dur, note_glide, pitch, expr, retake, [spk_embed] | pitch_cond, base_pitch |
| 音高预测器 | (合并入 *.pitch.onnx) | pitch_cond, steps | x_pred |
| 音高后处理 | (合并入 *.pitch.onnx) | x_pred, base_pitch | pitch_pred |
| 方差预处理 | (合并入 *.variance.onnx) | encoder_out, ph_dur, pitch, variances, retake, [spk_embed] | variance_cond |
| 多变量预测器 | (合并入 *.variance.onnx) | variance_cond, steps | xs_pred |
| 方差后处理 | (合并入 *.variance.onnx) | xs_pred | variance_pred 元组 |

音高/方差的扩散后端：GaussianDiffusion（DDPM）或 RectifiedFlow（RF）。骨干网络架构：WaveNet、LYNXNet、LYNXNet2。

### 1.2 声学模型（DiffSingerAcoustic / DiffSingerAcousticONNX）

| 子模块 | ONNX 文件 | 输入 | 输出 |
|---|---|---|---|
| FS2 + 辅助解码器 | (合并入单个 *.onnx) | tokens, durations, f0, variances...，[gender, velocity, spk_embed, languages] | condition, [aux_mel] |
| 扩散/RF 骨干网络 | (合并入同一 *.onnx) | condition, [x_aux, depth], steps | mel |

两种扩散类型：DDPM（1000步，可选 K 步浅扩散）或 Rectified Flow（T_start 控制）。

### 1.3 NSF-HiFiGAN 声码器

| 子模块 | ONNX 文件 | 输入 | 输出 |
|---|---|---|---|
| 生成器 | *.onnx | mel, f0 | waveform |

---

## 2. 现有导出流水线

当前流程（`deployment/exporters/`）：

```
.ckpt -> [构建模型] -> ONNX 封装模块 -> torch.onnx.export -> onnxsim -> 合并 -> 最终 .onnx
```

关键细节：
- 导出 ONNX 需要 PyTorch 1.13（opset 15，TorchScript 跟踪）
- ONNX opset 15
- 骨干网络在使用前经过 TorchScript 跟踪
- onnxsim 简化两次（条件投影提取之前和之后）

---

## 3. 量化与 GGML 计划

### 阶段 1：ONNX 量化脚本

**脚本**：`scripts/quantize_onnx.py`

对导出的 .onnx 文件应用动态量化（仅权重量化，激活保持 FP32）。

**目标**：读取声学模型（*.onnx）、方差模型（*.linguistic.onnx、*.dur.onnx、*.pitch.onnx、*.variance.onnx）、声码器（*.onnx）→ 应用 onnxruntime 的 quantize_dynamic() → 以 _q.onnx 后缀保存。

ONNX 量化类型：

| ONNX QType | 描述 | 位宽 |
|---|---|---|
| QUInt8 | 无符号 8-bit 逐张量非对称 | 8-bit |
| QInt8 | 有符号 8-bit 逐张量对称 | 8-bit |
| QUInt16 | 无符号 16-bit | 16-bit |

使用 `onnxruntime.quantization.quantize_dynamic()`。可选支持通过 CalibrationDataReader 进行静态量化。

### 阶段 2：CKPT 直接转 GGML 转换器

**脚本**：`scripts/convert_ckpt_to_ggml.py`

直接从原始 PyTorch .ckpt 检查点提取权重量化并导出为 GGML 格式，**跳过中间 ONNX 导出步骤**。

#### 2.1 CKPT 加载与张量提取

```
.ckpt -> torch.load() -> state_dict -> 按 prefix 分离子模型 -> 逐张量量化 -> GGML 二进制
```

**加载策略：**
1. 使用 `torch.load(ckpt_path, map_location='cpu')` 加载检查点
2. 从 state_dict 中按 prefix 分离子模型：
   - `model.fs2.*` → FastSpeech2 声学/方差编码器
   - `model.diffusion.*` → DDPM/RF 扩散模型（声学）
   - `model.aux_decoder.*` → 辅助解码器（浅扩散）
   - `model.pitch_predictor.*` → 音高预测器（方差模型）
   - `model.variance_predictor.*` → 多变量预测器（方差模型）
   - `model.fs2.dur_predictor.*` → 时长预测器（方差模型）
   - `generator.*` → 声码器生成器（NSF-HiFiGAN）
3. 去除 prefix 以得到与子模块对应的局部 state_dict

#### 2.2 张量名映射与维度记录

对于每个子模块，需要记录：
- 原始 PyTorch 张量名称 → GGML 张量名称
- 张量形状（从 state_dict 直接读取）
- 权重类型（Linear/Conv1d/Embedding/LayerNorm，从形状推断）
- 是否属于 conditioner projection（标记为可缓存节点）

**推断规则（从张量形状推断类型）：**

| 张量形状 | 推断类型 | 备注 |
|---|---|---|
| [oc, ic] | Linear/Conv1D 权重 | GEMM 或 Conv 权重 |
| [oc,] | Bias | 偏置项 |
| [vocab, dim] | Embedding 权重 | 嵌入表 |
| [dim,] | LayerNorm 权重 | 仿射变换 |
| [dim,] | LayerNorm bias | 仿射偏置 |

模型实例通常会提供 `named_parameters()` 来给出更准确的类型名称。converter 会优先使用此信息。

#### 2.3 量化策略

与阶段 3（ONNX→GGML）共享同一套 GGML 量化核函数（`ggml/quantize.py`）。

**智能量化选择：**
- Auto-attention / GEMM 大权重（≥4096 参数）：K-quants（Q4_K、Q5_K）
- Conv1D 权重：Q5_0、Q8_0（保留更好精度以维持波形质量）
- Embedding 表：Q4_0 或 F16
- LayerNorm / bias：F32（不量化，精度敏感）
- 小权重（<256）：F16

#### 2.4 输出格式

GGML 二进制格式（与阶段 3 兼容）：

```
magic (uint32) = 0x67676d6c ("ggml")
version (uint32)
n_tensors (uint64)
n_kv_pairs (uint64)
[kv_pairs ...]
[tensor_headers ...]
[tensor_data ...]
```

额外 KV 元数据：
- `general.name` → 模型名称（如 `ds_acoustic`）
- `general.type` → 量化类型
- `dsrx.submodule` → 子模块名称（`fs2`、`diffusion`、`dur`、`pitch`、`variance`、`generator`）
- `dsrx.source` → `ckpt`（标记来源）

#### 2.5 命令行接口

```
# 单个 CKPT 转换
python scripts/convert_ckpt_to_ggml.py --ckpt ckpt/model_ckpt_steps_1000000.ckpt ^
    --type q4_k --output models/acoustic/model_q4_k.ggml

# 全量化扫描（输出所有量化等级）
python scripts/convert_ckpt_to_ggml.py --ckpt ckpt/model_ckpt_steps_1000000.ckpt ^
    --all --output-dir models/acoustic/

# 自动分离子模型并生成对应的 GGML 文件
python scripts/convert_ckpt_to_ggml.py --ckpt ckpt/variance_ckpt.ckpt ^
    --split-submodules --all --output-dir models/variance/
```

**`--split-submodules` 模式**会自动识别 .ckpt 中的子模块（fs2、dur_predictor、pitch_predictor、variance_predictor），分别为每个子模块导出独立的 GGML 文件。

---

### 阶段 3：ONNX 转 GGML 转换器

**脚本**：`scripts/convert_onnx_to_ggml.py`

GGML 以扁平二进制格式存储张量，包含头部 + 量化数据。

**GGML 二进制格式（头部）：**
- magic: uint32 = 0x67676d6c（"ggml"）
- version: uint32
- n_tensors: uint64
- n_kv_pairs: uint64
- kv 键值对：key/value 字符串
- 每个张量的头部：n_dims、name_len、dtype、offset、dims...
- 张量数据：量化后的字节

**要支持的 GGML 量化类型：**

| ggml_type | 值 | 描述 | 每权重位数 |
|---|---|---|---|
| GGML_TYPE_F32 | 0 | FP32 未量化 | 32 |
| GGML_TYPE_F16 | 1 | FP16 半精度 | 16 |
| GGML_TYPE_Q4_0 | 2 | 4-bit 块量化（32元素/块） | 4.5 |
| GGML_TYPE_Q4_1 | 3 | 4-bit + scale+min（32元素/块） | 5.0 |
| GGML_TYPE_Q5_0 | 6 | 5-bit 块量化 | 5.5 |
| GGML_TYPE_Q5_1 | 7 | 5-bit + scale+min | 6.0 |
| GGML_TYPE_Q8_0 | 8 | 8-bit 块量化 | 9.5 |
| GGML_TYPE_Q2_K | 10 | 2-bit K-quant（256元素/超块） | 2.6 |
| GGML_TYPE_Q3_K | 11 | 3-bit K-quant（256元素/超块） | 3.4 |
| GGML_TYPE_Q4_K | 12 | 4-bit K-quant（256元素/超块） | 4.4 |
| GGML_TYPE_Q5_K | 13 | 5-bit K-quant（256元素/超块） | 5.4 |
| GGML_TYPE_Q6_K | 14 | 6-bit K-quant（256元素/超块） | 6.6 |
| GGML_TYPE_Q8_K | 15 | 8-bit K-quant（256元素/超块） | 8.5 |
| GGML_TYPE_IQ4_NL | 17 | 4-bit 重要性量化 | 4.5 |

**转换算法：**
1. 从 ONNX 读取权重张量（numpy）
2. 根据张量大小自动选择量化类型：
   - 大型 GEMM/Conv（>=4096）：K-quants（Q4_K..Q6_K）
   - 中型（>=1024）：Q5_0、Q5_1、Q8_0
   - 小型（<1024）：F16
   - 嵌入表：Q4_0 或 F16
3. 通过纯 NumPy 实现的 GGML 量化核函数应用块量化（与阶段 2 共享 `ggml/quantize.py`）
4. 写入头部 + 量化数据

**输出目录结构：**

```
models/
  acoustic/
    model_f16.ggml, model_q4_0.ggml, ..., model_q8_k.ggml
  variance/
    linguistic_f16.ggml, linguistic_q4_0.ggml, ..., dur_*.ggml, pitch_*.ggml, variance_*.ggml
  vocoder/
    nsf_hifigan_f16.ggml, nsf_hifigan_q4_0.ggml, ..., nsf_hifigan_q8_k.ggml
```

### 阶段 4：GGML 运行时推理库

**模块**：`ggml/`（Python 包）

组件：
- `loader.py`：解析 GGML 二进制头部、读取张量元数据、加载量化数据
- `quantize.py`：量化核函数（NumPy 实现），被阶段 2 和阶段 3 共享使用
- `dequantize.py`：反量化核函数（NumPy），支持所有类型
- `ops.py`：计算操作（带量化权重的矩阵乘法、conv1d、层归一化、SiLU 等）
- `graph.py`：图执行引擎（遍历 ONNX 计算图或从 CKPT 重建计算图）
- `config.py`：量化预设

反量化支持：以上所有 14 种量化类型，在矩阵乘法期间按块反量化为 FP32。

### 阶段 5：统一推理脚本

**脚本**：`scripts/ggml_infer.py`

**CLI 接口：**

```
# 单一量化推理（GGML 格式）
python scripts/ggml_infer.py --acoustic models/acoustic/model_q4_k.ggml ^
    --variance-linguistic models/variance/linguistic_q5_k.ggml ^
    --variance-dur models/variance/dur_q5_k.ggml ^
    --variance-pitch models/variance/pitch_q5_k.ggml ^
    --vocoder models/vocoder/nsf_hifigan_q8_0.ggml ^
    --input sample.ds --output output.wav

# 对所有量化级别进行基准测试
python scripts/ggml_infer.py --benchmark --input sample.ds --output-dir benchmarks/

# 使用 ONNX runtime 后端进行比较
python scripts/ggml_infer.py --backend onnx --input sample.ds --output output.wav

# 使用 ONNX 量化模型
python scripts/ggml_infer.py --backend onnx-quant --input sample.ds --output output.wav
```

支持：完整流水线推理、跨量化混合（方差使用 Q8_0 + 声学使用 Q5_K）、从 .ds 文件读取格式。

**基准测试 JSON 输出：**
- quant_type、model_size_mb、load_time_ms、infer_time_ms、peak_mem_mb、rtf

---

## 4. 模型来源对比

| 特性 | CKPT 直接转换（阶段 2） | ONNX 导出后转换（阶段 3） |
|---|---|---|
| 依赖 | PyTorch（已有依赖） | PyTorch 1.13 + ONNX + onnxsim |
| 流程 | .ckpt → GGML（一步到位） | .ckpt → ONNX → GGML（两步） |
| 计算图信息 | 需要额外提供或推断 | ONNX 本身包含完整计算图 |
| 推理兼容性 | 输出 .ggml 文件，格式相同 | 输出 .ggml 文件，格式相同 |
| 推荐场景 | 快速实验、纯推理部署 | 需要图优化、ONNX 生态工具 |

两种路径生成的 `.ggml` 文件格式完全相同，可以被同一套 GGML 推理库（阶段 4）加载推理。

---

## 5. 实现顺序

| 阶段 | 文件 | 描述 | 预估行数 |
|---|---|---|---|
| 1 | scripts/quantize_onnx.py | ONNX 动态量化 | ~200 |
| 2 | scripts/convert_ckpt_to_ggml.py | CKPT 直接转 GGML（绕过 ONNX） | ~500 |
| 3a | scripts/convert_onnx_to_ggml.py | ONNX 转 GGML 二进制转换 | ~800 |
| 3b | ggml/quantize.py, dequantize.py | 量化核函数（NumPy），被阶段 2/3a 共享 | ~600 |
| 4 | ggml/loader.py, ops.py, graph.py | GGML 运行时（加载器 + 计算操作） | ~1500 |
| 5 | scripts/ggml_infer.py | 统一推理脚本 | ~800 |
| 6 | scripts/benchmark.py | 基准测试与质量验证 | ~400 |

**估算新增代码总量：约 4800 行。**

---

## 6. 关键技术挑战

1. **骨干网络中的 Conv1d**：WaveNet 使用带条件投影的膨胀因果卷积。ONNX 导出为带 3D 权重张量的 Conv。GGML 反量化必须处理 GEMM（2D）和 Conv（3D/4D）两种布局。

2. **K-quant 实现**：Q2_K 到 Q8_K 使用分层量化，包含 256 元素超块和多个缩放因子。必须与 llama.cpp 的逐位精确输出匹配。

3. **频谱图归一化**：DDPM/RF 骨干网络使用 spec_min/spec_max 的归一化/反归一化。GGML 运行时必须精确复制这些操作。

4. **噪声确定性**：扩散采样需要与 PyTorch 等效的种子化随机数生成器，以保证可复现性。

5. **动态形状**：ONNX 使用动态轴（n_tokens、n_frames）。GGML 运行时必须处理可变长度输入。

6. **CKPT 无图信息**：.ckpt 文件只包含权重张量，不包含计算图。直接从 CKPT 转换时，需要额外的模型架构描述文件（config.yaml）来重建计算结构，否则 GGML 推理库需要预先知道图结构来加载权重。

---

## 7. 质量验证

- 逐层 MSE：量化张量与 FP32 原始张量的对比（要求 < 1e-4）
- 输出波形：FP32 ONNX 与 GGML 量化结果之间的互相关 / SNR
- 参考输出与量化输出之间的梅尔倒谱失真

---

## 8. 文件结构

```
scripts/
  quantize_onnx.py           # 阶段 1
  convert_ckpt_to_ggml.py    # 阶段 2（新增！）
  convert_onnx_to_ggml.py    # 阶段 3
  ggml_infer.py              # 阶段 5

ggml/
  __init__.py
  loader.py                  # GGML 二进制解析器
  quantize.py                # 量化核函数（阶段 2/3 共享）
  dequantize.py              # 反量化核函数
  ops.py                     # 计算操作
  graph.py                   # 图执行引擎
  config.py                  # 预设与配置

models/                      # 输出目录
  acoustic/
  variance/
  vocoder/

benchmarks/                  # 基准测试输出
  results/
```