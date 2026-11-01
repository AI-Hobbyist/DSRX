# DSRX `dslight` 可选前级 / 后级计划书

> 日期：2026-10-31　分支：`ds_light_backend`　基线提交：`4650459`
>
> **一句话**：新增一个**纯增量、可选**的轻量组合 —— 后级 backbone `dslight` + 前级 aux decoder `dslight`，
> 设计原则参考 Synthesizer V 的"小而美"（膨胀卷积 + 窄通道 + 短核 + 单次前向），
> **但完全不与 SV 的格式、键名、模型、数值兼容**。
>
> **默认行为不变**：现有 `wavenet / lynxnet / lynxnet2 / dit` 后级与 `convnext` 前级保持原样，旧配置、旧 checkpoint、
> 旧推理路径不受影响；`dslight` 只能通过配置显式选中。

---

## 1. 结论与项目边界

| 决策 | 方案 |
| --- | --- |
| 新增方式 | **增量追加**：只加注册项、新模块文件、新配置文件；不改动既有架构的数值行为 |
| 后级（backbone） | 新注册 `dslight`，与 `lynxnet2` / `dit` 并列，做成**可切选项** |
| 前级（aux decoder） | 新注册 `dslight`，与 `convnext` 并列，做成**可切选项** |
| 默认值 | `acoustic.yaml` 等既有配置**一行不改**；`convnext` 与 `lynxnet2` 仍是默认 |
| 旧权重兼容 | 不改 `architecture_fields` 契约；旧 checkpoint 不加载到 `dslight`，`dslight` checkpoint 也不会被旧架构误载 |
| 与 SV 的关系 | **只借鉴设计原则**，不实现任何 SV 兼容层（见 §10 硬约束清单） |
| 本计划交付 | 计划书 + 后续按 P0–P4 实现；本文不包含任何代码改动 |

**"可选"的准确含义**：同一个实验既可以继续训练 `lynxnet2 + convnext`，也可以新建 `dslight + dslight` 的对照实验；
两者可在同一仓库共存、可按实验切换、可各自导出；**不存在"默认替换"**。

---

## 2. 当前仓库事实与接入位置

| 现状 / 位置 | 对本计划的约束 |
| --- | --- |
| [后级注册表](modules/backbones/__init__.py#L8-L13)：`BACKBONES = {wavenet, lynxnet, lynxnet2, dit}`，由 `build_backbone()`（L15）实例化 | 追加一个键即可接入主构造链；**不能**另造一套模型系统 |
| [后级参数校验](modules/backbones/__init__.py#L18-L29)：仅 `dit` 走白名单严格校验，其余走 `filter_kwargs` 静默过滤 | `dslight` 必须像 `dit` 一样**严格拒绝未知参数**，避免拼错的超参被静默忽略 |
| [前级注册表](modules/aux_decoder/__init__.py#L7-L12)：`AUX_DECODERS = {'convnext': ...}`、`AUX_LOSSES = {'convnext': nn.L1Loss}` | 追加 `dslight` 键；**必须同时**提供 loss 注册项，否则 `build_aux_loss()` 报 KeyError |
| [前级构造](modules/aux_decoder/__init__.py#L15-L25)：`build_aux_decoder()` 用 `filter_kwargs(aux_decoder_args, decoder_cls)` 过滤 | 前级侧存在与后级相同的"静默丢参"风险，`dslight` 前级需自校验 |
| [前级契约](modules/aux_decoder/__init__.py#L28-L70)：`AuxDecoderAdaptor(in_dims, out_dims, num_feats, spec_min, spec_max, arch, args)`；`decoder(condition, infer=)` 返回 `[B, T, out_dims*num_feats]` | 新前级**只能**实现 `forward(condition, infer=False)`；反归一化、`num_feats>1` 的 reshape/transpose 由 Adaptor 负责，不要重复实现 |
| [声学模型装配](modules/toplevel.py#L58-L67)：`AuxDecoderAdaptor(in_dims=hparams['hidden_size'](256), out_dims=audio_num_mel_bins(128), num_feats=1, ...)` | `dslight` 前级默认 `in_dims=256, out_dims=128`；不要把 `hidden_size` 写死进模块 |
| [声学前向](modules/toplevel.py#L104-L127)：推理时 `aux_mel_pred = aux_decoder(condition, infer=True)` 后按 `mel2ph>0` 置零；训练时用 `aux_decoder_grad` 做条件梯度混合 | 掩码与梯度混合逻辑**不变**；`dslight` 只需产出同形状张量 |
| [后级张量契约](configs/dit/p0_baseline.yaml#L36-L44)：`spec [B,F,M,T]`、`diffusion_step [B]`、`condition [B,H,T]`、`output [B,F,M,T]`、`valid_mask [B,T]` | `dslight` 后级必须严格满足；`valid_mask` 由 `run_backbone()` 按 `supports_valid_mask` 决定是否传入 |
| [配置位置](configs/original/acoustic.yaml#L78)：`backbone_type: 'lynxnet2'`；[浅扩散参数](configs/original/acoustic.yaml#L97-L106)：`aux_decoder_arch: convnext` / `aux_decoder_args` | 新配置**另建文件**（如 `configs/dslight/acoustic.yaml`），不改原文件 |

### 2.1 现状参数账本（用于说明"为什么值得做可选档"）

前置假设：`hidden_size=256`、`audio_num_mel_bins=128`、`num_feats=1`。

**现有前级 `convnext`（C=512, L=6, k=7）**

| 部件 | 参数量 |
| --- | --- |
| `inconv` 256→512, k=7 | 918,016 |
| 每个 `ConvNeXtBlock`：dwconv 512·7 + LN + 512×2048 + 2048×512 + γ | ≈ 2,105,344 |
| × 6 块 | 12,632,064 |
| `outconv` 512→128, k=7 | 458,880 |
| **合计** | **≈ 14.01 M** |

即：**当前"前级"比多数后级还大**（默认 `lynxnet2` 为 1024 通道 × 6 层）。这是一个明确的优化空间，
也是本计划的直接动机之一。

---

## 3. 设计：`dslight` 后级（backbone）

### 3.1 张量契约（与既有 backbones 完全一致）

| 项 | 形状 / 说明 |
| --- | --- |
| 输入 `spec` | `[B, F, M, T]`（flatten 为 `[B, F*M, T]` 参与卷积） |
| 输入 `diffusion_step` | `[B]`（连续时间或离散步，沿用上层传入语义，不自行缩放） |
| 输入 `condition` | `[B, H, T]`（`H = hidden_size`） |
| 输出 | `[B, F, M, T]` |
| `valid_mask` | `supports_valid_mask = False`（沿用现有卷积后级的无掩码契约；卷积可跨相邻帧传播，输出掩码不能保证有效帧不受填充影响） |

### 3.2 结构

```
x[B,F*M,T] ──1×1──> h[C,T]
cond[B,H,T] ─1×1──> c[C,T]                       （逐层复用，只投影一次）
t[B] ─sinusoid─> Linear(Te→C) ─> e[C]             （全局时间投影，只算一次）

stem:  depthwise Conv1d(C, k=7, groups=C) + pointwise C→C
×N block（膨胀循环 d = 1,3,9,27,81）:
      h = depthwise Conv1d(C, k=7, groups=C, dilation=d)
      a, b = split(pointwise C→2C)
      h = concat(a * sigmoid(b), a)               （门控输出与未门控分支拼接）
      h = pointwise 2C→C
      h = h * (1 + e) + e + block_cond_proj(c)               （时间调制的残差支路）
      x = x + h                                   （残差）

head: 1×1  Conv1d(C → F*M) ──> reshape ──> [B,F,M,T]
```

**设计要点（借鉴 SV 的可解释部分，非复制其实现）**

1. **膨胀卷积替代全局注意力**：d=1/3/9/27/81，感受野指数增长，无 O(T²)，无 KV cache，CPU 上纯卷积。
2. **窄通道 + 短核**：C=192、k=7，单张权重只有几十 KB，缓存友好。
3. **深度可分离 + GLU 门控**：主算力在 `C→2C` 的逐点卷积上，参数与通道线性相关。
4. **条件与时间调制只做一次投影**：输入条件先投影一次，每层再用独立的 `C→C` 条件投影；时间投影只计算一次，缩放和偏移均由同一时间向量得到。

### 3.3 参数与算力预算（三档）

固定量：`F·M = 128`（mel bins）、`H = 256`（`hidden_size`）、`Te = 64`（时间嵌入维度）、`k = 7`。

| 档位 | C | N | 膨胀循环 | 参数量 | 每帧 MAC | 相对折中档 |
| --- | --- | --- | --- | --- | --- | --- |
| **速度优先** | 128 | 8 | 1, 3, 9, 27 | **≈ 0.76 M** | 0.75 M | 0.32× |
| **折中（默认）** | **192** | **12** | **1, 3, 9, 27, 81** | **≈ 2.39 M** | 2.36 M | 1.00× |
| **质量优先** | 256 | 15 | 1, 3, 9, 27, 81 | **≈ 5.18 M** | 5.14 M | 2.17× |

> 三档只改变 `num_channels / num_layers / dilation_cycle` 三个显式参数，不引入隐式 preset 机制
> （与 `p0_baseline.yaml` 的 "unknown args reject" 契约一致）。

折中档（C=192, N=12, F·M=128, H=256）明细：

| 部件 | 参数量 |
| --- | --- |
| `in_proj` 128→192, k=1 | 24,768 |
| `cond_proj` 256→192, k=1 | 49,344 |
| `time_mlp` 64→192 | 12,480 |
| `stem`（dw 192,k=7 + pw 192→192） | 38,592 |
| 每 block（dw 1,536 + pw1 74,112 + pw2 73,920 + cond 37,056） | 186,624 |
| × 12 block | 2,239,488 |
| `head` 192→128, k=1 | 24,704 |
| **合计** | **≈ 2.39 M** |

每帧 MAC ≈ 2.36 M；按 44.1 kHz / hop 512（≈86.1 帧/秒）估算，单采样步 ≈ 0.20 GMAC/s，
20 步 ≈ 4.07 GMAC/s。**注意：总成本仍由采样步数主导，本计划不改采样器与步数。**

---

## 4. 设计：`dslight` 前级（aux decoder）

### 4.1 契约（与 `convnext` 完全一致）

```python
class DSLightAuxDecoder(nn.Module):
    def __init__(self, in_dims, out_dims, /, *, num_channels=128, num_layers=4,
                 kernel_size=7, dilation_cycle=(1, 3, 9), dropout_rate=0.0): ...

    def forward(self, condition, infer=False):
        # condition: [B, T, in_dims]  ->  [B, T, out_dims]
        ...
```

- 输入/输出形状、`infer` 语义与 `ConvNeXtDecoder` 相同；
- **只负责**"条件 → 频谱"映射，不做反归一化（由 `AuxDecoderAdaptor.denorm_spec` 处理）；
- 不引入 `num_feats` 逻辑（Adaptor 已处理）。

### 4.2 参数与算力（三档）

固定量：`in_dims = 256`、`out_dims = 128`。

| 档位 | C | N | k | 参数量 | 每帧 MAC | 相对现有 `convnext`（14.01 M） |
| --- | --- | --- | --- | --- | --- | --- |
| **速度优先** | 96 | 3 | 5 | **≈ 0.20 M** | 0.20 M | 1.4% |
| **折中（默认）** | 128 | 4 | 7 | **≈ 0.42 M** | 0.41 M | 3.0% |
| **质量优先** | 192 | 6 | 7 | **≈ 1.12 M** | 1.11 M | 8.0% |

### 4.3 结构与账本（折中档）

```
condition[B,T,256] ─transpose─> [B,256,T]
  1×1  Conv1d(256 → 128)
  × 4 block（d = 1,3,9,...）:
      depthwise k=7 + 1×1 128→256 + 分割门控并拼接 + 1×1 256→128 + 残差
  k=7  Conv1d(128 → 128)
  transpose ─> [B,T,128]
```

| 部件 | 参数量 |
| --- | --- |
| `in_proj` 256→128, k=1 | 32,896 |
| 每 block（dw 1,024 + pw1 33,024 + pw2 32,896） | 66,944 |
| × 4 block | 267,776 |
| `out_proj` 128→128, k=7 | 114,816 |
| **合计** | **≈ 0.42 M** |

对比 `convnext` 的 ≈14.01 M：**参数量 −97%**，每帧 MAC 从 ≈3.6 M 降到 ≈0.30 M。

### 4.4 为什么前级可以这么小

前级只是**浅扩散的起点估计器**（`T_start=0.4` 附近的粗谱），精度要求低于终谱；
其上还有完整的扩散/流匹配阶段负责细节。把前级做小，对最终音质的影响远小于对速度/显存的影响，
这与 SV "把简单任务交给小网络" 的取向一致，但实现、训练目标、权重与 SV 无任何关系。

---

## 5. 配置（可选、按实验生效）

新增文件（**不改动 `configs/original/*.yaml`**）：

`configs/dslight/acoustic.yaml`（示例片段）

```yaml
backbone_type: 'dslight'
backbone_args:
  num_channels: 192
  num_layers: 12
  kernel_size: 7
  dilation_cycle: [1, 3, 9, 27, 81]
  cond_channels: 256        # = hidden_size
  time_embed_dim: 64
  dropout_rate: 0.0
  # 三档差异只体现在 num_channels / num_layers / dilation_cycle（见 §5.1），不加隐式 preset

shallow_diffusion_args:
  train_aux_decoder: true
  train_diffusion: true
  val_gt_start: false
  aux_decoder_arch: dslight
  aux_decoder_args:
    num_channels: 128
    num_layers: 4
    kernel_size: 7
    dilation_cycle: [1, 3, 9]
    dropout_rate: 0.0
  aux_decoder_grad: 0.1
```

### 5.1 三档总表（前级 + 后级）

| 档位 | 后级参数 | 前级参数 | **合计参数** | 每采样步 MAC/s | 20 步累计 | 参考用途 |
| --- | --- | --- | --- | --- | --- | --- |
| **速度优先** | 0.76 M | 0.20 M | **≈ 0.96 M** | 64 M | 1.28 GMAC/s | 端侧 / 低配 CPU、批量预处理、快速迭代 |
| **折中（默认）** | 2.39 M | 0.42 M | **≈ 2.80 M** | 204 M | 4.07 GMAC/s | 通用训练与推理 |
| **质量优先** | 5.18 M | 1.12 M | **≈ 6.30 M** | 443 M | 8.86 GMAC/s | 质量优先，仍可 CPU 实时 |

估算条件：44.1 kHz / hop 512 ≈ 86.13 帧/秒；20 步对应 `sampling_steps=20`；前级每次合成只跑 1 次。

**关键结论**：仅把前级从 `convnext` 换成 `dslight`（折中档），参数就从 14.01 M 降到 0.42 M
（**−13.59 M，−97%**），且不改变后级、采样器、训练目标与既有配置；
三档合计（0.96 M / 2.80 M / 6.30 M）都比现有前级单独一项更小。

### 5.2 对照开关

| 目标 | 配置 |
| --- | --- |
| 旧默认（不变） | `backbone_type: lynxnet2` + `aux_decoder_arch: convnext` |
| 只换后级 | `backbone_type: dslight` + `aux_decoder_arch: convnext` |
| 只换前级 | `backbone_type: lynxnet2` + `aux_decoder_arch: dslight` |
| 全轻量 | `backbone_type: dslight` + `aux_decoder_arch: dslight` |

---

## 6. 训练速度预测（it/s）

> 本节全部为**静态估算**，不是实测。P3/P4 必须用实测值替换本表（见 §6.6）。

### 6.1 预测模型

训练一个优化步 = **1 次前向 + 1 次反向**，前向包含：FastSpeech2 编码器（音素级）
+ **1 个随机扩散步**的后级 + 前级 + loss。按常规近似：

```
每步 FLOP ≈ 6 × 每步 MAC        （前向 2×MAC，反向 4×MAC）
it/s = 1 / (每步 FLOP / 有效 GPU 算力 + 固定开销)
```

假设与边界（全部显式列出，便于替换）：

| 项 | 取值 | 依据 |
| --- | --- | --- |
| 采样率 / hop | 44.1 kHz / 512 | 与仓库 vocoder 配置一致（≈86.1 帧/秒） |
| 音素 : 帧 | 1 : 5 | 歌唱场景常见区间（4–8）取中值；影响编码器项 |
| 编码器 | H=256、4 层、2 头、FFN 4H、k=3 | `base.yaml` L29–L36 + `EncSALayer` |
| 编码器算力 | ≈7.35 M MAC/音素 + 自注意力 `2·P²·H` | 结构推算 |
| 有效算力 | 20 / 35 TFLOP/s 两档 | RTX 4090 + bf16 autocast 的保守/乐观利用率 |
| 固定开销 | 30 ms/步 | DataLoader、优化器、loss、日志、Python；**估算值** |
| 未计入 | 验证、存盘、数据增强、vocoder 验证 | 由 `val_check_interval` 决定，不属于纯训练步 |

### 6.2 每帧算力对照（前向）

| 配置 | 后级 MAC/帧 | 前级 MAC/帧 | 合计 |
| --- | --- | --- | --- |
| 基线 `lynxnet2`(1024,6,k31) + `convnext`(512,6,k7) | 31.91 M | 13.99 M | **45.89 M** |
| `dslight` 速度优先 | 0.76 M | 0.20 M | **0.96 M** |
| `dslight` 折中 | 2.39 M | 0.42 M | **2.80 M** |
| `dslight` 质量优先 | 5.18 M | 1.12 M | **6.30 M** |

### 6.3 含编码器的每步 MAC 与加速比

| 配置 | 16k 帧/步 | 相对基线 | 50k 帧/步 | 相对基线 |
| --- | --- | --- | --- | --- |
| 基线 | 763.3 G | 1.00× | 2420.0 G | 1.00× |
| 速度优先 | 44.1 G | **17.3× 更省** | 172.5 G | **14.0× 更省** |
| 折中 | 73.6 G | **10.4× 更省** | 264.8 G | **9.1× 更省** |
| 质量优先 | 129.5 G | **5.9× 更省** | 439.4 G | **5.5× 更省** |

> 编码器是公共成本：批量越大、音素越多，自注意力 `P²` 项占比越高，加速比随批量略微下降。

### 6.4 it/s 预测

**有效算力 20 TFLOP/s（保守）**

| 配置 | 6144 帧 | 16k 帧 | 50k 帧 |
| --- | --- | --- | --- |
| 基线 | ≈ 8.5 | ≈ 3.9 | ≈ 1.3 |
| 速度优先 | ≈ 28.8 | ≈ 23.1 | ≈ 12.2 |
| **折中** | **≈ 26.2** | **≈ 19.2** | **≈ 9.1** |
| 质量优先 | ≈ 22.5 | ≈ 14.5 | ≈ 6.2 |

**有效算力 35 TFLOP/s（乐观）**

| 配置 | 6144 帧 | 16k 帧 | 50k 帧 |
| --- | --- | --- | --- |
| 基线 | ≈ 12.5 | ≈ 6.2 | ≈ 2.2 |
| 速度优先 | ≈ 30.6 | ≈ 26.6 | ≈ 16.8 |
| **折中** | **≈ 28.9** | **≈ 23.5** | **≈ 13.3** |
| 质量优先 | ≈ 26.1 | ≈ 19.2 | ≈ 9.5 |

**帧吞吐（20 TFLOP/s）**：基线 ≈62 k帧/s；速度优先 ≈370–612 k帧/s；折中 ≈307–457 k帧/s；质量优先 ≈232–309 k帧/s。

### 6.5 怎么读这张表

1. **折中档预期区间：16k 帧 19–24 it/s，50k 帧 9–13 it/s**（4090 量级）。
2. **it/s 提升远小于算力提升**：算力降了约 10×，但固定开销（30 ms）会吃掉大部分，
   `it/s` 上限约为 `1/0.03 ≈ 33`，所以 dslight 在小批量下会先"撞天花板"。
3. **更合理的用法是加大 batch**：dslight 权重与激活都小，可把 `max_batch_frames`
   从 16k 提到 32k–50k，墙钟吞吐（帧/秒）更高。
4. **瓶颈常常不在算子**：这类管线的 DataLoader/特征预处理很容易成为瓶颈；
   若实测明显低于预测，先查数据侧（`num_workers`、缓存、batch frames 抖动）。
5. **显存**：折中档权重约 11 MB + 优化器状态；按 12 层 × 帧数 × 192 × 4B 粗估，
   每 1 万帧激活约 0.2–0.3 GB（bf16 更低）；8 GB 卡跑 16k–32k 帧、24 GB 卡跑 50k 帧应有余量
   （**待 P4 实测确认**）。

### 6.6 必须做的校准（P3 门槛）

预测表不能替代实测。P3 至少完成：

1. 同一配置、同一设备、同一 DataLoader 设置下连续跑 **50–100 步**，丢弃前 10 步（warmup），取中位数。
2. 记录：`it/s`、`frames/s`、`max_memory_allocated`、`num_workers`、`accumulate_grad_batches`。
3. 用基线实测换算：`it/s_dslight ≈ 1 / (t_base_measured × ratio + t_overhead)`，
   `ratio` 取 §6.3 对应批量下的值；偏差 >30% 时以实测为准并回写本节。
4. 换 GPU / 精度 / 批量模板必须重测，**不得跨设备外推**本节数值。

---

## 7. 集成点清单（实现阶段）

| # | 文件 | 动作 | 风险 / 注意 |
| --- | --- | --- | --- |
| 1 | `modules/backbones/dslight.py` | 新增 `DSLightBackbone`（满足 §3.1 契约） | 必须可 `torch.jit.trace`；禁止数据相关控制流 |
| 2 | `modules/backbones/__init__.py` L8-L13 / L15 | 注册 `'dslight'`；为其加**严格参数白名单**（同 `dit` 分支） | 不要动其他键的行为 |
| 3 | `modules/aux_decoder/dslight.py` | 新增 `DSLightAuxDecoder` | 只用基础算子（Conv1d / Sigmoid / Split / Add） |
| 4 | `modules/aux_decoder/__init__.py` L7-L12 | 注册 `'dslight'` 到 `AUX_DECODERS` **与** `AUX_LOSSES`（沿用 `nn.L1Loss`） | 只加一组键，不改 `AuxDecoderAdaptor` |
| 5 | `configs/dslight/*.yaml` | 新增实验配置 | 不修改 `configs/original/*` |
| 6 | `deployment/exporters/acoustic_exporter.py` | 复用既有导出路径；确认 `dslight` 走 opset 15 基础算子分支 | 不得要求 SDPA / 自定义算子 |
| 7 | `inference/optimization.py` | 复用 `_velocity_backbones` 的 trace 逻辑（依赖 `velocity_fn` 属性） | `dslight` 作为 velocity 后级时必须 trace 成功；FP16 选择性替换只作用于 `nn.Linear`，本设计以 Conv 为主，需确认无遗漏收益/风险 |
| 8 | `tests/` | 新增单测：形状、参数量、mask 语义、注册与配置解析、ONNX trace | 无既有测试覆盖这些路径（见 CodeGraph 依赖结论） |

**未列出的文件一律不改**（尤其 `modules/toplevel.py`、`modules/core/*`、`training/*`）：
契约相同，因此无需改动；若实现中发现必须改，先回到本计划书说明原因。

---

## 8. 工作包与完成条件

P0 评审记录：用户确认使用门控输出与未门控分支拼接以保留 `2C` 中间维度；前级输出卷积使用 `kernel_size`（折中档为 7）。原配置和默认架构保持不变。时间调制复用同一个 `C` 维投影以匹配原参数账本。P0 后各阶段验证结果在此追加。

P1 验证：`conda activate diffs`，执行 `python -m unittest discover -s tests -p test_dslight_backbone.py`，3 项通过；三档后级参数量实测分别为 759,424、2,389,376、5,177,472。测试覆盖多特征形状、无掩码分派、trace 等价及未知参数拒绝。

P2 验证：`conda activate diffs`，执行 `python -m unittest discover -s tests -p test_dslight_aux.py`，3 项通过；三档前级参数量实测分别为 199,424、415,488、1,118,912。覆盖适配器训练和推理、反归一化、多特征形状、loss 注册及未知参数拒绝。

P3 验证：`diffs` 环境运行 `python -m unittest discover -s tests -p test_dslight_training_smoke.py`，2 项通过。合成固定批次训练 20 步，辅助损失 0.05332→0.01273，主损失 11.56686→0.78647；验证了权重路径、有限梯度和旧配置前反向。另以本机 OpenCpop 的 7 条训练、1 条验证录音预处理，运行真实 Lightning 训练 60 步，训练日志中两项损失整体下降；这是短程 smoke，不代表收敛或音质合格。可编辑模板为 `configs/dslight/template.yaml`。

P3 小样本设备校准：RTX 5060 Laptop GPU，FP32，batch=1、num_workers=0、accumulate_grad_batches=1，同一 7 条录音（351–583 帧），各运行 60 步、丢弃前 10 步；基线 `lynxnet2+convnext` 为 13.57 it/s、6289 frames/s、峰值 1,230,164,480 bytes，`dslight+dslight` 为 10.81 it/s、5183 frames/s、峰值 186,021,376 bytes。中位数分别按步时间及逐步帧吞吐计算。第一次测量该小批量 DSLight 未加速；重复测量结果分别为 17.61 和 14.75 it/s，速度受运行条件波动影响；计划原有 4090 / 16k–50k 帧预测不适用于此设备和批量，尚未获得对应规模的实测，不能跨设备外推。基准脚本为 `tests/dslight_benchmark.py`，本地配置位于忽略目录 `data/dslight_smoke/`，数据并非项目交付。

P4 实测（小样本边界）：使用本机 OpenCpop 7 条训练、1 条验证录音，RTX 5060 Laptop GPU 上 FP32、相同数据加载顺序和 AdamW 设置训练。下表的 L1 是单条验证录音的自然对数 mel 误差；60 步与 600 步是不同短程对照，不代表收敛模型或全数据集效果。

| 配置 / 训练步数 | 前级 mel L1 | 最终 mel L1 | CPU 单线程声学 RTF（20 步） |
| --- | ---: | ---: | ---: |
| `lynxnet2+convnext` / 60 | 1.17810 | 2.84107 | 1.6334 |
| `dslight+dslight` 折中 / 60 | 1.31350 | 3.58102 | 0.1421 |
| `dslight+dslight` 质量优先前级 / 60 | 1.29083 | 3.57560 | 0.1581 |
| `lynxnet2+convnext` / 600 | 1.19784 | 未测 | 未测 |
| `dslight+dslight` 质量优先前级 / 600 | 1.22433 | 未测 | 未测 |

60 步时折中和质量前级分别比基线高约 11.5% 和 9.6%，不达 5% 门槛；600 步质量前级高约 2.21%，在该单条验证样本通过该数值门槛。该样本不足以证明泛化质量，正式使用须以代表性验证集和收敛权重重测。RTF 仅计声学模型，不含 vocoder；CPU 线程数为 1，取热身后 3 次中位数。质量档配置 `configs/dslight/quality.yaml` 与数据模板组合使用。

P4 导出：在 `diffs`（PyTorch 2.11、ONNX 1.16.2、ONNX Runtime 1.23）直接调用 `DiffSingerAcousticExporter`，使用 60 步短训权重成功生成完整 opset 15 声学图；ONNX checker 通过，CPU Runtime 在 16 和 64 帧输出有限 `[1,T,128]`。前后级独立动态帧数值比较通过。`scripts/export.py` 入口要求 PyTorch 1.13，当前环境未验证该 CLI 入口。人工抽听依用户要求跳过，没有主观质量结论。

P4 扩展验证（未达质量门槛）：同一 OpenCpop 数据源的 20 条未用于训练的录音，按 mel 元素加权计算。7 条训练、600 步时，基线 L1 为 1.05026，质量档为 1.21546（高 15.7%）；进一步加参至前级 256 通道、8 层（2,414,976 参数）后为 1.15272（高 9.8%）。将训练扩展至 107 条、两组同样训练 600 步后，基线为 1.40710，加参档为 1.81484（高 29.0%）。同一 107 条训练录音和 20 条验证录音继续分别从头训练 3,000 步后，基线为 1.02351、加参档为 1.27218（高 24.3%）。这些数据不证明最终收敛效果，但当前更广的验证集未通过 ≤5% 门槛；P4 质量验收仍未完成，需使用充分训练的可比权重和代表性验证集重测。未把扩展配置设为默认。

P4 质量诊断：基于前述 107 条训练 / 20 条验证、3,000 步权重，原始完整模型前级 L1 分别为 1.02351 / 1.27218。交叉输入另一模型编码器输出后，旧编码器 + DSLight 前级为 4.53547，DSLight 编码器 + 旧前级为 1.55056；交叉组合存在分布不匹配，不能单独据此归因。进一步固定**同一个旧模型编码器**，在相同缓存条件上分别从头训练旧前级与 C=256/N=8 的 DSLight 前级各 1,500 步（AdamW 1e-4），20 条验证集 L1 为 1.00726 / 1.05335（差约 4.6%；一次独立重复的 DSLight 值为 1.05246）。该控制实验表明在固定条件下差距较小，提示联合训练条件值得进一步调查；它不等同于 DSLight 全模型训练结果，也不解除 P4 质量门槛。可用 `tests/dslight_frozen_encoder_probe.py --config <配置> --weights <旧模型 state_dict>` 复核，需匹配的数据和权重。

P4 联合训练梯度对照：在相同 107 条训练 / 20 条验证录音、3,000 步、FP32 AdamW 1e-4 和相同数据顺序下，将 C=256/N=8 DSLight 前级的 `aux_decoder_grad` 从 0.1 提至 1.0，前级 L1 从 1.27218 降至 1.10516，但仍比旧组合 1.02351 高约 8.0%，未达 ≤5% 门槛。固定随机种子按帧加权测量的验证主损失 L2：旧组合 0.24231、DSLight 原梯度 0.97587、梯度 1.0 为 0.97524；主损失差距仍在，此处不据此前级指标宣称质量达标。结果仅代表该数据与训练时长。`tests/dslight_benchmark.py` 支持 `--steps`、`--aux-decoder-grad` 和 `--save-weights`；`tests/dslight_quality_compare.py` 可在指定数据与权重上复核前级及主损失。

P4 后级诊断：相同 107 条训练 / 20 条验证、3,000 步权重，固定噪声并分别在 t=0.4/0.55/0.7/0.85/0.99 测试主损失 L2，旧组合为 0.08175/0.08721/0.11503/0.24605/1.01829，DSLight 加参前级组合为 0.93425/0.93043/0.95819/1.00356/1.05699。训练集前 20 条在 t=0.55 分别为 0.08100/0.92804，说明差距不只出现在所测验证集。验证集 t=0.55 的预测 RMS 为 1.081/0.418、预测与目标余弦相似度为 0.963/0.473；把频谱输入置零引起的输出变化 RMS 为 1.128/0.357。这些是诊断关联，不足以确定根因。将后级提高到原计划质量档 C=256/N=15 并训练 600 步后，验证集 t=0.55 L2 为 1.25324；同样 600 步时旧组合为 0.18838，原后级为 1.19893，未观察到仅增加后级参数的改善。`tests/dslight_timestep_probe.py` 可按指定配置、权重与数据重新测量；不同训练时长和设备不能直接外推。

P4 训练条件与结构诊断（实验未进入正式模型）：同一训练数据、验证集、AdamW 及 600 步下，学习率统一使用原声学配置的 6e-4 时，旧组合前级 L1 / 主损失 L2 为 1.71481 / 0.45182，DSLight 加参前级为 1.46486 / 1.09802。前级在此短训对照优于旧组合，但主损失仍较高，不能据此声称完整模型质量达标。`tests/dslight_benchmark.py --learning-rate` 可指定实验学习率。此前 3,000 步的初始化梯度探针在一条样本上观察到条件投影梯度 RMS 约为输入投影的 22 倍，仅为单样本关联证据。独立脚本中的两个单变量实验各训练 600 步：给后级 stem 加输入残差时，前级 L1 / 主 L2 为 1.81256 / 1.20421；将逐层条件投影缩放到 0.1 时为 1.68304 / 1.20808；原组合为 1.81484 / 1.20865。两项均未明显改善主损失，未修改正式结构。真实收敛及主损失差距仍需继续诊断。

P4 固定编码器后级对照：以旧组合 3,000 步权重的声学编码器固定生成条件，仅从头训练折中档 DSLight 后级 1,500 步，107 条训练 / 20 条验证、AdamW 6e-4、FP32，t=0.55 的验证主损失 L2 为 0.60291；旧组合已训练 3,000 步的后级为 0.08721。训练步数不同，因此不能据此量化最终质量差异；该实验表明单独固定编码器并未立即消除当前后级差距。`tests/dslight_frozen_backbone_probe.py` 提供可指定配置、旧模型权重和训练步数的复测入口；默认重建同样的对照。

P4 输入直连消融（仅独立实验，未改正式模型）：在此前 107 条训练 / 20 条验证、FP32 AdamW 1e-4、600 步条件下，临时令 DSLight 后级输出为原输出减去当前频谱状态（无新增参数），验证前级 L1 / 主损失 L2 为 1.79637 / 0.85297；未改结构的 DSLight 为 1.81484 / 1.20865，旧组合为 1.40710 / 0.38589。该单次短训对照中主损失下降但仍高于旧组合，前级质量也未达标；实验不能证明根因或质量合格。训练与验证临时脚本及权重保留在本机忽略目录 `data/dslight_eval100/`，正式模型结构与默认配置未变。若拟调整已冻结契约，须先形成充分对照并取得用户确认。

P4 权重隔离修复（部分）：非 LoRA 声学导出中，若显式启用 DSLight 后级或前级，使用严格 checkpoint 加载，拒绝旧后级或旧前级的错配；不改变旧架构导出的原非严格策略。`test_dslight_checkpoint_export.py` 在 `diffs` 中通过 5 项测试，覆盖匹配的 DSLight、旧后级错配、旧前级错配、仅前级 DSLight 错配及旧配置加载。后续以源 checkpoint 中 DSLight 前后级特征键拒绝反向导出为旧架构，并验证旧架构部分 checkpoint 仍按原宽松策略加载；共 8 项导出测试通过。该检查仅覆盖非 LoRA 声学导出，LoRA 及其他入口仍需单独审计，不能据此宣称权重隔离全部完成。

P4 调参记录：107 条训练 / 20 条验证、FP32 AdamW、600 步下，DSLight 加参前级（C=256/N=8）设置学习率 1.2e-3、`aux_decoder_grad=1.0`，前级 L1 / 后级主损失 L2 为 1.90650 / 1.00855；原组合学习率 6e-4 时为 1.71481 / 0.45182。两组学习率不同，是超参候选与原组合基线的短训比较，不是同超参实验；候选前级较基线高约 11.2%，后级主损失也更高，未达质量目标。权重位于本机忽略目录 `data/dslight_eval100/`，不能作为收敛或听感结论。

P4 后级独立学习率实验：同一 107 条训练 / 20 条验证、600 步、FP32 AdamW，原组合统一学习率 1e-4；DSLight 组合将后级设为 6e-4、其他模块 1e-4、`aux_decoder_grad=1.0`。原组合前级 L1 / 后级主 L2 为 1.40710 / 0.38589，DSLight 为 1.75128 / 1.09855，均未达到以原组合为基线的 5% 目标。`tests/dslight_benchmark.py --backbone-learning-rate` 可复核该参数分组；两组优化器参数不同，结论限本短训实验。

P4 LoRA 导出基座保护（部分）：对包含 DSLight 前级或后级的声学导出，在注入 LoRA 之前严格加载完整基座，拒绝旧架构及同架构尺寸不匹配；匹配 DSLight 基座可加载。`test_dslight_checkpoint_export.py` 共 11 项通过。此项只覆盖声学导出器的基座，后续 adapter、原生推理及 variance 等入口仍待单独审计，不能宣称所有 LoRA 路径已隔离。

P4 声学 LoRA 加载补充：用户授权修改计划外的原生声学推理入口及必要共享校验工具。对显式选用 DSLight 的声学模型，原生推理与导出在注入 LoRA 前严格加载基座；后续完整权重或纯 adapter 在加载前检查键和尺寸，包含原生推理优先 `.infer.ckpt` 的情况。`test_dslight_checkpoint_export.py` 和 `test_dslight_checkpoint_inference.py` 共 22 项通过，覆盖匹配基座、架构 / 尺寸错配、错误后续完整权重、合法纯 adapter 和旧配置加载。该记录不覆盖 variance 或旧架构 LoRA 反向加载 DSLight 权重，仍需审计。

P4 参数账本（构造实测）：以 `configs/templates/all_in_one_lynxnet2_muon.yaml` 为基础，词表 65，所有 duration/pitch/energy/breathiness/voicing/tension 开关开启，H=256，mel=128。DSLight 候选的声学后级设为 C=192/N=12，pitch 和 variance 后级各设为 C=128/N=8，声学前级设为 C=256/N=8；variance 仅做构造和后级张量形状 smoke，未完成 variance 训练与质量验证，不作为已验证模型。`tests/dslight_parameter_report.py` 可复核。

| 模块 | 旧配置：参数 / FP32 权重 | DSLight 候选：参数 / FP32 权重 |
| --- | ---: | ---: |
| 声学编码器 | 5.273 M / 20.12 MiB | 5.273 M / 20.12 MiB |
| 声学前级 | 14.009 M / 53.44 MiB | 2.415 M / 9.21 MiB |
| 声学后级 | 40.619 M / 154.95 MiB | 2.389 M / 9.11 MiB |
| **声学合计** | **59.902 M / 228.51 MiB** | **10.078 M / 38.44 MiB** |
| variance 编码器（含 duration） | 6.619 M / 25.25 MiB | 6.619 M / 25.25 MiB |
| pitch 后级 | 10.283 M / 39.22 MiB | 0.743 M / 2.83 MiB |
| variance 后级 | 5.832 M / 22.25 MiB | 0.739 M / 2.82 MiB |
| variance 其余条件嵌入 | 0.004 M / 0.01 MiB | 0.004 M / 0.01 MiB |
| **variance 合计** | **22.737 M / 86.74 MiB** | **8.105 M / 30.92 MiB** |
| **分开训练两模型之和 / all-in-one 总量** | **82.639 M / 315.24 MiB** | **18.182 M / 69.36 MiB** |

M 表示百万参数；权重体积按 FP32 每参数 4 字节、1 MiB = 1,048,576 字节计算，不含 checkpoint 的优化器状态、元数据与文件格式开销。表格四舍五入；精确参数整数以 `tests/dslight_parameter_report.py` 输出为准。候选 variance 比声学小仅是参数规模事实，不代表质量已验证。

这组 all-in-one 实现含独立声学和 variance 子模型，因此同配置下参数量等于分别构造的两者之和，并非共享编码器。词表、启用项或维度改变会改变数值。

训练显存参考：107 条样本测试中，RTX 5060 Laptop、FP32 AdamW、batch=1、209–688 帧、无梯度累积，`torch.cuda.max_memory_allocated` 声学旧配置为 1,233,726,976 字节（约 1,176.6 MiB），加参 DSLight 声学为 249,120,768 字节（约 237.6 MiB）；不包括 CUDA reserved 和其他进程。仅按 FP32 参数、梯度及 AdamW 两份状态合计 16 字节/参数估算下限：旧声学/variance/all-in-one 分别约 914/347/1261 MiB，DSLight 候选分别约 154/124/277 MiB。该下限不含激活、临时张量、框架及显存碎片，不能作为实际训练峰值；variance 和 all-in-one 的真实峰值尚未测量，需按实际 batch、帧长、精度、优化器、梯度累积和设备实测。

### 50 MB 内 all-in-one 体积参考（实验配置）

`configs/dslight/all_in_one_50mb.yaml` 是可复制修改的配置参考，以 50,000,000 字节作为更严格的上限，**不涉及 SV 格式或权重兼容**。保持全部 duration、pitch 和四项 variance 开关，词表 65；主隐藏维度 192，声学后级 C=128/N=8，声学前级 C=128/N=4，pitch 和 variance 后级均为 C=96/N=6，duration 预测器 C=128/N=3。构造结果如下：

| 模块 | 参数量 | FP32 纯权重体积 |
| --- | ---: | ---: |
| 声学编码器 | 2.972 M | 11.34 MiB |
| 声学前级 | 0.407 M | 1.55 MiB |
| 声学后级 | 0.751 M | 2.87 MiB |
| **声学合计** | **4.131 M** | **15.76 MiB** |
| variance 编码器（含 duration） | 3.242 M | 12.37 MiB |
| pitch 后级 | 0.331 M | 1.26 MiB |
| variance 后级 | 0.328 M | 1.25 MiB |
| **variance 合计（含其他嵌入）** | **3.903 M** | **14.89 MiB** |
| **all-in-one 合计** | **8.033 M** | **30.65 MiB（32.13 MB）** |

`test_dslight_size_reference.py` 验证模型的声学和 variance 合成前反向、有限梯度以及实际保存的 `state_dict` 小于 50,000,000 字节。词表、启用项和附加文件会改变体积；训练 checkpoint 若含优化器状态也会更大。该配置尚未经过真实训练、导出或质量验收，不能保证音质、训练性能或完整发行包小于 50 MB。FP32 AdamW 的参数、梯度、两个优化器状态合计下限约 122.6 MiB，未计激活及其他开销，真实训练峰值尚未测量。

#### 接近 50 MB 的质量优先参考

`configs/dslight/all_in_one_quality_50mb.yaml` 基于上述参考档增加容量，主隐藏维度为 224，声学后级 C=160/N=10、前级 C=192/N=6，pitch 和 variance 后级 C=112/N=6，duration 预测器 C=128/N=3；所有预测开关保持启用，词表 65。

| 模块 | 参数量 | FP32 纯权重体积 |
| --- | ---: | ---: |
| 声学编码器 | 4.041 M | 15.41 MiB |
| 声学前级 | 1.113 M | 4.24 MiB |
| 声学后级 | 1.414 M | 5.39 MiB |
| **声学合计** | **6.567 M** | **25.05 MiB** |
| variance 编码器（含 duration） | 4.331 M | 16.52 MiB |
| pitch 后级 | 0.445 M | 1.70 MiB |
| variance 后级 | 0.441 M | 1.68 MiB |
| **variance 合计（含其他嵌入）** | **5.220 M** | **19.91 MiB** |
| **all-in-one 合计** | **11.788 M** | **44.97 MiB（47.15 MB）** |

FP32 纯参数为 47,150,980 字节；本次实际序列化 `state_dict` 为 47,312,569 字节，小于 50,000,000 字节。体积依词表、开关、序列化版本及额外附件变化，含优化器的训练 checkpoint 不受此限制。合成声学和 variance 前反向与有限梯度测试通过；此处“质量优先”仅指相对小档增加容量，并非质量达标结论。尚未以该档完成真实训练、导出或质量验收，前级 ≤5% 门槛仍未通过。FP32 AdamW 参数、梯度与状态占用下限约 179.9 MiB，不含激活、缓存和其他开销；真实峰值未测。

#### 50 MB 参考档训练显存：合成批次实测

在 RTX 5060 Laptop GPU、PyTorch 2.11、FP32、AdamW、每流 batch=1、512 帧 / 64 音素、无梯度累积下，运行两次合成前反向与优化器更新；all-in-one 同时计算声学和 variance 两流。下表分别是 PyTorch 峰值 allocated / reserved，单位 MiB：

| 参考档 | 单独声学 | 单独 variance | all-in-one 两流 |
| --- | ---: | ---: | ---: |
| 紧凑档 | 100.0 / 122.0 | 95.3 / 116.0 | 177.5 / 214.0 |
| 接近 50 MB 质量优先参考档 | 148.3 / 186.0 | 120.8 / 148.0 | 249.3 / 306.0 |

脚本 `tests/dslight_memory_probe.py` 支持调整帧数及音素数。此实验使用合成输入和简化损失，并非真实数据、完整 Lightning 训练或显存上限；音素 / 帧长、batch、精度、优化器、梯度累积、后端缓存及其他进程均会影响实际占用。可用作小批量起点，正式训练应测量实际峰值。

GPU 推理优化验证：RTX 5060、`diffs` 环境下，`optimize_model_for_inference` 成功 trace / freeze `velocity_fn`，16、32 帧 warmup 通过，24 帧输出与原模型在 `rtol=1e-4, atol=1e-5` 内一致；全模型 8 个 Linear 被选择性 FP16 化，后级时间投影不足默认阈值，卷积保持原精度。复测脚本为 `tests/dslight_gpu_trace.py`。

| 阶段 | 内容 | 完成条件（可验证） |
| --- | --- | --- |
| **P0 契约冻结（本文档）** | 张量契约、注册方式、配置键、验收门槛 | 本文档评审通过；明确"默认不变" |
| **P1 后级 `dslight`** | 实现 + 注册 + 严格参数校验 + 单测 | ①`[B,F,M,T]` 出入形状一致；②三档参数量与 §3.3 预算一致（0.76 / 2.39 / 5.18 M，实测偏差 ≤2%）；③`supports_valid_mask=False` 且 `run_backbone` 路径可用；④`torch.jit.trace` 通过；⑤未知参数报错而非静默丢弃 |
| **P2 前级 `dslight`** | 实现 + 注册 + 单测 | ①`AuxDecoderAdaptor` 装配后 `forward(condition, infer=True/False)` 形状正确；②三档参数量与 §4.2 预算一致（0.20 / 0.42 / 1.12 M，实测偏差 ≤2%）；③`num_feats>1` 分支不回归（Adaptor 内部逻辑不变） |
| **P3 训练 smoke** | 新配置跑短程训练 | ①同一 batch 前反向无 NaN；②`aux mel L1` 与主 loss 均有下降趋势；③`lam aux` 权重路径生效；④旧配置回归 smoke 通过（证明未被影响） |
| **P4 推理 / 导出 / 对比** | native 推理 + ONNX + 对照表 | ①ONNX opset 15 导出成功，ONNX Runtime 1.23 可加载并输出同形状；②同一设备 CPU 单线程 RTF 记录；③与 `lynxnet2+convnext` 的 mel L1 对比表；④主观抽听记录（人工结论，不得由指标替代） |

**门槛定义（不得事后放宽）**

- 参数量：与 §3.3 / §4.2 / §5.1 的档位预算一致（后级 0.76 / 2.39 / 5.18 M，前级 0.20 / 0.42 / 1.12 M，
  合计 0.96 / 2.80 / 6.30 M），实测与估算偏差 >2% 时先查实现，不得直接放宽预算。
- 功能：所有**现有**配置与 checkpoint 的加载、推理、导出行为不变（回归验证）。
- 速度：P4 记录 `dslight` 与基线在**同设备同线程数**下的 RTF；若不优于基线，需在文档中如实记录并说明原因，不得改写为"已优化"。
- 质量：前级 mel L1 允许劣于 `convnext` ≤ 5%；超过则视为未达标（可调档位而不是放宽门槛）。

---

## 9. 风险与回退

| 风险 | 影响 | 处理 |
| --- | --- | --- |
| 容量不足导致细节变差 | 前级粗谱偏差增大 | 前级提供三档；折中档不达标时升到“质量优先”档（C=192, L=6，≈1.12 M），仍比 `convnext` 小 92% |
| 感受野仍不够 | 长句尾音/连音处不稳 | 扩充膨胀循环（加 243 或再叠一轮 1/3/9/27/81） |
| 采样步数主导成本 | 后级变小但端到端提升有限 | 本计划**不**改采样器/步数；仅记录基线，少步化留给后续独立计划 |
| 导出算子不兼容 | ONNX 失败 | 只用 Conv1d / Sigmoid / Split / Mul / Add；P4 前先跑 trace 单测 |
| 旧行为被污染 | 最严重 | 只做键追加；P3/P4 均跑旧配置回归 smoke；任何共享代码修改需单独说明并复验 |
| 名称混淆 | 维护成本 | `dslight` 前级与后级类名区分：`DSLightBackbone` / `DSLightAuxDecoder` |

**回退方式**：配置改回 `backbone_type: lynxnet2` + `aux_decoder_arch: convnext` 即可，无需删代码；
若 `dslight` 未达门槛，保留注册项但标记为 experimental，不进入默认配置。

---

## 10. 硬约束：不与 Synthesizer V 兼容

本节为**验收否决项**，任何实现若触碰即视为不合格：

1. **不实现、不引入** `.nofs` / `.dnni` 的读写、解析或转换代码；不复制 `NOFS::*`、`DNNIHelper` 之类逻辑。
2. **不使用** SV 的键名/哈希键（`prim0..5`、`modl*`、`moda*`、`f0model-dds`、`_gnc1v0` 等）作为本项目任何格式契约；
   参数命名沿用 PyTorch 默认（`weight` / `bias`）。
3. **不支持**导入 SV 声库、不支持导出为 SV 声库、不做音色转换或"从 SV 音库提取音色"。
4. **不以**"对齐 SV 数值、对齐 SV 听感、复现 SV 参数规模"作为验收标准。
5. **不引入** SV 的单文件虚拟文件系统、条目压缩表、名字哈希化、私有算子图序列化。
6. 允许并且仅允许借鉴以下**设计原则**：
   - 膨胀卷积栈（1/3/9/27/81）+ 1×1 卷积 替代全局注意力；
   - 窄通道（64–512）、短核（7–15）、单张权重小；
   - 非自回归、单次前向、显式时长/音高分离（本计划只在"前级估计 + 扩散修正"这一既有结构内体现）；
   - fp32 推理、无框架依赖、单张权重缓存友好。

---

## 11. 附：参数量核算方式（可复核）

```python
# 后级 dslight（折中档）
C, N, FM, H, Te = 192, 12, 128, 256, 64
p = FM*C + C                      # in_proj
p += H*C + C                      # cond_proj
p += Te*C + C                     # time_mlp
p += C*7 + C + C*C + C            # stem (dw + pw)
p += N * (C*7 + C + C*2*C + 2*C + 2*C*C + C + C*C + C)   # blocks
p += C*FM + FM                    # head
print(p)                          # ≈ 2.39e6

# 前级 dslight
C, N, IN, OUT, k = 128, 4, 256, 128, 7
q = IN*C + C
q += N * (C*k + C + C*2*C + 2*C + 2*C*C + C)
q += C*OUT*k + OUT
print(q)                          # ≈ 0.42e6

# 现有前级 convnext（对照）
C, N, IN, OUT, k = 512, 6, 256, 128, 7
r = IN*C*k + C
r += N * (C*k + C + 2*C + C*4*C + 4*C + 4*C*C + C + C)
r += C*OUT*k + OUT
print(r)                          # ≈ 14.0e6
```

三档计算脚本（可直接运行复核）：

```python
FM, H, Te, IN, OUT = 128, 256, 64, 256, 128

def backbone(C, N, k):
    p = (FM*C + C) + (H*C + C) + (Te*C + C) + (C*k + C + C*C + C)
    p += N * (C*k + 5*C*C + 5*C)
    p += C*FM + FM
    return p

def aux(C, N, k):
    p = IN*C + C
    p += N * (C*k + 4*C*C + 4*C)
    p += C*OUT*k + OUT
    return p

for name, b, a in [
    ('速度优先', (128, 8, 7), (96, 3, 5)),
    ('折中',     (192, 12, 7), (128, 4, 7)),
    ('质量优先', (256, 15, 7), (192, 6, 7)),
]:
    print(name, backbone(*b), aux(*a))
```

输出：

| 档位 | 后级 | 前级 | 合计 |
| --- | --- | --- | --- |
| 速度优先 | 759,424 | 199,424 | **958,848** |
| 折中 | 2,389,376 | 415,488 | **2,804,864** |
| 质量优先 | 5,177,472 | 1,118,912 | **6,296,384** |
| （对照）现有 `convnext` 前级 | — | 14,008,960 | 14,008,960 |

> 上述为静态估算；P1/P2 必须以 `sum(p.numel() for p in model.parameters())` 实测值写入验收记录，
> 估算与实测不一致时以实测为准。
