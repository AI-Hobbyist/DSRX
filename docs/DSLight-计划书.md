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
