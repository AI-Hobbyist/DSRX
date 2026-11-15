# 可选 AUX DiT

在启用 shallow diffusion 的配置中选择：

```yaml
shallow_diffusion_args:
  aux_decoder_arch: dit
  aux_decoder_args:
    num_channels: 192
    num_layers: 3
    num_heads: 3
    mlp_ratio: 4
    rope_base: 10000.0
    layer_norm_eps: 0.000001
    attention_dropout: 0.0
    mlp_dropout: 0.0
    use_gradient_checkpointing: true
```

`num_channels / num_heads` 必须为偶数整数。`aux_decoder_args` 与 `backbone_args` 一样，继承时整组替换，避免从 ConvNeXt/DSLight 混入 kernel_size 等参数。切回 `convnext` 或 `dslight` 时提供对应参数。

AUX 使用已有 DiTBlock、RoPE 和 adaLN，单次预测 mel；有效帧 condition 的均值驱动 adaLN。没有额外扩散采样。训练及 PyTorch 推理以 mel2ph>0 屏蔽 padding，保持原 L1、aux_decoder_grad、mel 归一化、shallow 起点和 Mini-DDSP 联合监督。ONNX 使用既有 batch-1、无 padding 的动态长度输入协议，并复用 DiTBlockONNX。

## 两份新配置

- `configs/templates/all_in_one_aux_dit.yaml`：复制现有 all_in_one_dit.yaml，提供轻量 AUX + 声学后级 + pitch/variance DiT；内置声码器开关沿原模板默认 false，需要时改为 true。
- `configs/experiments/all_in_one_mini_ddsp_cloud_aux_dit.yaml`：复制当前 data 云端配置，所有上述 backbone 设为 DiT，Mini-DDSP 已启用；保留原数据、验证来源、预处理、增强、Muon、学习率、联合训练和批量设置。

新配置为完整约 40 MB 量化权重目标缩小了网络尺寸：共享 encoder 2 层、hidden=160；声学 DiT 5 层、256 通道；AUX/pitch/variance DiT 各 3 层、192 通道；时长保持 ResNet，3 层、128 通道。原配置及已有模型均未改变。

旧预处理数据可直接使用：云端 `binary_data_dir` 沿用 `data/pre/binary/all_in_one_mini_ddsp_cloud`，本次配置的声学/唱法预处理指纹与原配置一致。不必为 AUX/backbone 更换重做预处理。需要修改词典或数据路径时，仍按原预处理规则处理。

用户指定的实机配置已从 `data/pre/configs/all_in_one_mini_ddsp_cloud.yaml` 原样复制为 `data/pre/configs/dit_all_in_one_mini_ddsp.yaml`，仅修改上述模型参数，保留 `200ep`、`1ep` 验证、原批量、增强和存档策略。此文件也已纳入版本控制，原文件不变。新架构与旧 checkpoint 的维度不同，须用新实验名：

```bash
python scripts/train.py --config data/pre/configs/dit_all_in_one_mini_ddsp.yaml --exp_name dit_all_in_one_mini_ddsp
```

第一次不需要 `--reset`；后续同命令续训这个新实验。不要复用旧实验名，否则保存的 config/checkpoint 会覆盖或不匹配新网络。云端批量设置沿原配置，完整 DiT 云端训练吞吐与长序列显存尚未实测；本地仅运行了有界的随机张量验证。

`--exp_name` 自动决定权重目录 `ckpt/dit_all_in_one_mini_ddsp/`，无需在配置中另设权重目录。

## INT8 大小与验证口径

DiT AUX 使用标准 nn.Linear，适合线性权重 INT8；LayerNorm、RoPE、注意力和合成等计算保留浮点。以 128 音素词表、内置 Mini-DDSP **开启**，实际 `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` 后保存 `{"state_dict": model.state_dict()}`：

| 配置 | 总参数量 | 实际推理权重大小 |
| --- | --- | --- |
| 通用模板 | 15,842,753 | 35,193,307 bytes（35.19 MB / 33.56 MiB） |
| 云端副本 | 17,179,841 | 39,783,705 bytes（39.78 MB / 37.94 MiB） |

总大小包含声学 encoder、DiT AUX/后级、完整唱法模型、时长、内置声码器及 noise_basis 等资源；仅 Linear 动态 INT8，卷积、embedding、非线性参数和资源仍为 FP32。词表/配置变化会改变大小。训练仍按原混合精度训练，含 optimizer 的训练 checkpoint 不能用这个推理权重预算衡量。

这是量化兼容性及尺寸验证，没有新增 INT8 推理 CLI 或量化模型发布格式。现有 CLI 仍读取正常训练 checkpoint；量化后的 state_dict 需要以同配置构造并量化模型后严格加载。测试确实执行量化权重保存/重载和唱法→声学→声码器 CPU 推理，不以文件大小代替可运行性。PyTorch 当前对 torch.ao API 有弃用提示，本次使用已安装版本实际可用实现，没有安装额外依赖。

`tests/test_aux_dit.py` 覆盖 CPU、CUDA BF16、padding 隔离（非零 attention gates）、梯度/重计算一致、严格 checkpoint 重载、DDPM/Reflow 训练/shallow 推理、真实 AUX 和 FS2+AUX ONNX/ORT 动态帧以及 INT8。随机权重只验证工程行为；训练收敛和训练后音质为 MANUAL/PENDING。

## 本地 Netron 测试产物

`artifacts/aux_dit/onnx/aux_dit_random.full.int8.onnx` 是完整声学（含 DiT AUX）+ 唱法 + Mini-DDSP 的单文件测试图，使用云端网络尺寸、128 音素词表、随机权重；文件 44,040,864 bytes（44.04 MB / 42.00 MiB）。它与上表 PyTorch packed state_dict 格式不同，包含 ONNX 图与量化元数据，不混用大小。

正式 all-in-one exporter 导出 FP32 图后，使用 ORT `quantize_dynamic` 对常量 MatMul/Gemm 及采样 If/Loop 子图进行 per-channel INT8，其他算子和必要资源仍为浮点。已检查 133 个 MatMulInteger、250 个 INT8 initializer；onnx.checker PASS，ORT CPU 9/13 帧完整预测唱法→声学→波形输出 PASS（测试手动给定时长，音高/唱法 retake 全开、声学 depth=0.6、采样各 2 步）。

产物脚本 `artifacts/aux_dit/export_netron.py`、`quantize_netron.py` 和 `onnx/int8-report.json` 保留在本地。ORT 量化器对包含 lexical capture 的子图父节点排序有问题，脚本保留全部节点及计算并按真实依赖重新排序后通过检查和运行。所有大文件和临时实验均在 ignored artifacts，不推送。此图供 Netron/工程验证，不能作为训练完成的声库。
