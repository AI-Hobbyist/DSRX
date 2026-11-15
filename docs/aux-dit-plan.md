# AUX DiT 模式实施计划

日期：2026-11-15。用户确认“前级”指 shallow diffusion 的 AUX mel 解码器。

## 范围与设计

- 新增 `shallow_diffusion_args.aux_decoder_arch: dit`，保留 ConvNeXt/DSLight。
- AUX 仍是 condition `[B,T,H]` 到归一化 mel `[B,T,M]` 的单次确定性预测，不增加噪声、采样步数或扩散损失。
- 复用现有 DiTBlock（RoPE、自注意力、adaLN、MLP）；有效帧条件的均值驱动 adaLN，替代扩散时间条件。保留 L1、反归一化、aux_decoder_grad、后级 shallow 起点和内置声码器联合监督路径。
- 显式传递 mel2ph 有效帧 mask，屏蔽 padding 的注意力和池化；旧 AUX 不改变调用语义或权重结构。
- 根据用户最终要求，两份新配置的 AUX、声学后级、pitch/variance backbone 均为 DiT。分别从 all_in_one_dit.yaml 和当前 data 云端训练配置复制，原文件不变；保留预处理参数和 binary_data_dir。
- 完整 all-in-one INT8 推理权重以约 40 MB 为目标：新配置采用 hidden_size=160、enc_layers=2、声学 DiT 5×256、AUX/pitch/variance DiT 3×192、原时长 ResNet 3×128。声码器结构及资源不变；标准 Linear 支持动态 INT8，其余保持浮点，实际序列化核验总大小。
- 支持现有 batch-1 动态帧 ONNX 部署协议，复用部署端 DiTBlockONNX。
- 新架构须新建实验或明确部分初始化；不能直接严格续训旧 AUX 权重。沿用原数据和预处理参数时无需因 AUX 网络更换重做二进制包。

## M1：计划与范围确认

验收：核对 AUX 工厂、训练/推理及部署调用路径，落盘本计划，检查 diff；提交并推送后进入 M2。

状态：完成；提交 3b2b364 已推送 ds_light_backend，远端 SHA 核对一致。用户明确授权后续实现提交推送。

## M2：实现与验证

1. 新增 AUX DiT、工厂和 L1 注册；按模型能力透传 mask。
2. 声学训练/推理接入；部署视图替换为兼容 opset 15 的注意力模块。
3. 提供可选配置、使用说明与回归测试。
4. 验收：先复现未注册模式失败；测试形状/反归一化、padding 隔离（非零 attention gates）、梯度/优化更新、checkpoint 重载、训练到 shallow 推理、真实 ONNX 导出和 ORT 动态帧数值一致；旧 AUX 回归通过。限定文件格式化，git diff --check。
5. CUDA 可用时执行有界 BF16 前后向和推理；不启动长期训练，不以随机权重结果宣称音质。听音/收敛记录 MANUAL/PENDING。
6. 更新计划、根目录 AGENT_HANDOFF.md，检查 git 状态/diff，提交并推送，核对远端 SHA。

状态：实现与自动验收完成。18 项最终回归 PASS；新增模式、padding mask、部署适配、两份独立配置均已落地，Black 限定修改范围格式化及 git diff --check PASS。等待本阶段提交推送。

实测：128 音素词表，显式开启内置 Mini-DDSP，Linear 动态 INT8、其余 FP32，推理 state_dict（不含 optimizer）模板 35,193,307 bytes、云端 39,783,705 bytes；严格重载后完整唱法→声学→声码器 CPU 推理通过。CUDA BF16 前后向、AUX 与完整 FS2+AUX opset15 导出/ORT 动态帧也通过。预处理指纹保持一致。上述使用随机权重，收敛与训练后音质 MANUAL/PENDING。

用户追加交付：本阶段提交推送后，fetch 比较分支提交关系；合并最新实现到 main 并推送，再从 main 创建并切换 dev。保留原有工作区改动。

用户追加 Netron 验证产物：已用正式 all-in-one exporter 导出完整云端尺寸随机权重三合一 ONNX，再对常量 MatMul/Gemm（含采样子图）做 per-channel 动态 INT8。44,040,864 bytes，133 个 MatMulInteger、250 个 INT8 initializer；onnx.checker 及 ORT 9/13 帧完整波形推理 PASS。ORT 量化器对子图 lexical capture 的父节点重排有缺陷，产物脚本仅按真实依赖恢复拓扑顺序，不删除节点或计算。文件位于 ignored artifacts/aux_dit/onnx/aux_dit_random.full.int8.onnx，不纳入 Git。
