# AUX DiT 模式实施计划

日期：2026-11-15。用户确认“前级”指 shallow diffusion 的 AUX mel 解码器。

## 范围与设计

- 新增 `shallow_diffusion_args.aux_decoder_arch: dit`，保留 ConvNeXt/DSLight。
- AUX 仍是 condition `[B,T,H]` 到归一化 mel `[B,T,M]` 的单次确定性预测，不增加噪声、采样步数或扩散损失。
- 复用现有 DiTBlock（RoPE、自注意力、adaLN、MLP）；有效帧条件的均值驱动 adaLN，替代扩散时间条件。保留 L1、反归一化、aux_decoder_grad、后级 shallow 起点和内置声码器联合监督路径。
- 显式传递 mel2ph 有效帧 mask，屏蔽 padding 的注意力和池化；旧 AUX 不改变调用语义或权重结构。
- 新增独立可选的 AUX DiT + DSLight 后级 all-in-one 模板；不修改用户 data 配置，不改唱法预测器、时长或声码器。
- 支持现有 batch-1 动态帧 ONNX 部署协议，复用部署端 DiTBlockONNX。
- 新架构须新建实验或明确部分初始化；不能直接严格续训旧 AUX 权重。沿用原数据和预处理参数时无需因 AUX 网络更换重做二进制包。

## M1：计划与范围确认

验收：核对 AUX 工厂、训练/推理及部署调用路径，落盘本计划，检查 diff；提交并推送后进入 M2。

状态：源码核对和计划完成，待提交推送。

## M2：实现与验证

1. 新增 AUX DiT、工厂和 L1 注册；按模型能力透传 mask。
2. 声学训练/推理接入；部署视图替换为兼容 opset 15 的注意力模块。
3. 提供可选配置、使用说明与回归测试。
4. 验收：先复现未注册模式失败；测试形状/反归一化、padding 隔离（非零 attention gates）、梯度/优化更新、checkpoint 重载、训练到 shallow 推理、真实 ONNX 导出和 ORT 动态帧数值一致；旧 AUX 回归通过。限定文件格式化，git diff --check。
5. CUDA 可用时执行有界 BF16 前后向和推理；不启动长期训练，不以随机权重结果宣称音质。听音/收敛记录 MANUAL/PENDING。
6. 更新计划、根目录 AGENT_HANDOFF.md，检查 git 状态/diff，提交并推送，核对远端 SHA。

状态：待实施。无需 Qt 或 WSL；所有测试前台 PowerShell Tee build.log，并立即检查退出码。
