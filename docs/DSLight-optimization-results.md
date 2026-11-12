# DSLight 优化实现与实测

日期：2026-11-12。模型为用户提供的 100000 步 all-in-one checkpoint；声学骨干160通道、10层、224维条件。原权重保持不变，所有生成包、特征和音频位于被忽略的 artifacts/。

## 已实现的路径

- 原生 eager Reflow/DDPM：每次采样显式准备各层条件，不在模型上保存跨乐句缓存。训练、普通 forward 与原导出入口保留。
- FP32 时间频率非持久化缓存，设备转换时重建，精度往返不污染频率；低精度沿原 arange/exp 路径。eval 调制用 addcmul，训练表达不变。
- 固定层数、核宽和膨胀周期的通道扫描，生成独立重训配置；不将旧权重不严格装入较窄网络。
- 真实留出录音的辅助输出/浅起点/实际 NFE/重建误差评估，不要求重标注或全量二值化。
- 仅 DSLight blocks 的 expand/project/condition 使用逐行 INT8 权重存储。深度卷积、时间与输出路径保留浮点；显式包格式和严格回载，禁止训练该格式。

## 已观察的收益与取舍

同一 CPU、单线程、同一真实留出特征、20步 Euler、3次测量中位数：

| 路径 | 秒 | 说明 |
| --- | ---: | --- |
| FP32 普通条件路径 | 0.45989 | 每步重复条件投影 |
| FP32 准备条件路径 | 0.39990 | 约少13.0%，数值一致性通过 |
| INT8 weight-only | 0.42928 | 比缓存FP32慢约7.3%，不是整数卷积核 |

时间有运行波动，不将一次样本结果推广成全系统提速承诺。此处包含声学前端与辅助解码器，不含声码器和特征提取。TorchScript 未保留准备接口时仍走普通路径；当前不声称 ONNX 循环也获得缓存提速。

相同声学 state_dict 包：FP32 **26,335,459字节**，INT8 weight-only **22,535,435字节**，约减少14.4%。不能与含 variance/优化器的105MB原训练checkpoint直接比较。实际运行会临时反量化，不能把磁盘收益等同为同等峰值内存收益。

同一CPU真实样本：FP32目标mel MAE **0.58856**，量化为 **0.59342**；量化相对FP32输出MAE **0.05199**。这不是零误差，不将量化设为默认。

宽度扫描：128通道922240参数，160通道1413888参数，192通道2009984参数。固定256帧、20次单线程随机权重微基准，中位12.56/16.29/19.92ms。128相对160少34.8%参数，耗时少22.9%；较窄权重仍需独立训练，尚未证明等音质。因此现有160权重继续使用。

## 少步评估

配套配置只指定一个留出样本2033001224，本次严格使用该样本；参考时长、音高和 variance 控制来自原特征链。这是声学重建评估，不是从乐谱预测全部控制的完整端到端验收。

CUDA上起点0.4的4/8/20步，实际NFE为4/8/20，mel MAE为0.52248/0.54938/0.57293；辅助解码器MAE0.45543。起点0.7的4/8/20步为0.52634/0.54338/0.55432。辅助路径NFE0。该样本没有显示更多步数降低重建误差，不据此认定辅助输出听感更好或更改默认20步。

CPU与CUDA的随机噪声序列不同，因此CPU量化报告与CUDA采样报告不能直接交叉比较误差。比较均在各自报告的同一设备、同一固定种子下进行。

## 复跑

在仓库根目录使用环境 Python，前台 PowerShell 每条命令接 `2>&1 | Tee-Object -FilePath build.log`，立即记录并检查 `$LASTEXITCODE`，失败读取日志。

```text
python scripts/profile_dslight_width.py --config ckpt/dslight/config.yaml --output artifacts/dslight_width
python scripts/evaluate_dslight_sampling.py --config ckpt/dslight/config.yaml --checkpoint ckpt/dslight/model_ckpt_steps_100000.ckpt --output artifacts/dslight_sampling
python scripts/evaluate_dslight_quantization.py --config ckpt/dslight/config.yaml --checkpoint ckpt/dslight/model_ckpt_steps_100000.ckpt --features artifacts/dslight_sampling --output artifacts/dslight_quantization --render-audio
```

INT8包使用 `modules.backbones.dslight_quantization.load_weight_only_package`，输入是已按配套配置构造并 eval 的浮点模型；不要把该包交给普通训练 checkpoint 加载器。原 checkpoint、架构和普通 ONNX 路径继续支持。

## 验证证据及待验证项

- 七项骨干测试通过，包括参数预算、trace、准备条件等价、乐句隔离、实际采样准备次数、精度往返、训练/eval表达对照。
- 两项量化测试通过：零行、训练拒绝、受保护层、包回载、动态长度、准备条件和trace对照。
- 原有opset15动态帧ONNX测试通过；这验证原浮点导出，不宣称INT8包已支持现有ONNX导出器。
- 真实权重严格加载、实际NFE断言、有限输出、量化包磁盘回载逐值一致，配套声码器五份波形均为有限值。
- 报告：artifacts/dslight_width/report.json、artifacts/dslight_sampling/sampling.json、artifacts/dslight_quantization/quantization.json。
- 音频：artifacts/dslight_quantization/sample_0_{fp32,fp32_4steps,int8_weight_only,auxiliary,reference_mel}.wav。

MANUAL/PENDING：人工听音。未完成：较窄网络的真实训练与质量等价验证、广泛录音覆盖、旧PyTorch1.13环境验证。当前不以随机网络速度代替真实质量，也不把本次实验称为完成了一个等音质128通道替代模型。
