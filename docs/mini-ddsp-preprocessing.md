# 内置 Mini-DDSP 的独立预处理

声码器仍内置于 all-in-one 模型，网络、完整 checkpoint 和单模型导出结构不变。声码器数据处理作为独立第三阶段，由原 all-in-one 入口自动调度；不新增独立声码器训练器或模型导出。

## 使用

开启 all_in_one.mini_ddsp.enabled 后，一条原预处理命令依次执行声学 → 唱法 → 声码器，再训练完整模型：

```bash
python scripts/binarize.py --config data/all_in_one_mini_ddsp_cloud.yaml --resume
python scripts/train.py --config data/all_in_one_mini_ddsp_cloud.yaml --exp_name all_in_one_mini_ddsp_cloud
```

只需重做声码器数据时，可运行 scripts/binarize_vocoder.py --config <配置> --resume。独立准备音频数据时，也可使用 configs/vocoder/mini_ddsp_opencpop.yaml；其公共设置集中在 configs/shared/opencpop_preprocessing.yaml，不继承 acoustic/variance 网络配置。读取 datasets 中的 raw_data_dir/wavs；也可在该数据集条目中显式设置 audio_dir。声码器不读取 CSV/DS 或字典。speaker 和相对音频名用于关联原样本，test_prefixes 决定留出来源。

统一使用 binary_data_dir。声学/唱法分别写 acoustic/variance，声码器写 vocoder 下的 train.data/train.meta、valid.data/valid.meta，使用 DiffSinger 同款 HDF5 indexed 格式。波形、mel、F0、UV、谐波幅度集中存入包内；谐波中间缓存只存在于本次命令的临时目录，结束后清理。没有 vocoder_binary_data_dir 设置。

## 配置与训练语义

采样率、mel、hop/FFT/window、F0 的 pe/pe_ckpt/f0_min/f0_max、谐波分离的 hnsep/hnsep_ckpt/hnsep_sample_rate 直接复用当前配置。独立流程不硬编码 VR。模板和实际云端配置显式列出这些设置；网络、Muon、BS64/50000 与原增强强度保持原状。data/all_in_one_mini_ddsp_cloud.yaml 原来的 mini_ddsp.enabled:false 保留；启用内置声码器训练时设为 true。本次测试仅在覆盖配置中启用。

声学恢复原来的 mel/曲线增强，不再做真实音频变调/变速、增强后重提 F0/唱法或波形配对缓存。声码器仅预处理原音频，内部进行 hnsep、F0、mel 和幅度标签提取。

训练仍是原来的 acoustic/variance 两路，声码器监督在 acoustic batch 内关联原录音。独立声码器包必须覆盖相应声学数据的原录音；未新增第三训练流。声学增强样本只对应原音频的声码器自身监督，不把变调/变速后的声学预测与原波形计算联合损失；原始样本继续执行声学主后级/AUX 到内置声码器的联合波形监督。

独立提取不要求 F0 逐位相等；实际对照 mel 差为 0，F0 最大差 0.000244140625 Hz。训练检查共享音频契约、源 ID、原始样本帧长，声码器自身训练消费包内 mel/F0，声学联合分支消费声学自己的 F0。

复用 ref_only/Mini-DDSP/data_loaders.py 的训练时随机裁片和自然对数音量增强；原参考没有变调/变速增强。设置位于 all_in_one.mini_ddsp：

```yaml
segment_frames: 128
training_augmentation:
  random_crop: true
  volume_enabled: true
  log_volume_min: -1.0
  log_volume_max: 1.0
```

随机 log_gain 的上限同时受原波形峰值约束；波形和幅度标签乘 exp(log_gain)，自然对数 mel 加 log_gain，F0/UV 不变。验证不做音量增强。波形联合分支保留原目标，不混用声码器自身的音量增强目标。幅度同步缩放是为了使本项目幅度监督与波形一致；原参考 loader 未同步放大幅度标签，不照搬其不一致行为。

## 验证与性能证据（2026-11-14）

固定同一音频、变调 +3 半音/速度 0.8 的本机 CUDA 剖析：拆分前配对增强预热后约 0.163～0.167 s，首次 1.252 s；拆分后增强中位数 enabled=false 0.05248 s、enabled=true 0.05615 s。旧路径额外执行两次 phase-vocoder time_stretch（其中一次来自 pitch_shift）并重提 F0/唱法，且增强在主进程串行执行。恢复的是该额外开销，不把本机单条增强速度冒充云端整段吞吐；4090 的 5～10 it/s 恢复情况 MANUAL/PENDING。

独立参考的谐波分离提前准备、提取器复用可以借鉴；其逐帧 CZT/complex128 没有本地加速证据，本次未更换幅度分析算法。声码器 worker 内复用 F0 提取器，原音频只处理一次，不离线扩增 4～5 倍波形。

40 项自动回归 PASS。以用户 data/all_in_one_mini_ddsp_cloud.yaml 为基准，仅覆盖本机路径、2 条训练+5 条 val、预处理 worker2、DataLoader0、冒烟步数：声学/唱法和独立声码器两进程出包通过；train acoustic8/variance2/vocoder2，valid 各5，ID关联及train/valid隔离、有限值、波形长度、源 WAV/CSV SHA256保持通过；完整 all-in-one CUDA bf16/Muon 两步训练、两路5条val、图/试听和step2 checkpoint通过。另用仅测试覆盖 warmup/ramp=0 的一步训练验证主后级/AUX 联合损失非零有限（6.23292/6.0928）。没有修改正式网络、优化器、批大小或正式 warmup/ramp。

用户实际文件中的字典 data/pre/dictionaries/zh.txt 和 samples/00_我多想说再见啊.ds 本机缺失；测试使用已有 opencpop-dsrx_cfg 字典与 samples/02_不老梦_variance.ds，保留实际文件引用不动。原 DS 的不匹配词段仍按已有逻辑警告跳过；没有把被跳过的段记为通过。

日志和报告：artifacts/preprocessing_performance/，包括 before.log、profile-summary.log、after.json、final-tests.log、reference-acoustic.log、reference-vocoder.log、reference-training.log、joint-training.log、verification.json。完整测试 checkpoint 在 ckpt/independent_vocoder_smoke/model_ckpt_steps_2.ckpt；它包含 acoustic/variance/vocoder，仅用于跑通，不是音质权重。长期收敛/音质训练 SKIPPED（用户决定），听音 MANUAL/PENDING。

最新版 CUDA Torch/setuptools 升级已按用户要求暂停；本次没有安装新 Torch，不把未验证的最新版升级混入本提交。

## 断点继续（2026-11-14 修正）

此前将独立阶段误实现为必须手动运行第二条命令，导致开启 Mini-DDSP 后原入口仍在唱法结束时退出。现由 AllInOneBinarizer 按顺序调用三阶段；独立只是处理职责和包的分离，不是拆模型或要求换训练入口。阶段日志明确打印 acoustic、variance、vocoder，关闭开关时明确打印跳过原因。

原命令加 --resume 即可。也可设置 binarization_args.resume:true；checkpoint_interval 默认16个源样本，设1可逐源保存。新运行无论是否启用复用都会写进度，之后可以 --resume 接续。源样本的全部增强作为一个提交组；中断后最多重做上一个检查点之后的源组，已提交记录不重提取、不重复追加。

进度使用同一二进制包的 .progress.pkl，完成后只保留 .complete.json。HDF5先flush再原子保存游标/元数据，恢复时回滚未提交尾部、恢复增强抽样随机状态；输入元数据、文件大小/修改时间及预处理参数的指纹变化会拒绝混用。恢复控制和worker数量不进入指纹，可以调整进程数继续。

此前版本没有指纹的完整 .data/.meta 可在 --resume 下做结构、数量和已有源ID检查后复用，并明确打印 legacy package。它无法追溯旧参数，使用此模式表示按相同数据/参数复用旧包；若旧参数或数据实际已变，应关闭resume重做。旧版本未完成且没有进度文件的部分无法凭空恢复。新检查点支持正常异常/Ctrl+C退出后的续跑，不承诺修复强杀/断电导致的HDF5文件损坏。

实际验证：已有声学/唱法包、缺声码器包时，一次原入口 --resume 自动处理声码器，旧 .data/.meta SHA256不变；再次运行全部复用。真实CUDA声码器在提交第一条后注入KeyboardInterrupt，再从1/2恢复，7条真实音频各提取一次、第一条记录逐值不变、声学/唱法哈希不变。回归同时覆盖增强参数稳定、未提交尾部回滚、旧完整包复用与参数变化拒绝。验证来源匹配DiffSinger的精确名称优先规则，避免 wav1 的留出规则误包含 wav10。

最终入口/续跑验收：45项回归PASS；原入口一次命令从零两进程完成全部三包，第二次全部复用，新包实际CUDA训练读取通过。日志 artifacts/preprocessing_resume/final-tests.log、fresh-all-stages.log、fresh-all-reused.log、fresh-training.log；真实中断报告 real-interruption-report.json，旧包哈希校验见 resume-missing-vocoder.log。

## 联合 loss 的监控口径

`training/ddsp_joint_loss` 和 `training/ddsp_aux_joint_loss` 与其他训练指标按同一 `log_interval` 刷新，记录该日志区间内未加权联合误差按有效样本数加权的均值。每个 batch 都参与收集，不再要求有效样本恰好出现在日志步上。`training/ddsp_joint_sample_count` 记录该区间实际参与联合项的样本总数，`training/ddsp_joint_factor` 仍记录当前实际联合权重。若整个区间都处于 warmup 或没有原始配对样本，则样本数为0，联合曲线缺测，不补零、不写NaN、不沿用旧值。反向传播仍使用原来的加权损失，不将这些监控指标额外加入优化目标。区间统计跨epoch收集，续训从本次启动后重新收集。

`validation/ddsp_joint_loss` 和 `validation/ddsp_aux_joint_loss` 使用固定噪声下的完整声学推理 mel，经当前内置声码器合成音频，计算固定中心片段的多尺度频谱误差。不使用真实 mel 作为浅扩散起点，不乘 warmup/ramp 或损失权重，从第 0 步即可比较。声码器从整句开始合成以保持相位与试听一致；只在中心片段计分以维持固定评估范围。验证额外执行完整声学推理，会增加验证耗时。固定噪声按验证 batch 序号设置，相同验证分批和采样设置下可比较，RNG 隔离不影响后续训练。

联合推理指标不加入 `validation/total_loss`；总量只汇总声学、唱法和真实 mel 的声码器监督项，避免动态权重改变验证总量。此指标仍使用数据集真实 F0/唱法条件，用于声学到声码器的质量评估，并非预测唱法的整链路指标。

标签沿用原名但计算口径已变，按用户选择重新生成 TensorBoard 日志，避免拼接旧曲线；无需删除 checkpoint 或重新预处理。长期收敛和音质由实际训练试听验收。

## 其他后级的 all-in-one 模板

以下五份模板均补齐共享音频、当前F0/hnsep设置、内置声码器选择、裁片/音量增强、checkpoint_interval/resume，以及一条原入口执行三阶段的命令：

- configs/templates/all_in_one_wavenet_adamw.yaml
- configs/templates/all_in_one_lynxnet2_muon.yaml
- configs/templates/all_in_one_dit.yaml
- configs/templates/all_in_one_dslight_50mb.yaml
- configs/templates/all_in_one_dslight_quality_50mb.yaml

保留 mini_ddsp.enabled:false 的原默认开关，用户启用后联合训练内置声码器。显式列出的 parselmouth/world 和音频值与原继承值相同，没有擅自替换提取算法；用户可在模板中改成实际使用的pe/hnsep。没有单独输出路径，没有更改独立声学/唱法模板。

五份配置级联加载、无重复YAML键、启用后内置声码器/损失模块构造通过；移除本次新配置项后，解析出的原配置逐项完全相等，包含原网络、优化器、批量和默认开关。报告 artifacts/all_in_one_template_update/report.json。未分别实际训练这五种后级，长期训练/音质验收仍按用户要求 SKIPPED；实际CUDA训练验证使用用户指定云端配置。
