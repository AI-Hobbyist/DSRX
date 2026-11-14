# 内置 Mini-DDSP 的独立预处理

声码器仍内置于 all-in-one 模型，网络、完整 checkpoint 和单模型导出结构不变。这里只拆开数据准备入口，不新增独立声码器训练器或模型导出。

## 使用

同一份配置可分别执行两条预处理命令，再训练完整模型：

```bash
python scripts/binarize.py --config data/all_in_one_mini_ddsp_cloud.yaml
python scripts/binarize_vocoder.py --config data/all_in_one_mini_ddsp_cloud.yaml
python scripts/train.py --config data/all_in_one_mini_ddsp_cloud.yaml --exp_name all_in_one_mini_ddsp_cloud
```

独立准备音频数据时，也可使用 configs/vocoder/mini_ddsp_opencpop.yaml；其公共设置集中在 configs/shared/opencpop_preprocessing.yaml，不继承 acoustic/variance 网络配置。读取 datasets 中的 raw_data_dir/wavs；也可在该数据集条目中显式设置 audio_dir。声码器不读取 CSV/DS 或字典。speaker 和相对音频名用于关联原样本，test_prefixes 决定留出来源。

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
