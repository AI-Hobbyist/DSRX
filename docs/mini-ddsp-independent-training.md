# Mini-DDSP 独立训练与 AIO 合并

独立任务只训练现有 MiniDDSP，不构造声学/唱法模型，不要求词典、音素或 acoustic/variance 数据包。复用现有真实 mel、F0、波形、谐波幅度标签和 MiniDDSPLoss，保留随机裁片、音量增强；独立训练仅含声码器频谱/幅度监督，不执行 AIO 联合损失。

## 预处理与训练

已有 `binary_data_dir/vocoder/{train,valid}.data/.meta` 及 `audio_contract.json` 时可复用，无需重做。默认路径仍为 `data/pre/binary/all_in_one_mini_ddsp_cloud/vocoder`。缺少包时：

```bash
python scripts/binarize_vocoder.py --config configs/vocoder/mini_ddsp_train.yaml --resume
```

独立训练使用原入口：

```bash
python scripts/train.py --config configs/vocoder/mini_ddsp_train.yaml --exp_name mini_ddsp_pretrain
```

同一份配置集中管理数据集路径、预处理、音频、声码器尺寸、增强和训练设置，声码器尺寸与 AIO 一致。`all_in_one.mini_ddsp` 是已有模型设置命名空间，`task_cls` 决定这里只实例化声码器。修改数据/音频设置后须按现有预处理规则核对数据包；训练时会校验音频契约。验证使用中心裁片、不做音量增强，并记录参考/预测试听。正常训练 checkpoint 仍由原 Lightning 管线保存，分类为 `vocoder`，嵌入声码器结构和音频参数供合并检查。

重复同一训练命令是正式训练续跑用法；本次不会为验证而反复启停、加载或重建模型。

## 从零训练 AIO 时用预训练声码器试听

在 AIO 训练配置中设置以下字段。将 `all_in_one.mini_ddsp.enabled` 改为
`false`，并保留同级的其他 AIO 设置。

```yaml
finetune_enabled: false
val_with_vocoder: true
vocoder: MiniDDSPVocoder
vocoder_ckpt: ckpt/mini_ddsp_pretrain/model_ckpt_steps_20000.ckpt
num_valid_plots: 5
all_in_one:
  enabled: true
  mini_ddsp:
    enabled: false
```

将路径换成实际独立训练权重，用新的 `--exp_name` 启动训练。验证通过原来的
试听入口合成音频，声码器不参与优化，也不会写入 AIO checkpoint。
独立声码器与 AIO 的音频参数必须一致。

## 合并为一个推理实验

选择匹配的独立声码器 checkpoint，以及一个已经包含内置 Mini-DDSP 的完整 AIO checkpoint。音频参数和声码器尺寸必须一致，声学网络可以是 17.2M 或 40M DiT：

```bash
python scripts/merge_vocoder.py \
  --aio ckpt/dit_all_in_one_mini_ddsp/model_ckpt_steps_18000.ckpt \
  --vocoder ckpt/mini_ddsp_pretrain/model_ckpt_steps_20000.ckpt \
  --out ckpt/dit_all_in_one_mini_ddsp_merged
```

步数仅为示例，换成实际文件。默认读取 `--aio` 同目录的完整 config.yaml；必要时 `--config` 指定与该 AIO 权重对应的配置。输出目录必须尚不存在，不覆盖原实验。工具会：

- 验证类别、音频/网络契约、完整声码器 state keys、shape、dtype。
- 仅替换 `model.vocoder.*`，包括合成资源；其余 checkpoint 权重保持原值。
- 输出 `model_ckpt_steps_0.ckpt`、完整配置及原 AIO 的词典/语言/说话人映射。
- 不携带旧优化器、调度器或 Trainer 恢复状态，记录来源与原训练步数。

随后仍用同一 AIO 推理入口，例如：

```bash
python scripts/benchmark_infer.py "samples/00_我多想说再见啊.ds" --exp dit_all_in_one_mini_ddsp_merged --device cuda
```

## 后续联合微调

合并权重是推理/初始化产物，不是完整训练恢复 checkpoint。声码器在真实 mel 上预训练，而 AIO 推理使用预测 mel，合并后可进行联合微调以适配输入分布。

复制原 AIO 训练配置，保留对应网络和数据设置，并填写：

```yaml
finetune_enabled: true
finetune_ckpt_path: ckpt/dit_all_in_one_mini_ddsp_merged/model_ckpt_steps_0.ckpt
finetune_ignored_params: []
```

使用另一个全新实验名（例如 `dit_all_in_one_mini_ddsp_joint`）启动。不要在合并目录原地执行训练续跑。需要多少微调和最终音质以实际验证/听音决定；合并本身不改变训练算法或自动保证音质。

## 本轮验证（2026-11-19）

- 使用已有真实 vocoder 数据包，经 `scripts/train.py` 完成 CPU FP32 的 1 次优化器更新、5 条验证样本及 checkpoint 保存。32 个参数张量的 Adam 状态均为 step 1，梯度均值有限且存在非零更新；TensorBoard 含参考/预测音频。
- 用该独立 checkpoint 与本地完整 AIO 权重执行实际合并：37 个声码器张量与独立权重逐项一致，453 个其余张量与原 AIO 逐项一致。产物不含旧优化器状态。
- `tests/test_vocoder_merge.py` 的 3 项测试通过，覆盖替换范围、音频契约/keys/shape/dtype 不兼容，以及配置/词典/映射打包和拒绝覆盖已有输出。
- 首次训练发现缺少永久 checkpoint 参数，已在新训练配置补齐，修正后训练通过。证据为 `artifacts/independent_vocoder/result.json`、验证日志及 `ckpt/independent_vocoder_functional`；这些路径是忽略的本地验证产物，1 step 权重不供正式音质使用。

生命周期/反复加载卸载/重建/恢复测试 SKIPPED（用户禁止）。CUDA/BF16 独立训练、长时间云端训练、合并后完整歌曲推理和最终音质 MANUAL/PENDING；本轮验证未宣称这些项目通过。
