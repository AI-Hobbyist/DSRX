# Variance 训练结合 Acoustic 验证合成计划

## 背景

当前项目把声学模型与 variance 模型分成两条训练/推理链路：

- `training/variance_task.py` 负责 dur / pitch / variance 预测训练，验证时已经会把 duration、pitch、各 variance 曲线写入 TensorBoard figure。
- `training/acoustic_task.py` 负责 mel 生成训练，验证时已经支持 `val_with_vocoder`，会把 mel figure 和 vocoder audio 写入 TensorBoard。
- `inference/ds_variance.py` 已经实现 `.ds` 输入到 `.ds` 输出的自动补全逻辑，内部顺序会根据模型开关和 `.ds` 字段缺失情况预测 dur、pitch、variance。
- `inference/ds_acoustic.py` 已经实现 `.ds` 输入到 mel / wav 的声学推理逻辑。

目标是在 variance 训练验证阶段增加一个可选的 `.ds` 合成验证：使用当前正在训练的 variance 模型补全测试 `.ds`，再使用一个已存在的 acoustic ckpt 合成 mel 和音频，并把结果写入 TensorBoard。整体流程为：

```text
dur -> pitch -> variance -> acoustic
```

其中任一部分未开启或输入已经具备对应字段时直接跳过。

## 配置设计

新增单个配置节点 `val_with_ds`。节点缺失、为空、没有 acoustic ckpt 目录或没有 `.ds` 文件时，整套功能直接关闭，不影响现有训练。

建议 schema：

```yaml
val_with_ds:
  acoustic_ckpt_dir: ckpt/acoustic_exp_name
  acoustic_ckpt_steps: null
  acoustic_vocoder: true
  release_acoustic_after_val: false
  release_variance_after_val: false
  overwrite_ds_dur: false
  overwrite_ds_pitch: false
  variance_ckpts:
    dur:
      ckpt_dir: null
      ckpt_steps: null
    pitch:
      ckpt_dir: null
      ckpt_steps: null
    variance:
      ckpt_dir: null
      ckpt_steps: null
  max_samples_per_val: null
  seed: -1
  spk:
    speaker_a:
      - samples/valid/a.ds
      - samples/valid/b.ds
    speaker_b:
      - samples/valid/c.ds
```

说明：

- `acoustic_ckpt_dir` 必填才启用 acoustic 合成；指向声学模型 ckpt 目录，目录内需要有 `config.yaml`、`model_ckpt_steps_*.ckpt`，多说话人时还需要 `spk_map.json`。
- `acoustic_ckpt_steps` 可选；为 `null` 时使用 acoustic ckpt 目录下最新 checkpoint。
- `acoustic_vocoder` 控制是否试听。为 `false` 时只记录 mel figure；为 `true` 时同时记录 audio。
- `release_acoustic_after_val` 控制每次 validation 结束后是否释放 acoustic 模型与 vocoder。默认 `false` 表示缓存复用；设为 `true` 可降低验证后的显存占用，但下次 validation 会重新加载 acoustic。
- `release_variance_after_val` 控制每次 validation 结束后是否释放 `variance_ckpts` 中加载的外部分阶段 variance 模型。默认 `false` 表示缓存复用；设为 `true` 可降低验证后的显存占用，但下次 validation 会重新加载这些外部 variance ckpt。当前正在训练的内存模型不会被释放。
- `overwrite_ds_dur` 控制 `.ds` 已经包含 `ph_dur` 时是否仍然重算 dur。默认 `false` 表示保留原 `.ds` 时长；如果缺少 `ph_dur`，仍会在可用时补全。
- `overwrite_ds_pitch` 控制 `.ds` 已经包含 `f0_seq` 时是否仍然重算 pitch。默认 `false` 表示保留原 `.ds` 音高；如果缺少 `f0_seq`，仍会在可用时补全。
- `variance_ckpts` 可选，用于 dur / pitch / variance 是三个独立模型的情况。某个阶段配置了 `ckpt_dir` 时，该阶段使用对应外部 variance ckpt；未配置时，若当前正在训练的模型支持该阶段，则使用当前内存模型；两者都没有时跳过该阶段。
- `max_samples_per_val` 可选，用于限制每次 validation 里实际合成的 `.ds` 数量，避免验证太慢。
- `seed` 可选，传给当前 variance 采样，默认 `-1` 表示不固定。
- `spk` 是 speaker 到 `.ds` 文件列表的映射，也是验证合成时唯一的 speaker 来源。即使 `.ds` 文件内部已经带有 `spk_mix` / `ph_spk_mix`，也必须被这里的 speaker 覆盖。这里的 speaker 名称必须与 acoustic 模型的 `spk_map.json` 中的 speaker 名称对应；如果当前 variance 模型也开启了 `use_spk_id`，同名 speaker 也需要存在于 variance 模型的 `spk_map.json`。

最小示例：

```yaml
val_with_ds:
  acoustic_ckpt_dir: ckpt/my_acoustic
  spk:
    alice:
      - samples/valid/alice_test.ds
```

## 实现方案

### 1. 配置读取与启用条件

在 `VarianceTask.__init__` 中读取：

```python
self.val_with_ds_cfg = hparams.get('val_with_ds') or {}
self.enable_val_with_ds = bool(
    self.val_with_ds_cfg.get('acoustic_ckpt_dir')
    and self.val_with_ds_cfg.get('spk')
)
```

节点缺失、`spk` 为空、`acoustic_ckpt_dir` 为空时直接关闭。这样不需要额外的 `enabled` 字段。

### 2. 新增验证 helper

建议新增 `training/variance_val_ds.py`，把功能从 `VarianceTask` 主体中拆出来，避免 `variance_task.py` 继续膨胀。

核心职责：

- 解析 `val_with_ds.spk`，读取每个 `.ds` 文件，统一转成 `List[OrderedDict]`。
- 对每个 speaker 给 param 强制覆盖 speaker 设置，忽略 `.ds` 文件内部已有 speaker：
  - `spk_mix = {spk: 1.0}`
  - `ph_spk_mix = {spk: 1.0}`
- 校验 `.ds` 路径存在、JSON 非空、`val_with_ds.spk` 中的 speaker 在 acoustic `spk_map.json` 中存在。
- 如果 variance 模型开启 speaker id，校验 speaker 在当前 variance `spk_map.json` 中也存在。
- 在 rank 0 执行 TensorBoard 写入；DDP 下避免每个 rank 都重复合成。

### 3. 当前 variance 模型补全 `.ds`

不要在训练中重新构造 `DiffSingerVarianceInfer`，因为当前训练中的模型权重在内存里，不一定已经保存为 checkpoint。

应复用 `VarianceTask.run_model(sample, infer=True)` 或抽一个轻量方法，把 `inference/ds_variance.py` 的 `.ds` preprocess / postprocess 逻辑改造成可注入模型的 helper：

- 复用 `DiffSingerVarianceInfer.preprocess_input()` 的 `.ds` 解析逻辑。
- 新增一个可传入 `variance_model`、`phoneme_dictionary`、`device` 的 runner，避免从 ckpt 重新加载 variance。
- 预测 flags 采用 `DiffSingerVarianceInfer.run_inference()` 的逻辑：
  - dur：`predict_dur` 开启且 `.ds` 缺少 `ph_dur`，或用户显式希望预测 dur。
  - pitch：`predict_pitch` 开启且 `.ds` 缺少 `f0_seq`，或后续 variance 需要 pitch。
  - variance：`predict_energy` / `predict_breathiness` / `predict_voicing` / `predict_tension` 中开启的字段，且 `.ds` 缺少对应字段。
- 每一步未开启或已有手工字段时跳过。

分阶段模型补充：

- 按固定顺序执行 `dur -> pitch -> variance`。
- 如果 `val_with_ds.variance_ckpts.<stage>.ckpt_dir` 非空，优先加载该阶段的外部 variance ckpt。
- 如果外部 ckpt 未配置，但当前训练中的模型支持该阶段，则复用当前内存模型。
- 如果两者都不可用，则跳过该阶段。

输出仍然是更新后的 param dict：

- dur 写 `ph_dur`
- pitch 写 `f0_seq`、`f0_timestep`
- variance 写 `{energy,breathiness,voicing,tension}` 与对应 `*_timestep`

### 4. 临时 acoustic hparams 上下文

`DiffSingerAcousticInfer` 和 acoustic model / vocoder 在构造与 forward 时都会读取全局 `utils.hparams.hparams`。训练中不能永久调用 `set_hparams()` 切换到 acoustic 配置，否则会污染当前 variance task。

需要实现一个上下文工具，例如：

```python
@contextmanager
def temporary_hparams(new_hparams):
    old_hparams = hparams.copy()
    hparams.clear()
    hparams.update(new_hparams)
    try:
        yield
    finally:
        hparams.clear()
        hparams.update(old_hparams)
```

加载 acoustic 配置时使用：

```python
acoustic_hp = set_hparams(
    config='',
    exp_name=acoustic_exp_name,
    print_hparams=False,
    global_hparams=False,
)
```

如果只提供目录 `ckpt/foo`，需要从目录名反推出 `exp_name=foo`，或补一个内部 loader 直接读取 `ckpt/foo/config.yaml` 并设置 `work_dir=ckpt/foo`。

acoustic 模型初始化、`.ds` preprocess、forward、vocoder 都必须包在 `temporary_hparams(acoustic_hp)` 内执行。

### 5. Acoustic 合成与 TensorBoard

复用 `DiffSingerAcousticInfer`：

- `preprocess_input(param)` 将补全后的 `.ds` 转为 acoustic batch。
- `forward_model(batch)` 生成 mel。
- 如果 `acoustic_vocoder: true`，调用 `run_vocoder(mel, f0=batch['f0'])` 生成 wav。

TensorBoard tag 建议：

```text
val_with_ds/{speaker}/{ds_stem}/mel
val_with_ds/{speaker}/{ds_stem}/audio
```

mel 图复用 `utils.plot.spec_to_figure`，参数来自 acoustic hparams：

- `mel_vmin`
- `mel_vmax`
- `audio_sample_rate`

audio 使用：

```python
self.logger.all_rank_experiment.add_audio(
    tag,
    wav,
    sample_rate=acoustic_hp['audio_sample_rate'],
    global_step=self.global_step,
)
```

### 6. Validation 调用点

在 `VarianceTask._on_validation_epoch_end()` 中触发合成最合适：

- 常规 batch validation 已经完成，loss / metric 已经聚合。
- 当前模型处于 eval / no_grad 场景，便于采样。
- 可以按 `val_check_interval` 控制频率，不额外引入新的训练循环。

建议结构：

```python
def _on_validation_epoch_end(self):
    if self.enable_val_with_ds:
        self.val_ds_runner.run(self)
```

runner 内部使用 `torch.no_grad()`，并保存/恢复 `self.model.training` 状态。

## 错误处理

启动时做快速校验，尽早失败：

- `val_with_ds` 不是 dict：报配置错误。
- `acoustic_ckpt_dir` 不存在：报清晰路径错误。
- acoustic `config.yaml` 不存在：报 acoustic ckpt 目录不完整。
- `.ds` 文件不存在或 JSON 为空：报对应文件名。
- speaker 不在 acoustic `spk_map.json`：报 speaker 与 acoustic 模型不匹配。
- variance `use_spk_id: true` 且 speaker 不在当前 variance `spk_map.json`：报 speaker 与 variance 模型不匹配。
- acoustic 开启 vocoder 但 `vocoder_ckpt` 不存在：报错或自动降级只记录 mel；建议默认报错，避免用户以为已经试听。

运行期单个 `.ds` 合成失败时，建议记录 warning 并继续其他样本；如果所有样本都失败，再抛出异常。

## 文件改动清单

- `configs/original/variance.yaml`
  - 增加注释形式的 `val_with_ds` 示例，默认关闭。
- `configs/templates/config_variance.yaml`
  - 增加同样的用户可编辑示例。
- `training/variance_task.py`
  - 初始化 `val_with_ds` runner。
  - 在 `_on_validation_epoch_end()` 调用 runner。
- `training/variance_val_ds.py`
  - 新增 `.ds` 验证合成 runner。
  - 包含配置校验、speaker 注入、variance 补全、acoustic 合成、TensorBoard 写入。
- 可选：`utils/hparams.py`
  - 如果需要直接从 `ckpt/foo/config.yaml` 加载 acoustic hparams，可补一个不污染全局 hparams 的 loader helper。

## 测试计划

1. 配置关闭回归：
   - 不写 `val_with_ds`。
   - 写 `val_with_ds: {}`。
   - 预期：训练与现状完全一致。

2. 配置校验：
   - acoustic ckpt 目录不存在。
   - `.ds` 文件不存在。
   - speaker 不在 acoustic `spk_map.json`。
   - 预期：报错信息指向具体字段和路径。

3. 单说话人 smoke test：
   - `val_with_ds.spk` 配一个 speaker 和一个短 `.ds`。
   - 使用很小 `num_valid_plots` / `max_samples_per_val: 1`。
   - 预期：TensorBoard 出现 mel figure；`acoustic_vocoder: true` 时出现 audio。

4. 阶段跳过测试：
   - `.ds` 已带 `ph_dur`，dur 跳过。
   - `.ds` 已带 `f0_seq`，pitch 跳过。
   - variance 某字段已存在，且没有显式要求重预测时跳过。
   - 预期：输出合成仍成功。

5. 多说话人测试：
   - `spk` 下配置两个 speaker。
   - 预期：TensorBoard tag 按 speaker 分组，且 speaker mix 注入正确。

6. DDP / 多卡 sanity：
   - 确认只有 rank 0 写 TensorBoard 或不同 rank 不会重复写同一个 tag。

## 风险与注意事项

- acoustic hparams 是最大风险点。必须保证 acoustic 模型和 vocoder 的构造、preprocess、forward、TensorBoard sample rate 都在临时 acoustic hparams 下完成，结束后恢复 variance hparams。
- acoustic 与 variance 的 speaker 名称要一致。验证合成时 speaker 只取自 `val_with_ds.spk`，并覆盖 `.ds` 文件内部已有 speaker。当前计划先强制同名校验，不做 speaker 映射；如果后续需要映射，可以把 `spk` 扩展为对象形式。
- 验证合成会明显增加 validation 时间，尤其是 vocoder 试听。建议提供 `max_samples_per_val`，并默认只在用户配置后启用。
- `.ds` 合成结果只写 TensorBoard，不默认落盘，避免训练目录产生大量临时文件。需要调试时可以再加 `save_intermediate_ds: true`。
