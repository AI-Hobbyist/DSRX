# All-in-One 训练与预处理模式计划书

## 目标

在现有 DiffSinger 训练体系中实现一个可选的 all-in-one 训练和预处理模式，使唱法 variance 模块与声学 acoustic 模块可以训练、验证、保存到同一个权重文件中，并提供一份 all-in-one 配置文件示例。

目标流程保持为：

```text
dur -> pitch -> variance -> acoustic + vocoder
```

其中 dur、pitch、variance 参数预测都属于原有 `DiffSingerVariance` 唱法/方差模型内部的分支；acoustic 是另一个顶层声学模块，使用同一个内存模型中的 acoustic 子模块生成 mel，vocoder 负责试听音频。all-in-one 模式必须是可选功能，不破坏现有独立 variance 训练、独立 acoustic 训练、`.ds` 推理和已有 checkpoint 兼容性。

## 当前项目流程梳理

### 训练入口

- `scripts/train.py` 读取 hparams 后根据 `task_cls` 动态导入训练任务。
- `configs/original/variance.yaml` 当前使用 `training.variance_task.VarianceTask`。
- `configs/original/acoustic.yaml` 当前使用 `training.acoustic_task.AcousticTask`。
- `basics/base_task.py` 负责通用训练循环、validation 生命周期、loss 聚合、checkpoint 与 Lightning Trainer 构建。

### 预处理入口

- `scripts/binarize.py` 根据 `binarizer_cls` 动态导入 binarizer 并执行 `process()`。
- `preprocessing/variance_binarizer.py` 负责 variance 训练需要的 duration、pitch、energy、breathiness、voicing、tension 等特征。
- `preprocessing/acoustic_binarizer.py` 负责 acoustic 训练需要的 mel、f0、speaker/language 等声学样本特征。

### 模型与验证

- `training/variance_task.py` 已包含同一个 variance 模型内的 dur、pitch、variance 参数预测分支，并在 validation 中记录 duration、pitch、variance 曲线。
- `training/acoustic_task.py` 已包含 mel 训练/验证逻辑，开启 `val_with_vocoder` 后会记录 mel 对比图和试听音频。
- `training/variance_val_ds.py` 已存在 `.ds` validation 相关扩展，适合参考数据读取、tag 命名、`.ds` 样本组织方式，但 all-in-one 模式不应复用其多模型联合推理验证路径。
- `inference/ds_variance.py` 和 `inference/ds_acoustic.py` 是离线推理路径，all-in-one validation 可以复用其中输入解析和后处理思想，但不能在 validation 中重新构造多个外部模型来完成联合验证。

## 设计原则

1. all-in-one 是新模式，不改变默认行为。
2. 同一个 Lightning task 管理同一个组合模型，checkpoint 中只保存一个 `state_dict`。
3. validation 使用当前训练中的内存模型，不依赖外部 variance/acoustic checkpoint。
4. 唱法模块开启时，唱法 validation 必须增加与 acoustic 的联合 validation 输出。
5. 联合 validation 的 TensorBoard 分类要能区分常规 acoustic、常规 variance、all-in-one 联合推理。
6. 仍然允许使用现有 valid sample batch 方式；可以额外支持 `.ds` 样本 validation。
7. 不复用现有“多个模型从多个 checkpoint 加载后串联推理”的验证流程。

## 配置设计

新增一份示例配置：

```text
configs/templates/config_all_in_one.yaml
```

同时新增一份 LoRA 微调示例配置：

```text
configs/finetune_templates/all_in_one_lora.yaml
```

建议新增或保留以下关键配置：

```yaml
base_config:
  - configs/original/base.yaml

task_cls: training.all_in_one_task.AllInOneTask
binarizer_cls: preprocessing.all_in_one_binarizer.AllInOneBinarizer

all_in_one:
  enabled: true
  train_dur: true
  train_pitch: true
  train_variance: true
  train_acoustic: true
  shared_encoder: false
  loss_weights:
    dur: 1.0
    pitch: 1.0
    variance: 1.0
    acoustic: 1.0
    aux_mel: 0.2

validation:
  joint_infer: true
  joint_infer_category: all_in_one_joint
  num_joint_valid_plots: 10
  val_with_vocoder: true
  val_with_ds:
    enabled: false
    ds_files: []
    ds_val_spks: []
    max_samples_per_val: null
    overwrite_ds_dur: false
    overwrite_ds_pitch: false
    overwrite_ds_var: false
```

与原配置的映射关系：

- `predict_dur` 对应 `all_in_one.train_dur`。
- `predict_pitch` 对应 `all_in_one.train_pitch`。
- `predict_energy`、`predict_breathiness`、`predict_voicing`、`predict_tension` 对应 `all_in_one.train_variance` 下的具体 variance 开关。
- `use_energy_embed`、`use_breathiness_embed`、`use_voicing_embed`、`use_tension_embed` 决定 acoustic 是否消费 variance 特征。
- `val_with_vocoder` 保留原语义，但在 all-in-one 联合 validation 中归入 `all_in_one_joint/audio/...` 分类。

注意：这里的 `train_dur`、`train_pitch`、`train_variance` 是 `DiffSingerVariance` 内部的分支级训练开关，不表示 dur 是独立于 variance 的顶层模型。all-in-one 的顶层结构应理解为：

```text
AllInOneModel
  variance
    linguistic encoder
    duration predictor
    pitch predictor
    variance-parameter predictor
  acoustic
    fs2 acoustic encoder
    aux decoder / diffusion or reflow
```

### All-in-One LoRA 配置模板

all-in-one 模式应提供独立 LoRA 模板，方便用户从完整 all-in-one checkpoint 或已有 acoustic/variance 迁移权重继续微调。模板建议放在：

```text
configs/finetune_templates/all_in_one_lora.yaml
```

建议内容：

```yaml
base_config:
  - configs/templates/config_all_in_one.yaml

work_dir: ckpt/all_in_one_lora

finetune_enabled: false
freezing_enabled: false

lora:
  enabled: true
  base_ckpt: ckpt/all_in_one_base/model_ckpt_steps_100000.ckpt
  rank: 8
  alpha: 16
  train_bias: false
  merge_before_export: true
  target_modules:
    - linear
  module_scope:
    acoustic: true
    dur: true
    pitch: true
    variance: true

all_in_one:
  enabled: true
  train_dur: true
  train_pitch: true
  train_variance: true
  train_acoustic: true
  loss_weights:
    dur: 1.0
    pitch: 1.0
    variance: 1.0
    acoustic: 1.0
    aux_mel: 0.2

validation:
  joint_infer: true
  joint_infer_category: all_in_one_joint_lora
  num_joint_valid_plots: 10
  val_with_vocoder: true
```

`module_scope` 用于表达 LoRA 作用范围：

- 全开：同时微调 acoustic 与唱法模块。
- 只开 `acoustic`：只做声学 LoRA 微调。
- 只开 `dur` / `pitch` / `variance`：只做唱法链路 LoRA 微调。

实现时可以将 `module_scope` 转换为实际的模块名前缀过滤，例如：

```text
acoustic -> model.acoustic.*
dur      -> model.variance.dur_predictor.* 或 model.dur_predictor.*
pitch    -> model.variance.pitch_predictor.* 或 model.pitch_predictor.*
variance -> model.variance.variance_predictor.* 或 model.variance_predictor.*
```

如果沿用现有 `BaseTask.build_model()` 的 LoRA 注入逻辑，需要扩展 `utils.lora.inject_lora()` 或调用侧过滤，让 `target_modules` 既能匹配层类型，也能受 `module_scope` 限制。否则 all-in-one LoRA 只能粗粒度注入所有匹配层，不利于只微调 acoustic 或只微调唱法。

## 预处理方案

### 新增 AllInOneBinarizer

新增：

```text
preprocessing/all_in_one_binarizer.py
```

职责：

- 复用 `AcousticBinarizer` 的 mel、f0、speaker、language、音频相关处理。
- 复用 `VarianceBinarizer` 的 dur、pitch、variance 特征处理。
- 产出同一个 `binary_data_dir`，确保每条样本同时拥有 acoustic 和 variance 训练所需字段。
- 对缺失字段按训练开关做校验：只训练 acoustic 时允许 variance 目标缺失；开启对应 variance 训练时必须存在对应 target 或可由预处理生成。

### 数据字段要求

all-in-one 样本至少需要支持：

- 文本/音素：`tokens`、`ph_dur`、`word_div` 等现有字段。
- 音高：`f0`、`note_midi`、`note_dur`、`note_rest`，按现有 pitch 流程保持兼容。
- variance：`energy`、`breathiness`、`voicing`、`tension` 以及对应 timestep/grid 信息。
- 声学：`mel`、`wav_fn` 或 vocoder 所需的 mel/f0 信息。
- 多说话人/多语言：复用 `spk_id`、`spk_map`、`lang_id`、`lang_map`。

### 兼容策略

- 保留 `AcousticBinarizer` 和 `VarianceBinarizer` 不变。
- `AllInOneBinarizer` 内部优先组合已有处理逻辑，而不是复制大段代码。
- 如果 acoustic 与 variance 当前 metadata 字段命名不一致，先增加字段适配层，再逐步收敛。

## 模型与任务方案

### 新增组合模型

新增或扩展：

```text
modules/all_in_one/
training/all_in_one_task.py
```

组合模型建议结构：

```text
AllInOneModel
  text encoder / speaker / language embeddings
  variance module
    duration predictor
    pitch predictor
    variance predictors
  acoustic decoder / diffusion or reflow module
```

第一阶段可以先采用“子模块组合、权重同文件保存”的方式，不强制共享 encoder。`shared_encoder: false` 作为保守默认值，降低对现有 acoustic/variance 结构的侵入。后续再评估是否做共享 encoder。

### AllInOneTask

新增：

```text
training/all_in_one_task.py
```

职责：

- 继承 `BaseTask`。
- 在 `_build_model()` 中构造组合模型。
- 在 `run_model(sample, infer=False)` 中按配置执行 dur、pitch、variance、acoustic。
- 训练时返回分类 loss：
  - `dur_loss`
  - `pitch_loss`
  - `var_loss`
  - `mel_loss`
  - `aux_mel_loss`
  - `all_in_one_joint_loss` 或 `joint_consistency_loss`，如后续增加联合一致性约束。
- validation 时同时记录独立子任务指标和联合推理指标。

### 参数量总结

当前 `BaseTask.build_model()` 会在模型构建、冻结、finetune/QAT 准备后调用 `self.print_arch()`，默认实现是 `utils.print_arch(self.model)`。因此 `AllInOneTask` 继承 `BaseTask` 后可以自然打印整体模型结构。

但 all-in-one 模式还需要额外打印分模块参数量总结表格，避免用户只能看到一个巨大的组合模型总量。建议在 `AllInOneTask` 中重写 `print_arch()`：

```python
@rank_zero_only
def print_arch(self):
    super().print_arch()
    print_all_in_one_param_summary(self.model)
```

建议表格字段：

```text
module             total_params   trainable_params   frozen_params   percent
shared             ...
variance_encoder   ...
variance_dur       ...
variance_pitch     ...
variance_params    ...
acoustic           ...
lora               ...
other              ...
total              ...
```

统计规则：

- `total` 必须等于 `self.model.parameters()` 的总参数量。
- `trainable_params` 必须反映冻结、finetune、LoRA 注入之后的最终 `requires_grad` 状态。
- 如果 `shared_encoder: true`，共享 encoder 不应重复计入 dur/pitch/variance/acoustic，应单独归入 `shared`。
- 如果开启 LoRA，应额外打印 LoRA 参数量和 LoRA 可训练参数占比。
- 如果只开启部分分支，关闭的分支不显示或显示为 `disabled`，但 variance 顶层模块与总参数量必须仍然准确。

实现方式建议：

- 给 `AllInOneModel` 提供稳定的 `named_submodules_for_summary()`，返回模块名到 `nn.Module` 的映射。
- 增加一个通用 helper，例如 `utils/model_summary.py`：

```python
def count_module_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable, total - trainable
```

- 为避免共享参数重复统计，分模块统计时按 `id(parameter)` 去重；总计以全模型去重结果为准。

## All-in-One Validation 方案

### 常规样本联合 validation

在 `AllInOneTask._validation_step()` 中保留当前样本 validation：

```text
sample -> dur -> pitch -> variance -> acoustic -> mel/vocoder
```

当唱法模块开启时，必须额外执行联合 inference 分支：

- dur 使用当前模型预测或 sample 中 ground truth，取决于配置。
- pitch 使用当前模型预测或 sample 中 ground truth。
- variance 使用当前模型预测或 sample 中 ground truth。
- acoustic 必须消费上述联合链路输出，生成 mel。

TensorBoard 分类建议：

```text
all_in_one_joint/mel/<sample_name>
all_in_one_joint/audio/<sample_name>
all_in_one_joint/loss/<loss_name>
all_in_one_variance/dur/<sample_name>
all_in_one_variance/pitch/<sample_name>
all_in_one_variance/<variance_name>/<sample_name>
```

mel 对比图至少包含：

- ground truth mel
- acoustic teacher-forcing 或常规 acoustic 输出
- joint inference 输出 mel
- 差异图或 diffmel

试听至少包含：

- ground truth 或 reference audio
- acoustic 常规输出 audio
- joint inference 输出 audio

### `.ds` 样本 validation

可选新增 `.ds` 模式，但它必须走 all-in-one 当前内存模型：

```text
.ds -> preprocess_input -> dur -> pitch -> variance -> acoustic -> vocoder
```

实现要求：

- 可以参考 `training/variance_val_ds.py` 的 `.ds` 文件读取、speaker 覆盖、tag 命名、样本数限制。
- 不允许使用 `variance_ckpts` 加载外部 dur/pitch/variance 模型。
- 不允许使用外部 acoustic checkpoint。
- acoustic 与 variance 均从当前 all-in-one checkpoint/内存模型中取权重。

## Checkpoint 方案

all-in-one checkpoint 仍使用 Lightning 默认保存，但 `state_dict` 中应包含所有子模块：

```text
model.dur_predictor.*
model.pitch_predictor.*
model.variance_predictor.*
model.acoustic.*
model.vocoder_adapter.*  # 如存在，仅保存可训练部分
```

加载策略：

- 新训练：从空模型开始。
- 迁移训练：允许从已有 variance 或 acoustic checkpoint 局部加载到对应 prefix。
- finetune：沿用 `BaseTask.load_finetune_ckpt()` 的 strict shape 与 ignored params 机制。
- 导出：后续新增 all-in-one exporter，或先提供拆分导出脚本从同一 checkpoint 中抽取 acoustic/variance 子模块。

## ONNX 导出方案

### 现有导出结构分析

当前 ONNX 导出入口在 `scripts/export.py`：

- `export acoustic` 使用 `DiffSingerAcousticExporter`。
- `export variance` 使用 `DiffSingerVarianceExporter`。
- acoustic exporter 会导出一个 acoustic ONNX，并生成 `dsconfig.yaml`、phoneme/language/speaker 等附件。
- variance exporter 会按现有编辑器预期拆分导出：
  - `<model_name>.linguistic.onnx`
  - `<model_name>.dur.onnx`，仅在 `predict_dur` 开启时导出
  - `<model_name>.pitch.onnx`，仅在 `predict_pitch` 开启时导出
  - `<model_name>.variance.onnx`，仅在 variance predictor 开启时导出
  - 对应 `dsconfig.yaml` 与附件

这说明现有编辑器兼容性主要依赖两个方面：

1. 导出的 ONNX 文件名、拆分方式、输入输出名、dynamic axes、后处理图保持不变。
2. `dsconfig.yaml` 中 acoustic、variance、phoneme、speaker、language、采样率、mel 参数等字段保持不变。

因此 all-in-one 模式第一版不应优先导出一个全链路大 ONNX。更稳妥的目标是：即使输入权重来自同一个 all-in-one checkpoint，也能导出和旧 acoustic/variance 模型一模一样结构的 ONNX 文件，以保证现有编辑器无需修改即可加载。

### 当前脚本缺口

现有 exporter 的权重加载逻辑默认是：

```text
load_ckpt(model, hparams['work_dir'], prefix_in_ckpt='model')
```

也就是说它假设 checkpoint 里的目标模型权重位于 `model.*`。all-in-one checkpoint 预计会变成：

```text
model.acoustic.*
model.variance.*
model.dur_predictor.*
model.pitch_predictor.*
model.variance_predictor.*
```

或者类似的组合模型 prefix。这样直接调用旧 `export acoustic` / `export variance` 会出现两个问题：

- exporter 构造的是旧的 `DiffSingerAcousticONNX` 或 `DiffSingerVarianceONNX`，但 checkpoint 里对应权重不再位于纯 `model.*`。
- all-in-one 配置中同时存在 acoustic 与 variance 配置，导出某个子模块时需要临时取出对应子配置和 state_dict prefix。

所以如果脚本没有实现 all-in-one 导出适配，需要新增该部分。

### 兼容导出目标

新增 all-in-one 兼容导出模式，要求：

```text
输入：
  ckpt/all_in_one_exp/model_ckpt_steps_xxx.ckpt

输出：
  artifacts/all_in_one_exp/acoustic/*.onnx 或旧 acoustic 单文件结构
  artifacts/all_in_one_exp/variance/*.linguistic.onnx
  artifacts/all_in_one_exp/variance/*.dur.onnx
  artifacts/all_in_one_exp/variance/*.pitch.onnx
  artifacts/all_in_one_exp/variance/*.variance.onnx
  dsconfig.yaml / phonemes / languages / speaker attachments
```

导出的 ONNX 图结构、输入输出名、文件命名和配置字段应与现有 `DiffSingerAcousticExporter`、`DiffSingerVarianceExporter` 保持一致。编辑器侧看到的仍然是原 acoustic/variance 模型产物，而不是 all-in-one 专用格式。

### 实现建议

新增一个导出入口：

```bash
python scripts/export.py all-in-one --exp all_in_one_exp --ckpt 100000 --out artifacts/all_in_one_exp
```

或在现有入口上增加参数：

```bash
python scripts/export.py acoustic --exp all_in_one_exp --from_all_in_one --submodule acoustic
python scripts/export.py variance --exp all_in_one_exp --from_all_in_one --submodule variance
```

推荐第一版新增独立 `all-in-one` 命令，内部顺序调用兼容 exporter：

```text
AllInOneCompatExporter
  export_acoustic_compatible()
    构造 DiffSingerAcousticONNX
    从 all-in-one checkpoint 中抽取 acoustic prefix
    重映射为旧 acoustic exporter 期望的 model.* 权重
    复用 DiffSingerAcousticExporter 的 export_model/export_attachments

  export_variance_compatible()
    构造 DiffSingerVarianceONNX
    从 all-in-one checkpoint 中抽取 variance/dur/pitch prefix
    重映射为旧 variance exporter 期望的 model.* 权重
    复用 DiffSingerVarianceExporter 的 export_model/export_attachments
```

需要新增的能力：

- exporter 支持 `state_dict_override` 或 `ckpt_prefix_map`，避免只能从 `hparams['work_dir']` + `prefix_in_ckpt='model'` 加载。
- all-in-one 配置中明确子模块 prefix，例如：

```yaml
all_in_one:
  export_prefixes:
    acoustic: model.acoustic
    variance: model.variance
```

- 如果第一版组合模型直接复用旧子模型作为属性，应尽量让 prefix 简单稳定：

```text
model.acoustic.*
model.variance.*
```

这样导出时可以把：

```text
model.acoustic.fs2.*
model.acoustic.diffusion.*
```

重映射为：

```text
model.fs2.*
model.diffusion.*
```

variance 同理，把：

```text
model.variance.fs2.*
model.variance.dur_predictor.*
model.variance.pitch_predictor.*
model.variance.variance_predictor.*
```

重映射为旧 exporter 期望的：

```text
model.fs2.*
model.dur_predictor.*
model.pitch_predictor.*
model.variance_predictor.*
```

### LoRA 导出兼容

all-in-one LoRA 微调也需要支持兼容导出：

- 若 `lora.merge_before_export: true`，先把 all-in-one checkpoint 中对应子模块 LoRA merge 到子模型，再导出旧结构 ONNX。
- 若只微调 acoustic LoRA，则 acoustic ONNX 使用 merge 后权重，variance ONNX 可从 base 或 all-in-one 原始子模块导出。
- 若只微调唱法 LoRA，则 variance ONNX 使用 merge 后权重，acoustic ONNX 可从 base 或 all-in-one 原始子模块导出。
- 导出日志必须打印实际加载的 base checkpoint、LoRA checkpoint、子模块 prefix，避免用户误以为导出了完整 all-in-one 图。

### 验收标准

- 使用 all-in-one checkpoint 能导出与现有 acoustic exporter 相同结构的 acoustic ONNX。
- 使用 all-in-one checkpoint 能导出与现有 variance exporter 相同结构的 linguistic/dur/pitch/variance ONNX。
- 导出的 `dsconfig.yaml` 能被现有编辑器按旧模型格式读取。
- 对同一输入样本，兼容拆分 ONNX 链路的输出应与 all-in-one PyTorch 子模块输出在可接受误差内一致。
- 不要求第一版支持单个全链路 all-in-one ONNX；该能力可以作为后续增强。

## 实施阶段

### 阶段 1：完整流程分析与接口确认

- 梳理 `AcousticTask` 和 `VarianceTask` 的 sample 字段、loss 输出、validation 图/音频写入。
- 梳理 `AcousticBinarizer` 和 `VarianceBinarizer` 的输出字段差异。
- 梳理 `DiffSingerVarianceInfer` 和 `DiffSingerAcousticInfer` 中可复用的数据解析与后处理函数。
- 明确 all-in-one batch schema，并写入开发注释或文档。

### 阶段 2：预处理合并

- 新增 `AllInOneBinarizer`。
- 统一 metadata、speaker map、language map、binary dataset 字段。
- 增加字段完整性检查。
- 用小数据集跑一次 binarize，确认 acoustic/variance 字段可以被同一个 dataset 读取。

### 阶段 3：组合模型与任务

- 新增 `AllInOneTask` 和组合模型。
- 接入 dur、pitch、variance、acoustic 四类 loss。
- 支持按配置关闭任意子模块训练。
- 确认 checkpoint 中所有开启模块写入同一个权重文件。
- 启动训练时打印 all-in-one 分模块参数量表格和总参数量。
- LoRA 模式下额外打印 LoRA 总参数、可训练参数和占全模型比例。

### 阶段 4：联合 validation

- 在 all-in-one task 中实现常规样本联合 validation。
- 开启唱法模块时，强制增加联合 mel 对比图、试听音频和 joint loss 分类。
- 可选实现 `.ds` validation，但只使用当前 all-in-one 内存模型。
- TensorBoard tag 与现有 acoustic/variance tag 保持可区分。

### 阶段 5：配置示例与文档

- 新增 `configs/templates/config_all_in_one.yaml`。
- 新增 `configs/finetune_templates/all_in_one_lora.yaml`。
- 在文档中说明从现有 acoustic/variance 配置迁移到 all-in-one 的字段对应关系。
- 在文档中说明 all-in-one LoRA 的 `module_scope`、`target_modules`、`base_ckpt`、`merge_before_export` 用法。
- 给出最小配置、启用唱法模块配置、多 speaker 配置示例。
- 给出 all-in-one LoRA 全模型微调、只微调 acoustic、只微调唱法链路三个示例。

### 阶段 6：ONNX 兼容导出

- 新增 `scripts/export.py all-in-one` 或等价参数。
- 新增 `AllInOneCompatExporter`，从同一个 all-in-one checkpoint 拆出 acoustic 与 variance 子模块。
- 复用现有 acoustic/variance ONNX 图导出与优化逻辑，保持文件结构、输入输出和附件格式不变。
- 增加 prefix 重映射和 LoRA merge 处理。
- 用现有编辑器加载导出产物做兼容性验证。

### 阶段 7：验证与回归

- 跑 `scripts/binarize.py --config configs/templates/config_all_in_one.yaml` 的最小样本流程。
- 跑 `scripts/train.py --config configs/templates/config_all_in_one.yaml` 的 smoke test。
- 检查 validation 输出：
  - dur/pitch/variance 曲线正常。
  - acoustic mel 对比图正常。
  - all-in-one joint mel 对比图正常。
  - all-in-one joint audio 正常。
  - loss 分类包含唱法、声学、联合推理相关项。
- 确认关闭 all-in-one 后，原 variance/acoustic 训练配置行为不变。

## 风险与注意事项

- 全局 `hparams` 当前被许多推理和模型组件读取，all-in-one validation 中不能临时切换到外部 acoustic/variance 配置。
- 如果共享 encoder，会显著扩大改动面；建议第一版先做子模块组合。
- `.ds` validation 容易误用外部 checkpoint，需要在配置 schema 和代码断言中显式禁止。
- vocoder 通常不训练，只在 validation 试听时加载；需要控制显存占用。
- 联合 inference 比普通 validation 慢，应提供 `num_joint_valid_plots`、`max_samples_per_val` 等限制。

## 验收标准

- 提供 `config_all_in_one.yaml` 示例。
- 提供 `all_in_one_lora.yaml` LoRA 微调模板。
- 使用 all-in-one task 训练时，dur、pitch、variance、acoustic 权重保存在同一个 checkpoint。
- all-in-one 开训时打印 acoustic、dur、pitch、variance、shared、LoRA 和 total 参数量总结。
- 开启唱法模块后，validation 产生 `dur -> pitch -> variance -> acoustic + vocoder` 的联合输出。
- TensorBoard 中能区分普通 variance、普通 acoustic、all-in-one joint 三类结果。
- `.ds` validation 如实现，只使用当前 all-in-one 权重，不依赖外部模型联合推理。
- all-in-one checkpoint 可以分模块导出旧结构 ONNX，并兼容现有编辑器。
- 原有 `VarianceTask`、`AcousticTask`、`VarianceBinarizer`、`AcousticBinarizer` 的默认行为保持兼容。
