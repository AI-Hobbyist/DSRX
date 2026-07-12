# TensorBoard 验证指标布局迁移计划

## 目标

为训练脚本增加一个启动参数：

```bash
--tb_layout flat
--tb_layout grouped
```

其中：

- `flat`：默认值，保持现有 TensorBoard tag 风格，例如 `diffmel_0`、`tension_3`。
- `grouped`：启用 speaker/sample 分组 tag，例如 `aco_diffmel_alex_en/sample_001`。

这样可以在不破坏旧日志习惯和外部脚本兼容性的前提下，让需要多 speaker、多条 val 对比的训练任务切换到更清晰的分组视图。

## 背景

当前已有验证指标大多使用扁平编号：

```text
diffmel_0
auxmel_0
gt_0
diff_0
dur_0
pitch_0
tension_0
```

这种方式与 acoustic 原始实现一致：每条 valid 样本按全局 `data_idx` 单独写一个 figure/audio。但当同一个 speaker 有多条 val 时，TensorBoard 里只看到 `_0`, `_1`, `_2`，不容易判断它们属于哪个 speaker，也不方便同 speaker 内横向比较。

`val_with_ds` 目前已经更适合分组展示：

```text
var_mel_<spk>/<ds_file>
var_audio_<spk>/<ds_file>
```

本计划把这种布局抽象成可选模式，并推广到 acoustic 与 variance 的已有验证指标。

## 启动参数设计

### 参数

```bash
--tb_layout {flat,grouped}
```

### 默认值

```bash
--tb_layout flat
```

默认保持现状，避免旧训练流程、旧 TensorBoard 阅读习惯、自动化脚本被突然改变。

### 含义

`flat`：

```text
diffmel_0
diff_0
tension_0
var_mel_0
```

`grouped`：

```text
aco_diffmel_<spk>/<sample_name>
aco_audio_diff_<spk>/<sample_name>
var_tension_<spk>/<sample_name>
var_mel_<spk>/<ds_file>
var_audio_<spk>/<ds_file>
```

## 配置流转方案

训练入口解析 `--tb_layout` 后写入 `hparams`：

```python
hparams['tb_layout'] = args.tb_layout
```

各 task 内部通过：

```python
hparams.get('tb_layout', 'flat')
```

决定 tag 生成方式。

如果项目的启动参数已经有统一的 hparams 覆盖机制，优先复用现有机制；否则在训练入口显式加 argparse 参数。参数名保持 CLI 风格 `--tb_layout`，hparams key 使用同名 `tb_layout`。

## Tag 生成规范

新增共享 helper，建议放在 `utils/tensorboard_utils.py`：

```python
import re


def sanitize_tb_tag_part(value, fallback='item'):
    value = re.sub(r'[^0-9A-Za-z._-]+', '_', str(value)).strip('._-')
    return value or fallback


def flat_tb_tag(prefix, index):
    return f'{sanitize_tb_tag_part(prefix, "metric")}_{int(index)}'


def grouped_tb_tag(prefix, speaker, item):
    return (
        f'{sanitize_tb_tag_part(prefix, "metric")}_'
        f'{sanitize_tb_tag_part(speaker, "spk")}/'
        f'{sanitize_tb_tag_part(item, "item")}'
    )


def validation_tb_tag(layout, prefix, index, speaker=None, item=None):
    if layout == 'grouped':
        return grouped_tb_tag(prefix, speaker, item)
    return flat_tb_tag(prefix, index)
```

注意：

- TensorBoard 用 `/` 表示层级，因此 `speaker` 和 `item` 必须 sanitize。
- image 和 audio 不共用同一个 tag，避免不同插件类型互相干扰。
- 同一 speaker 下如果 sample name 重复，需要追加去重后缀。

## Acoustic 指标迁移

涉及文件：

```text
training/acoustic_task.py
```

涉及函数：

- `plot_mel`
- `plot_wav`

### flat 布局

保持现有行为：

```text
diffmel_0
auxmel_0
gt_0
aux_0
diff_0
```

### grouped 布局

改为：

```text
aco_diffmel_<spk>/<sample_name>
aco_auxmel_<spk>/<sample_name>

aco_audio_gt_<spk>/<sample_name>
aco_audio_aux_<spk>/<sample_name>
aco_audio_diff_<spk>/<sample_name>
```

数据来源：

- `spk`: `self.valid_dataset.metadata['spk_names'][data_idx]`
- `sample_name`: `self.valid_dataset.metadata['names'][data_idx]`

原有图内标题继续保留：

```text
<spk> - <sample_name>
```

## Variance 已有曲线指标迁移

涉及文件：

```text
training/variance_task.py
```

涉及函数：

- `plot_dur`
- `plot_pitch`
- `plot_curve`

### flat 布局

保持现有行为：

```text
dur_0
pitch_0
energy_0
breathiness_0
voicing_0
tension_0
```

### grouped 布局

改为：

```text
var_dur_<spk>/<sample_name>
var_pitch_<spk>/<sample_name>
var_energy_<spk>/<sample_name>
var_breathiness_<spk>/<sample_name>
var_voicing_<spk>/<sample_name>
var_tension_<spk>/<sample_name>
```

数据来源：

- `spk`: `self.valid_dataset.metadata['spk_names'][data_idx]`
- `sample_name`: `self.valid_dataset.metadata['names'][data_idx]`

## val_with_ds 指标迁移

涉及文件：

```text
training/variance_val_ds.py
```

### flat 布局

为了和 acoustic/variance 的默认兼容策略一致，flat 下使用编号：

```text
var_mel_0
var_audio_0
var_mel_1
var_audio_1
```

### grouped 布局

使用当前推荐结构：

```text
var_mel_<spk>/<ds_file>
var_audio_<spk>/<ds_file>
```

数据来源：

- `spk`: `val_with_ds.ds_val_spks`
- `ds_file`: `.ds` 文件名去掉 `.ds` 后缀

`.ds` 文件名去重逻辑保留：如果不同路径下有同名 `.ds`，追加父目录或序号，避免同一 tag 覆盖。

## 实施步骤

1. 在训练入口增加 `--tb_layout` 参数，choices 为 `flat` 和 `grouped`，默认 `flat`。
2. 将参数写入 `hparams['tb_layout']`。
3. 新增共享 tag helper。
4. 改造 `training/acoustic_task.py`：
   - `plot_mel` 根据 `tb_layout` 生成 mel tag。
   - `plot_wav` 根据 `tb_layout` 生成 audio tag。
5. 改造 `training/variance_task.py`：
   - `plot_dur`、`plot_pitch`、`plot_curve` 根据 `tb_layout` 生成 tag。
6. 改造 `training/variance_val_ds.py`：
   - flat 下使用编号 tag。
   - grouped 下使用 `var_mel_<spk>/<ds>` 和 `var_audio_<spk>/<ds>`。
7. 运行检查：
   - `python -m py_compile training/acoustic_task.py training/variance_task.py training/variance_val_ds.py`
   - `git diff --check`
8. 分别用 `--tb_layout flat` 和 `--tb_layout grouped` 做短验证。

## 验收标准

### flat

- TensorBoard tag 与现有风格一致。
- 旧脚本和旧阅读习惯不受影响。
- `val_with_ds` 也使用扁平编号，不出现 speaker 分组。

### grouped

- 同一个 speaker 的多条 valid 样本在同一 TensorBoard 折叠组下。
- acoustic mel/audio 分别出现在 `aco_*` 和 `aco_audio_*` 分组。
- variance 曲线分别出现在 `var_dur_*`、`var_pitch_*`、`var_tension_*` 等分组。
- `val_with_ds` 出现在 `var_mel_*` 和 `var_audio_*` 分组。
- 每条样本仍然独立写 figure/audio，不做拼接或多子图合并。
- 图内标题仍为 `<speaker> - <sample_name>` 或 `<speaker> - <ds_file>`。

## 兼容与风险

- 默认 `flat` 能最大限度保持兼容。
- grouped 会产生新 tag；旧 event 不会被重命名，TensorBoard 中可能同时看到旧 tag 和新 tag，直到使用新日志目录或清理旧 event。
- 如果外部脚本读取固定 tag，例如 `diffmel_0`，需要在使用 grouped 前同步更新。
- 如果同一 speaker 下 sample name 重复，必须做 tag 去重，否则同一步可能覆盖。
- `tb_layout` 只改变 TensorBoard 展示结构，不改变验证样本选择逻辑；`num_valid_plots` 仍按全局 `data_idx` 限制前 N 条样本。
