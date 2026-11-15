# All-in-one Mini-DDSP 导出

从仓库根目录运行，实验必须开启 `all_in_one.mini_ddsp.enabled`，并提供完整 all-in-one checkpoint：

```powershell
python scripts/export.py all-in-one --exp all_in_one_mini_ddsp_cloud --out artifacts/cloud_export --name cloud
```

默认同时导出以下视图，采样率和 mel 定义来自训练配置：

| 视图 | 文件 | 用途 |
|---|---|---|
| 完整波形模型 | `cloud.full.onnx` | 时长、音高、唱法预测 → 声学 → 内置声码器，一次 ORT 调用输出波形 |
| 完整 mel-only | `cloud.full_mel_only.onnx` | 相同预测及手动控制，输出 mel，无声码器权重/噪声资源 |
| 声学及声码器 | `cloud.waveform.onnx` | 已有时长/F0/唱法时直接合成，输出 mel/波形/采样长度 |
| 分离模型 | `acoustic/`、`variance/` | 保留原有分离调度与显式外部声码器使用方式 |

`--split-only` 不生成两个完整模型，仅导出分离视图。原 `acoustic`、`variance` 导出命令保留。

完整模型的所有神经网络权重及 Mini-DDSP 噪声基底均内嵌，无需额外 checkpoint、外部声码器、噪声文件或仓库源码。文本转音素、音符/词语对齐及 token 编码仍由宿主根据导出的词典、phonemes/languages JSON 完成。图输入是编码后的乐谱张量，并非 `.ds` 文件字节。

## 输入与手动控制

完整图沿用现有 variance 导出模型的单句批量约定，batch=1；音素数、词数、音符数、帧数动态。调用前读取 ORT `get_inputs()`；实际暴露的语言、唱法、glide、speaker 字段随配置决定。

| 输入 | dtype / 形状 | 含义 |
|---|---|---|
| `tokens` | int64 `[1,P]` | 音素 ID |
| `word_div` | int64 `[1,W]` | 各词的音素个数，和为 P |
| `word_dur` | int64 `[1,W]` | 各词帧数 |
| `ph_midi` | int64 `[1,P]` | 由乐谱词音高分配至音素的 MIDI 值，供时长预测 |
| `note_midi` / `note_dur` | float32 / int64 `[1,N]` | 音符 MIDI 与帧数；休止音高按现有宿主规则插值 |
| `note_rest` | bool `[1,N]` | 休止符标记（启用 melody encoder 时） |
| `manual_durations` | int64 `[1,P]` | 手动音素帧数 |
| `use_manual_durations` | bool 标量 | true 使用手动时长；false 使用预测时长 |
| `manual_f0` | float32 `[1,T]` | 手动 F0，单位 Hz，允许 0 |
| `pitch_retake` | bool `[1,T]` | true 预测，false 精确保留该帧 manual_f0 |
| `expr` | float32 `[1,T]` | 音高表现力，通常为 1 |
| `manual_breathiness` 等 | float32 `[1,T]` | 各已启用唱法参数，单位同训练配置 |
| `variance_retake` | bool `[1,T,V]` | true 预测，false 精确保留该帧手动参数 |
| `gender` / `velocity` | float32 `[1,T]` | 音色偏移和速度条件；默认 0 / 1 |
| `depth` | float32 标量 | 声学采样深度；0 为仅 AUX，正值启用后级 |
| `pitch_steps` / `variance_steps` / `acoustic_steps` | int64 标量 | 各采样器的正整数步数 |

T 为 `note_dur.sum()`，也应等于 `word_dur.sum()`；手动时长使用相同帧轴。预测时长沿原 RhythmRegulator 按词归一化、取整，再按原推理语义裁剪或延长到音符总帧数。`variance_retake` 的最后一维顺序是启用的 `energy, breathiness, voicing, tension` 子集；当前云端配置 energy 关闭，因此 V=3。

所有自动模式仍需提供形状正确的手动占位张量，值可设为合法常量。全自动：`use_manual_durations=false`，两个 retake 全 true。全手动：前者 true，两个 retake 全 false。混合模式可逐帧、逐唱法参数选择。

## 输出

完整波形图输出 `predicted_durations`（按词归一化/取整的预测）、实际使用且已对齐总帧数的 `durations`、`predicted_pitch`（MIDI）、实际 `pitch`（MIDI）和 `f0`（Hz）、各 `predicted_<唱法参数>` 与实际唱法参数，以及 `mel [1,T,M]`、`waveform [1,T*hop_size]`、`waveform_lengths [1]`。手动模式下预测出口仍可用于对比；实际出口是声学与声码器真正消费的值，手动时长不依赖自动预测能否展开。mel-only 图保留所有控制出口，去掉波形出口及声码器资源。

## 功能验证

```powershell
python scripts/validate_all_in_one_onnx_controls.py --deploy artifacts/cloud_export --report artifacts/cloud_export_controls.json
```

验证器把完整图复制到独立临时目录，使用 `python -I`、默认 ORT 优化执行，无分离模型及源代码依赖。覆盖 1/9/10/11/257/1025 帧、多音素/多词、自动/手动/混合控制、精确手动值保留及波形长度。另由 `scripts/validate_mini_ddsp_onnx.py` 对真实 checkpoint 执行 eager 波形/确定性声学对照及显式 NSF 同 mel 契约检查。

这些验证证明导出与控制接口能运行，不证明模型已收敛。长期训练、训练后音质/F0收益与人工听音按用户决定记录为 SKIPPED 或 MANUAL/PENDING，不能由随机权重测试替代。
