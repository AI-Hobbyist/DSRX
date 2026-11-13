"""Compose variance, manual controls, acoustic and waveform into one ONNX graph."""

import copy
from pathlib import Path

import onnx
import torch
from torch import nn

from deployment.exporters.onnx_export import export_onnx
from deployment.modules.fastspeech2 import LengthRegulator


class DurationControls(nn.Module):
    def __init__(self):
        super().__init__()
        self.lr = LengthRegulator()

    def align(self, durations, frames):
        alignment = self.lr(durations)
        positions = torch.arange(frames, device=durations.device)
        positions = positions.clamp(max=alignment.shape[1] - 1)
        alignment = alignment[:, positions]
        phones = torch.arange(1, durations.shape[1] + 1, device=durations.device)[
            None, :, None
        ]
        return (alignment[:, None, :] == phones).long().sum(dim=2)

    def forward(
        self,
        raw_durations,
        word_div,
        word_dur,
        note_dur,
        manual_durations,
        use_manual_durations,
    ):
        ph2word = self.lr(word_div)
        words = torch.arange(1, word_dur.shape[1] + 1, device=word_dur.device)[
            None, :, None
        ]
        membership = ph2word[:, None, :] == words
        durations = raw_durations * (ph2word > 0)
        totals = (membership * durations[:, None, :]).sum(dim=2)
        ratio = word_dur.float() / totals.clamp(min=1e-5)
        scaled = durations * torch.gather(ratio, 1, ph2word - 1)
        predicted = self.align(scaled.round().long(), note_dur.sum())
        chosen = torch.where(use_manual_durations, manual_durations, predicted)
        return predicted, self.align(chosen, note_dur.sum())


class PitchInput(nn.Module):
    def forward(self, manual_f0):
        return 69 + 12 * torch.log2(manual_f0.clamp(min=1e-5) / 440)


class PitchControls(nn.Module):
    def forward(self, predicted_pitch, manual_pitch, manual_f0, pitch_retake):
        pitch = torch.where(pitch_retake, predicted_pitch, manual_pitch)
        f0 = torch.where(pitch_retake, 440 * 2 ** ((pitch - 69) / 12), manual_f0)
        return pitch, f0


class SingingControls(nn.Module):
    def forward(self, predicted, manual, retake):
        return tuple(
            torch.where(retake[:, :, index], prediction, value)
            for index, (prediction, value) in enumerate(zip(predicted, manual))
        )


def _rename_values(graph, mapping):
    for node in graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = mapping.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = mapping.get(name, name)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                _rename_values(attribute.g, mapping)
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        value.name = mapping.get(value.name, value.name)


def compose_components(components, outputs):
    """Bind named tensors, including lexical captures inside If/Loop subgraphs."""
    nodes, initializers, inputs, values = [], [], {}, {}
    produced = set()
    result = copy.deepcopy(components[0][0])
    for index, (model, bindings) in enumerate(components):
        prefix = f"stage{index}."
        component = onnx.compose.add_prefix(model, prefix)
        mapping = {prefix + key: value for key, value in bindings.items()}
        _rename_values(component.graph, mapping)
        nodes.extend(component.graph.node)
        initializers.extend(component.graph.initializer)
        for value in component.graph.input:
            inputs.setdefault(value.name, copy.deepcopy(value))
        for value in component.graph.output:
            values[value.name] = copy.deepcopy(value)
        produced.update(name for node in component.graph.node for name in node.output)
        produced.update(value.name for value in component.graph.initializer)
    result.graph.CopyFrom(
        onnx.helper.make_graph(
            nodes,
            "all_in_one",
            [value for name, value in inputs.items() if name not in produced],
            [values[name] for name in outputs],
            initializers,
        )
    )
    onnx.checker.check_model(result)
    return result


def _export_bridge(model, args, path, inputs, outputs, axes):
    export_onnx(
        model.eval(),
        args,
        path,
        input_names=inputs,
        output_names=outputs,
        dynamic_axes=axes,
        opset_version=15,
    )
    return onnx.load(path)


def export_full_graphs(
    directory, cache, variance_config, acoustic_path, waveform_path, name
):
    """Retain split views while adding complete automatic/manual pipeline views."""
    directory, cache = Path(directory), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    if not variance_config.get("predict_dur"):
        raise ValueError("Complete all-in-one export requires duration prediction")
    variance_dir = directory / "variance"
    graphs = {
        key: onnx.load(variance_dir / variance_config[key])
        for key in ("linguistic", "dur", "pitch", "variance")
    }
    curves = [
        name
        for name in ("energy", "breathiness", "voicing", "tension")
        if variance_config.get(f"predict_{name}", False)
    ]
    durations = torch.tensor([[2, 1, 3]])
    frames = 6
    duration_inputs = [
        "raw_durations",
        "word_div",
        "word_dur",
        "note_dur",
        "manual_durations",
        "use_manual_durations",
    ]
    duration_outputs = ["predicted_durations", "durations"]
    axes = {
        key: {1: "n_tokens"}
        for key in ["raw_durations", "manual_durations"] + duration_outputs
    }
    axes.update(
        {
            "word_div": {1: "n_words"},
            "word_dur": {1: "n_words"},
            "note_dur": {1: "n_notes"},
        }
    )
    duration_bridge = _export_bridge(
        DurationControls(),
        (
            durations.float(),
            torch.tensor([[2, 1]]),
            torch.tensor([[3, 3]]),
            torch.tensor([[3, 3]]),
            durations,
            torch.tensor(False),
        ),
        cache / "durations.onnx",
        duration_inputs,
        duration_outputs,
        axes,
    )
    f0 = torch.full((1, frames), 220.0)
    pitch_input = _export_bridge(
        PitchInput(),
        (f0,),
        cache / "pitch_input.onnx",
        ["manual_f0"],
        ["manual_pitch"],
        {key: {1: "n_frames"} for key in ("manual_f0", "manual_pitch")},
    )
    pitch_names = ["predicted_pitch", "manual_pitch", "manual_f0", "pitch_retake"]
    pitch_bridge = _export_bridge(
        PitchControls(),
        (f0.clone(), f0.clone(), f0.clone(), torch.ones_like(f0, dtype=torch.bool)),
        cache / "pitch_controls.onnx",
        pitch_names,
        ["pitch", "f0"],
        {key: {1: "n_frames"} for key in pitch_names + ["pitch", "f0"]},
    )
    predicted_names = [f"predicted_{curve}" for curve in curves]
    manual_names = [f"manual_{curve}" for curve in curves]
    singing_inputs = predicted_names + manual_names + ["variance_retake"]
    singing_bridge = _export_bridge(
        SingingControls(),
        (
            tuple(f0.clone() for _ in curves),
            tuple(f0.clone() for _ in curves),
            torch.ones(1, frames, len(curves), dtype=torch.bool),
        ),
        cache / "singing_controls.onnx",
        singing_inputs,
        curves,
        {key: {1: "n_frames"} for key in singing_inputs + curves},
    )
    identity = lambda model: {
        value.name: value.name
        for value in list(model.graph.input) + list(model.graph.output)
    }
    components = [(graphs["linguistic"], identity(graphs["linguistic"]))]
    bindings = identity(graphs["dur"])
    bindings["ph_dur_pred"] = "raw_durations"
    components += [
        (graphs["dur"], bindings),
        (duration_bridge, identity(duration_bridge)),
        (pitch_input, identity(pitch_input)),
    ]
    bindings = identity(graphs["pitch"])
    bindings.update(
        ph_dur="durations",
        pitch="manual_pitch",
        retake="pitch_retake",
        steps="pitch_steps",
        pitch_pred="predicted_pitch",
    )
    components += [(graphs["pitch"], bindings), (pitch_bridge, identity(pitch_bridge))]
    bindings = identity(graphs["variance"])
    bindings.update(
        ph_dur="durations", retake="variance_retake", steps="variance_steps"
    )
    bindings.update({curve: f"manual_{curve}" for curve in curves})
    bindings.update({f"{curve}_pred": f"predicted_{curve}" for curve in curves})
    components += [
        (graphs["variance"], bindings),
        (singing_bridge, identity(singing_bridge)),
    ]
    controls = (
        ["predicted_durations", "durations", "predicted_pitch", "pitch", "f0"]
        + predicted_names
        + curves
    )
    paths = {}
    for view, path, outputs in [
        ("full", waveform_path, ["mel", "waveform", "waveform_lengths"]),
        ("full_mel_only", acoustic_path, ["mel"]),
    ]:
        acoustic = onnx.load(path)
        bindings = identity(acoustic)
        bindings["steps"] = "acoustic_steps"
        complete = compose_components(
            components + [(acoustic, bindings)], controls + outputs
        )
        target = directory / f"{name}.{view}.onnx"
        onnx.save(complete, target)
        paths[view] = target.name
    return paths
