"""Single-model all-in-one deployment, with optional split component views."""

import copy
import json
from pathlib import Path

import onnx
import torch
from torch import nn
import yaml

from basics.base_exporter import BaseExporter
from deployment.exporters.acoustic_exporter import DiffSingerAcousticExporter
from deployment.exporters.variance_exporter import DiffSingerVarianceExporter
from deployment.exporters.onnx_export import export_onnx
from deployment.modules.toplevel import DiffSingerAcousticONNX, DiffSingerVarianceONNX
from modules.vocoders.mini_ddsp import MiniDDSP
from utils import load_ckpt
from utils.hparams import hparams, mini_ddsp_config


class AllInOneAcousticONNX(DiffSingerAcousticONNX):
    @property
    def category(self):
        return "all_in_one"


class AllInOneVarianceONNX(DiffSingerVarianceONNX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from deployment.modules.fastspeech2 import LengthRegulator

        self.lr = LengthRegulator()

    @property
    def category(self):
        return "all_in_one"


class AllInOneAcousticExporter(DiffSingerAcousticExporter):
    def build_model(self):
        model = (
            AllInOneAcousticONNX(
                vocab_size=len(self.phoneme_dictionary),
                out_dims=hparams["audio_num_mel_bins"],
                cross_lingual_token_idx=sorted(
                    {
                        self.phoneme_dictionary.encode_one(phone)
                        for phone in self.phoneme_dictionary.cross_lingual_phonemes
                    }
                ),
            )
            .eval()
            .to(self.device)
        )
        load_ckpt(
            model,
            hparams["work_dir"],
            ckpt_steps=self.ckpt_steps,
            prefix_in_ckpt="model.acoustic",
            strict=True,
            device=self.device,
            prefer_inference=True,
        )
        return model


class AllInOneVarianceExporter(DiffSingerVarianceExporter):
    def build_model(self):
        model = (
            AllInOneVarianceONNX(
                vocab_size=len(self.phoneme_dictionary),
                cross_lingual_token_idx=sorted(
                    {
                        self.phoneme_dictionary.encode_one(phone)
                        for phone in self.phoneme_dictionary.cross_lingual_phonemes
                    }
                ),
            )
            .eval()
            .to(self.device)
        )
        load_ckpt(
            model,
            hparams["work_dir"],
            ckpt_steps=self.ckpt_steps,
            prefix_in_ckpt="model.variance",
            strict=True,
            device=self.device,
            prefer_inference=True,
        )
        model.build_smooth_op(self.device)
        return model


class MiniDDSPWaveformView(nn.Module):
    def __init__(self, vocoder):
        super().__init__()
        self.vocoder = vocoder

    def forward(self, mel, f0):
        waveform = self.vocoder(mel, f0)
        lengths = torch.ones_like(f0[:, 0], dtype=torch.long) * (
            mel.shape[1] * self.vocoder.hop_size
        )
        return waveform, lengths


def export_vocoder_graph(vocoder, path):
    mel = torch.full(
        (1, 10, vocoder.mel_embed.in_features),
        -6.0,
        device=next(vocoder.parameters()).device,
    )
    f0 = torch.full((1, 10), 220.0, device=mel.device)
    export_onnx(
        MiniDDSPWaveformView(vocoder).eval(),
        (mel, f0),
        path,
        input_names=["mel", "f0"],
        output_names=["waveform", "waveform_lengths"],
        dynamic_axes={
            "mel": {0: "batch", 1: "n_frames"},
            "f0": {0: "batch", 1: "n_frames"},
            "waveform": {0: "batch", 1: "n_samples"},
            "waveform_lengths": {0: "batch"},
        },
        opset_version=15,
    )


def merge_waveform_graph(acoustic, vocoder):
    """Share the acoustic mel and F0 directly inside one self-contained graph."""
    result = copy.deepcopy(acoustic)
    graph = onnx.compose.add_prefix(vocoder, "mini_ddsp.")
    mapping = {
        "mini_ddsp.mel": "mel",
        "mini_ddsp.f0": "f0",
        "mini_ddsp.waveform": "waveform",
        "mini_ddsp.waveform_lengths": "waveform_lengths",
    }
    for node in graph.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = mapping.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = mapping.get(name, name)
    result.graph.node.extend(graph.graph.node)
    result.graph.initializer.extend(graph.graph.initializer)
    for output in graph.graph.output:
        output.name = mapping[output.name]
        result.graph.output.append(output)
    # Both graphs use the existing opset15 toolchain and embedded initializers.
    onnx.checker.check_model(result)
    return result


class DiffSingerAllInOneExporter(BaseExporter):
    def __init__(self, device="cpu", cache_dir=None, ckpt_steps=None, split_only=False):
        super().__init__(device=device, cache_dir=cache_dir)
        if not mini_ddsp_config(hparams).get("enabled", False):
            raise ValueError("Waveform all-in-one export requires Mini-DDSP enabled")
        if hparams.get("lora", {}).get("enabled", False):
            raise ValueError(
                "All-in-one export requires a full checkpoint with LoRA disabled"
            )
        self.ckpt_steps = ckpt_steps
        self.split_only = split_only

    def export(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        acoustic = AllInOneAcousticExporter(
            device=self.device,
            cache_dir=self.cache_dir / "acoustic",
            ckpt_steps=self.ckpt_steps,
        )
        acoustic.export(path / "acoustic")
        acoustic_config_path = path / "acoustic/dsconfig.yaml"
        acoustic_config = yaml.safe_load(
            acoustic_config_path.read_text(encoding="utf-8")
        )
        acoustic_config.pop("vocoder", None)
        acoustic_config["view"] = "mel-only"
        acoustic_config_path.write_text(
            yaml.safe_dump(acoustic_config, sort_keys=False), encoding="utf-8"
        )
        mel_path = path / "acoustic" / acoustic_config["acoustic"]
        vocoder = MiniDDSP(hparams).eval().to(self.device)
        load_ckpt(
            vocoder,
            hparams["work_dir"],
            ckpt_steps=self.ckpt_steps,
            prefix_in_ckpt="model.vocoder",
            strict=True,
            device=self.device,
            prefer_inference=True,
        )
        vocoder_path = self.cache_dir / "mini_ddsp.onnx"
        export_vocoder_graph(vocoder, vocoder_path)
        waveform_path = path / (hparams["exp_name"] + ".waveform.onnx")
        onnx.save(
            merge_waveform_graph(onnx.load(mel_path), onnx.load(vocoder_path)),
            waveform_path,
        )
        vocoder_path.unlink()
        variance = AllInOneVarianceExporter(
            device=self.device,
            cache_dir=self.cache_dir / "variance",
            ckpt_steps=self.ckpt_steps,
        )
        variance.export(path / "variance")
        full_views = {}
        if not self.split_only:
            from deployment.exporters.all_in_one_graph import export_full_graphs

            variance_config = yaml.safe_load(
                (path / "variance/dsconfig.yaml").read_text(encoding="utf-8")
            )
            full_views = export_full_graphs(
                path,
                self.cache_dir / "controls",
                variance_config,
                mel_path,
                waveform_path,
                hparams["exp_name"],
            )
        metadata = {
            "format": "dsrx-all-in-one-mini-ddsp",
            "version": 2,
            **full_views,
            "default_view": "full" if full_views else "waveform",
            "manual_controls": {
                "use_manual_durations": "true selects manual_durations; false predicts durations",
                "pitch_retake": "true predicts F0; false preserves manual_f0 at that frame",
                "variance_retake": "true predicts a curve; false preserves its manual value",
                "duration_unit": "frames",
                "f0_unit": "Hz",
            },
            "opset": 15,
            "waveform": waveform_path.name,
            "mel_only": str(mel_path.relative_to(path).as_posix()),
            "variance_config": "variance/dsconfig.yaml",
            "acoustic_config": "acoustic/dsconfig.yaml",
            "outputs": ["mel", "waveform", "waveform_lengths"],
            "default_external_vocoder": False,
            "sample_rate": hparams["audio_sample_rate"],
            "hop_size": hparams["hop_size"],
            "num_mel_bins": hparams["audio_num_mel_bins"],
            "fft_size": hparams["fft_size"],
            "win_size": hparams["win_size"],
            "fmin": hparams["fmin"],
            "fmax": hparams["fmax"],
            "mel_base": "e",
            "noise_resource_version": vocoder.resource_version,
            "optional_external_backend": "NSF-HiFiGAN with exactly matching audio contract",
        }
        (path / "manifest.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"| exported self-contained all-in-one deployment => {path}")
