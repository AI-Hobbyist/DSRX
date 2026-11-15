import copy
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from deployment.exporters.onnx_export import export_onnx
from deployment.modules.dit import DiTBlockONNX
from deployment.modules.toplevel import DiffSingerAcousticONNX
from modules.aux_decoder import AuxDecoderAdaptor, build_aux_decoder, build_aux_loss
from modules.aux_decoder.dit import DiTAuxDecoder
from modules.toplevel import DiffSingerAcoustic, DiffSingerAllInOne
from utils.binarization_checkpoint import preprocessing_signature
from utils.hparams import hparams, set_hparams

SMALL_ARGS = {
    "num_channels": 16,
    "num_layers": 2,
    "num_heads": 2,
    "mlp_ratio": 2,
    "use_gradient_checkpointing": True,
}
TEMPLATE = "configs/templates/all_in_one_aux_dit.yaml"
CLOUD = "configs/experiments/all_in_one_mini_ddsp_cloud_aux_dit.yaml"


def activate_attention(model):
    # Zero adaLN gates hide attention/mask bugs in an untrained DiT.
    with torch.no_grad():
        for block in model.blocks:
            block.adaLN_modulation[1].weight.normal_(std=0.03)
            block.adaLN_modulation[1].bias.normal_(std=0.03)


def small_acoustic_config(diffusion_type="reflow"):
    config = set_hparams(config=TEMPLATE, print_hparams=False)
    config.update(
        hidden_size=16,
        enc_layers=1,
        num_heads=2,
        dropout=0.0,
        audio_num_mel_bins=8,
        use_lang_id=False,
        use_spk_id=False,
        use_key_shift_embed=False,
        use_speed_embed=False,
        use_energy_embed=False,
        use_breathiness_embed=False,
        use_voicing_embed=False,
        use_tension_embed=False,
        diffusion_type=diffusion_type,
        sampling_steps=2,
        timesteps=4,
        K_step=4,
        K_step_infer=2,
        diff_speedup=1,
        infer=False,
    )
    config["backbone_args"] = dict(SMALL_ARGS, time_embed_dim=16)
    config["shallow_diffusion_args"]["aux_decoder_args"] = SMALL_ARGS.copy()
    hparams.update(config)
    return config


def acoustic_inputs(device="cpu"):
    return {
        "txt_tokens": torch.tensor([[1, 2, 3], [1, 2, 3]], device=device),
        "mel2ph": torch.tensor(
            [[1, 1, 2, 2, 3, 0, 0], [1, 1, 2, 2, 3, 3, 3]], device=device
        ),
        "f0": torch.full((2, 7), 220.0, device=device),
    }


class AuxDiTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(1234)
        self.previous_hparams = copy.deepcopy(hparams)

    def tearDown(self):
        hparams.clear()
        hparams.update(self.previous_hparams)

    def test_factory_and_adaptor(self):
        self.assertIsInstance(build_aux_loss("dit"), nn.L1Loss)
        for features in (1, 2):
            with self.subTest(features=features):
                lower = [0.0] * 4 if features == 1 else [[0.0] * 4] * features
                upper = [2.0] * 4 if features == 1 else [[2.0] * 4] * features
                model = AuxDecoderAdaptor(
                    6, 4, features, lower, upper, "dit", SMALL_ARGS
                )
                model.eval()
                condition = torch.randn(2, 9, 6)
                raw = model(condition)
                result = model(condition, infer=True)
                shape = (2, 9, 4) if features == 1 else (2, 2, 9, 4)
                self.assertEqual(result.shape, shape)
                torch.testing.assert_close(result, raw + 1)
                torch.testing.assert_close(result, model(condition, infer=True))

    def test_padding_isolation_with_active_attention(self):
        model = build_aux_decoder(6, 4, "dit", SMALL_ARGS).eval()
        activate_attention(model)
        condition = torch.randn(2, 9, 6, requires_grad=True)
        mask = torch.tensor([[True] * 5 + [False] * 4, [True] * 9])
        output = model(condition, valid_mask=mask)
        changed = condition.detach().clone()
        changed[0, 5:] = 1000 * torch.randn(4, 6)
        torch.testing.assert_close(output, model(changed, valid_mask=mask))
        torch.testing.assert_close(output[0, :5], model(condition[:1, :5])[0])
        self.assertEqual(torch.count_nonzero(output[0, 5:]).item(), 0)
        output.square().sum().backward()
        self.assertEqual(torch.count_nonzero(condition.grad[0, 5:]).item(), 0)
        self.assertGreater(model.blocks[0].attn.qkv.weight.grad.abs().sum().item(), 0)

    def test_validation(self):
        for args, error in (
            ({"num_layer": 3}, TypeError),
            ({"num_layers": 0}, ValueError),
            ({"num_channels": 15, "num_heads": 2}, ValueError),
            ({"num_channels": 12, "num_heads": 4}, ValueError),
            ({"mlp_ratio": 0}, ValueError),
            ({"attention_dropout": 1.0}, ValueError),
            (None, ValueError),
        ):
            with self.subTest(args=args), self.assertRaises(error):
                build_aux_decoder(6, 4, "dit", args)
        model = build_aux_decoder(6, 4, "dit", SMALL_ARGS)
        for mask in (torch.ones(2, 5), torch.zeros(2, 5, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                model(torch.randn(2, 5, 6), valid_mask=mask)

    def test_checkpointing_gradients_and_reload(self):
        model = build_aux_decoder(6, 4, "dit", SMALL_ARGS).train()
        activate_attention(model)
        direct = copy.deepcopy(model)
        direct.use_gradient_checkpointing = False
        condition = torch.randn(2, 7, 6, requires_grad=True)
        other_condition = condition.detach().clone().requires_grad_()
        model(condition).square().mean().backward()
        direct(other_condition).square().mean().backward()
        torch.testing.assert_close(condition.grad, other_condition.grad)
        for (name, parameter), (_, other) in zip(
            model.named_parameters(), direct.named_parameters()
        ):
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            torch.testing.assert_close(parameter.grad, other.grad)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = build_aux_decoder(6, 4, "dit", SMALL_ARGS).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        model.eval()
        torch.testing.assert_close(restored(condition), model(condition))

    def test_acoustic_training_and_shallow_source(self):
        for diffusion_type in ("ddpm", "reflow"):
            with self.subTest(diffusion_type=diffusion_type):
                small_acoustic_config(diffusion_type)
                model = DiffSingerAcoustic(vocab_size=16, out_dims=8).train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                inputs = acoustic_inputs()
                target = torch.randn(2, 7, 8)
                before = (
                    model.aux_decoder.decoder.output_projection.weight.detach().clone()
                )
                for _ in range(2):
                    optimizer.zero_grad()
                    output = model(
                        **inputs, gt_mel=target, infer=False, return_clean=True
                    )
                    loss = (
                        build_aux_loss("dit")(
                            output.aux_out, model.aux_decoder.norm_spec(target)
                        )
                        + (output.diff_out[0] - output.diff_out[1]).square().mean()
                    )
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    optimizer.step()
                self.assertFalse(
                    torch.equal(
                        before, model.aux_decoder.decoder.output_projection.weight
                    )
                )
                self.assertGreater(
                    model.aux_decoder.decoder.blocks[0]
                    .attn.qkv.weight.grad.abs()
                    .sum()
                    .item(),
                    0,
                )
                sources = []

                def capture_source(module, args, kwargs):
                    sources.append(kwargs["src_spec"].detach().clone())

                handle = model.diffusion.register_forward_pre_hook(
                    capture_source, with_kwargs=True
                )
                model.eval()
                with torch.no_grad():
                    result = model(**inputs, infer=True)
                handle.remove()
                torch.testing.assert_close(sources[0], result.aux_out)
                self.assertEqual(result.diff_out.shape, target.shape)
                self.assertTrue(torch.isfinite(result.diff_out).all())
                self.assertEqual(torch.count_nonzero(result.aux_out[0, 5:]).item(), 0)

    def test_config_copies_and_preprocessing_reuse(self):
        for original, copied in (
            ("configs/templates/all_in_one_dit.yaml", TEMPLATE),
            ("data/all_in_one_mini_ddsp_cloud.yaml", CLOUD),
        ):
            with self.subTest(config=copied):
                baseline = set_hparams(config=original, print_hparams=False)
                if copied == CLOUD:
                    # The cloud copy captures the user's enabled local vocoder setting.
                    baseline["all_in_one"]["mini_ddsp"]["enabled"] = True
                expected = copy.deepcopy(baseline)
                original_signature = preprocessing_signature([], ["mel", "f0"])
                config = set_hparams(config=copied, print_hparams=False)
                expected["base_config"] = config["base_config"]
                for key in (
                    "hidden_size",
                    "enc_layers",
                    "backbone_type",
                    "backbone_args",
                    "dur_prediction_args",
                    "pitch_prediction_args",
                    "variances_prediction_args",
                ):
                    expected[key] = config[key]
                expected["shallow_diffusion_args"]["aux_decoder_arch"] = "dit"
                expected["shallow_diffusion_args"]["aux_decoder_args"] = config[
                    "shallow_diffusion_args"
                ]["aux_decoder_args"]
                self.assertEqual(config, expected)
                self.assertEqual(
                    original_signature, preprocessing_signature([], ["mel", "f0"])
                )
                self.assertEqual(config["binary_data_dir"], baseline["binary_data_dir"])
                self.assertEqual(config["backbone_type"], "dit")
                for key in ("pitch_prediction_args", "variances_prediction_args"):
                    self.assertEqual(config[key]["backbone_type"], "dit")
                    for setting in (
                        "repeat_bins",
                        "total_repeat_bins",
                        "pitd_norm_min",
                        "pitd_norm_max",
                    ):
                        self.assertEqual(
                            config[key].get(setting), baseline[key].get(setting)
                        )
                model = DiffSingerAllInOne(128, config["audio_num_mel_bins"])
                self.assertIsInstance(model.acoustic.aux_decoder.decoder, DiTAuxDecoder)
                self.assertEqual(
                    hasattr(model, "vocoder"),
                    config["all_in_one"]["mini_ddsp"]["enabled"],
                )

    def test_full_model_int8_weight_budget(self):
        for path in (TEMPLATE, CLOUD):
            with self.subTest(config=path):
                config = set_hparams(config=path, print_hparams=False)
                config["all_in_one"]["mini_ddsp"]["enabled"] = True
                hparams.update(config)
                model = DiffSingerAllInOne(128, config["audio_num_mel_bins"]).eval()
                hparams["sampling_steps"] = 2
                quantized = torch.ao.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
                storage = io.BytesIO()
                torch.save({"state_dict": quantized.state_dict()}, storage)
                parameters = sum(parameter.numel() for parameter in model.parameters())
                print(
                    f"{path}: {parameters} parameters; complete Linear INT8 + float resources = {storage.tell()} bytes"
                )
                self.assertLess(storage.tell(), 42_000_000)
                self.assertIn("vocoder.noise_basis", quantized.state_dict())
                storage.seek(0)
                quantized.load_state_dict(
                    torch.load(storage, weights_only=True)["state_dict"], strict=True
                )
                tokens = torch.tensor([[1, 2, 3]])
                mel2ph = torch.tensor([[1] * 4 + [2] * 4 + [3] * 4])
                languages = torch.ones_like(tokens)
                speaker = torch.zeros(1, dtype=torch.long)
                with torch.no_grad():
                    _, pitch, variances = quantized.variance(
                        tokens,
                        midi=torch.full_like(tokens, 60),
                        ph2word=torch.tensor([[1, 1, 2]]),
                        ph_dur=torch.tensor([[4, 4, 4]]),
                        word_dur=torch.tensor([[8, 4]]),
                        mel2ph=mel2ph,
                        note_midi=torch.tensor([[60.0, 62.0]]),
                        note_rest=torch.zeros(1, 2, dtype=torch.bool),
                        note_dur=torch.tensor([[8, 4]]),
                        mel2note=torch.tensor([[1] * 8 + [2] * 4]),
                        base_pitch=torch.full((1, 12), 60.0),
                        spk_id=speaker,
                        languages=languages,
                        infer=True,
                    )
                    f0 = 440.0 * 2 ** ((60.0 + pitch) / 12 - 69.0 / 12)
                    mel = quantized.acoustic(
                        tokens,
                        mel2ph=mel2ph,
                        f0=f0,
                        key_shift=torch.zeros(1, 1),
                        speed=torch.ones(1, 1),
                        spk_embed_id=speaker,
                        languages=languages,
                        **variances,
                        infer=True,
                    ).diff_out
                    waveform = quantized.vocoder(mel, f0)
                self.assertEqual(waveform.shape, (1, 12 * config["hop_size"]))
                self.assertTrue(torch.isfinite(waveform).all())

    def test_fs2_aux_deployment_view_dynamic_frames(self):
        small_acoustic_config()
        native = DiffSingerAcoustic(16, 8).eval()
        activate_attention(native.aux_decoder.decoder)
        deployment = DiffSingerAcousticONNX(16, 8, cross_lingual_token_idx=[]).eval()
        deployment.load_state_dict(native.state_dict(), strict=True)
        view = deployment.view_as_fs2_aux()
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            path = Path(directory) / "fs2_aux.onnx"
            tokens = torch.tensor([[1, 2, 3]])
            durations = torch.tensor([[3, 2, 4]])
            export_onnx(
                view,
                (tokens, durations, torch.full((1, 9), 220.0), {}),
                str(path),
                input_names=["tokens", "durations", "f0"],
                output_names=["condition", "aux_mel"],
                dynamic_axes={
                    "tokens": {1: "tokens_count"},
                    "durations": {1: "tokens_count"},
                    "f0": {1: "frames"},
                    "condition": {1: "frames"},
                    "aux_mel": {1: "frames"},
                },
                opset_version=15,
            )
            onnx.checker.check_model(onnx.load(path))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            for durations in (torch.tensor([[1, 2, 3]]), torch.tensor([[7, 5, 8]])):
                frames = int(durations.sum())
                f0 = torch.full((1, frames), 220.0)
                actual = session.run(
                    None,
                    {
                        "tokens": tokens.numpy(),
                        "durations": durations.numpy(),
                        "f0": f0.numpy(),
                    },
                )
                mel2ph = torch.repeat_interleave(torch.arange(1, 4), durations[0])[None]
                expected_condition = native.fs2(tokens, mel2ph, f0)
                expected_aux = native.aux_decoder(expected_condition, infer=True)
                np.testing.assert_allclose(
                    actual[0], expected_condition.numpy(), rtol=1e-4, atol=1e-5
                )
                np.testing.assert_allclose(
                    actual[1], expected_aux.numpy(), rtol=1e-4, atol=1e-5
                )

    def test_opset15_dynamic_frames_and_int8_runtime(self):
        small_acoustic_config()
        native = DiffSingerAcoustic(16, 8).eval()
        activate_attention(native.aux_decoder.decoder)
        deployment = DiffSingerAcousticONNX(16, 8, cross_lingual_token_idx=[]).eval()
        deployment.load_state_dict(native.state_dict(), strict=True)
        self.assertIsInstance(deployment.aux_decoder.decoder.blocks[0], DiTBlockONNX)
        decoder = deployment.aux_decoder
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            path = Path(directory) / "aux.onnx"
            export_onnx(
                decoder,
                (torch.randn(1, 9, 16), True),
                str(path),
                input_names=["condition"],
                output_names=["mel"],
                dynamic_axes={"condition": {1: "frames"}, "mel": {1: "frames"}},
                opset_version=15,
            )
            onnx.checker.check_model(onnx.load(path))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            for frames in (1, 7, 16):
                condition = torch.randn(1, frames, 16)
                expected = native.aux_decoder(condition, infer=True)
                actual = session.run(None, {"condition": condition.numpy()})[0]
                np.testing.assert_allclose(
                    actual, expected.numpy(), rtol=1e-4, atol=1e-5
                )

            # Quantize standard Linear weights while keeping attention/normalization float.
            quantized = torch.ao.quantization.quantize_dynamic(
                native.aux_decoder, {nn.Linear}, dtype=torch.qint8
            )
            self.assertFalse(
                any(type(module) is nn.Linear for module in quantized.modules())
            )
            condition = torch.randn(2, 13, 16)
            mask = torch.tensor([[True] * 8 + [False] * 5, [True] * 13])
            expected = native.aux_decoder(condition, valid_mask=mask)
            actual = quantized(condition, valid_mask=mask)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertLess((actual - expected).abs().max().item(), 0.1)
            self.assertEqual(torch.count_nonzero(actual[0, 8:]).item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_bfloat16(self):
        small_acoustic_config()
        model = DiffSingerAcoustic(16, 8).train().cuda()
        inputs = acoustic_inputs("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                **inputs, gt_mel=torch.randn(2, 7, 8, device="cuda"), infer=False
            )
            loss = (
                output.aux_out.float().square().mean()
                + output.diff_out[0].float().square().mean()
            )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, parameter in model.aux_decoder.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(**inputs, infer=True)
        self.assertTrue(torch.isfinite(result.diff_out).all())


if __name__ == "__main__":
    unittest.main()
