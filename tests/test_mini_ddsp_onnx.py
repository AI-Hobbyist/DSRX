import pathlib
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch

from deployment.exporters.all_in_one_exporter import export_vocoder_graph
from deployment.exporters.onnx_export import export_onnx
from deployment.exporters.all_in_one_graph import (
    DurationControls,
    PitchControls,
    SingingControls,
)
from deployment.modules.fastspeech2 import LengthRegulator
from modules.fastspeech.tts_modules import RhythmRegulator
from modules.vocoders.mini_ddsp import MiniDDSP
from tests.test_mini_ddsp import config


class MiniDDSPOnnxTests(unittest.TestCase):
    def test_manual_duration_does_not_require_valid_automatic_predictions(self):
        manual = torch.tensor([[2, 3]])
        predicted, actual = DurationControls()(
            torch.zeros(1, 2),
            torch.tensor([[2]]),
            torch.tensor([[5]]),
            torch.tensor([[5]]),
            manual,
            torch.tensor(True),
        )
        torch.testing.assert_close(predicted, torch.zeros(1, 2, dtype=torch.long))
        torch.testing.assert_close(actual, manual)
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "manual_duration.onnx"
            args = (
                torch.ones(1, 2),
                torch.tensor([[2]]),
                torch.tensor([[5]]),
                torch.tensor([[5]]),
                manual,
                torch.tensor(True),
            )
            names = ["raw", "word_div", "word_dur", "note_dur", "manual", "override"]
            export_onnx(
                DurationControls(),
                args,
                path,
                input_names=names,
                output_names=["predicted", "actual"],
                opset_version=15,
            )
            values = {name: value.numpy() for name, value in zip(names, args)}
            values["raw"] = np.zeros((1, 2), np.float32)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(path), options)
            _, actual = session.run(None, values)
            np.testing.assert_array_equal(actual, manual.numpy())

    def test_manual_controls_and_native_duration_normalization(self):
        raw = torch.tensor([[3.0, 1.0, 2.0, 4.0]])
        word_div = torch.tensor([[2, 2]])
        word_dur = torch.tensor([[3, 6]])
        note_dur = torch.tensor([[2, 3, 4]])
        manual = torch.tensor([[1, 2, 3, 3]])
        predicted, actual = DurationControls()(
            raw, word_div, word_dur, note_dur, manual, torch.tensor(False)
        )
        expected = RhythmRegulator()(raw, torch.tensor([[1, 1, 2, 2]]), word_dur)
        torch.testing.assert_close(predicted, expected)
        torch.testing.assert_close(actual, expected)
        _, actual = DurationControls()(
            raw, word_div, word_dur, note_dur, manual, torch.tensor(True)
        )
        torch.testing.assert_close(actual, manual)
        mask = torch.tensor([[True, False, True, False]])
        manual_f0 = torch.tensor([[220.0, 0.0, 220.0, 330.0]])
        pitch, f0 = PitchControls()(
            torch.full((1, 4), 69.0), torch.full((1, 4), 60.0), manual_f0, mask
        )
        torch.testing.assert_close(f0, torch.tensor([[440.0, 0.0, 440.0, 330.0]]))
        torch.testing.assert_close(pitch, torch.tensor([[69.0, 60.0, 69.0, 60.0]]))
        first, second = SingingControls()(
            (torch.ones(1, 4), torch.ones(1, 4) * 2),
            (torch.zeros(1, 4), torch.zeros(1, 4)),
            torch.stack((mask, ~mask), dim=2),
        )
        torch.testing.assert_close(first, mask.float())
        torch.testing.assert_close(second, (~mask).float() * 2)

    def test_length_regulator_single_token_and_zero_durations(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "length_regulator.onnx"
            model = LengthRegulator()
            export_onnx(
                model,
                (torch.tensor([[2, 1, 3]]),),
                path,
                input_names=["durations"],
                output_names=["alignment"],
                dynamic_axes={
                    "durations": {0: "batch", 1: "tokens"},
                    "alignment": {0: "batch", 1: "frames"},
                },
                opset_version=15,
            )
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(path), options)
            for durations in ([[1]], [[0, 1]], [[2, 0, 3]], [[1, 1], [0, 2]]):
                values = torch.tensor(durations)
                actual = session.run(None, {"durations": values.numpy()})[0]
                np.testing.assert_array_equal(actual, model(values).numpy())

    def test_opset15_dynamic_rates_lengths_and_batches(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as temporary:
            for rate in (24000, 44100, 48000):
                model = MiniDDSP(config(rate)).eval()
                path = pathlib.Path(temporary) / f"vocoder_{rate}.onnx"
                export_vocoder_graph(model, path)
                graph = onnx.load(path)
                onnx.checker.check_model(graph)
                self.assertTrue(
                    all(not value.external_data for value in graph.graph.initializer)
                )
                self.assertTrue(
                    any(
                        "noise_basis" in value.name for value in graph.graph.initializer
                    )
                )
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                session = ort.InferenceSession(
                    str(path), sess_options=options, providers=["CPUExecutionProvider"]
                )
                for frames in (1, 9, 10, 11, 257, 1025):
                    with self.subTest(sample_rate=rate, frames=frames):
                        mel = torch.randn(2, frames, 8) - 6
                        f0 = torch.linspace(120, 430, frames)[None, :].repeat(2, 1)
                        f0[1] = 0
                        actual, lengths = session.run(
                            None, {"mel": mel.numpy(), "f0": f0.numpy()}
                        )
                        with torch.no_grad():
                            expected = model(mel, f0).numpy()
                        self.assertEqual(actual.shape, (2, frames * 16))
                        np.testing.assert_array_equal(lengths, [frames * 16] * 2)
                        np.testing.assert_allclose(
                            actual, expected, rtol=2e-4, atol=2e-5
                        )


if __name__ == "__main__":
    unittest.main()
