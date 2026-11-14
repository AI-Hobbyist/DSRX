import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from basics.base_binarizer import BaseBinarizer
from preprocessing.mini_ddsp_cache import pairing_id
from preprocessing.vocoder_binarizer import VocoderBinarizer
from utils.hparams import hparams


class FeatureHarness(BaseBinarizer):
    def __init__(self, directory, sources, fail_on=None):
        self.binary_data_dir = directory
        self.sources = sources
        self.fail_on = fail_on
        self.calls = []
        self.data_attrs = ["tokens", "mel", "f0", "sample_id", "augmentation_id"]
        self.spk_map = {"speaker": 0}
        self.binarization_args = hparams["binarization_args"]

    def meta_data_iterator(self, prefix):
        yield from self.sources

    def process_item(self, name, metadata, settings):
        if name == self.fail_on:
            raise KeyboardInterrupt()
        self.calls.append(name)
        identity, augmentation = pairing_id(f"speaker:{name.split(':')[1]}")
        return {
            "name": name,
            "spk_id": 0,
            "spk_name": "speaker",
            "ph_text": "a b",
            "seconds": 0.02,
            "length": 2,
            "tokens": np.array([1, 2]),
            "mel": np.full((2, 3), float(ord(name[-1])), np.float32),
            "f0": np.full(2, 220, np.float32),
            "sample_id": np.bytes_(identity),
            "augmentation_id": np.bytes_(augmentation),
        }

    def arrange_data_augmentation(self, iterator):
        return {
            name: [{"func": self.augment, "kwargs": {"shift": random.uniform(-3, 3)}}]
            for name, _ in iterator
        }

    def augment(self, item, shift):
        result = copy.deepcopy(item)
        result["mel"] += shift
        result["augmentation_id"] = np.bytes_(pairing_id("", key_shift=shift)[1])
        return result


class PreprocessingResumeTests(unittest.TestCase):
    def setUp(self):
        self.previous = copy.deepcopy(hparams)
        self.rng_state = random.getstate()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = []
        for name in "abc":
            path = self.root / f"{name}.wav"
            path.touch()
            self.sources.append(
                (f"0:{name}", {"wav_fn": str(path), "spk_name": "speaker"})
            )
        hparams.clear()
        hparams.update(
            {
                "binary_data_dir": str(self.root / "vocoder-output"),
                "datasets": [],
                "audio_sample_rate": 44100,
                "binarization_args": {"num_workers": 0, "checkpoint_interval": 1},
            }
        )

    def tearDown(self):
        hparams.clear()
        hparams.update(self.previous)
        random.setstate(self.rng_state)
        self.temporary.cleanup()

    def test_feature_resume_preserves_augmentation_and_rolls_back_uncommitted_tail(
        self,
    ):
        random.seed(123)
        clean = FeatureHarness(self.root / "clean", self.sources)
        clean.process_dataset("train", apply_augmentation=True)
        random.seed(123)
        interrupted = FeatureHarness(self.root / "partial", self.sources, fail_on="0:b")
        with self.assertRaises(KeyboardInterrupt):
            interrupted.process_dataset("train", apply_augmentation=True)
        self.assertEqual(interrupted.calls, ["0:a"])
        with h5py.File(self.root / "partial/train.data", "a") as source:
            source.create_dataset("2/uncommitted", data=np.ones(1))
        hparams["binarization_args"]["resume"] = True
        resumed = FeatureHarness(self.root / "partial", self.sources)
        resumed.process_dataset("train", apply_augmentation=True)
        self.assertEqual(resumed.calls, ["0:b", "0:c"])
        with h5py.File(self.root / "clean/train.data", "r") as clean_data, h5py.File(
            self.root / "partial/train.data", "r"
        ) as resumed_data:
            self.assertEqual(set(clean_data), set(resumed_data))
            for index in clean_data:
                self.assertEqual(set(clean_data[index]), set(resumed_data[index]))
                for key in clean_data[index]:
                    np.testing.assert_array_equal(
                        clean_data[index][key][()], resumed_data[index][key][()]
                    )

    def test_completed_legacy_features_are_reused_without_extraction(self):
        harness = FeatureHarness(self.root / "complete", self.sources)
        harness.process_dataset("valid")
        (self.root / "complete/valid.complete.json").unlink()
        hparams["binarization_args"]["resume"] = True
        with patch.object(
            harness, "process_item", side_effect=AssertionError("recomputed")
        ):
            harness.process_dataset("valid")
        self.assertTrue((self.root / "complete/valid.complete.json").exists())

    def test_changed_preprocessing_config_is_rejected(self):
        harness = FeatureHarness(self.root / "complete", self.sources)
        harness.process_dataset("valid")
        hparams["binarization_args"]["resume"] = True
        hparams["audio_sample_rate"] = 48000
        with self.assertRaisesRegex(ValueError, "inputs/config changed"):
            harness.process_dataset("valid")

    def test_vocoder_resumes_only_remaining_audio(self):
        sources = [
            {
                "path": metadata["wav_fn"],
                "sample_id": pairing_id(f"speaker:{name.split(':')[1]}")[0],
                "augmentation_id": pairing_id("")[1],
                "spk_id": 0,
            }
            for name, metadata in self.sources
        ]

        def extracted(source):
            return {
                "sample_id": np.bytes_(source["sample_id"]),
                "augmentation_id": np.bytes_(source["augmentation_id"]),
                "length": 2,
                "spk_id": 0,
                "waveform": np.ones(1024, np.float32),
                "mel": np.zeros((2, 3), np.float32),
                "f0": np.full(2, 220, np.float32),
                "uv": np.zeros(2, bool),
                "amplitude": np.ones((2, 4), np.float32),
            }

        vocoder = VocoderBinarizer()
        with patch.object(
            vocoder,
            "process_item",
            side_effect=[extracted(sources[0]), KeyboardInterrupt()],
        ):
            with self.assertRaises(KeyboardInterrupt):
                vocoder.process_dataset("train", sources)
        hparams["binarization_args"]["resume"] = True
        with patch.object(vocoder, "process_item", side_effect=extracted) as process:
            vocoder.process_dataset("train", sources)
        self.assertEqual(
            [call.args[0]["path"] for call in process.call_args_list],
            [source["path"] for source in sources[1:]],
        )
        with patch.object(
            vocoder, "process_item", side_effect=AssertionError("recomputed")
        ):
            vocoder.process_dataset("train", sources)


if __name__ == "__main__":
    unittest.main()
