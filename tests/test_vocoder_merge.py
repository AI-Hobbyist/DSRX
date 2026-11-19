"""Checkpoint tensor replacement and its confirmed compatibility error paths."""

import copy
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from scripts.merge_vocoder import merge_experiment
from utils.hparams import set_hparams
from utils.mini_ddsp_checkpoint import mini_ddsp_contract, merge_vocoder_weights


class VocoderMergeTests(unittest.TestCase):
    def setUp(self):
        self.config = set_hparams(
            config="configs/vocoder/mini_ddsp_train.yaml",
            print_hparams=False,
            global_hparams=False,
        )
        self.config.update(
            dictionaries={"zh": "unused"}, use_spk_id=True, use_lang_id=True
        )
        self.aio = {
            "category": "all_in_one",
            "global_step": 18000,
            "optimizer_states": [{"stale": True}],
            "state_dict": {
                "model.acoustic.weight": torch.tensor([1.0]),
                "model.variance.weight": torch.tensor([2.0]),
                "model.vocoder.control.weight": torch.zeros(2, 3),
                "model.vocoder.synthesis_version": torch.tensor(3),
            },
        }
        self.donor = {
            "category": "vocoder",
            "global_step": 20000,
            "mini_ddsp_contract": mini_ddsp_contract(self.config),
            "state_dict": {
                "model.control.weight": torch.ones(2, 3),
                "model.synthesis_version": torch.tensor(3),
            },
        }

    def test_replaces_only_vocoder_and_omits_training_state(self):
        merged = merge_vocoder_weights(self.aio, self.donor, self.config)
        for name, original in self.aio["state_dict"].items():
            if name.startswith("model.vocoder."):
                expected = self.donor["state_dict"][
                    name.replace("model.vocoder.", "model.", 1)
                ]
            else:
                expected = original
            self.assertTrue(torch.equal(merged["state_dict"][name], expected), name)
        self.assertTrue(
            torch.equal(
                self.aio["state_dict"]["model.vocoder.control.weight"],
                torch.zeros(2, 3),
            )
        )
        self.assertNotIn("optimizer_states", merged)
        self.assertEqual(merged["global_step"], 0)

    def test_rejects_incompatible_contract_keys_shapes_and_dtype(self):
        bad = copy.deepcopy(self.donor)
        bad["mini_ddsp_contract"]["hop_size"] += 1
        with self.assertRaisesRegex(ValueError, "contract"):
            merge_vocoder_weights(self.aio, bad, self.config)
        bad = copy.deepcopy(self.donor)
        del bad["state_dict"]["model.synthesis_version"]
        with self.assertRaisesRegex(ValueError, "state keys"):
            merge_vocoder_weights(self.aio, bad, self.config)
        bad = copy.deepcopy(self.donor)
        bad["state_dict"]["model.control.weight"] = torch.ones(3, 2)
        with self.assertRaisesRegex(ValueError, "shape"):
            merge_vocoder_weights(self.aio, bad, self.config)
        bad["state_dict"]["model.control.weight"] = torch.ones(
            2, 3, dtype=torch.float16
        )
        with self.assertRaisesRegex(ValueError, "dtype"):
            merge_vocoder_weights(self.aio, bad, self.config)

    def test_packages_inference_assets_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aio_dir = root / "aio"
            aio_dir.mkdir()
            aio_path = aio_dir / "model_ckpt_steps_18000.ckpt"
            torch.save(self.aio, aio_path)
            donor_path = root / "vocoder.ckpt"
            torch.save(self.donor, donor_path)
            (aio_dir / "config.yaml").write_text(
                yaml.safe_dump(self.config), encoding="utf-8"
            )
            assets = {
                "dictionary-zh.txt": "a aa\n",
                "spk_map.json": '{"singer": 0}',
                "lang_map.json": '{"zh": 0}',
            }
            for name, contents in assets.items():
                (aio_dir / name).write_text(contents, encoding="utf-8")
            output = merge_experiment(aio_path, donor_path, root / "merged")
            merged = torch.load(output, weights_only=False)
            self.assertEqual(merged["category"], "all_in_one")
            self.assertTrue(
                torch.equal(
                    merged["state_dict"]["model.vocoder.control.weight"],
                    torch.ones(2, 3),
                )
            )
            for name, contents in assets.items():
                self.assertEqual(
                    (output.parent / name).read_text(encoding="utf-8"), contents
                )
            config = yaml.safe_load(
                (output.parent / "config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(config["base_config"], [])
            self.assertFalse(config["finetune_enabled"])
            with self.assertRaises(FileExistsError):
                merge_experiment(aio_path, donor_path, output.parent)


if __name__ == "__main__":
    unittest.main()
