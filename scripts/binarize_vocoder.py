"""Prepare an audio-only indexed Mini-DDSP dataset."""

import argparse
import os
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parents[1]
os.environ["PYTHONPATH"] = str(root_dir)
sys.path.insert(0, str(root_dir))

from utils.hparams import hparams, set_hparams

set_hparams()


def binarize():
    from preprocessing.vocoder_binarizer import VocoderBinarizer

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume", action="store_true")
    args, _ = parser.parse_known_args()
    if args.resume:
        hparams.setdefault("binarization_args", {})["resume"] = True
    VocoderBinarizer().process()


if __name__ == "__main__":
    binarize()
