"""Prepare an audio-only indexed Mini-DDSP dataset."""

import os
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parents[1]
os.environ["PYTHONPATH"] = str(root_dir)
sys.path.insert(0, str(root_dir))

from utils.hparams import set_hparams

set_hparams()


def binarize():
    from preprocessing.vocoder_binarizer import VocoderBinarizer

    VocoderBinarizer().process()


if __name__ == "__main__":
    binarize()
