import pathlib

from preprocessing.acoustic_binarizer import AcousticBinarizer
from preprocessing.variance_binarizer import VarianceBinarizer
from utils.hparams import hparams


class AllInOneBinarizer:
    def process(self):
        binary_data_dir = pathlib.Path(hparams["binary_data_dir"])
        # Waveform supervision is prepared by scripts/binarize_vocoder.py.
        # Enabling the built-in vocoder must not change feature augmentation.
        AcousticBinarizer(binary_data_dir=binary_data_dir / "acoustic").process()
        VarianceBinarizer(binary_data_dir=binary_data_dir / "variance").process()
