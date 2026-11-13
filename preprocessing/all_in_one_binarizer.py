import pathlib
import tempfile

from preprocessing.acoustic_binarizer import AcousticBinarizer
from preprocessing.variance_binarizer import VarianceBinarizer
from utils.hparams import hparams, mini_ddsp_config


class AllInOneBinarizer:
    def process(self):
        binary_data_dir = pathlib.Path(hparams["binary_data_dir"])
        if mini_ddsp_config(hparams).get("enabled", False):
            from preprocessing.vocoder_binarizer import VocoderBinarizer

            binary_data_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".mini-ddsp-", dir=binary_data_dir
            ) as cache:
                hparams["_mini_ddsp_cache_dir"] = cache
                try:
                    AcousticBinarizer(
                        binary_data_dir=binary_data_dir / "acoustic"
                    ).process()
                    VarianceBinarizer(
                        binary_data_dir=binary_data_dir / "variance"
                    ).process()
                    VocoderBinarizer(
                        binary_data_dir=binary_data_dir / "vocoder"
                    ).process()
                finally:
                    hparams.pop("_mini_ddsp_cache_dir", None)
            return
        AcousticBinarizer(binary_data_dir=binary_data_dir / "acoustic").process()
        VarianceBinarizer(binary_data_dir=binary_data_dir / "variance").process()
