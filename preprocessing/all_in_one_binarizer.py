import pathlib

from preprocessing.acoustic_binarizer import AcousticBinarizer
from preprocessing.variance_binarizer import VarianceBinarizer
from utils.hparams import hparams, mini_ddsp_config


class AllInOneBinarizer:
    def process(self):
        binary_data_dir = pathlib.Path(hparams["binary_data_dir"])
        print("| Preprocessing stage 1/3: acoustic", flush=True)
        AcousticBinarizer(binary_data_dir=binary_data_dir / "acoustic").process()
        print("| Preprocessing stage 2/3: variance", flush=True)
        VarianceBinarizer(binary_data_dir=binary_data_dir / "variance").process()
        if mini_ddsp_config(hparams).get("enabled", False):
            from preprocessing.vocoder_binarizer import VocoderBinarizer

            print("| Preprocessing stage 3/3: vocoder", flush=True)
            VocoderBinarizer().process()
        else:
            print(
                "| Vocoder preprocessing disabled: all_in_one.mini_ddsp.enabled=false",
                flush=True,
            )
