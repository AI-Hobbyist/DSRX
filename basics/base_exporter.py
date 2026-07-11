import json
import pathlib
import shutil
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from utils.hparams import hparams
from utils.precision import (
    PrecisionContext,
    resolve_dtype,
    VALID_EXPORT_PRECISIONS,
)


class BaseExporter:
    def __init__(
            self,
            device: Union[str, torch.device] = None,
            cache_dir: Path = None,
            precision: str = 'fp32',
            quantize: bool = False,
            **kwargs
    ):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir: Path = cache_dir.resolve() if cache_dir is not None \
            else Path(__file__).parent.parent / 'deployment' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Precision settings
        if precision == 'all':
            self.export_precisions = ['fp32', 'fp16', 'bf16']
        elif precision in VALID_EXPORT_PRECISIONS:
            self.export_precisions = [precision]
        else:
            raise ValueError(
                f"Invalid precision '{precision}'. "
                f"Valid: {sorted(VALID_EXPORT_PRECISIONS)}"
            )
        self.export_quantize = quantize
        self._precision_ctx = PrecisionContext('fp32', device_type=self.device.type)

    def _get_precision_suffix(self, precision: str) -> str:
        """Return file suffix for a given precision (e.g., '_fp16')."""
        if precision == 'fp32':
            return ''
        return f'_{precision}'

    def _convert_model_to_precision(self, model: nn.Module, precision: str) -> nn.Module:
        """Convert model to target precision for export."""
        if precision == 'fp32':
            return model
        elif precision == 'fp16':
            return model.half()
        elif precision == 'bf16':
            return model.to(torch.bfloat16)
        else:
            raise ValueError(f"Unsupported export precision: {precision}")

    @staticmethod
    def _convert_onnx_to_fp16(onnx_model, keep_io_types: bool = True):
        """Convert float32 ONNX model to float16."""
        import onnx
        from onnxconverter_common import float16
        return float16.convert_float_to_float16(onnx_model, keep_io_types=keep_io_types)

    @staticmethod
    def _quantize_onnx_static(model_path: str, output_path: str, calibration_reader=None):
        """Apply static int8 quantization to an ONNX model."""
        from onnxruntime.quantization import quantize_static, QuantType
        if calibration_reader is None:
            from utils.calibration import create_calibration_reader
            calibration_reader = create_calibration_reader()
        quantize_static(
            model_input=model_path,
            model_output=output_path,
            calibration_data_reader=calibration_reader,
            quant_format=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
        )

    @staticmethod
    def _quantize_onnx_dynamic(model_path: str, output_path: str):
        """Apply dynamic int8 quantization to an ONNX model."""
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )

    # noinspection PyMethodMayBeStatic
    def build_spk_map(self) -> dict:
        if hparams['use_spk_id']:
            with open(Path(hparams['work_dir']) / 'spk_map.json', 'r', encoding='utf8') as f:
                spk_map = json.load(f)
            assert isinstance(spk_map, dict) and len(spk_map) > 0, 'Invalid or empty speaker map!'
            assert len(spk_map) == len(set(spk_map.values())), 'Duplicate speaker id in speaker map!'
            return spk_map
        else:
            return {}

    # noinspection PyMethodMayBeStatic
    def build_lang_map(self) -> dict:
        lang_map_fn = pathlib.Path(hparams['work_dir']) / 'lang_map.json'
        if lang_map_fn.exists():
            with open(lang_map_fn, 'r', encoding='utf8') as f:
                lang_map = json.load(f)
            assert isinstance(lang_map, dict) and len(lang_map) > 0, 'Invalid or empty language map!'
            assert len(lang_map) == len(set(lang_map.values())), 'Duplicate language id in language map!'
            return lang_map
        else:
            return {}

    def build_model(self) -> nn.Module:
        """
        Creates an instance of nn.Module and load its state dict on the target device.
        """
        raise NotImplementedError()

    def export_model(self, path: Path):
        """
        Exports the model to ONNX format.
        :param path: the target model path
        """
        raise NotImplementedError()

    # noinspection PyMethodMayBeStatic
    def export_dictionaries(self, path: Path):
        dicts = hparams.get('dictionaries')
        if dicts is not None:
            for lang in dicts.keys():
                fn = f'dictionary-{lang}.txt'
                shutil.copy(pathlib.Path(hparams['work_dir']) / fn, path)
                print(f'| export dictionary => {path / fn}')
        else:
            fn = 'dictionary.txt'
            shutil.copy(pathlib.Path(hparams['work_dir']) / fn, path)
            print(f'| export dictionary => {path / fn}')

    def export_attachments(self, path: Path):
        """
        Exports related files and configs (e.g. the dictionary) to the target directory.
        :param path: the target directory
        """
        raise NotImplementedError()

    def export(self, path: Path):
        """
        Exports all the artifacts to the target directory.
        :param path: the target directory
        """
        raise NotImplementedError()
