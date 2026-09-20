import importlib
import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

from utils.hparams import set_hparams, hparams
from utils.aux_dataset import (
    AUX_MODULES,
    build_aux_datasets,
    get_aux_binary_data_dir,
    parse_aux_dataset_config
)

set_hparams()


def binarize():
    binarizer_name = hparams["binarizer_cls"]
    pkg = ".".join(binarizer_name.split(".")[:-1])
    cls_name = binarizer_name.split(".")[-1]
    binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
    enabled_modules = {
        name for name in AUX_MODULES
        if hparams.get(f'predict_{name}', False)
    }
    aux_config = parse_aux_dataset_config(
        hparams.get('aux_datasets', {}), enabled_modules=enabled_modules
    )
    if aux_config is None:
        aux_datasets = None
    elif binarizer_name != 'preprocessing.variance_binarizer.VarianceBinarizer':
        raise ValueError('aux_datasets is only supported by the variance binarizer.')
    else:
        aux_datasets = build_aux_datasets(aux_config)

    print("| Binarizer: ", binarizer_cls)
    binarizer_cls().process()
    if aux_datasets is None:
        return
    aux_binary_data_dir = get_aux_binary_data_dir(hparams['binary_data_dir'])
    print('| Auxiliary binary data dir: ', aux_binary_data_dir)
    binarizer_cls(
        datasets=aux_datasets,
        binary_data_dir=aux_binary_data_dir
    ).process()


if __name__ == '__main__':
    binarize()
