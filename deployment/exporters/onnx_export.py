import inspect

import torch


def export_onnx(*args, **kwargs):
    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
        kwargs['dynamo'] = False
    return torch.onnx.export(*args, **kwargs)