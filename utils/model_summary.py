from collections import OrderedDict

import torch.nn as nn


def count_unique_params(parameters):
    seen = set()
    total = 0
    trainable = 0
    for param in parameters:
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return total, trainable, total - trainable


def count_module_params(module: nn.Module):
    return count_unique_params(module.parameters())


def find_lora_params(module: nn.Module):
    for name, param in module.named_parameters():
        if 'lora_' in name:
            yield param


def format_param_count(value: int):
    if value >= 1_000_000:
        return f'{value / 1_000_000:.3f}M'
    if value >= 1_000:
        return f'{value / 1_000:.3f}K'
    return str(value)


def print_param_summary(rows, title='Parameter Summary'):
    print(f'| {title}:')
    header = ('module', 'total', 'trainable', 'frozen', 'percent')
    widths = [16, 14, 14, 14, 10]
    print('| ' + ' '.join(h.ljust(w) for h, w in zip(header, widths)))
    print('| ' + ' '.join('-' * w for w in widths))
    total_params = next((row[1] for row in rows if row[0] == 'total'), 0)
    for name, total, trainable, frozen in rows:
        percent = '0.00%'
        if total_params > 0:
            percent = f'{total / total_params * 100:.2f}%'
        values = (
            name,
            format_param_count(total),
            format_param_count(trainable),
            format_param_count(frozen),
            percent,
        )
        print('| ' + ' '.join(v.ljust(w) for v, w in zip(values, widths)))


def build_module_param_summary(model: nn.Module, named_modules: OrderedDict):
    rows = []
    assigned_params = set()
    for name, module in named_modules.items():
        params = list(module.parameters())
        total, trainable, frozen = count_unique_params(params)
        rows.append((name, total, trainable, frozen))
        assigned_params.update(id(param) for param in params)

    other_params = [
        param
        for param in model.parameters()
        if id(param) not in assigned_params
    ]
    other = count_unique_params(other_params)
    if other[0] > 0:
        rows.append(('other', *other))

    lora = count_unique_params(find_lora_params(model))
    if lora[0] > 0:
        rows.append(('lora', *lora))

    total = count_module_params(model)
    rows.append(('total', *total))
    return rows
