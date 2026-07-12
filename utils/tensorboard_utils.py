import re


def sanitize_tb_tag_part(value, fallback='item'):
    value = re.sub(r'[^0-9A-Za-z._-]+', '_', str(value)).strip('._-')
    return value or fallback


def flat_tb_tag(prefix, index):
    return f'{sanitize_tb_tag_part(prefix, "metric")}_{int(index)}'


def grouped_tb_tag(prefix, speaker, item):
    return (
        f'{sanitize_tb_tag_part(prefix, "metric")}_'
        f'{sanitize_tb_tag_part(speaker, "spk")}/'
        f'{sanitize_tb_tag_part(item, "item")}'
    )


def validation_tb_tag(layout, prefix, index, speaker=None, item=None):
    if layout == 'grouped':
        return grouped_tb_tag(prefix, speaker, item)
    return flat_tb_tag(prefix, index)
