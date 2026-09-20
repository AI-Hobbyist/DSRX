import sys
import json
import random
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.acoustic_task import AcousticTask
from training.all_in_one_task import AllInOneTask
from training.variance_task import VarianceTask
from preprocessing import all_in_one_binarizer
from inference.ds_acoustic import DiffSingerAcousticInfer
from inference.ds_variance import DiffSingerVarianceInfer
from utils.hparams import hparams, set_hparams
from utils.variance_validation import (
    load_validation_sources, prepare_variance_segment,
    validate_variance_validation_config
)


def configure_smoke_model():
    set_hparams('configs/templates/all_in_one_dit.yaml', print_hparams=False)
    hparams['hidden_size'] = 16
    hparams['audio_num_mel_bins'] = 4
    hparams['spec_min'] = [-12]
    hparams['spec_max'] = [0]
    hparams['enc_layers'] = 1
    hparams['num_heads'] = 2
    hparams['use_shallow_diffusion'] = False
    hparams['backbone_args'] = {
        'num_layers': 1,
        'num_channels': 16,
        'num_heads': 2,
        'mlp_ratio': 2,
        'time_embed_dim': 16,
        'use_gradient_checkpointing': False
    }
    for key in ('pitch_prediction_args', 'variances_prediction_args'):
        hparams[key]['backbone_type'] = 'dit'
        hparams[key]['backbone_args'] = {
            'num_layers': 1,
            'num_channels': 16,
            'num_heads': 2,
            'mlp_ratio': 2,
            'time_embed_dim': 16,
            'use_gradient_checkpointing': False
        }


def acoustic_sample():
    frames = 4
    return {
        'size': 1,
        'tokens': torch.tensor([[1, 2, 3]]),
        'mel2ph': torch.tensor([[1, 1, 2, 2]]),
        'mel': torch.randn(1, frames, 4).clamp(-11, -1),
        'f0': torch.full((1, frames), 220.0),
        'energy': torch.full((1, frames), -24.0),
        'breathiness': torch.full((1, frames), -48.0),
        'voicing': torch.full((1, frames), -24.0),
        'tension': torch.zeros(1, frames),
        'key_shift': torch.zeros(1, 1),
        'speed': torch.ones(1, 1)
    }


def variance_sample():
    frames = 4
    return {
        'size': 1,
        'tokens': torch.tensor([[1, 2, 3]]),
        'ph_dur': torch.tensor([[2, 2, 0]]),
        'ph2word': torch.tensor([[1, 1, 0]]),
        'midi': torch.tensor([[60, 62, 0]]),
        'mel2ph': torch.tensor([[1, 1, 2, 2]]),
        'note_midi': torch.tensor([[60.0, 62.0]]),
        'note_rest': torch.tensor([[False, False]]),
        'note_dur': torch.tensor([[2.0, 2.0]]),
        'mel2note': torch.tensor([[1, 1, 2, 2]]),
        'base_pitch': torch.tensor([[60.0, 60.0, 62.0, 62.0]]),
        'pitch': torch.tensor([[60.1, 59.9, 62.1, 61.9]]),
        'energy': torch.full((1, frames), -24.0),
        'breathiness': torch.full((1, frames), -48.0),
        'voicing': torch.full((1, frames), -24.0),
        'tension': torch.zeros(1, frames)
    }


def main():
    configure_smoke_model()
    hparams['predict_dur'] = False
    task = AllInOneTask()
    assert task.val_with_variance_enabled is False
    assert task.model.category == 'all_in_one'
    assert hparams['predict_dur'] is True
    assert task.model.variance.predict_dur is True
    assert task.model.variance.variance_prediction_list == [
        'energy', 'breathiness', 'voicing', 'tension'
    ]

    acoustic_losses = AcousticTask.run_model(
        task, acoustic_sample(), model=task.model.acoustic
    )
    variance_losses = VarianceTask.run_model(
        task, variance_sample(), model=task.model.variance
    )
    assert set(acoustic_losses) == {'mel_loss'}
    assert set(variance_losses) == {'dur_loss', 'pitch_loss', 'var_loss'}
    losses = {**acoustic_losses, **variance_losses}
    assert all(torch.isfinite(loss) for loss in losses.values())

    sum(losses.values()).backward()
    assert any(parameter.grad is not None for parameter in task.model.acoustic.parameters())
    assert any(parameter.grad is not None for parameter in task.model.variance.parameters())

    minimal_segment = prepare_variance_segment({
        'ph_seq': 'SP n i h ao SP',
        'note_seq': 'rest C4 D4 rest',
        'note_dur': '0.1 0.2 0.2 0.1'
    }, 'zh', {})
    assert minimal_segment['ph_num'] == '2 2 1 1'
    assert minimal_segment['note_slur'] == '0 0 0 0'
    text_segment = prepare_variance_segment({
        'text': 'SP ni hao SP',
        'ph_seq': 'this value must be replaced',
        'note_seq': 'rest C4 D4 rest',
        'note_dur': '0.1 0.2 0.2 0.1'
    }, 'zh', {'ni': ['n', 'i'], 'hao': ['h', 'ao']})
    assert text_segment['ph_seq'] == 'SP n i h ao SP'
    assert text_segment['ph_num'] == '1 2 2 1'

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        dictionaries = {}
        sources = {'enable': True}
        for language in ('zh', 'en'):
            dictionary_path = temp_dir / f'{language}.txt'
            dictionary_path.write_text('word\tw er d\n', encoding='utf-8')
            source_path = temp_dir / f'{language}.ds'
            source_path.write_text(json.dumps([{
                'text': 'word', 'note_seq': 'C4', 'note_dur': '0.2'
            }]), encoding='utf-8')
            dictionaries[language] = dictionary_path
            sources[language] = [source_path]
        validate_variance_validation_config(sources, dictionaries)
        first_language, _ = load_validation_sources(
            sources, dictionaries, rng=random.Random(0)
        )
        second_language, _ = load_validation_sources(
            sources, dictionaries, previous_language=first_language,
            rng=random.Random(0)
        )
        assert first_language != second_language

    variance_infer = DiffSingerVarianceInfer(
        device='cpu', predictions=set(), model=task.model.variance
    )
    acoustic_infer = DiffSingerAcousticInfer(
        device='cpu', load_vocoder=False, model=task.model.acoustic
    )
    assert variance_infer.model is task.model.variance
    assert acoustic_infer.model is task.model.acoustic

    output_dirs = []

    class FakeBinarizer:
        def __init__(self, binary_data_dir):
            output_dirs.append(Path(binary_data_dir).name)

        def process(self):
            pass

    acoustic_binarizer = all_in_one_binarizer.AcousticBinarizer
    variance_binarizer = all_in_one_binarizer.VarianceBinarizer
    try:
        all_in_one_binarizer.AcousticBinarizer = FakeBinarizer
        all_in_one_binarizer.VarianceBinarizer = FakeBinarizer
        all_in_one_binarizer.AllInOneBinarizer().process()
    finally:
        all_in_one_binarizer.AcousticBinarizer = acoustic_binarizer
        all_in_one_binarizer.VarianceBinarizer = variance_binarizer
    assert output_dirs == ['acoustic', 'variance']
    print('All-in-one training validation passed.')


if __name__ == '__main__':
    main()
