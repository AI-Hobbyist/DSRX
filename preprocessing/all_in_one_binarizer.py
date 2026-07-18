import librosa
import numpy as np
import torch

from preprocessing.variance_binarizer import VARIANCE_ITEM_ATTRIBUTES, VarianceBinarizer
from utils.binarizer_utils import get_mel2ph_torch, get_mel_torch
from modules.pe import initialize_pe
from utils.hparams import hparams


ALL_IN_ONE_ITEM_ATTRIBUTES = sorted(set(VARIANCE_ITEM_ATTRIBUTES + [
    'mel',
    'f0',
    'key_shift',
    'speed',
]))

pitch_extractor = None


class AllInOneBinarizer(VarianceBinarizer):
    def __init__(self):
        super().__init__()
        self.data_attrs = ALL_IN_ONE_ITEM_ATTRIBUTES
        assert hparams['mel_base'] == 'e', (
            "Mel base must be set to 'e' for all-in-one acoustic training."
        )

    @torch.no_grad()
    def process_item(self, item_name, meta_data, binarization_args):
        processed_input = super().process_item(item_name, meta_data, binarization_args)
        if processed_input is None:
            return None

        waveform, _ = librosa.load(meta_data['wav_fn'], sr=hparams['audio_sample_rate'], mono=True)
        mel = get_mel_torch(
            waveform, hparams['audio_sample_rate'], num_mel_bins=hparams['audio_num_mel_bins'],
            hop_size=hparams['hop_size'], win_size=hparams['win_size'], fft_size=hparams['fft_size'],
            fmin=hparams['fmin'], fmax=hparams['fmax'],
            device=self.device
        )
        length = mel.shape[0]
        if length != processed_input['length']:
            length = min(length, processed_input['length'])
            mel = mel[:length]
            processed_input['length'] = length
            for key in ['mel2ph', 'pitch', 'uv', 'base_pitch', 'energy', 'breathiness', 'voicing', 'tension']:
                if key in processed_input:
                    processed_input[key] = processed_input[key][:length]
        if 'mel2ph' not in processed_input:
            ph_dur_sec = torch.FloatTensor(meta_data['ph_dur']).to(self.device)
            processed_input['mel2ph'] = get_mel2ph_torch(
                self.lr, ph_dur_sec, length, self.timestep, device=self.device
            ).cpu().numpy()

        global pitch_extractor
        if pitch_extractor is None:
            pitch_extractor = initialize_pe(self.device)
        f0, uv = pitch_extractor.get_pitch(
            waveform, samplerate=hparams['audio_sample_rate'], length=length,
            hop_size=hparams['hop_size'], f0_min=hparams['f0_min'], f0_max=hparams['f0_max'],
            interp_uv=True
        )
        if uv.all():
            print(f"Skipped '{item_name}': empty gt f0")
            return None

        if isinstance(mel, torch.Tensor):
            mel = mel.cpu().numpy()
        processed_input['mel'] = mel.astype(np.float32, copy=False)
        processed_input['f0'] = f0.astype(np.float32)
        if hparams.get('use_key_shift_embed', False):
            processed_input['key_shift'] = 0.0
        if hparams.get('use_speed_embed', False):
            processed_input['speed'] = 1.0
        return processed_input
