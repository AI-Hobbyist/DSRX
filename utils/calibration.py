"""
ONNX quantization calibration data reader for DSRX acoustic models.

Provides calibration data pipelines for:
  - Static quantization (requires representative calibration data)
  - The acoustic model's FS2 encoder + diffusion backbone

Usage::

    from utils.calibration import AcousticCalibrationDataReader

    reader = AcousticCalibrationDataReader(
        model=model,
        num_samples=100,
        seq_len_range=(10, 200),
    )
    from onnxruntime.quantization import quantize_static
    quantize_static('model.onnx', 'model_int8.onnx', reader)
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from utils.hparams import hparams


class AcousticCalibrationDataReader:
    """Calibration data reader for ONNX Runtime static quantization.

    Generates representative dummy inputs that cover the expected
    dynamic range of the acoustic model (FS2 encoder + diffusion).

    For real-world calibration with actual dataset distributions,
    provide ``real_samples`` — a list of (tokens, durations, f0,
    variances, kwargs) tuples from the binarized dataset.

    Attributes:
        num_samples: Total number of calibration batches.
        seq_len_range: (min, max) token lengths to generate.
        frame_multiplier: Factor to multiply token length for frame count.
        batch_size: Batch size for each calibration iteration.
    """

    def __init__(
        self,
        model: torch.nn.Module = None,
        num_samples: int = 100,
        seq_len_range: Tuple[int, int] = (5, 50),
        frame_multiplier: int = 10,
        batch_size: int = 1,
        device: str = 'cpu',
        real_samples: Optional[List[dict]] = None,
        seed: int = 42,
    ):
        self.model = model
        self.num_samples = num_samples
        self.seq_len_range = seq_len_range
        self.frame_multiplier = frame_multiplier
        self.batch_size = batch_size
        self.device = device
        self.real_samples = real_samples
        self._rng = np.random.RandomState(seed)
        self._iter_index = 0
        self._cache: Optional[List[dict]] = None

        # Determine what inputs the model expects
        self._input_names = self._resolve_input_names()

    def _resolve_input_names(self) -> List[str]:
        """Determine the ordered input names based on hparams."""
        names = ['tokens', 'durations', 'f0']

        if self.model is not None:
            variance_list = getattr(self.model.fs2, 'variance_embed_list', [])
        else:
            variance_list = []
            for v_name in ['energy', 'breathiness', 'voicing', 'tension']:
                if hparams.get(f'use_{v_name}_embed', False):
                    variance_list.append(v_name)
        names.extend(variance_list)

        if hparams.get('use_key_shift_embed', False):
            names.append('gender')
        if hparams.get('use_speed_embed', False):
            names.append('velocity')
        if hparams.get('use_spk_id', False):
            names.append('spk_embed')
        if hparams.get('use_lang_id', False):
            names.append('languages')

        return names

    def _generate_dummy_batch(self, n_tokens: int) -> dict:
        """Generate a single dummy calibration batch."""
        n_frames = n_tokens * self.frame_multiplier
        hidden = hparams.get('hidden_size', 256)
        num_mel = hparams.get('audio_num_mel_bins', 128)

        feed: Dict[str, np.ndarray] = {}

        for name in self._input_names:
            if name == 'tokens':
                feed[name] = self._rng.randint(
                    1, 100, size=(self.batch_size, n_tokens)
                ).astype(np.int64)
            elif name == 'durations':
                base = n_frames // n_tokens
                feed[name] = np.full(
                    (self.batch_size, n_tokens), base, dtype=np.int64
                )
            elif name == 'f0':
                feed[name] = self._rng.uniform(
                    100, 800, size=(self.batch_size, n_frames)
                ).astype(np.float32)
            elif name == 'gender':
                feed[name] = self._rng.uniform(
                    -1, 1, size=(self.batch_size, n_frames)
                ).astype(np.float32)
            elif name == 'velocity':
                feed[name] = self._rng.uniform(
                    0.5, 2.0, size=(self.batch_size, n_frames)
                ).astype(np.float32)
            elif name == 'spk_embed':
                feed[name] = self._rng.randn(
                    self.batch_size, n_frames, hidden
                ).astype(np.float32)
            elif name == 'languages':
                feed[name] = np.zeros(
                    (self.batch_size, n_tokens), dtype=np.int64
                )
            else:
                # Variance embeddings
                feed[name] = self._rng.randn(
                    self.batch_size, n_frames
                ).astype(np.float32)

        return feed

    def _prepare_cache(self):
        """Pre-generate all calibration batches."""
        if self._cache is not None:
            return

        self._cache = []

        if self.real_samples:
            for sample in self.real_samples[:self.num_samples]:
                self._cache.append(self._real_to_feed(sample))
        else:
            min_len, max_len = self.seq_len_range
            for i in range(self.num_samples):
                # Vary sequence length to cover dynamic axes
                t = i / max(self.num_samples - 1, 1)
                n_tokens = int(min_len + t * (max_len - min_len))
                self._cache.append(self._generate_dummy_batch(n_tokens))

    def _real_to_feed(self, sample: dict) -> dict:
        """Convert a real binarized sample to ONNX feed dict."""
        # Subclass / override for real dataset integration
        raise NotImplementedError(
            "Real sample calibration requires implementing _real_to_feed()."
        )

    # ------------------------------------------------------------------
    # onnxruntime.quantization.CalibrationDataReader interface
    # ------------------------------------------------------------------

    def get_next(self) -> Optional[dict]:
        """Return next calibration batch or None if done."""
        self._prepare_cache()
        if self._iter_index >= len(self._cache):
            return None
        batch = self._cache[self._iter_index]
        self._iter_index += 1
        return batch

    def rewind(self):
        """Reset iteration index for another pass."""
        self._iter_index = 0

    def set_range(self, start_index: int, end_index: int):
        self._iter_index = start_index
        self.num_samples = end_index - start_index

    # ------------------------------------------------------------------
    # Iterator protocol (convenience)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        self._prepare_cache()
        for i in range(self.num_samples):
            yield self._cache[i]

    def __len__(self) -> int:
        return self.num_samples


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_calibration_reader(
    model=None,
    num_samples: int = 100,
    device: str = 'cpu',
) -> AcousticCalibrationDataReader:
    """Create a calibration reader with sensible defaults for DSRX."""
    return AcousticCalibrationDataReader(
        model=model,
        num_samples=num_samples,
        seq_len_range=(5, 50),
        frame_multiplier=10,
        batch_size=1,
        device=device,
    )
