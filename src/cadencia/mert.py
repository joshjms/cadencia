"""Frozen, revision-pinned MERT-95M extraction with content-addressed caching."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import librosa
import numpy as np

from .audio import read_audio


@dataclass(frozen=True)
class MertConfig:
    model_id: str = 'm-a-p/MERT-v1-95M'
    revision: str = '12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'
    extraction_version: int = 3
    layer: int = 12
    core_seconds: float = 4.0
    context_seconds: float = 0.4

    @classmethod
    def v2(cls, *, full_song=False):
        return cls(model_id='m-a-p/MERT-v2-FullSong' if full_song else 'm-a-p/MERT-v2-30s',
                   revision='7001f7657276238629e61c7746ce70735e0604c0' if full_song else '0c0c8c6f6e04dcc39d96f94e0be9978a74848906',
                   extraction_version=4,layer=24,core_seconds=360. if full_song else 28.,context_seconds=0. if full_song else 1.)

    @property
    def is_v2(self): return self.model_id.startswith('m-a-p/MERT-v2-')

    @property
    def feature_dim(self): return 1024 if self.is_v2 else 768

    @property
    def frame_hz(self): return 25 if self.is_v2 else 75

    def __post_init__(self):
        if self.model_id not in ('m-a-p/MERT-v1-95M','m-a-p/MERT-v2-30s','m-a-p/MERT-v2-FullSong') or len(self.revision)!=40 or any(c not in '0123456789abcdef' for c in self.revision):
            raise ValueError('Use a supported MERT model and immutable revision')
        if self.is_v2:
            if self.extraction_version!=4 or not 1<=self.layer<=24:
                raise ValueError('MERT-v2 requires extraction v4 and layers 1..24')
            expected=(360.,0.) if self.model_id.endswith('FullSong') else (28.,1.)
            if (self.core_seconds,self.context_seconds)!=expected:
                raise ValueError('Unsupported MERT-v2 chunk policy')
        elif self.extraction_version not in (2,3) or not 0<=self.layer<=12 or self.core_seconds!=4. or self.context_seconds!=.4:
            raise ValueError('Invalid MERT-v1 extraction configuration')


class FrozenMert:
    """Extract time-resolved 768-dimensional features; never train MERT weights.

    Loads lazily, so reading cached features does not load MERT or need a network.
    Each chunk is normalized by the official Wav2Vec2 feature extractor.
    """

    def __init__(self, *, config=None, device='cpu'):
        self.config = config or MertConfig()
        self.device = device
        self.model = self.processor = None

    def _load(self):
        if self.model is not None:
            return
        try:
            from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor
            from huggingface_hub import hf_hub_download
            import torch
        except ImportError as error:
            raise RuntimeError('Install MERT dependencies: uv sync --extra placement') from error
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.config.model_id, revision=self.config.revision)
        config = AutoConfig.from_pretrained(self.config.model_id, revision=self.config.revision,
                                           trust_remote_code=True)
        if self.config.is_v2:
            # The official v2 code uses torchaudio's mel frontend and SDPA.
            try:
                import torchaudio
            except ImportError as error:
                raise RuntimeError('MERT-v2 requires torchaudio matching the installed torch build') from error
        options = dict(attn_implementation='sdpa') if self.config.is_v2 else {}
        model = AutoModel.from_config(config, trust_remote_code=True, code_revision=self.config.revision, **options)
        if self.config.is_v2:
            from safetensors.torch import load_file
            weights_path = hf_hub_download(self.config.model_id,'model.safetensors',revision=self.config.revision)
            weights = load_file(weights_path)
        else:
            weights_path = hf_hub_download(self.config.model_id, 'pytorch_model.bin', revision=self.config.revision)
            weights = torch.load(weights_path, map_location='cpu', weights_only=True)
        # Transformers' old loading path bypasses torch's weight-norm migration
        # hooks on recent PyTorch and silently initializes positional convolution.
        # Migrate these two keys explicitly, then require an EXACT state match.
        expected = model.state_dict()
        for old, new in [('weight_g', 'parametrizations.weight.original0'),
                         ('weight_v', 'parametrizations.weight.original1')]:
            prefix = 'encoder.pos_conv_embed.conv.'
            if prefix + new in expected and prefix + old in weights:
                weights[prefix + new] = weights.pop(prefix + old)
        model.load_state_dict(weights, strict=True)
        self.model = model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()

    def extract(self, path, *, cache_dir):
        import torch

        path, cache_dir = Path(path), Path(cache_dir)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        identity = dict(audio_sha256=digest, extractor=asdict(self.config), version=self.config.extraction_version)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cached = cache_dir / f'{key}.npz'
        if cached.is_file():
            return read_features(cached, identity=identity), cached
        self._load()
        samples, rate, _ = read_audio(path)
        audio = librosa.resample(samples.mean(axis=1), orig_sr=rate, target_sr=24000)
        duration = len(samples) / rate
        # Seven convolution layers: stride 320, receptive field 400 samples.
        # Timestamp each feature at its receptive-field center, not chunk duration / N.
        frames, times = [], []
        core, context = int(self.config.core_seconds*24000), int(self.config.context_seconds*24000)
        if self.config.model_id.endswith('FullSong') and duration>360:
            raise ValueError('FullSong extraction supports up to 360 seconds; use v2-30s for longer recordings')
        with torch.inference_mode():
            for start in range(0, len(audio), core):
                begin, end = max(0, start - context), min(len(audio), start + core + context)
                segment = audio[begin:end]
                minimum = 1025 if self.config.is_v2 else 400
                if len(segment) < minimum:
                    segment = np.pad(segment, (0, minimum - len(segment)))
                inputs = self.processor(segment, sampling_rate=24000, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                output = self.model(**inputs, output_hidden_states=True)
                layer = self.config.layer-1 if self.config.is_v2 else self.config.layer
                hidden = output.hidden_states[layer][0].float().cpu().numpy()
                # v2: centered 10ms mel frames, then kernel-2 resampling
                # strides 1,2,2 => centers at 20ms + index*40ms.
                stamp = (begin + (480 if self.config.is_v2 else 199.5) + np.arange(len(hidden)) * (960 if self.config.is_v2 else 320)) / 24000
                if self.config.is_v2:
                    valid = output.feature_attention_mask[0].cpu().numpy().astype(bool)
                    hidden,stamp=hidden[valid],stamp[valid]
                keep = (stamp >= start / 24000) & (stamp < min(duration, (start + core) / 24000))
                frames.append(hidden[keep])
                times.append(stamp[keep])
        matrix = np.concatenate(frames).astype(np.float16 if self.config.extraction_version >= 3 else np.float32)
        features = dict(features=matrix, times=np.concatenate(times), duration_s=np.array(duration),
                        identity=np.array(json.dumps(identity, sort_keys=True)))
        validate_features(features)
        cache_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = cached.with_suffix('.npy')
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix='.npy', delete=False) as temporary:
            temporary_matrix = Path(temporary.name)
            np.save(temporary, matrix, allow_pickle=False)
        try:
            temporary_matrix.replace(matrix_path)
        finally:
            temporary_matrix.unlink(missing_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix='.npz', delete=False) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **{k: features[k] for k in ('times', 'duration_s', 'identity')})
        try:
            temporary_path.replace(cached)
        finally:
            temporary_path.unlink(missing_ok=True)
        features['features'] = np.load(matrix_path, mmap_mode='r', allow_pickle=False)
        return features, cached


def validate_features(value):
    x, times, duration = value['features'], value['times'], float(value['duration_s'])
    config = MertConfig(**json.loads(str(value['identity']))['extractor'])
    if (x.ndim != 2 or x.shape != (len(times), config.feature_dim) or not len(times)
            or x.dtype not in (np.float32, np.float16) or not np.isfinite(x).all()
            or not np.isfinite(times).all() or not np.isfinite(duration) or duration <= 0
            or (np.diff(times) <= 0).any() or times[0] < 0 or times[-1] >= duration):
        raise ValueError('Invalid MERT cache arrays')


def read_features(path, *, identity=None):
    with np.load(path, allow_pickle=False) as arrays:
        value = {k: arrays[k] for k in ('times', 'duration_s', 'identity')}
        if 'features' in arrays.files:
            value['features'] = arrays['features']  # Legacy compressed cache.
    if 'features' not in value:
        value['features'] = np.load(Path(path).with_suffix('.npy'), mmap_mode='r', allow_pickle=False)
    validate_features(value)
    if identity is not None and json.loads(str(value['identity'])) != identity:
        raise ValueError('MERT cache identity mismatch')
    return value
