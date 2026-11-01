"""Causal event-history decoder with cross-attention to local frozen MERT frames."""

from dataclasses import asdict, dataclass
from functools import lru_cache

import torch
from torch import nn

EMPTY, TAP, HOLD_START, HOLD_END, RELEASE_TAP, RELEASE_HOLD = range(6)
ACTIONS = ['empty', 'tap', 'hold_start', 'hold_end', 'release_tap', 'release_hold']
LEGACY_CHECKPOINT_FORMAT = 'cadencia.placement.v1'
CHECKPOINT_FORMAT = 'cadencia.placement.v2'
PATTERN_COUNT = 6 ** 4
PATTERN_ENCODING = 'lane0 + 6*lane1 + 36*lane2 + 216*lane3'


@dataclass(frozen=True)
class PlacementConfig:
    d_model: int = 128
    heads: int = 4
    layers: int = 2
    history_events: int = 64
    audio_window_s: float = 2.0
    grid_hz: int = 50
    dropout: float = 0.1
    joint_patterns: bool = True
    audio_dim: int = 768
    audio_hz: int = 75
    use_mel: bool = False
    use_rhythm: bool = False

    def __post_init__(self):
        if type(self.use_rhythm) is not bool or (self.use_rhythm and not self.use_mel):
            raise ValueError('Rhythm conditioning requires the high-resolution mel architecture')
        if type(self.use_mel) is not bool or (self.use_mel and (self.grid_hz!=100 or self.audio_dim!=1024 or self.audio_hz!=25)):
            raise ValueError('Mel branch requires 100 Hz queries and MERT-v2 features')
        if type(self.joint_patterns) is not bool:
            raise ValueError('joint_patterns must be a boolean')
        if (self.audio_dim not in (768,1024) or self.audio_hz not in (25,75) or self.d_model <= 0 or self.heads <= 0 or self.d_model % self.heads
                or self.layers < 1 or self.history_events < 1 or self.grid_hz not in (50,100)
                or self.audio_window_s != 2.0 or not 0 <= self.dropout < 1):
            raise ValueError('Invalid placement model configuration')


class PlacementTransformer(nn.Module):
    """Predict one whole-pattern distribution [batch, 1296].

    joint_patterns=False preserves the legacy [batch, 4, 6] checkpoint layout.
    Target lane actions are never included in the inputs.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or PlacementConfig()
        c = self.config
        self.lane_actions = nn.ModuleList([nn.Embedding(6, c.d_model) for _ in range(4)])
        self.history_timing = nn.Linear(4, c.d_model)
        self.conditioning = nn.Sequential(nn.Linear(8, c.d_model), nn.GELU(), nn.Linear(c.d_model, c.d_model))
        if c.use_rhythm:
            self.rhythm_projection = nn.Sequential(nn.Linear(6,c.d_model),nn.GELU(),nn.Linear(c.d_model,c.d_model))
        self.query = nn.Parameter(torch.zeros(1, 1, c.d_model))
        self.audio = nn.Linear(c.audio_dim, c.d_model)
        self.audio_time = nn.Linear(1, c.d_model)
        if c.use_mel:
            self.mel_encoder = nn.Sequential(nn.Conv1d(64,c.d_model,3,padding=1),nn.GELU(),nn.Conv1d(c.d_model,c.d_model,3,padding=1))
            self.mel_time = nn.Linear(1,c.d_model)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=c.d_model, nhead=c.heads, dim_feedforward=c.d_model * 4,
            dropout=c.dropout, batch_first=True), num_layers=c.layers)
        self.output = nn.Linear(c.d_model, PATTERN_COUNT if c.joint_patterns else 24)

    def forward(self, history, history_timing, history_padding, condition,
                audio, audio_times, audio_padding, audio_indices=None, mel=None, mel_padding=None, rhythm=None):
        result = self.encode(history, history_timing, history_padding, condition,
                             audio, audio_times, audio_padding, audio_indices, mel, mel_padding, rhythm)
        logits = self.output(result)
        return logits if self.config.joint_patterns else logits.reshape(-1, 4, 6)

    def history_tokens(self, history, history_timing, history_padding, condition):
        # Right-padded history followed by an unmasked target query. This keeps
        # the query visible even at the beginning of a chart (empty history).
        events = sum(embedding(history[..., lane]) for lane, embedding in enumerate(self.lane_actions))
        events = events + self.history_timing(history_timing)
        query = self.query.expand(len(history), -1, -1) + self.conditioning(condition)[:, None]
        tokens = torch.cat([events, query], dim=1)
        padding = torch.cat([history_padding, torch.zeros(len(history), 1, dtype=torch.bool, device=history.device)], dim=1)
        # Ignore padded query positions as well as padded keys; padded history
        # rows can attend to the target query, preventing all-masked softmax rows.
        mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool,
                                    device=tokens.device), diagonal=1)
        mask = mask[None].expand(len(history), -1, -1).clone()
        mask[:, :-1, -1] = ~history_padding
        mask = mask.repeat_interleave(self.config.heads, dim=0)
        return tokens, padding, mask

    def encode(self, history, history_timing, history_padding, condition,
               audio, audio_times, audio_padding, audio_indices=None, mel=None, mel_padding=None, rhythm=None):
        tokens, padding, mask = self.history_tokens(history, history_timing, history_padding, condition)
        if self.config.use_rhythm:
            if rhythm is None:raise ValueError('Rhythm-conditioned checkpoint requires rhythm inputs')
            tokens = tokens.clone()
            tokens[:,-1] = tokens[:,-1] + self.rhythm_projection(rhythm)
        if not torch.is_autocast_enabled():
            audio = audio.to(self.audio.weight.dtype)
        memory = self.audio(audio)
        if audio_indices is not None:
            memory = memory[audio_indices]
        memory = memory + self.audio_time(audio_times[..., None])
        if self.config.use_mel:
            if mel is None or mel_padding is None:
                raise ValueError('High-resolution checkpoint requires mel windows and masks')
            local = self.mel_encoder(mel.masked_fill(mel_padding[...,None],0).to(self.audio.weight.dtype).transpose(1,2)).transpose(1,2)
            offsets = torch.arange(-50,51,device=local.device,dtype=local.dtype)/100
            local = local + self.mel_time(offsets[None,:,None])
            memory = torch.cat([memory,local],dim=1)
            audio_padding = torch.cat([audio_padding,mel_padding],dim=1)
        result = self.decoder(tokens, memory, tgt_mask=mask,
                              tgt_key_padding_mask=padding, memory_key_padding_mask=audio_padding)
        return result[:, -1]


def legal_actions(held):
    """A held lane may remain held, release, or release and immediately restart."""
    held = held.bool()
    mask = torch.zeros(*held.shape, 6, dtype=torch.bool, device=held.device)
    mask[..., EMPTY] = True
    mask[..., TAP] = mask[..., HOLD_START] = ~held
    mask[..., HOLD_END] = mask[..., RELEASE_TAP] = mask[..., RELEASE_HOLD] = held
    return mask


def encode_patterns(actions):
    """Encode [..., 4] lane actions as one base-six token (lane zero is least significant)."""
    if actions.ndim < 1 or actions.shape[-1] != 4 or actions.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8):
        raise ValueError('Pattern actions must be an integer tensor ending in four lanes')
    if ((actions < 0) | (actions >= 6)).any():
        raise ValueError('Pattern actions must be between 0 and 5')
    bases = torch.tensor([1,6,36,216], device=actions.device)
    return (actions.long() * bases).sum(-1)


def decode_patterns(tokens):
    """Decode joint token IDs to [..., 4] lane actions."""
    if tokens.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8):
        raise ValueError('Pattern tokens must be integers')
    if ((tokens < 0) | (tokens >= PATTERN_COUNT)).any():
        raise ValueError('Pattern token outside the vocabulary')
    bases = torch.tensor([1,6,36,216], device=tokens.device)
    return (tokens.long()[...,None] // bases) % 6


@lru_cache(maxsize=8)
def _pattern_tables(device):
    """Tiny fixed tables for all 16 possible held-lane states."""
    device = torch.device(device)
    patterns = decode_patterns(torch.arange(PATTERN_COUNT,device=device))
    states = ((torch.arange(16,device=device)[:,None] // torch.tensor([1,2,4,8],device=device)) % 2).bool()
    active = states[:,None,:]
    legal = ((patterns[None] == EMPTY)
             | ((patterns[None] == TAP) | (patterns[None] == HOLD_START)) & ~active
             | (patterns[None] >= HOLD_END) & active).all(-1)
    final = torch.where(active, patterns[None] == HOLD_END,
                        (patterns[None] == EMPTY) | (patterns[None] == TAP)).all(-1)
    return legal, final


def legal_patterns(held, *, final_step=False):
    if held.shape[-1] != 4:
        raise ValueError('Held state must contain four lanes')
    states = (held.long() * torch.tensor([1,2,4,8],device=held.device)).sum(-1)
    table = _pattern_tables(str(held.device))[1 if final_step else 0]
    return table[states]


def mask_logits(logits, held, *, final_step=False):
    if logits.ndim == held.ndim and logits.shape[-1] == PATTERN_COUNT:
        return logits.masked_fill(~legal_patterns(held, final_step=final_step), float('-inf'))
    return logits.masked_fill(~legal_actions(held), float('-inf'))


def checkpoint_config(checkpoint):
    """Determine head layout from the checkpoint format, including pre-flag v1 files."""
    version = checkpoint.get('format')
    if version not in (CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT) or checkpoint.get('actions') != ACTIONS:
        raise ValueError('Unsupported placement checkpoint')
    joint = version == CHECKPOINT_FORMAT
    config = dict(checkpoint['config'])
    if config.get('joint_patterns', joint) != joint:
        raise ValueError('Checkpoint format and prediction head disagree')
    if joint and (checkpoint.get('pattern_count') != PATTERN_COUNT or checkpoint.get('pattern_encoding') != PATTERN_ENCODING):
        raise ValueError('Unsupported joint pattern vocabulary')
    config['joint_patterns'] = joint
    return PlacementConfig(**config)


def model_metadata(model):
    joint = model.config.joint_patterns
    metadata = dict(format=CHECKPOINT_FORMAT if joint else LEGACY_CHECKPOINT_FORMAT,
                    config=asdict(model.config), actions=ACTIONS)
    if joint:
        metadata.update(pattern_count=PATTERN_COUNT,pattern_encoding=PATTERN_ENCODING,
                        objective='joint_pattern_cross_entropy',class_weighting='none')
    return metadata
