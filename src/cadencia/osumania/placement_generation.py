"""Autoregressive four-lane generation from a frozen-MERT placement checkpoint."""

import hashlib
from pathlib import Path
import tempfile

import librosa
import numpy as np

from ..audio import read_audio
from ..mert import FrozenMert, MertConfig
from ..placement import PlacementTransformer, mask_logits, checkpoint_config, decode_patterns
from ..placement import TAP, HOLD_START, HOLD_END, RELEASE_TAP, RELEASE_HOLD
from ..placement_data import advance_held, make_sample, tempo_at
from .artifact import write_artifact


class PlacementModel:
    def __init__(self, path, *, checkpoint=None, device='cpu'):
        import torch

        self.device = torch.device(device)
        checkpoint = checkpoint if checkpoint is not None else torch.load(path, map_location='cpu', weights_only=True)
        from ..interval import FORMAT, IntervalTransformer
        from ..placement import PlacementConfig
        from ..planner import HIGH_RES_FORMAT, FORMAT as PLANNED_FORMAT, PlannedTransformer, BarPlanner, PLANNER_FORMAT
        self.planned = checkpoint.get('format') in (PLANNED_FORMAT,HIGH_RES_FORMAT)
        self.intervals = checkpoint.get('format') == FORMAT or self.planned
        if self.planned:
            saved_planner = checkpoint['planner']
            if saved_planner['format'] != PLANNER_FORMAT or saved_planner['mert'] != checkpoint['mert']:
                raise ValueError('Incompatible embedded planner')
            self.planner = BarPlanner(saved_planner.get('audio_dim',768),saved_planner.get('use_rhythm',False)).to(self.device)
            self.planner.load_state_dict(saved_planner['model'],strict=True)
            self.planner.eval()
        self.config = PlacementConfig(**checkpoint['config']) if self.intervals else checkpoint_config(checkpoint)
        if self.config.use_rhythm:
            from ..rhythm import SPEC
            if checkpoint.get('rhythm_backend')!=SPEC:raise ValueError('Rhythm backend differs from training checkpoint')
        self.model = (PlannedTransformer if self.planned else IntervalTransformer if self.intervals else PlacementTransformer)(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.extractor = FrozenMert(config=MertConfig(**checkpoint['mert']), device=str(self.device))
        self.checkpoint_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.alignment_prior_sha256 = checkpoint.get('alignment_prior_sha256')
        from ..beat_alignment import DEFAULT_PRIOR
        snapshot_prior = Path(path).with_name('alignment_prior.json')
        self.alignment_prior_path = snapshot_prior if snapshot_prior.is_file() else DEFAULT_PRIOR

    def predict(self, path, *, difficulty, bpm=None, cache_dir=None, decoding=None, seed=42, temperature=1., align_beats=None, alignment_max_shift_ms=25., repeat_sections=False, repeat_variation=.2):
        import torch

        from ..repetition import RepetitionConfig
        repetition_config = RepetitionConfig(variation=repeat_variation)
        if type(repeat_sections) is not bool:
            raise ValueError('repeat_sections must be a boolean')
        if repeat_sections and not self.config.use_rhythm:
            raise ValueError('Repeated sections require a rhythm-enabled checkpoint')
        alignment_enabled = self.config.use_mel if align_beats is None else align_beats
        if type(alignment_enabled) is not bool:
            raise ValueError('align_beats must be a boolean or None')
        decoding = decoding or ('sample' if self.config.joint_patterns else 'argmax')
        if decoding not in ('sample','argmax') or (decoding == 'sample' and not self.config.joint_patterns):
            raise ValueError('Sampling requires a joint-pattern checkpoint; legacy models use argmax')
        if isinstance(temperature,bool) or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be positive and finite')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('seed must be an integer between 0 and 2**63-1')
        random = torch.Generator(device='cpu').manual_seed(seed)
        path = Path(path)
        if isinstance(difficulty, bool) or not np.isfinite(difficulty) or difficulty <= 0:
            raise ValueError('difficulty must be a positive osu!mania star-rating target')
        if bpm is not None and (isinstance(bpm, bool) or not np.isfinite(bpm) or bpm <= 0):
            raise ValueError('bpm must be positive and finite')
        cache_dir = Path(cache_dir) if cache_dir else path.parent / 'mert_cache'
        features, _ = self.extractor.extract(path, cache_dir=cache_dir)
        duration = float(features['duration_s'])
        beat_offset = 0.
        timing = None
        if self.config.use_rhythm:
            from ..rhythm import analyze_rhythm
            features['rhythm'],_=analyze_rhythm(path,cache_dir.parent/'rhythm_cache',device=str(self.device))
            timing=features['rhythm']['timing_map']
            bpm=float(np.median([point['bpm'] for point in timing]))
            beat_offset=timing[0]['start_s']
        if (self.config.use_mel or alignment_enabled) and timing is None:
            from ..high_resolution import tracked_timing,extract_mel
            timing = tracked_timing(path,bpm_hint=bpm)
            bpm = float(np.median([point['bpm'] for point in timing]))
            beat_offset = timing[0]['start_s']
        if self.config.use_mel:
            from ..high_resolution import extract_mel
            features['mel'],_=extract_mel(path,cache_dir.parent/'mel_cache')
        if bpm is None:
            samples, rate, _ = read_audio(path)
            mono = librosa.resample(samples.mean(axis=1), orig_sr=rate, target_sr=22050)
            estimated, beats = librosa.beat.beat_track(y=mono, sr=22050, hop_length=441)
            bpm = float(np.asarray(estimated).ravel()[0])
            if not np.isfinite(bpm) or bpm <= 0 or not len(beats):
                raise ValueError('Could not estimate tempo; supply bpm= explicitly')
            beat_offset = float(beats[0] * 441 / 22050)
        # Estimated tempo is constant for arbitrary audio in v1; cached charts
        # train with their complete timing maps, including tempo changes.
        times = np.arange(int(np.ceil(duration * self.config.grid_hz))) / self.config.grid_hz
        times = times[times < duration]
        timing = timing or [dict(start_s=beat_offset, beat_position=0., bpm=bpm)]
        bpms, beat_positions = tempo_at(times, timing)
        if self.planned:
            from ..planner import bar_inputs
            bars, row_bars = bar_inputs(features,times,beat_positions,bpms,difficulty)
            with torch.inference_mode():
                plans = self.planner.predict(torch.from_numpy(bars[None]).to(self.device))[0].cpu().numpy()
        held = np.zeros(4, dtype=bool)
        history_actions, history_times, history_beats = [], [], []
        notes, active = [], {}
        with torch.inference_mode():
            for index, time_s in enumerate(times):
                # Explicit interval endpoints release lanes independently of model output.
                released = np.zeros(4, dtype=np.int64)
                if self.intervals:
                    for lane in list(active):
                        if active[lane]['end_s'] <= time_s:
                            active.pop(lane)
                            held[lane] = False
                            released[lane] = HOLD_END
                    if released.any() and history_times and history_times[-1] == float(time_s):
                        raise ValueError('Duplicate history timestamp')
                    # Include releases in this query's held state; add them to
                    # history together with any new heads after predicting.
                sample = make_sample(time_s, beat_positions[index], bpms[index], difficulty, held,
                                     history_actions, history_times, history_beats, features, self.config)
                inputs = {k: torch.from_numpy(v[None]).to(self.device) for k, v in sample.items()}
                if self.planned:
                    inputs['plan'] = torch.from_numpy(plans[row_bars[index]][None]).to(self.device)
                final_step = index == len(times) - 1
                if self.intervals:
                    from ..interval import legal_pattern_logits, decode
                    output = self.model(**inputs)
                    probability = float((output['event'] / temperature).sigmoid().item())
                    event = float(torch.rand((), generator=random)) < probability if decoding == 'sample' else probability >= .5
                    actions = np.zeros(4, dtype=np.int64)
                    confidence = probability
                    if event and not held.all():
                        logits = legal_pattern_logits(output['pattern'], torch.from_numpy(held[None]).to(self.device))
                        probabilities = (logits / temperature).softmax(-1)[0].cpu()
                        token = torch.multinomial(probabilities, 1, generator=random).squeeze(0) if decoding == 'sample' else probabilities.argmax()
                        actions = decode(token + 1).numpy()
                        confidence *= float(probabilities[token])
                    for lane, action in enumerate(actions):
                        if held[lane] or not action:
                            continue
                        note = dict(time_s=float(time_s), lane=lane, probability=confidence, kind='hold' if action == HOLD_START else 'tap')
                        if action == HOLD_START:
                            beats = float(np.expm1(np.clip(output['duration'][0, lane].float().item(), 0, 8)))
                            from ..high_resolution import time_at_beat
                            length = max(1 / self.config.grid_hz,time_at_beat(float(beat_positions[index])+beats,timing)-float(time_s))
                            # Snap endpoints to the same grid used for query/history.
                            endpoint = min(duration, round((time_s + length) * self.config.grid_hz) / self.config.grid_hz)
                            note['end_s'] = float(endpoint)
                            active[lane] = note
                            held[lane] = True
                        notes.append(note)
                    combined = actions.copy()
                    for lane in np.flatnonzero(released):
                        combined[lane] = {0: HOLD_END, TAP: RELEASE_TAP, HOLD_START: RELEASE_HOLD}[int(actions[lane])]
                    if combined.any():
                        history_actions.append(combined); history_times.append(float(time_s)); history_beats.append(float(beat_positions[index]))
                        history_actions = history_actions[-self.config.history_events:]
                        history_times = history_times[-self.config.history_events:]
                        history_beats = history_beats[-self.config.history_events:]
                    continue
                logits = mask_logits(self.model(**inputs), torch.from_numpy(held[None]).to(self.device),
                                     final_step=final_step)
                probabilities = (logits / temperature).softmax(-1)[0].cpu()
                if self.config.joint_patterns:
                    token = torch.multinomial(probabilities,1,generator=random).squeeze(0) if decoding == 'sample' else probabilities.argmax(-1)
                    actions = decode_patterns(token).numpy()
                    confidence = float(probabilities[token])
                else:
                    probabilities = probabilities.numpy()
                    actions = probabilities.argmax(-1)
                    if final_step:
                        actions[held] = HOLD_END
                        actions[~held & (actions == HOLD_START)] = TAP
                for lane, action in enumerate(actions):
                    if action in (HOLD_END, RELEASE_TAP, RELEASE_HOLD):
                        start_note = active.pop(lane)
                        start_note['end_s'] = float(time_s)
                    if action in (TAP, HOLD_START, RELEASE_TAP, RELEASE_HOLD):
                        note = dict(time_s=float(time_s), lane=lane,
                                    probability=confidence if self.config.joint_patterns else float(probabilities[lane, action]),
                                    kind='hold' if action in (HOLD_START, RELEASE_HOLD) else 'tap')
                        notes.append(note)
                        if note['kind'] == 'hold':
                            active[lane] = note
                held = advance_held(held, actions)
                if actions.any():
                    history_actions.append(actions.copy()); history_times.append(float(time_s))
                    history_beats.append(float(beat_positions[index]))
                    history_actions = history_actions[-self.config.history_events:]
                    history_times = history_times[-self.config.history_events:]
                    history_beats = history_beats[-self.config.history_events:]
        if self.intervals:
            from ..interval import merge_notes
            notes = merge_notes(notes)
        extras = dict(timing_map=timing) if self.config.use_mel or alignment_enabled else {}
        if self.config.use_rhythm:
            analysis=features['rhythm']
            extras['rhythm']=dict(spec=analysis['spec'],downbeat_times=analysis['downbeat_times'].tolist(),bar_boundaries_detected=analysis['bar_boundaries_detected'])
        result = dict(**extras,notes=notes, duration_s=duration, candidates=len(times) * 4, threshold=0.,
                    title=path.stem, model_sha256=self.checkpoint_sha256,
                    difficulty=float(difficulty), bpm=float(bpm), beat_offset_s=beat_offset,
                    decoding=decoding, seed=seed, temperature=float(temperature),
                    candidate_policy=('Calibrated event timing + conditional joint onsets + explicit hold intervals; same-lane occupancy filtering' if self.intervals else '50 Hz grid; frozen MERT + generated history; constrained joint-pattern ' + decoding + '; '
                                      'note confidence is the complete pattern probability' if self.config.joint_patterns else
                                      '50 Hz grid; frozen MERT + generated event history; constrained per-lane argmax'))

        if repeat_sections:
            from ..repetition import reuse_patterns
            result = reuse_patterns(result, features, config=repetition_config, seed=seed)
        if alignment_enabled:
            from ..beat_alignment import align_chart,audio_onsets,load_prior
            _,prior_hash=load_prior(self.alignment_prior_path)
            if self.alignment_prior_sha256 and prior_hash!=self.alignment_prior_sha256:
                raise ValueError('Alignment prior differs from training checkpoint')
            result=align_chart(result,prior_path=self.alignment_prior_path,max_shift_ms=alignment_max_shift_ms,onsets=audio_onsets(path),rhythm_evidence=features.get('rhythm'))
        return result

    def create_chart(self, mp3_file, *, out, difficulty, bpm=None, cache_dir=None, decoding=None, seed=42, temperature=1., align_beats=None, alignment_max_shift_ms=25., repeat_sections=False, repeat_variation=.2):
        import soundfile as sf

        out = Path(out)
        if out.exists():
            raise FileExistsError(out)
        result = self.predict(mp3_file, difficulty=difficulty, bpm=bpm, cache_dir=cache_dir,
                              decoding=decoding, seed=seed, temperature=temperature,
                              align_beats=align_beats,alignment_max_shift_ms=alignment_max_shift_ms,
                              repeat_sections=repeat_sections,repeat_variation=repeat_variation)
        with tempfile.TemporaryDirectory(prefix='cadencia-placement-') as temporary:
            playback = Path(temporary) / 'audio.wav'
            samples, rate, _ = read_audio(Path(mp3_file))
            sf.write(playback, samples, rate, subtype='PCM_16')
            write_artifact(out, result, playback)
        return out
