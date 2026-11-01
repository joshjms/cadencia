"""Frozen Beat This! beat/downbeat analysis and audio-only decoder conditioning."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from .audio import read_audio
from .high_resolution import timing_from_beats

CHECKPOINT_SHA256='8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331'
SPEC=dict(backend='beat-this',package_version='1.0',checkpoint='final0',checkpoint_sha256=CHECKPOINT_SHA256,postprocessor='minimal',frame_hz=50,cache_version=1)
FIELDS=['beat_probability','downbeat_probability','onset_strength','sin_bar_phase','cos_bar_phase','bar_confidence']


@lru_cache(maxsize=2)
def tracker(device):
    import torch
    from beat_this.inference import Audio2Frames
    path=Path(torch.hub.get_dir())/'checkpoints/beat_this-final0.ckpt'
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.hub.download_url_to_file('https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp/download?path=%2F&files=final0.ckpt',str(path),hash_prefix=CHECKPOINT_SHA256,progress=False)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:
        raise ValueError('Beat This checkpoint checksum mismatch')
    model=Audio2Frames(checkpoint_path=str(path),device=device)
    model.model.requires_grad_(False)
    return model


def bar_positions(times,beats,downbeats):
    """Use detected bar boundaries, extrapolating partial edge bars.

    With fewer than two downbeats, fall back to four-beat bars with confidence
    zero, rather than claiming that the first beat is a detected downbeat.
    """
    times=np.asarray(times,float);beats=np.asarray(beats,float);downbeats=np.asarray(downbeats,float)
    if len(downbeats)>=2:
        positions=np.interp(times,downbeats,np.arange(len(downbeats)))
        before=times<downbeats[0];after=times>downbeats[-1]
        positions[before]=(times[before]-downbeats[0])/(downbeats[1]-downbeats[0])
        positions[after]=len(downbeats)-1+(times[after]-downbeats[-1])/(downbeats[-1]-downbeats[-2])
        return positions,True
    beat_pos=np.interp(times,beats,np.arange(len(beats)))
    for mask,origin,position,delta in [(times<beats[0],beats[0],0.,beats[1]-beats[0]),(times>beats[-1],beats[-1],len(beats)-1,beats[-1]-beats[-2])]:
        beat_pos[mask]=position+(times[mask]-origin)/delta
    anchor=float(np.interp(downbeats[0],beats,np.arange(len(beats)))) if len(downbeats) else 0.
    return (beat_pos-anchor)/4,False


def from_predictions(beat_prob,down_prob,beats,downbeats,onset,onset_times,duration):
    beat_prob=np.asarray(beat_prob,float);down_prob=np.asarray(down_prob,float)
    beats=np.asarray(beats,float);downbeats=np.asarray(downbeats,float)
    timing=timing_from_beats(beats)
    if len(downbeats) and ((np.diff(downbeats)<=0).any() or not np.isfinite(downbeats).all()):raise ValueError('Invalid downbeats')
    if len(beat_prob)!=len(down_prob) or not len(beat_prob) or not np.isfinite(beat_prob).all() or not np.isfinite(down_prob).all() or (beat_prob<0).any() or (beat_prob>1).any() or (down_prob<0).any() or (down_prob>1).any():raise ValueError('Invalid rhythm probabilities')
    times=np.arange(int(np.ceil(duration*100)))/100
    times=times[times<duration];frame_times=np.arange(len(beat_prob))/50
    bars,detected=bar_positions(times,beats,downbeats)
    beat=np.interp(times,frame_times,beat_prob);down=np.interp(times,frame_times,down_prob)
    strength=np.interp(times,onset_times,onset)
    strength=np.clip(strength/max(1e-8,float(np.quantile(strength,.95))),0,1)
    confidence=np.interp(times,downbeats,np.interp(downbeats,frame_times,down_prob)) if detected else np.zeros(len(times))
    features=np.column_stack([beat,down,strength,np.sin(2*np.pi*bars),np.cos(2*np.pi*bars),confidence]).astype(np.float32)
    # Retain ordinary beat phase while anchoring integer beat numbers at a
    # detected downbeat, when available. Variable bar lengths use bars separately.
    if len(downbeats):
        anchor=int(np.argmin(np.abs(beats-downbeats[0])))
        for point in timing:point['beat_position']-=anchor
    return dict(times=times,features=features,bar_positions=bars,beat_times=beats,downbeat_times=downbeats,
                timing_map=timing,bar_boundaries_detected=detected,spec=SPEC)


def analyze_rhythm(path,cache_dir,*,device='cpu'):
    import librosa
    import torch
    from beat_this.model.postprocessor import Postprocessor
    path,cache_dir=Path(path),Path(cache_dir)
    identity=dict(audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),spec=SPEC)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();cached=cache_dir/f'{key}.npz'
    if cached.exists():return read_rhythm(cached),cached
    samples,rate,_=read_audio(path);mono=samples.mean(1)
    with torch.inference_mode():
        beat_logits,down_logits=tracker(device)(mono,rate)
        beats,downbeats=Postprocessor(type='minimal')(beat_logits,down_logits)
        beat_prob=beat_logits.sigmoid().cpu().numpy();down_prob=down_logits.sigmoid().cpu().numpy()
    duration=len(samples)/rate
    beats=beats[(beats>=0)&(beats<duration)];downbeats=downbeats[(downbeats>=0)&(downbeats<duration)]
    signal=librosa.resample(mono,orig_sr=rate,target_sr=24000)
    onset=librosa.onset.onset_strength(y=signal,sr=24000,hop_length=240,n_fft=1024)
    result=from_predictions(beat_prob,down_prob,beats,downbeats,onset,np.arange(len(onset))/100,duration)
    cache_dir.mkdir(parents=True,exist_ok=True)
    metadata=dict(timing_map=result['timing_map'],bar_boundaries_detected=result['bar_boundaries_detected'],spec=SPEC,identity=identity)
    with tempfile.NamedTemporaryFile(dir=cache_dir,suffix='.npz',delete=False) as f:
        temp=Path(f.name)
        np.savez_compressed(f,**{k:result[k] for k in ['times','features','bar_positions','beat_times','downbeat_times']},metadata=np.array(json.dumps(metadata)))
    temp.replace(cached)
    return result,cached


def read_rhythm(path):
    with np.load(path,allow_pickle=False) as arrays:
        result={k:arrays[k] for k in ['times','features','bar_positions','beat_times','downbeat_times']}
        metadata=json.loads(str(arrays['metadata']))
    if metadata['spec']!=SPEC:raise ValueError('Rhythm cache/backend mismatch')
    if result['features'].shape!=(len(result['times']),6) or not np.isfinite(result['features']).all() or (np.diff(result['times'])<=0).any():raise ValueError('Invalid rhythm cache')
    result.update(metadata);return result


def rhythm_at(analysis,times):
    return np.column_stack([np.interp(times,analysis['times'],analysis['features'][:,i]) for i in range(6)]).astype(np.float32)


def beat_confidence(analysis,times):
    activations=np.interp(analysis['beat_times'],analysis['times'],analysis['features'][:,0])
    return np.interp(times,analysis['beat_times'],activations)
