"""10ms acoustic windows and piecewise beat timing for Cadencia v1."""
import hashlib
from pathlib import Path
import numpy as np
import librosa
from .audio import read_audio

MEL_RATE=100
MEL_BINS=64
MEL_WINDOW=101


def extract_mel(path,cache_dir):
    path,cache_dir=Path(path),Path(cache_dir)
    identity=hashlib.sha256(path.read_bytes()+b'cadencia.mel.v1:24000:1024:240:64:log1p10:power2:center').hexdigest()
    cached=cache_dir/f'{identity}.npy'
    if not cached.exists():
        samples,rate,_=read_audio(path)
        mono=librosa.resample(samples.mean(1),orig_sr=rate,target_sr=24000)
        power=librosa.feature.melspectrogram(y=mono,sr=24000,n_fft=1024,hop_length=240,n_mels=MEL_BINS,power=2.,center=True)
        mel=np.log1p(10*power.T).astype(np.float16)
        cache_dir.mkdir(parents=True,exist_ok=True)
        import tempfile,os
        with tempfile.NamedTemporaryFile(dir=cache_dir,suffix='.npy',delete=False) as f:
            temp=Path(f.name);np.save(f,mel,allow_pickle=False)
        os.replace(temp,cached)
    matrix=np.load(cached,mmap_mode='r',allow_pickle=False)
    if matrix.ndim!=2 or matrix.shape[1]!=MEL_BINS or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError('Invalid high-resolution mel cache')
    return matrix,cached


def mel_windows(matrix,times):
    centers=np.floor(np.asarray(times)*MEL_RATE+.5).astype(int)
    indices=centers[:,None]+np.arange(-50,51)[None]
    valid=(indices>=0)&(indices<len(matrix))
    windows=np.asarray(matrix[np.clip(indices,0,len(matrix)-1)],np.float32)
    windows[~valid]=0
    return dict(mel=windows,mel_padding=~valid)


def tracked_timing(path, *, bpm_hint=None):
    samples,rate,_=read_audio(path)
    mono=librosa.resample(samples.mean(1),orig_sr=rate,target_sr=24000)
    onset=librosa.onset.onset_strength(y=mono,sr=24000,hop_length=240)
    # A time-varying tempo estimate allows local acceleration/deceleration.
    tempo=librosa.feature.tempo(onset_envelope=onset,sr=24000,hop_length=240,
                                start_bpm=bpm_hint or 120.,aggregate=None)
    _,frames=librosa.beat.beat_track(onset_envelope=onset,sr=24000,hop_length=240,bpm=tempo,trim=False)
    beats=librosa.frames_to_time(frames,sr=24000,hop_length=240)
    if len(beats)<2 or (np.diff(beats)<=0).any():
        raise ValueError('Could not track at least two beats; use a track with a detectable rhythm')
    return timing_from_beats(beats)


def timing_from_beats(beats):
    beats=np.asarray(beats,float)
    if len(beats)<2 or not np.isfinite(beats).all() or beats[0]<0 or (np.diff(beats)<=0).any():
        raise ValueError('Invalid beat timeline')
    # The first tracked beat is phase zero, not an asserted musical downbeat.
    return [dict(start_s=float(t),beat_position=float(i),bpm=float(60/d))
            for i,(t,d) in enumerate(zip(beats,np.r_[np.diff(beats),np.diff(beats)[-1]]))]


def time_at_beat(beat,timing):
    origins=np.array([p['beat_position'] for p in timing])
    i=max(0,int(np.searchsorted(origins,beat,side='right')-1));p=timing[i]
    return p['start_s']+(beat-p['beat_position'])*60/p['bpm']
