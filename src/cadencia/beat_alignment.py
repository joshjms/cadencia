"""Conservative MAP timing refinement from a training-only circular beat-phase prior."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .high_resolution import time_at_beat

DEFAULT_PRIOR=Path(__file__).with_name('resources')/'beat_prior.json'


def load_prior(path=DEFAULT_PRIOR):
    data=Path(path).read_bytes();prior=json.loads(data)
    probability=np.asarray(prior['phase_probability'],float)
    if (prior.get('schema')!='cadencia.beat_alignment_prior.v1' or prior.get('split')!='train'
            or probability.shape!=(120,) or not np.isfinite(probability).all()
            or (probability<0).any() or not np.isclose(probability.sum(),1)):
        raise ValueError('Alignment requires a normalized, training-only 120-bin phase prior')
    # Circular smoothing respects that phases zero and one are the same beat.
    probability=.6*probability+.2*np.roll(probability,1)+.2*np.roll(probability,-1)
    return probability,hashlib.sha256(data).hexdigest()


def phase_probability(beat,probabilities):
    x=(beat%1)*len(probabilities)-.5
    lo=int(np.floor(x));fraction=x-lo
    return (1-fraction)*probabilities[lo%len(probabilities)]+fraction*probabilities[(lo+1)%len(probabilities)]


def beat_at_time(time,timing):
    i=max(0,int(np.searchsorted([p['start_s'] for p in timing],time,side='right')-1))
    p=timing[i];return p['beat_position']+(time-p['start_s'])*p['bpm']/60


def audio_onsets(path):
    import librosa
    from .audio import read_audio
    samples,rate,_=read_audio(path)
    mono=librosa.resample(samples.mean(1),orig_sr=rate,target_sr=24000)
    envelope=librosa.onset.onset_strength(y=mono,sr=24000,hop_length=120,n_fft=512)
    frames=librosa.onset.onset_detect(onset_envelope=envelope,sr=24000,hop_length=120)
    strengths=envelope[frames]
    scale=float(np.quantile(strengths,.9)) if len(strengths) else 1.
    return frames*120/24000,np.clip(strengths/max(scale,1e-8),0,1)


def align_chart(chart, *, prior_path=DEFAULT_PRIOR,max_shift_ms=25.,onsets=None,rhythm_evidence=None):
    """Move each chord as one unit, keeping duration, lane order and note count.

    The original time is always a candidate. MAP selection uses empirical phase
    probability, optional acoustic peak evidence, and a movement penalty. No
    sampling or new notes. Conservative forward constraints keep future original
    times feasible and avoid hold overlaps, even for touching hold/tap boundaries.
    """
    if not np.isfinite(max_shift_ms) or not 0<max_shift_ms<=50:
        raise ValueError('Alignment maximum shift must be in (0,50] ms')
    timing=chart.get('timing_map')
    if not timing:raise ValueError('Alignment needs a tracked beat timing_map')
    probabilities,digest=load_prior(prior_path)
    limit=max_shift_ms/1000
    notes=[dict(n) for n in chart['notes']]
    groups={}
    for i,n in enumerate(notes):groups.setdefault(n['time_s'],[]).append(i)
    group_times=sorted(groups)
    future={}
    for lane in range(4):
        lane_indices=[i for i,n in enumerate(notes) if n['lane']==lane]
        for a,b in zip(lane_indices,lane_indices[1:]):future[a]=notes[b]['time_s']
    occupied=[-1.]*4;previous_group=-1.;moved=0;shifts=[]
    peak_times,peak_strengths=(np.asarray(v) for v in onsets) if onsets is not None else (np.array([]),np.array([]))
    if len(peak_times)!=len(peak_strengths) or not np.isfinite(peak_times).all() or not np.isfinite(peak_strengths).all():
        raise ValueError('Invalid acoustic onset evidence')
    for position,time in enumerate(group_times):
        indices=groups[time];beat=beat_at_time(time,timing)
        confidence=1.
        if rhythm_evidence is not None:
            from .rhythm import beat_confidence
            confidence=float(np.clip(beat_confidence(rhythm_evidence,[time])[0],0,1))
        candidates={time}
        for denominator in [1,2,3,4,6,8,12,16]:
            center=int(np.floor(beat*denominator))
            for number in range(center-1,center+3):
                candidate=time_at_beat(number/denominator,timing)
                if abs(candidate-time)<=limit:candidates.add(candidate)
        candidates.update(float(t) for t in peak_times[np.abs(peak_times-time)<=limit])
        def score(candidate):
            prior=np.log(max(1e-8,phase_probability(beat_at_time(candidate,timing),probabilities)))
            evidence=float(np.max(peak_strengths*np.exp(-.5*((peak_times-candidate)/.008)**2))) if len(peak_times) else 0.
            return .6*confidence*prior+2*evidence-1.5*((candidate-time)/limit)**2
        def feasible(candidate):
            delta=candidate-time
            if candidate<0 or candidate>=chart['duration_s'] or candidate<=previous_group+1e-6:return False
            if position+1<len(group_times) and candidate>=group_times[position+1]-1e-6:return False
            for index in indices:
                n=notes[index]
                if candidate<occupied[n['lane']]-1e-9:return False
                end=n.get('end_s',time)+delta
                if n.get('kind')=='hold' and end>chart['duration_s']:return False
                if end>future.get(index,chart['duration_s'])+1e-9:return False
            return True
        if confidence<.2:candidates={time}
        valid=[c for c in candidates if feasible(c)]
        if not valid:raise ValueError('Input chart has overlapping or unordered notes')
        best=max(valid,key=lambda c:(score(c),-abs(c-time)))
        if score(best)<score(time)+.15:best=time
        delta=best-time
        if abs(delta)>1e-8:moved+=1
        shifts.append(delta*1000)
        for index in indices:
            n=notes[index];n['time_s']=best
            if n.get('kind')=='hold':n['end_s']+=delta
            occupied[n['lane']]=n.get('end_s',best+1e-6)
        previous_group=best
    result=dict(chart,notes=notes)
    result['alignment']=dict(schema='cadencia.beat_alignment.v1',prior_sha256=digest,
                             onset_groups=len(groups),moved_groups=moved,max_shift_ms=max_shift_ms,
                             mean_absolute_shift_ms=float(np.mean(np.abs(shifts))) if shifts else 0.)
    result['candidate_policy']=chart.get('candidate_policy','')+'; training-prior MAP beat/onset alignment'
    return result
