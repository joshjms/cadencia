"""Descriptive grid/onset alignment diagnostics for existing generated artifacts."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
from cadencia.osumania.artifact import read_artifact
from cadencia.high_resolution import tracked_timing
from cadencia.beat_alignment import audio_onsets
if __package__:
    from .profile_beat_offsets import timing_observations
else:
    from profile_beat_offsets import timing_observations


def summary(values):
    values=np.asarray(values,float)
    return dict(count=len(values),signed_median_ms=float(np.median(values)),median_absolute_ms=float(np.median(np.abs(values))),p90_absolute_ms=float(np.quantile(np.abs(values),.9)),within_2ms=float(np.mean(np.abs(values)<=2)),within_10ms=float(np.mean(np.abs(values)<=10)),within_20ms=float(np.mean(np.abs(values)<=20)))


def nearest_offsets(times,reference):
    reference=np.asarray(reference)
    if not len(reference):return np.full(len(times),np.nan)
    i=np.searchsorted(reference,times)
    before=reference[np.maximum(i-1,0)];after=reference[np.minimum(i,len(reference)-1)]
    return (times-np.where(np.abs(times-before)<np.abs(times-after),before,after))*1000


def analyze(path):
    chart,audio=read_artifact(path)
    original=chart.get('timing_map') or [dict(start_s=chart.get('beat_offset_s',0.),beat_position=0.,bpm=chart['bpm'])]
    # Work per onset group so large chords do not dominate timing evidence.
    groups={}
    for note in chart['notes']:groups.setdefault(note['time_s'],note)
    notes=[dict(n,lane=0,start_s=n['time_s']) for n in groups.values()]
    trace=dict(lanes={'0':notes},timing_map=original)
    model=timing_observations(trace)
    with tempfile.TemporaryDirectory() as directory:
        wav=Path(directory)/'audio.wav';wav.write_bytes(audio)
        tracked=tracked_timing(wav,bpm_hint=chart['bpm'])
        peaks,strengths=audio_onsets(wav)
    trace['timing_map']=tracked;local=timing_observations(trace)
    times=model['time_s'];onset_offsets=nearest_offsets(times,peaks)
    strong=peaks[strengths>=.3];strong_offsets=nearest_offsets(times,strong)
    beat_times=np.array([p['start_s'] for p in tracked])
    constant_beats=chart.get('beat_offset_s',0.)+np.arange(-10,int(chart['duration_s']*chart['bpm']/60)+20)*60/chart['bpm']
    grid_drift=nearest_offsets(beat_times,constant_beats)
    sections=[]
    for start in np.arange(0,chart['duration_s'],30):
        selected=(times>=start)&(times<start+30)
        beat_selected=(beat_times>=start)&(beat_times<start+30)
        if selected.any():sections.append(dict(start_s=float(start),heads=int(selected.sum()),common_grid=summary(model['common_grid_offset_ms'][selected]),nearest_onset=summary(onset_offsets[selected]),tracked_beat_vs_constant_median_ms=float(np.median(grid_drift[beat_selected])) if beat_selected.any() else None))
    output=dict(file=str(path),notes=len(chart['notes']),holds=sum(n.get('kind')=='hold' for n in chart['notes']),onset_groups=len(times),duration_s=chart['duration_s'],conditioning_stars=chart['difficulty'],generation_bpm=chart['bpm'],generation_offset_s=chart.get('beat_offset_s',0.),generation_grid=summary(model['common_grid_offset_ms']),tracked_grid=summary(local['common_grid_offset_ms']),nearest_acoustic_onset=summary(onset_offsets),nearest_strong_acoustic_onset=summary(strong_offsets),tracked_beat_vs_constant=summary(grid_drift),sections=sections)
    arrays=dict(time_s=times,generation_grid_ms=model['common_grid_offset_ms'],tracked_grid_ms=local['common_grid_offset_ms'],nearest_onset_ms=onset_offsets,tracked_beat_s=beat_times,tracked_vs_constant_ms=grid_drift)
    return output,arrays


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('charts',type=Path,nargs='+');p.add_argument('--out',type=Path,default=Path('data/analysis/generated_alignment'));args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    reports=[];arrays=[]
    for i,path in enumerate(args.charts):
        print('Analyzing',path.name,flush=True)
        report,data=analyze(path);reports.append(report);arrays.append(data);np.savez_compressed(args.out/f'chart_{i}.npz',**data)
        print(json.dumps({k:v for k,v in report.items() if k!='sections'}),flush=True)
    (args.out/'report.json').write_text(json.dumps(dict(caveat='Tracked beats and detected onset peaks are estimates, not timing ground truth. Signed onset difference is note time minus nearest detected peak. Residual to common subdivisions does not identify intended rhythmic position. No charts modified.',charts=reports),indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(reports),3,figsize=(16,3.5*len(reports)),layout='constrained',squeeze=False)
    for row,(report,data) in enumerate(zip(reports,arrays)):
        title=Path(report['file']).name.split(' (')[0]
        axes[row,0].hist(data['generation_grid_ms'],bins=np.arange(-51,52,2),weights=np.ones(len(data['time_s']))/len(data['time_s'])*100,alpha=.65,label='Generation grid')
        axes[row,0].hist(data['tracked_grid_ms'],bins=np.arange(-51,52,2),weights=np.ones(len(data['time_s']))/len(data['time_s'])*100,histtype='step',linewidth=1.5,label='Tracked grid')
        axes[row,0].set(title=title,xlabel='Nearest common subdivision residual (ms)',ylabel='Onset groups per bin (%)');axes[row,0].legend()
        axes[row,1].scatter(data['time_s'],data['nearest_onset_ms'],s=4,alpha=.4)
        axes[row,1].set(xlabel='Song time (s)',ylabel='Note − nearest acoustic peak (ms)',ylim=(-150,150))
        axes[row,2].plot(data['tracked_beat_s'],data['tracked_vs_constant_ms'],linewidth=1)
        axes[row,2].set(xlabel='Song time (s)',ylabel='Tracked beat − nearest constant beat (ms)')
        for ax in axes[row]:ax.axhline(0,color='black',alpha=.2);ax.grid(alpha=.15)
    fig.suptitle('Generated-chart timing diagnostics — beat/onset estimates are not ground truth',fontsize=14)
    fig.savefig(args.out/'comparison.png',dpi=150)
    print('Saved',args.out/'comparison.png')


if __name__=='__main__':main()
