"""Empirical beat-to-note distributions from original cached chart timestamps."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from cadencia.placement_data import tempo_at


def timing_observations(trace):
    """Use original heads, not the quantized 20/10ms training grids."""
    notes=[n for lane in trace['lanes'].values() for n in lane if n.get('analysis_eligible',True)]
    notes.sort(key=lambda n:(n['start_s'],n['lane']))
    times=np.array([n['start_s'] for n in notes],float)
    if not len(times):raise ValueError('No eligible note heads')
    bpms,beats=tempo_at(times,trace['timing_map'])
    # Recompute in float64: tempo_at returns float32 for model input packing,
    # which loses sub-ms precision late in long songs.
    points=trace['timing_map'];starts=np.array([p['start_s'] for p in points],float)
    index=np.maximum(0,np.searchsorted(starts,times,side='right')-1)
    origins=np.array([p['beat_position'] for p in points],float)
    tempos=np.array([p['bpm'] for p in points],float)
    beats=origins[index]+(times-starts[index])*tempos[index]/60
    bpms=tempos[index]
    # Invert candidate beat positions through the complete tempo map, so nearest
    # beats/subdivisions remain correct on either side of a tempo change.
    def at(positions):
        i=np.maximum(0,np.searchsorted(origins,positions,side='right')-1)
        return starts[i]+(positions-origins[i])*60/tempos[i]
    def nearest(divisions):
        low=np.floor(beats*divisions)/divisions;high=low+1/divisions
        lo_time,hi_time=at(low),at(high)
        choose_high=(hi_time-times)<=(times-lo_time)
        position=np.where(choose_high,high,low)
        return position,np.where(choose_high,hi_time,lo_time)
    nearest_beat,nearest_time=nearest(1)
    _,quarter_time=nearest(4)
    # Common musical grids: straight through 1/16 beat and triplets through
    # 1/12 beat. This is a reference-grid residual, not proven expressive timing.
    references=np.array([nearest(d)[1] for d in [1,2,3,4,6,8,12,16]])
    selected=np.abs(references-times).argmin(0)
    common=references[selected,np.arange(len(times))]
    # A separate conservative categorical view picks the coarsest subdivision
    # within 2 ms, allowing the source .osu's millisecond timestamp rounding.
    category=np.full(len(times),'other',dtype=object)
    for d in [1,2,3,4,6,8,12,16]:
        _,candidate=nearest(d)
        matched=(category=='other')&(np.abs(times-candidate)<=.002)
        category[matched]=f'1/{d} beat'
    return dict(time_s=times,lane=np.array([n['lane'] for n in notes]),
                kind=np.array(['hold_start' if n['kind']=='hold' else 'tap' for n in notes]),
                bpm=bpms,phase=beats%1,nearest_beat_offset_beats=beats-nearest_beat,
                nearest_beat_offset_ms=(times-nearest_time)*1000,
                sixteenth_note_offset_ms=(times-quarter_time)*1000,
                common_grid_offset_ms=(times-common)*1000,subdivision=category,
                onset_group=np.r_[True,np.diff(times)>1e-9])


def hist(values,edges):
    counts,_=np.histogram(values,bins=edges)
    if int(counts.sum())!=len(values):raise ValueError('Histogram dropped samples')
    return dict(edges=edges.tolist(),counts=counts.tolist(),probability=(counts/max(1,len(values))).tolist())


def summarize(arrays):
    if not len(arrays['phase']):return dict(heads=0)
    result=dict(heads=len(arrays['phase']))
    result['phase_within_beat']=hist(arrays['phase'],np.linspace(0,1,121))
    result['nearest_beat_offset_beats']=hist(arrays['nearest_beat_offset_beats'],np.linspace(-1,1,241))
    for key in ['nearest_beat_offset_ms','sixteenth_note_offset_ms','common_grid_offset_ms']:
        values=arrays[key]
        bound=max(5.,float(np.ceil(np.max(np.abs(values))/5)*5))
        result[key]=hist(values,np.arange(-bound,bound+5,5.))
        result[key]['quantiles_ms']=dict(zip(['p01','p05','p50','p95','p99'],np.quantile(values,[.01,.05,.5,.95,.99]).tolist()))
    labels,counts=np.unique(arrays['subdivision'],return_counts=True)
    result['coarsest_subdivision_within_2ms']={str(k):dict(count=int(v),probability=float(v/len(arrays['phase']))) for k,v in zip(labels,counts)}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/placement_dataset_v1'))
    p.add_argument('--charts',type=Path,default=Path('data/osumania/charts'))
    p.add_argument('--split',choices=['all','train','validation','test'],default='all')
    p.add_argument('--out',type=Path,default=Path('data/analysis/beat_offsets'))
    args=p.parse_args();manifest_path=args.data/'manifest.json';manifest=json.loads(manifest_path.read_text())
    entries=[e for e in manifest['charts'] if args.split=='all' or e['split']==args.split]
    collected=[];chart_counts={}
    args.out.mkdir(parents=True,exist_ok=True)
    fields=['chart_id','split','stars','time_s','lane','kind','bpm','phase','nearest_beat_offset_beats','nearest_beat_offset_ms','sixteenth_note_offset_ms','common_grid_offset_ms','subdivision','onset_group']
    with (args.out/'observations.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for e in entries:
            trace_path=args.charts/str(e['chart_id'])/'trace.json';raw=trace_path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=e['trace_sha256']:raise ValueError(f'Source trace changed: {trace_path}')
            values=timing_observations(json.loads(raw));n=len(values['phase'])
            for i in range(n):
                writer.writerow(dict(chart_id=e['chart_id'],split=e['split'],stars=e['difficulty'],**{k:v[i] for k,v in values.items()}))
            values['split']=np.full(n,e['split']);values['star_band']=np.full(n,int(e['difficulty']))
            collected.append(values);chart_counts[e['split']]=chart_counts.get(e['split'],0)+1
    all_values={k:np.concatenate([v[k] for v in collected]) for k in collected[0]}
    def subset(mask):return {k:v[mask] for k,v in all_values.items()}
    result=dict(schema='cadencia.beat_offset_distribution.v1',charts=len(entries),chart_counts=chart_counts,split=args.split,
                source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                reference='Annotated source chart timing map; quarter-note beat. Original timestamps, no training-grid quantization. Positive offset means after the nearest beat.',
                caveat='Descriptive chart-to-annotated-grid offsets, not audio-onset latency or measured expressive timing. All-split profiles must not become training priors; use --split train.',
                note_weighted=summarize(all_values),onset_group_weighted=summarize(subset(all_values['onset_group'])),
                by_kind={kind:summarize(subset(all_values['kind']==kind)) for kind in ['tap','hold_start']},
                by_split={split:summarize(subset(all_values['split']==split)) for split in np.unique(all_values['split'])},
                by_star_band={str(int(band)):summarize(subset(all_values['star_band']==band)) for band in np.unique(all_values['star_band'])})
    (args.out/'distribution.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,10),layout='constrained')
    for ax,key,label in zip(axes,['phase_within_beat','nearest_beat_offset_ms','common_grid_offset_ms'],['Position within annotated beat (0 = beat, 0.5 = offbeat)','Time from nearest annotated beat (ms; positive = later)','Time from nearest common subdivision (ms)']):
        for name,color in [('note_weighted','#2864b7'),('onset_group_weighted','#d37b22')]:
            h=result[name][key];ax.stairs(np.array(h['probability'])*100,h['edges'],label='Each note' if name=='note_weighted' else 'Each chord/onset group',color=color)
        ax.set(xlabel=label,ylabel='Probability per bin (%)');ax.grid(alpha=.2);ax.legend()
    axes[2].set_xlim(-40,40)
    fig.suptitle(f'Beat-to-note timing: {len(entries):,} charts, {len(all_values["phase"]):,} heads\nOriginal chart timing; tap and hold starts; {args.split} splits',fontsize=14)
    fig.savefig(args.out/'distribution.png',dpi=160)
    print(json.dumps(dict(charts=result['charts'],heads=result['note_weighted']['heads'],onset_groups=result['onset_group_weighted']['heads'],subdivisions=result['note_weighted']['coarsest_subdivision_within_2ms'],quantiles=result['note_weighted']['common_grid_offset_ms']['quantiles_ms']),indent=2))
    print(args.out/'distribution.png')


if __name__=='__main__':main()
