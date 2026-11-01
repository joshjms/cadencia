import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from cadencia.interval import encode
from cadencia.phrase_training import PhraseDataset, PhraseSampler, PhraseBatchSampler, transition_loss, grouped_transition_loss, forward_rows
from cadencia.planner import PlannedDataset, PlannedTransformer
from tests.test_chart_pretraining import dataset
from tests.test_joint_placement import config, inputs


def test_phrase_loss_has_finite_gradients_and_captures_order():
    target = torch.tensor([[1,0,0,0],[0,1,0,0],[1,0,0,0],[0,0,1,0]])
    logits = torch.full((4,80),-20.)
    logits.scatter_(1,(encode(target)-1)[:,None],20.)
    logits.requires_grad_()
    beats = torch.arange(4)*.25
    held = torch.zeros(4,4,dtype=torch.bool)
    same,pairs = transition_loss(logits,target,held,beats)
    reordered = target[[0,2,1,3]]
    other,_ = transition_loss(logits,reordered,held,beats)
    assert same < 1e-5 and other > .1 and pairs == 3
    other.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_empty_phrase_zero_loss_is_differentiable():
    logits = torch.randn(8,80,requires_grad=True)
    loss,pairs = transition_loss(logits,torch.zeros(8,4,dtype=torch.long),torch.zeros(8,4,dtype=torch.bool),torch.arange(8)/10)
    loss.backward()
    assert loss == 0 and pairs == 0 and torch.isfinite(logits.grad).all()


def test_phrase_windows_cover_charts_without_crossing_and_sampling_is_reproducible(dataset):
    root,_ = dataset
    base = PlannedDataset(root,'train')
    # The fixture's planner features can be supplied without a network here.
    base.plans = [(np.zeros((1,4),np.float32),np.zeros(e['rows'],int)) for e in base.entries]
    windows = PhraseDataset(base,beats=4.,training=True)
    covered = []
    for chart,start,end,offset in windows.windows:
        covered.extend(range(start,end))
        batch = windows[(2,len(covered)%len(windows))]
        assert 'phrase_beats' in batch and (batch['inclusion'] > 0).all()
    assert covered == list(range(base.entries[0]['rows']))
    a,b = windows[(3,0)],windows[(3,0)]
    torch.testing.assert_close(a['target'],b['target'])
    torch.testing.assert_close(a['phrase_beats'],b['phrase_beats'])
    # Every human onset survives negative sampling.
    expected = base._load(0)[0]['onsets'][:windows.windows[0][2]].any(-1).sum()
    assert int(a['target'].any(-1).sum()) == expected
    sampler = PhraseSampler(len(windows));first=list(sampler)
    assert {i for _,i in first} == set(range(len(windows)))
    assert all(epoch==0 for epoch,_ in first)
    assert all(epoch==1 for epoch,_ in sampler)


def test_chunked_forward_matches_and_checkpointing_backpropagates():
    from cadencia.interval import IntervalTransformer
    model = IntervalTransformer(config()).eval()
    batch = inputs()
    # Test helper uses unpacked per-query memory; convert it to packed form.
    n,frames,dim = batch['audio'].shape
    batch['audio'] = torch.randn(n*frames,dim)
    batch['audio_indices'] = torch.arange(n*frames).reshape(n,frames)
    expected = model(**batch)
    actual = forward_rows(model,batch,1,checkpointed=True)
    for key in expected: torch.testing.assert_close(expected[key],actual[key],rtol=1e-4,atol=1e-5)
    actual['event'].sum().backward()
    assert model.audio.weight.grad.abs().sum() > 0


def test_phrase_train_resume_and_inference_loading(dataset,tmp_path):
    root,_ = dataset
    planner=tmp_path/'planner.pt';initial=tmp_path/'initial.pt';out=tmp_path/'phrase.pt'
    def run(*args):
        result=subprocess.run([sys.executable,*map(str,args)],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    run('scripts/train_bar_planner.py','--data',root,'--output',planner,'--epochs',1)
    run('scripts/train_intervals.py','--data',root,'--planner',planner,'--output',initial,'--epochs',1,'--workers',0,'--batch-size',128)
    args=['scripts/train_phrases.py','--data',root,'--planner',planner,'--output',out,'--workers',0,'--chunk-rows',64]
    run(*args,'--initialize',initial,'--epochs',1)
    saved=torch.load(out,weights_only=True)
    assert saved['architecture_version']=='v1' and saved['epochs_completed']==1
    assert saved['phrase_training']['weight']==.05
    assert saved['phrase_training']['recipe']=='mixed_phrases_v2'
    assert saved['baseline_validation']['base_loss'] > 0
    assert saved['best_loss'] <= saved['baseline_validation']['base_loss']
    assert saved['initialization']['source_best_epoch']==1
    from cadencia.osumania import load_model
    assert load_model(out).planned
    run(*args,'--resume',out.with_suffix('.last.pt'),'--epochs',2)
    last=torch.load(out.with_suffix('.last.pt'),weights_only=True)
    assert last['epochs_completed']==2 and len(last['history'])==2
    assert 'scheduler' in last and 'optimizer' in last
    assert last['history'][-1]['validation']['phrase_transition'] >= 0


def test_mixed_phrase_batch_preserves_all_rows_ids_and_packed_audio(dataset):
    root,_ = dataset
    base = PlannedDataset(root,'train')
    base.plans = [(np.zeros((1,4),np.float32),np.zeros(e['rows'],int)) for e in base.entries]
    phrases = PhraseDataset(base,training=True)
    sampler = PhraseBatchSampler(phrases,128)
    batches = list(sampler)
    assert sorted(i for batch in batches for _,i in batch) == list(range(len(phrases)))
    assert all(epoch==0 for batch in batches for epoch,_ in batch)
    assert all(epoch==1 for batch in sampler for epoch,_ in batch)
    keys = [(0,i) for i in range(len(phrases))]
    packed = phrases.__getitems__(keys)
    assert int(packed['timing_normalizer']) == len(base)
    assert len(torch.unique(packed['phrase_ids'])) == len(phrases)
    for i,key in enumerate(keys):
        one = phrases[key]
        selected = packed['phrase_ids'] == i
        for name in ['history','history_timing','target','duration_target','phrase_beats','inclusion']:
            torch.testing.assert_close(packed[name][selected],one[name])
        torch.testing.assert_close(packed['audio'][packed['audio_indices'][selected]],one['audio'][one['audio_indices']])


def test_mixed_batches_keep_chart_boundaries_and_note_weights(dataset):
    root,_ = dataset
    base = PlannedDataset(root,'train')
    base.entries.append(dict(base.entries[0],chart_id=99))
    base.ends = np.cumsum([e['rows'] for e in base.entries])
    base.plans = [(np.zeros((1,4),np.float32),np.zeros(e['rows'],int)) for e in base.entries]
    phrases = PhraseDataset(base,training=False)
    left = next(i for i,w in enumerate(phrases.windows) if w[0]==0)
    right = next(i for i,w in enumerate(phrases.windows) if w[0]==1)
    batch = phrases.__getitems__([(0,left),(0,right)])
    assert batch['phrase_ids'].max()==1
    for phrase in (0,1):
        beats = batch['phrase_beats'][batch['phrase_ids']==phrase]
        assert (beats[1:]>beats[:-1]).all()
    assert int(batch['timing_normalizer'])==len(batch['target'])
    model = PlannedTransformer(base.config).eval()
    from scripts.train_intervals import run_epoch
    together = run_epoch(model,[batch],torch.device('cpu'),phrase_weight=.05)
    separate = run_epoch(model,[phrases[(0,left)],phrases[(0,right)]],torch.device('cpu'),phrase_weight=.05)
    for key in ['timing','pattern','duration','phrase_transition','base_loss']:
        assert together[key]==pytest.approx(separate[key],rel=1e-5,abs=1e-6)


def test_transition_loss_does_not_connect_independent_phrases():
    torch.manual_seed(4)
    logits = torch.randn(5,80,requires_grad=True)
    target = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])
    held = torch.zeros(5,4,dtype=torch.bool)
    beats = torch.tensor([0.,1.,0.,.25,1.])
    ids = torch.tensor([0,0,1,1,1])
    a,na = transition_loss(logits[:2],target[:2],held[:2],beats[:2])
    b,nb = transition_loss(logits[2:],target[2:],held[2:],beats[2:])
    mixed,pairs,windows = grouped_transition_loss(logits,target,held,beats,ids)
    torch.testing.assert_close(mixed,(a*na+b*nb)/(na+nb))
    assert pairs==3 and windows==2
    mixed.backward()
    assert torch.isfinite(logits.grad).all()


def test_fixed_grid_denominator_and_joint_note_weighting():
    from cadencia.interval import losses
    output = dict(event=torch.tensor([0.,2.,-1.],requires_grad=True),pattern=torch.randn(3,80,requires_grad=True),duration=torch.zeros(3,4,requires_grad=True))
    target = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,0]])
    inclusion = torch.tensor([1.,1.,.1])
    _,parts = losses(output,target,torch.zeros(3,4),torch.zeros(3,4,dtype=torch.bool),inclusion,timing_normalizer=torch.tensor(20.))
    bce = torch.nn.functional.binary_cross_entropy_with_logits(output['event'],target.any(-1).float(),reduction='none')
    torch.testing.assert_close(parts['timing'],(bce/inclusion).sum()/20)
    torch.testing.assert_close(parts['pattern'],torch.nn.functional.cross_entropy(output['pattern'][:2],encode(target[:2])-1))
    (parts['timing']+parts['pattern']).backward()
    assert torch.isfinite(output['event'].grad).all()
