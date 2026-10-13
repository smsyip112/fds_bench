#!/usr/bin/env python3
"""Non-regression tests for scenario labels, information use and common-path audit."""
import tempfile,sys
from pathlib import Path
import numpy as np
from benchmark import layout,shared_safe_reachability,load_causal_forecast,rollout

for level,seed,deadline in [('M1',21011,90),('M2',31011,120)]:
    e=layout(level,seed,family='easy')
    s=layout(level,seed,family='branch_dispute')
    assert shared_safe_reachability(e,range(4),deadline) is not None
    assert shared_safe_reachability(s,range(4),deadline) is None
    assert all(shared_safe_reachability(s,[m],deadline) is not None for m in range(4))
    assert all(np.array_equal(s.gt[0,0],s.gt[i,0]) for i in range(1,4))
    assert np.all((s.hyp>=0)&(s.hyp<=1))
    with tempfile.TemporaryDirectory() as td:
        try:load_causal_forecast(s,td)
        except FileNotFoundError:pass
        else:raise AssertionError('missing export accepted')
        np.savez_compressed(Path(td)/f'{level}_{seed}.npz',
            hazard_prob=s.hyp,free_mask=s.free,origin_t=0,causal=1,trained_on_test=0,model_id='test-export')
        load_causal_forecast(s,td)
        np.savez_compressed(Path(td)/f'{level}_{seed}.npz',
            hazard_prob=s.gt,free_mask=s.free,origin_t=0,causal=0,trained_on_test=0,model_id='ground-truth')
        try:load_causal_forecast(s,td)
        except ValueError:pass
        else:raise AssertionError('leakage export accepted')
print('PASS: benchmark family, feasibility, causal prefix and forecast import checks')
