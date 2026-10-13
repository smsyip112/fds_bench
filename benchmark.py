#!/usr/bin/env python3
"""M1/M2 NO-FDS exploratory benchmark for FIRE.

Truth: independent synthetic dynamic hazard process, NOT FDS and NOT the saved
V1 model. Predictor: explicit scenario bank; drop-in exported forecasts can
replace it after checkpoint/source validation. All policies have the same local
sensor, map, time, actions and physical constraints; no policy sees true future
except the ORACLE diagnostic reference.  See README.md for severe limitations.
"""
import argparse, csv, json, heapq
from dataclasses import dataclass
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@dataclass
class World:
    level: str
    seed: int
    G: nx.Graph
    free: np.ndarray
    start: tuple
    goal: tuple
    sources: tuple
    gt: np.ndarray     # [4, T, H, W] generated independently of policy
    hyp: np.ndarray    # [4, T, H, W] forecasts, procedural, no trained model


def layout(level, seed, T=125, family="easy"):
    rng=np.random.default_rng(seed)
    if level=='M1':
        width,height=37,27
        xs=[2,10,18,26,34];ys=[3,10,17,24]
        gap_plan=[(10,18,26),(17,10,18)]
        ignition=9; spread=3.7
    else:
        width,height=55,39
        xs=[2,11,20,29,38,47,52];ys=[3,11,19,27,35]
        gap_plan=[(11,20,29),(19,11,20),(27,29,38)]
        ignition=12; spread=4.1
    if family not in ('easy','branch_dispute'):
        raise ValueError(family)
    if family=='branch_dispute':
        if level=='M1':
            gap_plan=[(10,18,26),(17,18,26)]
            ignition=4;spread=1.30
        else:
            gap_plan=[(11,20,38),(19,20,38),(27,20,38)]
            ignition=5;spread=1.25
    G=nx.Graph()
    for y in ys:
        for x in range(xs[0],xs[-1]+1):
            G.add_node((x,y))
            if x>xs[0]:G.add_edge((x-1,y),(x,y))
    for x in xs:
        for y in range(ys[0],ys[-1]+1):
            G.add_node((x,y))
            if y>ys[0]:G.add_edge((x,y-1),(x,y))
    # Open-space shortcut between major corridors is genuinely sealed by
    # fixed walls, not invented only for the active policy.
    for y,x0,x1 in gap_plan:
        for x in range(x0,x1):
            if G.has_edge((x,y),(x+1,y)):G.remove_edge((x,y),(x+1,y))
    # Random extra doors / obstacles, but never disconnect the traversable map.
    cells=list(G.edges())
    for _ in range(10 if level=='M1' else 22):
        u,v=cells[int(rng.integers(len(cells)))]
        if G.has_edge(u,v):
            G.remove_edge(u,v)
            if not nx.is_connected(G):G.add_edge(u,v)
    start=(xs[0],ys[1]); goal=(xs[-1],ys[-2])
    if not nx.has_path(G,start,goal):raise AssertionError('layout disconnected')
    free=np.zeros((height,width),dtype=np.uint8)
    for x,y in G.nodes:free[y,x]=1
    ix=2 if level=='M1' else 3
    if family=='branch_dispute':
        src=[(xs[ix],ys[0]),(xs[ix],ys[-1]),(xs[ix+1],ys[0]),(xs[ix+1],ys[-1])]
    else:
        src=[(xs[ix],ys[0]),(xs[ix],ys[-1]),(xs[-2],ys[1]),(xs[1],ys[-2])]
    # Synthetic propagation along traversable graph; deliberately NOT physics.
    # Truth includes hidden spread perturbation. Forecast bank uses nominal
    # propagation: prediction and truth do not share perfect trajectories.
    gt=np.zeros((4,T,height,width),dtype=np.float32)
    hyp=np.zeros_like(gt)
    for mode,source in enumerate(src):
        if family=='branch_dispute':
            # Simplified building zone resistance, NOT smoke physics:
            # vertical transfer is slower, so a source on one bypass does
            # not necessarily poison the other before the robot arrives.
            dist=nx.single_source_dijkstra_path_length(G,source,
                    weight=lambda a,b,d: 8.0 if a[0]==b[0] else 1.0)
        else:
            dist=nx.single_source_shortest_path_length(G,source)
        ign=ignition + (mode%2)
        true_rate=spread*(1+float(rng.uniform(-.16,.16)))
        nom_rate=spread
        for (x,y),d in dist.items():
            tau_true=ign+true_rate*d
            tau_nom=ign+nom_rate*d
            tt=np.arange(T,dtype=np.float32)
            gt[mode,:,y,x]=np.clip((tt-tau_true)/6.0,0,1)
            hyp[mode,:,y,x]=np.clip((tt-tau_nom)/6.0,0,1)
    return World(level,seed,G,free,start,goal,tuple(src),gt,hyp)



def load_causal_forecast(world, folder):
    """Read ONE causal prefix forecast bank for a world (no silent fallback).

    Importing weights/replaying FIRE predictor is outside this prototype.
    Export a test-world independent array first; this checks only the handoff.
    If the predictor saw held-out future snapshots, the test is INVALID.
    """
    p=Path(folder)/f'{world.level}_{world.seed}.npz'
    if not p.is_file():
        raise FileNotFoundError(f'Missing forecast {p}; do not substitute truth')
    with np.load(p,allow_pickle=False) as z:
        need=['hazard_prob','free_mask','origin_t','causal','trained_on_test','model_id']
        missing=[k for k in need if k not in z.files]
        if missing:raise ValueError(f'{p}: missing keys {missing}')
        h=z['hazard_prob'];free=z['free_mask']
        if h.shape !=world.hyp.shape or not np.array_equal(free,world.free):
            raise ValueError('forecast shape/geometry do not match world')
        if not np.all(np.isfinite(h)) or h.min()<0 or h.max()>1:
            raise ValueError('hazard_prob must be finite and within [0,1]')
        if int(z['origin_t'])!=0 or int(z['causal'])!=1 or int(z['trained_on_test'])!=0:
            raise ValueError('strict initial-prefix contract failed: origin_t==0, causal==1, trained_on_test==0 required')
        if str(z['model_id'])=='procedural-ground-truth':
            raise ValueError('ground truth is not an admissible predictor')
        world.hyp=h.astype(np.float32,copy=True)
    return str(p)


def shared_safe_reachability(world, modes, deadline, safety_threshold=.50):
    """Audit known-truth geometry for a single OPEN-LOOP route safe in all modes.

    This is an offline diagnostic (never called by a navigation policy).
    Finding a common safe path rules out mandatory active info acquisition
    for success under this simplistic known-deadline threshold definition.
    Its absence is NOT proof that active information is necessary.
    """
    start=world.start
    if any(world.gt[m,0,start[1],start[0]]>=safety_threshold for m in modes):return None
    frontier={start}
    for t in range(deadline+1):
        if world.goal in frontier:return t
        if t==deadline:break
        next_front=set()
        for pos in frontier:
            for nxt in list(world.G.neighbors(pos))+[pos]:
                x,y=nxt
                if all(world.gt[m,t+1,y,x]<safety_threshold for m in modes):
                    next_front.add(nxt)
        frontier=next_front
        if not frontier:break
    return None

def observe(world, mode, pos, t, radius=5):
    # Same range/line-of-sight and deterministic noise for all policies.
    x,y=pos
    nodes=[pos]
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        u=pos
        for r in range(radius):
            v=(u[0]+dx,u[1]+dy)
            if not world.G.has_edge(u,v):break
            nodes.append(v);u=v
    # Sensor observation independent from candidate forecasts.
    vals=[]
    for px,py in nodes:
        k=(world.seed*7717 + mode*1777 + t*951 + px*41 + py*83)&0xffffffff
        noise=np.random.default_rng(k).normal(0,0.035)
        vals.append((px,py,float(np.clip(world.gt[mode,t,py,px]+noise,0,1))))
    return vals


def update_posterior(world, logw, obs, t):
    # Conditional likelihood under nominal scenario forecasts; not trained.
    # Cap evidence by subsampling to reduce the pretend independence of
    # highly-correlated nearby cells.
    selected=obs[::max(1,len(obs)//5)]
    for x,y,value in selected:
        pred=world.hyp[:,t,y,x]
        logw += -0.5*((value-pred)/0.105)**2
    logw -= logw.max()
    p=np.exp(logw);p/=p.sum()
    return np.log(np.maximum(p,1.e-20))


def shortest_path(world,pos,goal,t,policy,posterior,seen,oracle_mode=None):
    geom=nx.single_source_shortest_path_length(world.G,pos)
    if policy=='oracle':
        if oracle_mode is None:raise ValueError('oracle must receive hidden mode (diagnostic only)')
        weights=None
    else:weights=posterior
    danger={}
    for x,y in world.G.nodes:
        at=min(world.hyp.shape[1]-1,t+int(geom.get((x,y),200)))
        if policy=='oracle':
            risk=float(world.gt[oracle_mode,at,y,x])
        elif policy in ('reactive','geometry'):
            o=seen.get((x,y),None)
            risk=0.0 if policy=='geometry' or o is None else (o[0] if t-o[1]<=6 else .0)
        else:
            risk=float(np.dot(weights,world.hyp[:,at,y,x]))
        danger[(x,y)]=risk
    if policy=='preserve':
        # A small penalty for non-junction nodes in the same graph. This is
        # only a conservative heuristic, NOT a certified safe option set.
        def weight(u,v,d):return 1.0+19.0*danger[v]**2+(0.22 if world.G.degree(v)<=2 else 0)
    else:
        risk_weight=19.0 if policy!='geometry' else 0.0
        def weight(u,v,d):return 1.0+risk_weight*danger[v]**2
    return nx.shortest_path(world.G,pos,goal,weight=weight)


def choose_probe(world,pos,t,posterior):
    # Active variant pays real movement, has the SAME sensing radius and
    # model forecasts. It may choose one waypoint for additional information.
    if max(posterior)>.86:return None
    candidates=list(world.G.nodes)
    dist=nx.single_source_shortest_path_length(world.G,pos,cutoff=12)
    # Sample positions before the final third of the map; probed information
    # must arrive while the robot could still change route.
    cand=[p for p in candidates if p in dist and 3<=dist[p]<=12 and p[0]<world.goal[0]-8]
    best=(-1,None)
    for p in cand:
        at=min(world.hyp.shape[1]-1,t+dist[p])
        x,y=p
        vals=world.hyp[:,at,y,x]
        var=float(np.dot(posterior,(vals-np.dot(posterior,vals))**2))
        # use variance of all locally visible cells, not ground-truth hazard
        signal=var
        for u in world.G.neighbors(p):
            xx,yy=u;a=world.hyp[:,at,yy,xx]
            signal+=.3*float(np.dot(posterior,(a-np.dot(posterior,a))**2))
        priority=signal/(3.5+float(dist[p]))
        if priority>best[0]:best=(priority,p)
    return best[1] if best[0]>.0015 else None


def rollout(world,mode,policy,deadline=110,trace=False):
    t=0;pos=world.start;logw=np.log(np.full(4,.25))
    seen={};path=[pos];danger=0.;exposure=0.;scans=0;probe_target=None
    probed=False;fail='timeout';new_observations=0
    while t<deadline:
        obs=observe(world,mode,pos,t)
        new_observations+=len(obs)
        if policy not in ('geometry','reactive'):
            logw=update_posterior(world,logw,obs,t)
        for x,y,value in obs:seen[(x,y)]=(value,t)
        posterior=np.exp(logw);posterior/=posterior.sum()
        if pos==world.goal:
            fail='success';break
        if policy=='active' and not probed and probe_target is None and t<=24:
            target=choose_probe(world,pos,t,posterior)
            if target is not None:
                probe_target=target;scans+=1
        if probe_target==pos:
            probe_target=None;probed=True
        target=probe_target if probe_target is not None else world.goal
        plan_policy='passive' if policy=='active' else policy
        try:
            plan=shortest_path(world,pos,target,t,plan_policy,posterior,seen,oracle_mode=mode if policy=='oracle' else None)
        except (nx.NetworkXNoPath,nx.NodeNotFound):
            fail='no-path';break
        if len(plan)<2:
            if probe_target is not None:
                probe_target=None;probed=True;continue
            fail='stuck';break
        pos=plan[1];t+=1;path.append(pos)
        x,y=pos;h=float(world.gt[mode,t,y,x])
        danger+=h;exposure+=float(h>=.5)
        # Independent true hazard is used ONLY for outcome evaluation.
        if h>=.91:
            fail='hazard-stop';break
    success=int(pos==world.goal and fail in ('success','timeout'))
    if success:fail='success'
    return dict(success=success,reason=fail,travel=t,exposure=exposure,integrated_hazard=round(danger,3),
                scans=scans,posterior_true=round(float(np.exp(logw)[mode]/np.exp(logw).sum()),4),
                observations=new_observations,trace=path if trace else [])


def run(nmaps,seed,out,forecast_dir=None,family="easy"):
    out.mkdir(parents=True,exist_ok=True)
    rows=[];shown=None;audit=[]
    policies=['geometry','reactive','passive','active','preserve','oracle']
    for lvl in ['M1','M2']:
        for mi in range(nmaps):
            w=layout(lvl,seed+mi+ (10000 if lvl=='M2' else 0),family=family)
            if forecast_dir is not None:load_causal_forecast(w,forecast_dir)
            deadline=90 if lvl=='M1' else 120
            common=shared_safe_reachability(w,range(4),deadline)
            per_world=[shared_safe_reachability(w,[m],deadline) for m in range(4)]
            audit.append(dict(level=lvl,map_seed=w.seed,common_safe_exists=int(common is not None),
                              earliest_common_safe_t=common,each_scenario_safe=int(all(x is not None for x in per_world)),
                              possible_information_necessity_candidate=int(common is None and all(x is not None for x in per_world))))
            for mode in range(4):
                for policy in policies:
                    r=rollout(w,mode,policy,deadline=90 if lvl=='M1' else 120,
                              trace=bool(lvl=='M1' and mi==0 and mode==0))
                    rows.append(dict(level=lvl,map_seed=w.seed,mode=mode,policy=policy,
                                     **{k:v for k,v in r.items() if k!='trace'}))
                    if lvl=='M1' and mi==0 and mode==0:
                        if shown is None:shown=(w,{})
                        shown[1][policy]=r['trace']
        print('completed',lvl,flush=True)
    with (out/'trial_results.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    with (out/'geometry_necessity_audit.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=audit[0].keys());wr.writeheader();wr.writerows(audit)
    sm=[]
    for lvl in ['M1','M2']:
        for policy in policies:
            data=[r for r in rows if r['level']==lvl and r['policy']==policy]
            sm.append(dict(level=lvl,policy=policy,n=len(data),success_rate=round(float(np.mean([r['success'] for r in data])),3),
                           mean_travel=round(float(np.mean([r['travel'] for r in data])),2),
                           mean_exposure=round(float(np.mean([r['exposure'] for r in data])),2),
                           mean_integrated_hazard=round(float(np.mean([r['integrated_hazard'] for r in data])),2),
                           hazard_fail_rate=round(float(np.mean([r['reason']=='hazard-stop' for r in data])),3),
                           scan_rate=round(float(np.mean([r['scans']>0 for r in data])),3)))
    with (out/'summary.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=sm[0].keys());wr.writeheader();wr.writerows(sm)
    conf=dict(dataset_type='NO_FDS_PROCEDURAL_SYNTHETIC',predictor='nominal_scenario_bank_NOT_V1',levels=['M1','M2'],
              n_maps_per_level=nmaps,modes_per_map=4,seed=seed,policies=policies,
              forecast_source='PROCEDURAL_BANK' if forecast_dir is None else str(forecast_dir),family=family,
              policy_observation='same local line-of-sight range 5, Gaussian noise std 0.035',
              deadline_M1=90,deadline_M2=120,forbidden='DO NOT CLAIM FDS OR TRAINED MODEL RESULTS')
    (out/'config.json').write_text(json.dumps(conf,indent=2))
    fig,ax=plt.subplots(figsize=(9,5))
    xx=np.arange(len(policies));bar=.34
    for i,lvl in enumerate(['M1','M2']):
        vals=[next(r['success_rate'] for r in sm if r['level']==lvl and r['policy']==p) for p in policies]
        ax.bar(xx+(i-.5)*bar,vals,bar,label=lvl)
    ax.set(ylim=(0,1.05),ylabel='Goal arrival rate',title='Exploratory M1/M2 procedural-hazard benchmark (NOT FDS)')
    ax.set_xticks(xx,policies);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'success_comparison.png',dpi=160);plt.close(fig)
    if shown:
        world,paths=shown
        fig,axes=plt.subplots(1,2,figsize=(13,5.4))
        ax=axes[0];a=world.free
        ax.imshow(a,origin='lower',cmap='Greys',vmin=0,vmax=1)
        for i,(x,y) in enumerate(world.sources):ax.plot(x,y,marker='x',ms=8,label=f'possible source {i}')
        ax.scatter(*world.start,s=80,marker='o',label='start');ax.scatter(*world.goal,s=80,marker='*',label='goal')
        ax.set_title('M1 open corridors / scenario source hypotheses');ax.legend(fontsize=6,loc='upper center',ncol=2)
        ax=axes[1];t=29
        ax.imshow(world.gt[0,t],origin='lower',cmap='inferno',vmin=0,vmax=1)
        for p,route in paths.items():
            if route:
                points=np.array(route);ax.plot(points[:,0],points[:,1],label=p,lw=1.7)
        ax.plot(*world.start,'co');ax.plot(*world.goal,'g*');ax.legend(fontsize=8)
        ax.set_title(f'True hazard at t={t}, actual cumulative route traces')
        fig.tight_layout();fig.savefig(out/'example_map_and_routes.png',dpi=170);plt.close(fig)
    print(json.dumps(sm,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--n-maps',type=int,default=6)
    p.add_argument('--seed',type=int,default=21011);p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results')
    p.add_argument('--forecast-dir',type=Path,default=None,help='Causal initial-prefix NPZ files; errors if any missing')
    p.add_argument('--family',choices=['easy','branch_dispute'],default='easy')
    args=p.parse_args();run(args.n_maps,args.seed,args.out,args.forecast_dir,args.family)

if __name__=='__main__':main()
