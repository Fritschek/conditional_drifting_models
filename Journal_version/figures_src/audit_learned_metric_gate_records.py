"""Independent NumPy audit of the saved October SSPA gate; no Torch imports.

Run from any directory with NumPy installed. Requires the restored gate archive.
Checks sufficient statistics and archived summaries, not individual simulator draws.
Writes the pinned evidence JSON under Journal_version/evidence.
"""
import json,csv,hashlib,itertools,collections
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]
p=root/'results/learned_metric_gate_sspa_seed7_20261009/results.json'
d=json.loads(p.read_text()); rows=d['rows']; tasks=d['losses']; m=d['manifest']
methods=['fiber_sinkhorn','wgan','ddim10']; cheap=['swd','rbf_cross_value_squared','mean_derivative_cross','covariance_derivative_cross','moments_cross_trace','fourth_derivative_cross']; scores=cheap+['rbf_cross_trace']
maxerr=collections.defaultdict(float)
def check(name,a,b,tol=1e-10):
 e=float(np.max(np.abs(np.array(a)-np.array(b))))
 maxerr[name]=max(maxerr[name],e)
 assert np.allclose(a,b,rtol=tol,atol=tol),(name,e)
def stats(x):
 x=np.array(x,dtype=float);return x.mean(),x.std(ddof=1)/np.sqrt(len(x))
def sign(x):
 mean,se=stats(x);return int(np.sign(mean)) if abs(mean)>3*se+1e-12 else 0
mi={(r['samples'],r['anchor'],r['method'],r['repeat']):r for r in rows}
ti={(r['samples'],r['anchor'],r['method'],r['repeat'],r['probe']):r for r in tasks}
assert len(mi)==len(rows)==192 and len(ti)==len(tasks)==1536
assert set(mi)==set(itertools.product([512,2048],range(3),['analytic']+methods,range(8)))
assert set(ti)=={k+(p['name'],) for k in mi for p in m['probes']}
noise_mineig=float('inf')
for r in rows:
 for kind in ['rbf','moments','augmented']:
  s=np.array(r[kind+'_self_matrix']); c=np.array(r[kind+'_cross_matrix']); v=np.array(r[kind+'_noise_matrix'])
  check('self_average',s,(np.array(r[kind+'_split1_matrix'])+r[kind+'_split2_matrix'])/2)
  check('noise_identity',v,s-c)
  check('cross_trace',r[kind+'_cross_trace'],np.trace(c[1:,1:]))
  check('self_trace',r[kind+'_self_trace'],np.trace(s[1:,1:]))
  check('noise_trace',r[kind+'_estimated_noise_trace'],np.trace(v[1:,1:]))
  noise_mineig=min(noise_mineig,float(np.linalg.eigvalsh(v[1:,1:]).min()))
 for prefix in ['mean_value','mean_derivative','covariance_value','covariance_derivative','fourth_value','fourth_derivative']:
  a=np.array(r[prefix+'_split1']); b=np.array(r[prefix+'_split2'])
  check('moment_cross',r[prefix+'_cross'],np.sum(a*b))
  check('moment_noise',r[prefix+'_noise'],np.sum((a-b)**2)/2)
  check('moment_self',r[prefix+'_self'],(np.sum(a*a)+np.sum(b*b))/2)
for r in tasks:
 g=np.array(r['reference_gradient']); e1=np.array(r['gradient_split1'])-g; e2=np.array(r['gradient_split2'])-g
 check('task_cross',r['cross_squared_error'],e1@e2)
 check('task_self',r['self_squared_error'],(e1@e1+e2@e2)/2)
 check('task_noise',r['estimated_noise_squared'],np.sum((e1-e2)**2)/2)
 check('task_absolute_error',r['absolute_gradient_error'],np.linalg.norm((e1+e2)/2))
report=root/'results/learned_metric_gate_report_20261009'
for row in csv.DictReader((report/'metric_summary.csv').open()):
 key=(int(row['samples']),int(row['anchor']),row['method']); group=[mi[key+(r,)] for r in range(8)]
 for name,value in group[0].items():
  if isinstance(value,(int,float)) and name not in ['samples','anchor','repeat']:
   mean,se=stats([r[name] for r in group]);check('metric_csv',float(row[name+'_mean']),mean);check('metric_csv',float(row[name+'_mc_se']),se)
pooled={}
for row in csv.DictReader((report/'task_summary.csv').open()):
 key=(int(row['samples']),int(row['anchor']),row['method']); probe=row['probe'];group=[ti[key+(r,probe)] for r in range(8)]
 gs=np.array([(np.array(r['gradient_split1'])+r['gradient_split2'])/2 for r in group]);g=np.array(group[0]['reference_gradient'])
 e=np.linalg.norm(gs.mean(0)-g); pooled[key+(probe,)]=float(e)
 check('task_csv',float(row['pooled_absolute_error']),e)
 check('task_csv',float(row['gradient_mean_mc_se_l2']),np.linalg.norm(gs.std(0,ddof=1)/np.sqrt(8)))
 check('task_csv',float(row['reference_gradient_norm']),np.linalg.norm(g))
 for name in ['absolute_gradient_error','cross_squared_error','self_squared_error','estimated_noise_squared']:
  mean,se=stats([r[name] for r in group]);check('task_csv',float(row[name+'_mean']),mean);check('task_csv',float(row[name+'_mc_se']),se)
comparisons={}; per_kind={}; refinement=[]
for n in [512,2048]:
 comparisons[n]={s:{'agrees':0,'opposes':0,'unresolved':0} for s in scores};adds=resolved=0
 per_kind[n]={k:{s:{'agrees':0,'opposes':0,'unresolved':0} for s in scores} for k in ['rbf_section','quadratic','logistic','cosine']}
 for a in range(3):
  unresolved=informative=0
  for left,right in itertools.combinations(methods,2):
   ss={s:sign([mi[n,a,left,r][s]-mi[n,a,right,r][s] for r in range(8)]) for s in scores}
   for p0 in m['probes']:
    probe=p0['name'];order=sign([ti[n,a,left,r,probe]['cross_squared_error']-ti[n,a,right,r,probe]['cross_squared_error'] for r in range(8)])
    if not order: unresolved+=1;continue
    resolved+=1
    cheap_agrees=[s for s in cheap if ss[s]==order]
    informative+=bool(ss['rbf_cross_trace']==order and len(cheap_agrees)<len(cheap))
    adds+=bool(ss['rbf_cross_trace']==order and not cheap_agrees)
    for s in scores:
     k='unresolved' if not ss[s] else 'agrees' if ss[s]==order else 'opposes'
     comparisons[n][s][k]+=1;per_kind[n][p0['kind']][s][k]+=1
  if n==512: refinement.append(dict(anchor=a,refine=bool(unresolved or informative),unresolved_targets=unresolved,potentially_informative=informative))
 comparisons[n]['resolved_target_contrasts']=resolved;comparisons[n]['candidate_added_information']=adds
assert refinement==m['refinement']
checks=json.loads((report/'checks.json').read_text())
for s in scores: assert comparisons[2048][s]==checks['comparator_counts'][s]
assert comparisons[2048]['candidate_added_information']==checks['candidate_added_information']==0
winners=collections.Counter(); kwinners=collections.Counter(); bounds=[]; op_order=[]
for a in range(3):
 for pr in m['probes']:
  winner=min(methods,key=lambda model:pooled[2048,a,model,pr['name']]);winners[winner]+=1
  if pr['kind']=='rbf_section': kwinners[winner]+=1
 for model in methods:
  g=np.mean([np.array(mi[2048,a,model,r]['rbf_cross_matrix'])[1:,1:] for r in range(8)],axis=0)
  tr=float(np.trace(g)); op=float(np.linalg.eigvalsh(g)[-1]); ordinary=np.mean([mi[2048,a,model,r]['rbf_embedding_derivative_op'] for r in range(8)])
  op_order.append({'anchor':a,'method':model,'mean_cross_trace':tr,'largest_eigenvalue_mean_cross':op,'mean_ordinary_operator_norm':float(ordinary)})
  for pr in m['probes']:
   if pr['kind']=='rbf_section':bounds.append({'anchor':a,'method':model,'probe':pr['name'],'pooled_error':pooled[2048,a,model,pr['name']],'error_squared_over_cross_trace':pooled[2048,a,model,pr['name']]**2/tr})
source_matches={s['path']:hashlib.sha256((root/s['path']).read_bytes()).hexdigest()==s['sha256'] for s in m['sources']}
assert all(source_matches.values())
seeds=[100000000+n*100000+a*1000000+r*10000+role*1000+model*10+split for n in [512,2048] for a in range(3) for r in range(8) for role in range(3) for model in range(4) for split in range(2) if role>0 or model==0]
assert len(seeds)==len(set(seeds))
result={'commit':'90a444912ecfb060e4b5e2fa33cbe868c02824c3','raw_results_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'verification_scope':'Independent NumPy recomputation from saved sufficient statistics; no sample/Jacobian recomputation, Torch tests or simulation rerun. Largest-eigenvalue and bound-utilization summaries are post-hoc diagnostics, not unbiased norm estimates or calibrated bounds.','record_counts':{'metric':len(rows),'task':len(tasks)},'maximum_absolute_check_residuals':dict(maxerr),'minimum_noise_matrix_eigenvalue':noise_mineig,'source_hash_matches':source_matches,'distinct_seed_count':len(seeds),'refinement':refinement,'contrasts_by_N':comparisons,'comparators_by_loss_kind':per_kind,'pooled_error_winners':dict(winners),'kernel_section_error_winners':dict(kwinners),'operator_diagnostics':op_order,'kernel_section_bound_utilization':bounds}
(root/'Journal_version/evidence/learned_metric_gate_independent_audit_20261010.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'residuals':dict(maxerr),'noise_mineig':noise_mineig,'counts':comparisons[2048],'kernel_section_counts':per_kind[2048]['rbf_section'],'winners':dict(winners),'kernel_winners':dict(kwinners),'operator':op_order,'bound_ratio_range':[min(r['error_squared_over_cross_trace'] for r in bounds),max(r['error_squared_over_cross_trace'] for r in bounds)]},indent=2))
