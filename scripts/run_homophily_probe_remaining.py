import json, subprocess, sys
from pathlib import Path
configs = [
    'reed98_risk_router_homophily.yaml',
    'amherst41_ce_only.yaml',
    'amherst41_risk_router_homophily.yaml',
]
root = Path('configs/oracle/homophily_probe')
out = Path('records/homophily_probe_oracle_2026-05-01')
for name in configs:
    cfg = root / name
    variant = name.replace('.yaml', '')
    cmd = [sys.executable, 'run_node.py', '--config', str(cfg), '--output-dir', str(out), '--variant', variant]
    print('\nRUN', ' '.join(cmd), flush=True)
    subprocess.check_call(cmd)
rows=[]
for p in out.rglob('summary.json'):
    o=json.loads(p.read_text(encoding='utf-8'))
    best=max(o['runs'], key=lambda r:r['best_select_score'])
    rows.append({
      'dataset':o['dataset'], 'variant':o['variant'], 'metric':o['metric'],
      'best_select':o.get('best_select_score'), 'best_seed':o.get('best_select_seed'),
      'best_val':best.get('best_val_score'), 'best_test':best.get('best_test_score'), 'best_epoch':best.get('best_epoch'),
      'regime_mean':best.get('best_regime_mean'), 'gate_mean':best.get('best_gate_mean'),
      'path_val':best.get('best_path_val_scores'), 'path_test':best.get('best_path_test_scores'), 'summary':str(p)
    })
rows=sorted(rows, key=lambda r:(r['dataset'], r['variant']))
(out/'aggregate_homophily_probe.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
print('\nRESULTS')
for r in rows:
    print(r['dataset'], r['variant'], 'select', r['best_select'], 'seed', r['best_seed'], 'val', r['best_val'], 'test', r['best_test'], 'regime', r['regime_mean'])
