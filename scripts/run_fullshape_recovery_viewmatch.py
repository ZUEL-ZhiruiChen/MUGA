import json
import subprocess
import sys
from pathlib import Path

CONFIGS = [
    'actor_fullshape_viewmatch.yaml',
    'cornell_fullshape_viewmatch.yaml',
    'chameleon_filtered_directed_fullshape_viewmatch_balanced.yaml',
    'squirrel_filtered_fullshape_viewmatch.yaml',
    'squirrel_filtered_directed_fullshape_viewmatch.yaml',
]
ROOT = Path('configs/oracle/fullshape_recovery_viewmatch')
OUT = Path('records/fullshape_recovery_viewmatch_oracle_2026-04-30')
OUT.mkdir(parents=True, exist_ok=True)

for name in CONFIGS:
    cfg = ROOT / name
    variant = name.replace('.yaml', '')
    cmd = [sys.executable, 'run_node.py', '--config', str(cfg), '--output-dir', str(OUT), '--variant', variant]
    print('\nRUN', ' '.join(cmd), flush=True)
    subprocess.check_call(cmd)

rows = []
for p in OUT.rglob('summary.json'):
    o = json.loads(p.read_text(encoding='utf-8'))
    run = o['runs'][0]
    rows.append({
        'dataset': o['dataset'],
        'variant': o['variant'],
        'metric': o['metric'],
        'best_select': run['best_select_score'],
        'best_val': run['best_val_score'],
        'best_test': run['best_test_score'],
        'best_epoch': run['best_epoch'],
        'regime_mean': run.get('best_regime_mean'),
        'gate_mean': run.get('best_gate_mean'),
        'path_val': run.get('best_path_val_scores'),
        'path_test': run.get('best_path_test_scores'),
        'summary': str(p),
    })
rows = sorted(rows, key=lambda r: r['dataset'])
print('\nRESULTS')
for r in rows:
    print(r['dataset'], r['best_select'], 'val', r['best_val'], 'test', r['best_test'], 'epoch', r['best_epoch'], 'regime', r['regime_mean'])
(OUT / 'aggregate_viewmatch.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
