import json, subprocess, sys
from pathlib import Path

configs = [
    'actor_fullshape.yaml',
    'texas_fullshape.yaml',
    'cornell_fullshape.yaml',
    'wisconsin_fullshape.yaml',
    'chameleon_filtered_fullshape.yaml',
    'chameleon_filtered_directed_fullshape.yaml',
    'squirrel_filtered_fullshape.yaml',
    'squirrel_filtered_directed_fullshape.yaml',
]
root = Path('configs/oracle/fullshape_all')
out = Path('records/fullshape_all_oracle_2026-04-30')
out.mkdir(parents=True, exist_ok=True)
for cfg_name in configs:
    cfg = root / cfg_name
    variant = cfg_name.replace('.yaml','')
    cmd = [sys.executable, 'run_node.py', '--config', str(cfg), '--output-dir', str(out), '--variant', variant]
    print('\nRUN', ' '.join(cmd), flush=True)
    subprocess.check_call(cmd)

rows=[]
for p in out.rglob('summary.json'):
    o=json.loads(p.read_text(encoding='utf-8'))
    rows.append({
        'dataset': o['dataset'],
        'variant': o['variant'],
        'metric': o.get('metric'),
        'best_val': o.get('best_val_mean'),
        'best_test': o.get('best_test_mean'),
        'best_select': o.get('best_select_score'),
        'seed': o.get('best_select_seed'),
        'path': str(p),
    })
rows.sort(key=lambda r:r['dataset'])
(out/'aggregate_small.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
print('\nSMALL AGGREGATE')
for r in rows:
    print(r)
