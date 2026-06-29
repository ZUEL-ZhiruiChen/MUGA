import json, subprocess, sys
from pathlib import Path

configs = [
    'amazon_ratings_fullshape.yaml',
    'minesweeper_fullshape.yaml',
    'tolokers_fullshape.yaml',
    'roman_empire_fullshape.yaml',
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
(out/'aggregate_all.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
print('\nFULL AGGREGATE')
for r in rows:
    print(r)
