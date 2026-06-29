import json, subprocess, sys
from pathlib import Path
configs = [
 'actor_fullshape_freeze.yaml',
 'cornell_fullshape_freeze.yaml',
 'squirrel_filtered_fullshape_freeze.yaml',
 'squirrel_filtered_directed_fullshape_freeze.yaml',
 'chameleon_filtered_directed_fullshape_balanced_freeze.yaml',
]
root=Path('configs/oracle/fullshape_recovery_freeze')
out=Path('records/fullshape_recovery_freeze_oracle_2026-04-30')
out.mkdir(parents=True, exist_ok=True)
for name in configs:
    cfg=root/name
    variant=name.replace('.yaml','')
    cmd=[sys.executable,'run_node.py','--config',str(cfg),'--output-dir',str(out),'--variant',variant]
    print('\nRUN',' '.join(cmd),flush=True)
    subprocess.check_call(cmd)
rows=[]
for p in out.rglob('summary.json'):
    o=json.loads(p.read_text(encoding='utf-8'))
    rows.append((o['dataset'],o['variant'],o['metric'],o.get('best_val_mean'),o.get('best_test_mean'),o.get('best_select_score'),o.get('best_select_seed'),str(p)))
for r in sorted(rows): print(r)
(out/'aggregate_freeze.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
