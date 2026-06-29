import json, subprocess, sys
from pathlib import Path
configs = [
 'chameleon_filtered_directed_fullshape_cematch.yaml',
 'chameleon_filtered_directed_fullshape_cematch_calib.yaml',
]
root=Path('configs/oracle/fullshape_recovery_viewmatch')
out=Path('records/fullshape_recovery_viewmatch_oracle_2026-04-30')
for name in configs:
    cfg=root/name
    variant=name.replace('.yaml','')
    cmd=[sys.executable,'run_node.py','--config',str(cfg),'--output-dir',str(out),'--variant',variant]
    print('\nRUN',' '.join(cmd),flush=True)
    subprocess.check_call(cmd)
for p in sorted((out/'chameleon_filtered_directed').glob('*/summary.json')):
    o=json.loads(p.read_text(encoding='utf-8'))
    r=o['runs'][0]
    print(p.parent.name, r['best_select_score'], 'val', r['best_val_score'], 'test', r['best_test_score'], 'epoch', r['best_epoch'], 'regime', r.get('best_regime_mean'), 'gate', r.get('best_gate_mean'))
