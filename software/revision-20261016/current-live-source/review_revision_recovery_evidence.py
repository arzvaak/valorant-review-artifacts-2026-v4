"""Inspect real completed execution records affected by recovery review findings."""
import argparse
from pathlib import Path
import json
import torch
from review_revision_diagnostics import STORE,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise RuntimeError('Preserve prior evidence')
    root=STORE/'manuscript-review-player-fits-20261016-v1'
    rows=[]
    for path in sorted(root.glob('rolling_*/*/*/2026*/frozen.json')):
        r=read(path);directory=path.parent;attempt=directory/f"batch-{r['batch']}"
        best=attempt/'best.pt';last=attempt/'last.pt'
        best_state=torch.load(best,map_location='cpu',weights_only=False)
        last_state=torch.load(last,map_location='cpu',weights_only=False)
        telemetry=[json.loads(line) for line in (attempt/'epochs.jsonl').read_text().splitlines()]
        expected=None;value=float('inf')
        for epoch in telemetry:
            if epoch['stopping_log_loss']<value-1e-6:expected=epoch['epoch'];value=epoch['stopping_log_loss']
        record=dict(seed_receipt=str(path),epochs=len(telemetry),recorded_best_epoch=r['best_epoch'],checkpoint_best_epoch=best_state['epoch']+1,expected_best_epoch=expected,
            last_best=last_state['best'],telemetry_best=value,
            checkpoint_hash_matches=sha(best)==r['best_checkpoint_sha256'],prediction_hash_matches=sha(directory/'predictions.npz')==r['predictions'])
        if not (r['best_epoch']==best_state['epoch']+1==expected and last_state['best']==value and record['checkpoint_hash_matches'] and record['prediction_hash_matches']):raise RuntimeError('Affected checkpoint discrepancy: '+str(path))
        rows.append(record)
    if len(rows)!=816:raise RuntimeError('Unexpected completed seed inventory')
    feedback=STORE/'manuscript-review-feedback-20261016-v3';manifest=read(feedback/'generated-code.json')
    for name,h in manifest['files'].items():
        if sha(feedback/name)!=h:raise RuntimeError('Original generated execution code changed')
    dependencies={name:dict(current=sha(Path(__file__).parent/name),original=sha(STORE/'later-evaluation-20261011-v1/code'/name)) for name in ('later_reference_state.py','later_residual_features.py','later_retained_inference.py','later_score.py')}
    if any(r['current']!=r['original'] for r in dependencies.values()):raise RuntimeError('Original feedback dependency changed')
    save(args.output,dict(code=sha(__file__),completed_seed_records=rows,feedback_generated_manifest=sha(feedback/'generated-code.json'),feedback_dependencies=dependencies,
        interpretation='Actual completed executions have consistent selected checkpoints and unchanged generator dependencies. This does not simulate interrupted execution or claim a test suite passed. Future recovery fixes carry new code identity; original numerical artifacts are preserved.'))
    print('COMPLETED EXECUTION EVIDENCE',len(rows),'seeds; original feedback dependencies unchanged',flush=True)


if __name__=='__main__':main()
