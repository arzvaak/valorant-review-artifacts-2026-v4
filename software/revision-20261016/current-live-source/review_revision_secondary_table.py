"""Typeset the completed secondary inventory in separate readable blocks."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    completion=read(args.results.parent/'completion.json')
    if sha(args.results)!=completion['results']:raise RuntimeError('Completed metric inventory changed')
    records=read(args.results)['metrics']
    lines=['# Complete additional secondary metric inventory','','Descriptive point estimates condition on saved fits. AUC and ECE use equal-event weights; balanced accuracy gives the two target classes equal row weight. Canonical-orientation intercept/slope diagnose original probabilities without modifying them. No new hypothesis family or selection is introduced. Coefficient intervals for selected main procedures appear elsewhere; none is implied for these additional points. Rating, player and removal decisions use probability greater than 0.5; feedback rows preserve their saved decisions. Frozen 2 h and 6 h rating histories have different admission delays; online arms use 6 h throughout. Labels below abbreviate map-winner and series-winner tasks.']
    for task in ('map_winner','series_winner'):
        for group in ('ratings','players','removals','later_feedback'):
            rows=[r for r in records if r['task']==task and r['group']==group]
            if not rows:continue
            lines+=['','## '+task.replace('_winner','').title()+' / '+group.replace('_',' '),'',f"Rows/events: {rows[0]['rows']:,}/{rows[0]['events']:,}.",'','| Procedure | Balanced accuracy | AUC | Intercept | Slope | ECE 5 / 10 / 20 |','|---|---:|---:|---:|---:|---|']
            for r in rows:
                d=r['calibration_diagnostic'];fmt=lambda x:'unresolved' if x is None else f'{x:.5f}'
                label=r['condition'].replace('_',' ').replace('/',' / ').replace('-',' ')
                lines.append('| '+' | '.join([label,fmt(r['balanced_accuracy']),fmt(r['equal_event_weighted_roc_auc']),fmt(d['intercept']),fmt(d['slope']),' / '.join(fmt(r['ece_'+str(b)]) for b in (5,10,20))])+' |')
    args.output.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
