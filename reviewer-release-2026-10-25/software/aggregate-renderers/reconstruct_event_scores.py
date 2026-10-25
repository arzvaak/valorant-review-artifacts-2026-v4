"""Reconstruct declared score summaries from shared complete-event aggregates.

This is a scientific reconstruction command, not a test harness. It does not
read source rows, fit models or evaluate protected data. Standard library only.
"""
import argparse
import collections
import csv
from pathlib import Path


def reconstruct(source,output):
    groups=collections.defaultdict(list)
    with source.open(newline='') as f:
        for row in csv.DictReader(f):groups[row['analysis'],row['task'],row['procedure']].append(row)
    fields=('analysis','task','procedure','events','rows','event_brier','row_brier','event_log_loss','row_log_loss','event_accuracy','row_accuracy')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for (analysis,task,procedure),events in sorted(groups.items()):
            n=sum(int(e['rows']) for e in events);E=len(events)
            result=dict(analysis=analysis,task=task,procedure=procedure,events=E,rows=n)
            for source_field,metric in [('brier_sum','brier'),('log_loss_sum','log_loss'),('correct_sum','accuracy')]:
                result['event_'+metric]=sum(float(e[source_field])/int(e['rows']) for e in events)/E
                result['row_'+metric]=sum(float(e[source_field]) for e in events)/n
            w.writerow(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();reconstruct(a.input,a.output)
