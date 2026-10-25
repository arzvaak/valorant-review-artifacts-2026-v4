"""Plot all seven online references from shared, already-scored aggregates."""
import argparse
import csv
from pathlib import Path


def render(source, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    with Path(source).open() as f:
        rows=list(csv.DictReader(f))
    names=['Earlier residual','Online-rich residual','Dynamic strength','K20 raw','K20 calibrated','Glicko-2 calibrated','Player Gaussian calibrated']
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(8.5,4.9),sharex='col')
    for i,task in enumerate(['map_winner','series_winner']):
        for j,metric in enumerate(['event_brier','event_log_loss']):
            ax=axes[i,j]
            for y,name in enumerate(names):
                r=next(x for x in rows if x['task']==task and x['procedure']==name)
                ax.plot(float(r[metric]),y,'o',color='#0072B2' if 'residual' in name else '#D55E00',markersize=4)
            ax.set_yticks(range(len(names)),names if j==0 else ['']*len(names))
            ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.set_title(('Maps' if i==0 else 'Series')+' / '+('Brier' if j==0 else 'log loss'))
            if i==1:ax.set_xlabel('Event-equal loss (lower is better)')
            ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:
        fig.savefig(destination/('online-rating-display.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();render(args.input,args.output)
