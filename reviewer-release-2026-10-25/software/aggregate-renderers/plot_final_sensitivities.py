"""Reconstruct the final history figure using only shared aggregate CSVs."""
import argparse
import csv
from pathlib import Path


def render(csv_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    with Path(csv_path).open() as stream:
        rows = list(csv.DictReader(stream))
    names = {'linear': 'Logistic', 'catboost': 'CatBoost', 'xgboost': 'XGBoost', 'mlp': 'MLP', 'tabm': 'TabM'}
    colors = ['#0072B2', '#D55E00', '#009E73']
    arms = [('fresh20', 'fresh2h15m', 'Advancing 2 h / 15 min'),
            ('clocks40', 'frozen6h30m', 'Frozen 6 h / 30 min'),
            ('clocks40', 'frozen24h60m', 'Frozen 24 h / 60 min')]
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharex=True, sharey=True)
    for ax, task in zip(axes, ['map_winner', 'series_winner']):
        ax.axvline(0, color='#777777', linewidth=.8)
        for i, family in enumerate(names):
            for j, (report, arm, _) in enumerate(arms):
                matches = [r for r in rows if r['correction_family'] == report and r['policy'] == arm and r['task'] == task and r['a'] == 'BN-'+family and r['b'] == 'B-'+family and r['metric'] == 'brier']
                if not matches:
                    continue
                c = matches[0]; y = i+(j-1)*.2
                ax.plot([float(c['ci_low']),float(c['ci_high'])],[y,y],color=colors[j],linewidth=1.4)
                ax.plot(float(c['difference']),y,'o',color=colors[j],markersize=5,
                        markerfacecolor=colors[j] if c['conservative_rejection']=='True' else 'white')
        ax.set_title('After-picks maps' if task=='map_winner' else 'Before-play series')
        ax.set_yticks(range(5),list(names.values()))
        ax.tick_params(axis='y',labelleft=True)
        ax.set_xlabel('Event-equal Brier difference (BN − B)')
        ax.grid(axis='x',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis()
    fig.legend([Line2D([0],[0],color=c,marker='o') for c in colors],[a[2] for a in arms],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.02))
    fig.tight_layout(rect=(0,.12,1,1))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/'history-sensitivities.pdf',bbox_inches='tight')
    fig.savefig(output/'history-sensitivities.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    render(args.input,args.output)
