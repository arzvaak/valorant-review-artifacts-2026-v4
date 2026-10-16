"""Render all twelve saved later loss contrasts; no fitting or rescoring."""
import argparse, hashlib, json, platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main(source, output):
    if output.exists(): raise FileExistsError('Preserve prior figures')
    data=json.loads(source.read_text()); rows=data['primary_contrasts']; output.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.16,4.7));fig.subplots_adjust(left=.19,right=.98,bottom=.23,top=.84,hspace=1,wspace=.65)
    seen=[];parents=['original_residual','dynamic_strength','online_elo_k20'];names=['Earlier residual','Dynamic strength','K20 reference']
    for i,task in enumerate(['map_winner','series_winner']):
        for j,metric in enumerate(['brier','log_loss']):
            ax=axes[i,j];selected=[next(r for r in rows if r['task']==task and r['metric']==metric and r['parent']==parent) for parent in parents]
            for y,r in enumerate(selected):
                seen.append(r['id']);color=('#17695e' if r['difference']<0 else '#a54b36') if r['conservative_rejection'] else '#64717c'
                ax.plot(r['ci95'],[y,y],color=color,lw=1);ax.plot(r['difference'],y,'o',ms=4,mec=color,mfc=color if r['conservative_rejection'] else 'white')
            bounds=[v for r in rows if r['metric']==metric for v in r['ci95']]+[0];span=max(bounds)-min(bounds);ax.set_xlim(min(bounds)-.06*span,max(bounds)+.06*span);ax.set_yticks(range(3),names);ax.invert_yaxis();ax.axvline(0,color='#24384a',ls='--',lw=.8);ax.grid(axis='x',color='#e1e5e9',lw=.5);ax.set_axisbelow(True);ax.xaxis.set_major_locator(MaxNLocator(3));ax.set_xlabel('Rich residual minus reference loss')
            ax.set_title(('Maps: 1,613 rows' if task=='map_winner' else 'Series: 734 rows')+' · '+('Brier' if metric=='brier' else 'Log loss'))
    if sorted(seen)!=sorted(r['id'] for r in rows):raise ValueError('Incomplete source contrast coverage')
    fig.text(.02,.97,'Retained models on 23 later events',fontsize=10,va='top');fig.text(.02,.90,'All 12 primary comparisons: no corrected support',fontsize=8.5,va='top')
    fig.text(.02,.035,'Pointwise 95% complete-event intervals; negative favors rich residual.\nHollow points: no conservative Holm support; not equivalence.\nRetrospective, unreviewed cohort; score-only feedback policy; conditional on saved fits.',fontsize=8,va='bottom')
    for ext in ['png','svg','pdf']:fig.savefig(output/('later-loss-contrasts-journal.'+ext),**({'dpi':220} if ext=='png' else {'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {}))
    plt.close(fig)
    caption='All twelve primary later rich-residual comparisons on 1,613 valid maps and 734 series, each from 23 events. Differences use equal-event Brier and log loss; negative favors rich residual. Pointwise 95% complete-event bootstrap intervals condition on retained fits and omit refitting uncertainty. Hollow markers retain non-rejection under the twelve-contrast conservative Monte Carlo/Holm family; they do not imply equivalence. The negative pointwise series interval against the earlier residual does not establish corrected support. Scales are shared across tasks within each loss and include zero. The retrospective cohort is unreviewed; new observations are score-only and do not update histories. Schedule-derived availability and series roster proxies remain limitations. No new scoring or inference is performed.'
    (output/'caption.md').write_text(caption+'\n');(output/'code').mkdir();(output/'code/later_journal_figure.py').write_bytes(Path(__file__).read_bytes())
    spec={'source_sha256':sha(source),'code_sha256':sha(Path(__file__)),'environment':{'python':platform.python_version(),'matplotlib':matplotlib.__version__},'size_inches':[7.16,4.7],'minimum_font_points':8,'grammar':{'data':'12 saved primary contrast records','x':'equal-event candidate minus reference loss','y':'reference procedure','facets':'task rows / loss columns','layers':'pointwise CI, saved point, zero line','scales':'shared across tasks within loss','transform':'none; source intervals/decisions copied'},'caption':caption,'exact_plotted_rows':rows,'new_scoring_or_fitting':False,'outputs':{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    (output/'specification.json').write_text(json.dumps(spec,indent=2)+'\n');print(json.dumps({'output':str(output),'contrasts':len(seen)}))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();main(args.source,args.output)
