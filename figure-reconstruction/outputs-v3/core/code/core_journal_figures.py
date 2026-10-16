"""Render every saved core contrast at journal width; no fits or rescoring."""
import argparse, hashlib, json, platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main(source, output):
    if output.exists(): raise FileExistsError('Preserve earlier figure releases')
    data=json.loads(source.read_text());output.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    families=['linear','catboost','xgboost','mlp','tabm']; names=['Logistic','CatBoost','XGBoost','MLP','TabM'];tasks=['map_winner','series_winner'];metrics=['brier','log_loss']
    increments=['BN-B','BC-B','BNC-BN','BNC-BC','BN-BC','BNL-BN','BNLC-BNC','operational']
    explanations={'BN-B':'General numeric histories added to context','BC-B':'Agent/map channels added to context','BNC-BN':'Agent/map channels added to context and general history','BNC-BC':'General numeric histories added to context and agent/map channels','BN-BC':'General-history versus agent/map-channel packages; not a single addition','BNL-BN':'Linked histories added without agent/map channels','BNLC-BNC':'Linked histories added with agent/map channels','operational':'Frozen selector versus three declared references'}
    spec={'source':str(source),'source_sha256':sha(source),'code_sha256':sha(Path(__file__)),'environment':{'python':platform.python_version(),'matplotlib':matplotlib.__version__},'width_inches':7.16,'minimum_font_points':8,'new_scoring_or_fitting':False,'figures':{}}
    captions=['# Complete core comparison figures','', 'All 152 saved task/loss contrasts are retained. No model, score, interval or corrected decision is recalculated.','']
    plotted=[]
    for inc in increments:
        selected=[r for r in data['contrasts'] if (r['correction_family']=='operational12' if inc=='operational' else r['id'].split('/')[2]==inc)]
        fig,axes=plt.subplots(2,2,figsize=(7.16,4.7));fig.subplots_adjust(left=.15,right=.98,bottom=.22,top=.84,hspace=1,wspace=.75)
        for i,task in enumerate(tasks):
            for j,metric in enumerate(metrics):
                ax=axes[i,j]; rows=[r for r in selected if r['task']==task and r['metric']==metric]
                labels=['vs '+{'elo_k20':'K20 Elo','B-linear':'B logistic','B-tabm':'B TabM'}[r['b']] for r in rows] if inc=='operational' else [names[families.index(r['id'].split('/')[1])] for r in rows]
                for y,r in enumerate(rows):
                    color=('#17695e' if r['difference']<0 else '#a54b36') if r['conservative_rejection'] else '#64717c'
                    ax.plot(r['ci95'],[y,y],color=color,lw=1)
                    ax.plot(r['difference'],y,'o',ms=4,mec=color,mfc=color if r['conservative_rejection'] else 'white')
                ax.set_yticks(range(len(rows)),labels);ax.invert_yaxis();ax.axvline(0,color='#24384a',ls='--',lw=.8);ax.grid(axis='x',color='#e1e5e9',lw=.5);ax.set_axisbelow(True)
                bounds=[v for r in selected if r['metric']==metric for v in r['ci95']]+[0];span=max(bounds)-min(bounds);ax.set_xlim(min(bounds)-.07*span,max(bounds)+.07*span);ax.xaxis.set_major_locator(MaxNLocator(3));ax.set_xlabel('Loss difference');ax.set_title(('Maps after picks' if task=='map_winner' else 'Series before play')+' · '+('Brier' if metric=='brier' else 'Log loss'))
        fig.text(.02,.975,inc+' · '+explanations[inc],fontsize=9,va='top')
        fig.legend(handles=[Line2D([],[],marker='o',lw=0,color='#17695e',label='Corrected benefit'),Line2D([],[],marker='o',lw=0,color='#a54b36',label='Corrected harm'),Line2D([],[],marker='o',lw=0,color='#64717c',markerfacecolor='white',label='Unresolved')],loc='upper center',bbox_to_anchor=(.52,.93),ncol=3,frameon=False,fontsize=8)
        fig.text(.02,.045,'Pointwise 95% complete-event intervals; uncertainty conditional on saved fits.\nNegative favors the first procedure. Corrected decisions use the original inference family.\nMaps: 55,943 rows / 1,049 events; series: 25,186 rows / 1,047 events.',fontsize=8,va='bottom')
        name='core-'+inc.lower()+'-journal'
        for suffix in ['png','svg','pdf']:
            fig.savefig(output/(name+'.'+suffix),**({'dpi':220} if suffix=='png' else {'metadata':{'CreationDate':None,'ModDate':None}} if suffix=='pdf' else {}))
        plt.close(fig)
        caption=f'{inc}: {explanations[inc]}. Broad nonglobal retrospective development evidence, with event-equal probability-loss differences. Map support is 55,943 rows / 1,049 events; series support is 25,186 rows / 1,047 events. Intervals are pointwise 95% complete-event bootstrap intervals conditional on saved fitted forecasts; they omit refitting uncertainty. Filled green/red points retain conservative Holm benefit/harm decisions within the original family; hollow points are unresolved, not equivalent. Approximate bootstrap sampling validity, dependence and adaptive development remain limitations. Panels share task scales within each loss and include zero. B=context, N=general numeric histories, C=agent AND map channels, L=linked histories. C is not a pure agent-composition effect. No new scores or inference are computed.'
        spec['figures'][name]={'caption':caption,'size_inches':[7.16,4.7],'plotted_rows':selected};captions.extend(['## '+name,'',caption,'']);plotted.extend(r['id'] for r in selected)
    if sorted(plotted)!=sorted(r['id'] for r in data['contrasts']):raise ValueError('Incomplete or repeated source contrast coverage')
    (output/'captions.md').write_text('\n'.join(captions)+'\n');(output/'code').mkdir();(output/'code/core_journal_figures.py').write_bytes(Path(__file__).read_bytes());spec['outputs']={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()};(output/'specification.json').write_text(json.dumps(spec,indent=2)+'\n');print(json.dumps({'figures':len(spec['figures']),'source_contrasts':len(plotted),'output':str(output)}))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();main(args.source,args.output)
