"""Grammar-explicit figures from saved revision aggregate records only."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator

from review_revision_diagnostics import STORE, TASKS, read, save, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostics',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--feedback-results',type=Path)
    parser.add_argument('--online-calibration',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'pdf.fonttype':42})
    figures=[]
    def finish(fig,name,records,caption):
        for suffix in ('pdf','svg','png'):fig.savefig(args.output/(name+'.'+suffix),bbox_inches='tight',dpi=200)
        figures.append(dict(name=name,records=records,caption=caption,outputs={p.name:sha(p) for p in args.output.glob(name+'.*')}))
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.16,3.9));ax.set(xlim=(-.01,1.01),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.003',linewidth=.7,edgecolor='#486170',facecolor=color))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    def arrow(a,b,y):ax.annotate('',xy=(b,y),xytext=(a,y),arrowprops=dict(arrowstyle='->',lw=.8,color='#486170',shrinkA=0,shrinkB=0))
    ax.text(0,.99,'Prediction horizons and chronology',fontsize=10,va='top')
    for i,label in enumerate(['Fit\nweights','Early stop\nweights','Select\nprocedure','Calibrate\nprobabilities','Evaluate\nforecasts']):
        x=i*.205;box(x,.80,.17,.12,label,'#e8eef2')
        if i<4:arrow(x+.180,x+.192,.86)
    ax.text(0,.74,'Core histories admit training observations only across every downstream phase.',fontsize=8)
    for y,title,middle in [(.50,'Map winner after picks','Known map + observed\nplayers and agents'),(.25,'Series winner before play','Team pair + strictly prior\nroster and usage proxies')]:
        ax.text(0,y+.155,title,fontsize=9)
        box(0,y,.29,.12,'Admitted prior\nperformance / strength','#edf0f2')
        box(.36,y,.29,.12,middle,'#e4f1ed')
        box(.73,y,.26,.12,'Freeze forecast\nthen score target','#f5ece1')
        arrow(.303,.348,y+.06);arrow(.663,.718,y+.06)
    ax.text(0,.16,'Cutoff: schedule - 15 min. Core release proxy: schedule + 2 h; online proxy: + 6 h.',fontsize=8)
    ax.text(0,.105,'Equal-time forecasts precede updates. Online history admission is a separate regime.\nLater comparisons retain model weights, thresholds and calibrators.',fontsize=8,linespacing=1.4,va='top')
    finish(fig,'chronology',[],'Stage labels distinguish fitted decisions from history admission. Core downstream phases retain training-only state; online/later feedback policies are separately declared. Roster and release clocks remain availability proxies.')
    data={task:read(args.diagnostics/(task+'-diagnostics.json')) for task in TASKS}
    fig,axes=plt.subplots(2,2,figsize=(7.16,4.0),sharex='col')
    records=[];labels=['Earlier residual','Dynamic strength','Matched online Elo']
    for i,task in enumerate(TASKS):
        for j,metric in enumerate(('brier','log_loss')):
            ax=axes[i,j]
            for y,reference in enumerate(('earlier','dynamic','online_elo')):
                r=next(r for r in data[task]['contrasts'] if r['scope']=='broad_nonglobal' and r['a']=='rich' and r['b']==reference)
                value=r[metric];point=value['event_difference'];lo,hi=value['event_ci95']
                ax.errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt='o',color='#236b8e',capsize=3,ms=4,lw=1)
                records.append(dict(task=task,reference=reference,metric=metric,**value))
            ax.axvline(0,color='#666666',lw=.7);ax.set_yticks(range(3),labels if j==0 else ['']*3);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_title(('Maps' if i==0 else 'Series')+' / '+('Brier' if j==0 else 'log loss'))
            if i==1:ax.set_xlabel('Rich minus reference event-equal loss')
    fig.tight_layout()
    finish(fig,'online-probability-losses',records,'Broad nonglobal online-rich comparisons on original matched targets: 55943 maps/1049 events and 25186 series/1047 events. Points are event-equal paired loss differences, negative favoring rich. Intervals use 131072 complete-event resamples, seed 20261016, conditional on saved fits; pointwise exploratory evidence without original-family corrected decisions. Both task rows share each loss scale; every declared reference appears.')
    inputs={task:sha(args.diagnostics/(task+'-diagnostics.json')) for task in TASKS}
    if args.feedback_results:
        result=read(args.feedback_results);inputs['feedback']=sha(args.feedback_results)
        fig,axes=plt.subplots(2,2,figsize=(7.16,3.9),sharex='col');records=[]
        arms=('rich_residual','original_residual','dynamic_strength','online_elo_k20')
        for i,task in enumerate(TASKS):
            for j,metric in enumerate(('brier','log_loss')):
                ax=axes[i,j]
                for y,arm in enumerate(arms):
                    r=next(r for r in result['contrasts'] if r['task']==task and r['a']==arm+'_new' and r['metric']==metric)
                    point=r['difference'];lo,hi=r['ci95'];ax.errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt='o',color='#236b8e',capsize=3,ms=4,lw=1)
                    records.append(r)
                ax.axvline(0,color='#666666',lw=.7);ax.grid(axis='x',alpha=.2)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.set_yticks(range(4),['Rich residual','Earlier residual','Dynamic strength','K20'] if j==0 else ['']*4);ax.invert_yaxis()
                ax.set_title(('Maps' if i==0 else 'Series')+' / '+('Brier' if j==0 else 'log loss'))
                if i==1:ax.set_xlabel('Feedback minus score-only event-equal loss')
        fig.tight_layout()
        finish(fig,'later-feedback-losses',records,'Matched retained-weight feedback on the original 1,613 valid maps and 734 series across 23 events. Eighteen source-header regional events admit six-hour prior feedback; five unresolved events remain score-only. Pointwise 95% complete-event intervals use 131,072 draws, seed 20261021. Negative favors feedback. None of the 16 probability-loss contrasts passes conservative Holm. Loss scales match across horizons; all original scoring events are retained.')
    if args.online_calibration:
        result=read(args.online_calibration);inputs['online_calibration']=sha(args.online_calibration)
        fig,axes=plt.subplots(1,2,figsize=(7.16,2.65),sharex=True);records=[]
        names=('rich','earlier','dynamic','online_elo')
        for i,task in enumerate(TASKS):
            ax=axes[i]
            for y,name in enumerate(names):
                r=next(r for r in result['diagnostics'] if r['task']==task and r['package']==name)
                point=r['point'][1];lo,hi=r['coefficient_ci95'][1]
                ax.errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt='o',color='#236b8e',capsize=3,ms=4,lw=1);records.append(r)
            ax.axvline(1,color='#666666',lw=.7);ax.set_yticks(range(4),['Rich residual','Earlier residual','Dynamic strength','K20'] if i==0 else ['']*4);ax.invert_yaxis()
            ax.set_title('Maps' if i==0 else 'Series');ax.set_xlabel('Canonical diagnostic calibration slope');ax.grid(axis='x',alpha=.2)
        fig.tight_layout();finish(fig,'online-calibration-slopes',records,'Reporting-only canonical calibration slopes on 55,943 maps from 1,049 events and 25,186 series from 1,047 events. Pointwise 95% complete-event intervals use 2,048 draws, seed 20261024, with zero failed draws. The reference slope is one. Diagnostic fits never change forecasting probabilities. Complete tables retain intercept and 5/10/20-bin ECE intervals.')
        fig,axes=plt.subplots(1,2,figsize=(7.16,2.5),sharex=True);records=[]
        for i,task in enumerate(TASKS):
            ax=axes[i]
            for y,name in enumerate(('selector','block_elo','chance')):
                r=next(r for r in data[task]['metrics'] if r['scope']=='broad_nonglobal' and r['package']==name)
                ax.plot([100*r['row_accuracy'],100*r['row_tie_averaged_accuracy']],[y,y],color='#9aa5ac',lw=2)
                ax.scatter(100*r['row_accuracy'],y,color='#a34c3b',s=20,label='Deterministic ties' if y==0 and i==0 else None)
                ax.scatter(100*r['row_tie_averaged_accuracy'],y,color='#236b8e',s=20,label='Half credit at ties' if y==0 and i==0 else None)
                records.append(dict(task=task,**r))
            ax.set_yticks(range(3),['Core selector','Core K20','Fixed 0.5'] if i==0 else ['']*3);ax.invert_yaxis();ax.set_title('Maps' if i==0 else 'Series');ax.set_xlabel('Ordinary accuracy (%)');ax.grid(axis='x',alpha=.2)
        fig.legend(loc='upper center',bbox_to_anchor=(.56,1.05),ncol=2,frameon=False);fig.tight_layout()
        finish(fig,'exact-tie-contribution',records,'Original deterministic decision accuracy and exact-tie-averaged accuracy on the same broad cohort. Each p=0.5 tie receives half credit in the latter; non-tied decisions and probability losses are unchanged. Counts and actual tied-subset Team A fractions accompany the complete table. These are descriptive decision diagnostics, not randomized accuracy measurements.')
    save(args.output/'specification.json',dict(code=sha(__file__),inputs=inputs,grammar=dict(data='saved aggregate contrasts',transformation='identity; no filtering of favorable results',mapping='loss difference and complete-event interval',facets='task by loss',scales='common x scale within loss; visible zero',uncertainty='conditional pointwise event bootstrap'),figures=figures))
    print('Revision chronology and probability-loss figures rendered',flush=True)


if __name__=='__main__':main()
