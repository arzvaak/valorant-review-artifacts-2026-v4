"""Journal-width figures from scored evidence; no source access or fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyBboxPatch


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def player_label(row):
    a, b = row['a'], row['b']
    architecture = {'deep_sets': 'DS', 'set_transformer': 'ST', 'mlp': 'MLP', 'tabm': 'TabM'}
    ar, af = a.split('-', 1)
    if b == 'selector':
        return ar + ' ' + architecture[af] + ' − Sel'
    br, bf = b.split('-', 1)
    if ar == 'BNLC' and br == 'BN' and af == bf:
        return architecture[af] + ' full − general'
    assert ar == br
    return ar + ' ' + architecture[af] + ' − ' + architecture[bf]


def main(core, online, output, repository):
    assert not output.exists(), 'Preserve every earlier figure release'
    inputs = {'players': core / 'report/players/results.json', 'removals': core / 'report/removals/results.json', 'online': online / 'results.json', 'methods': repository / 'docs/paper-study/methods-and-evaluation.md'}
    data = {k: read(p) for k, p in inputs.items() if k != 'methods'}
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    spec = {'version': 'journal-width-evidence/1', 'code_sha256': sha(Path(__file__)), 'inputs': {k: {'path': str(p), 'sha256': sha(p)} for k, p in inputs.items()}, 'environment': {'python': platform.python_version(), 'matplotlib': matplotlib.__version__}, 'width_inches': 7.16, 'minimum_declared_font_points': 8, 'new_fits_or_scores': False, 'protected_or_raw_source_access': False, 'figures': {}}
    captions = ['# Journal-width figure captions and label keys', '', 'These are full-width scientific figures. Keep their declared width to preserve eight-point text. Captions carry inferential limits and full contrast identities; no observation or interval is removed to shorten a label.', '']

    def save(fig, name, caption, rows, geometry):
        for suffix in ('svg', 'pdf', 'png'):
            kwargs = {'dpi': 220} if suffix == 'png' else {'metadata': {'CreationDate': None, 'ModDate': None}} if suffix == 'pdf' else {}
            fig.savefig(output / (name + '.' + suffix), **kwargs)
        plt.close(fig)
        spec['figures'][name] = {'size_inches': list(fig.get_size_inches()), 'minimum_font_points': 8, 'caption': caption, 'geometry': geometry, 'plotted_rows': rows}
        captions.extend(['## ' + name, '', caption, ''])

    for stage in ('players', 'removals'):
        for task in ('map_winner', 'series_winner'):
            fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.3))
            fig.subplots_adjust(left=.19, right=.98, bottom=.22, top=.86, wspace=1.15)
            used, keys, support = [], [], set()
            for ax, metric in zip(axes, ('brier', 'log_loss')):
                rows = [r for r in data[stage]['contrasts'] if r['task'] == task and r['metric'] == metric]
                assert len(rows) == (12 if stage == 'players' else 10)
                labels = []
                for y, row in enumerate(rows):
                    lo, hi = row['ci95']
                    ax.plot([lo, hi], [y, y], color='#4d5965', linewidth=1)
                    ax.plot(row['difference'], y, 'o', color='#24384a', markersize=3.5)
                    label = player_label(row) if stage == 'players' else 'coverage only' if row['a'] == 'coverage' else row['a'].removeprefix('drop_').replace('schedule_stability', 'schedule').replace('map_side', 'map side')
                    labels.append(label)
                    support.add((row['rows'], row['events']))
                    used.append({k: row[k] for k in ('id', 'a', 'b', 'task', 'metric', 'difference', 'ci95', 'rows', 'events', 'conservative_rejection')} | {'display_label': label})
                    if metric == 'brier':
                        keys.append((label, row['a'], row['b']))
                ax.set_yticks(range(len(rows)), labels)
                ax.invert_yaxis()
                ax.axvline(0, color='#24384a', linestyle='--', linewidth=.8)
                bounds = [v for r in data[stage]['contrasts'] if r['metric'] == metric for v in r['ci95']] + [0]
                span = max(bounds) - min(bounds)
                ax.set_xlim(min(bounds) - .06 * span, max(bounds) + .06 * span)
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.grid(axis='x', color='#dce1e6', linewidth=.5)
                ax.set_axisbelow(True)
                ax.set_title('Brier score' if metric == 'brier' else 'Log loss')
                ax.set_xlabel('Loss difference')
            title = ('Player-set controls' if stage == 'players' else 'Conditional removals') + ' · ' + ('maps after picks' if task == 'map_winner' else 'series before play')
            fig.text(.015, .97, title, fontsize=9, va='top')
            fig.text(.015, .08, 'Pointwise 95% event intervals; saved-fit uncertainty only.\nNegative favors the first procedure. No corrected support; not equivalence.', fontsize=8, va='bottom')
            support_text = '; '.join(f'{n:,} rows / {e:,} events' for n, e in sorted(support))
            caption = f"{title}. Broad nonglobal development evidence: {support_text}. All {len(used)} task/loss contrasts are plotted with pointwise 95% complete-event bootstrap intervals conditional on saved fits, omitting refitting uncertainty. Panels share scales across tasks within each loss. None has conservative Holm support in its declared family; a pointwise interval excluding zero is not an adjusted decision. Nonsignificance does not establish equivalence. "
            caption += 'DS = Deep Sets; ST = Set Transformer; Sel = frozen selector. BN uses general numeric histories; BNLC jointly adds linked histories and composition. Full-minus-general and set-versus-summary contrasts change more than one factor.' if stage == 'players' else 'Each drop and coverage-only procedure is compared with its fixed selected full reference. Removal inference is conditional procedure analysis, not causal gameplay evidence; eight blocks provide support.'
            name = stage + '-' + task + '-journal'
            save(fig, name, caption, used, 'points and complete horizontal intervals; Cartesian zero reference; no truncation')
            captions.extend(['| Display label | First procedure | Reference |', '|---|---|---|', *[f'| {label} | {a} | {b} |' for label, a, b in keys], ''])

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5))
    fig.subplots_adjust(left=.17, right=.98, bottom=.25, top=.73, wspace=.85)
    used = []
    bounds = [100 * v for r in data['online']['contrasts'] if r['scope'] == 'broad_nonglobal' and r['a'] == 'online_rich_residual' and r['b'] in ('previous_residual', 'online_elo') for field in ('row_ci95', 'event_ci95') for v in r[field]] + [0]
    span = max(bounds) - min(bounds)
    for ax, task in zip(axes, ('map_winner', 'series_winner')):
        rows = [r for r in data['online']['contrasts'] if r['task'] == task and r['scope'] == 'broad_nonglobal' and r['a'] == 'online_rich_residual' and r['b'] in ('previous_residual', 'online_elo')]
        assert len(rows) == 2
        for y, row in enumerate(rows):
            for offset, weighting, color, marker in ((-.13, 'row', '#17695e', 'o'), (.13, 'event', '#955e19', 's')):
                lo, hi = [100 * v for v in row[weighting + '_ci95']]
                ax.plot([lo, hi], [y + offset, y + offset], color=color, linewidth=1)
                ax.plot(100 * row[weighting + '_difference'], y + offset, marker, markersize=4, color=color, label=('Row weighting' if weighting == 'row' else 'Event weighting') if y == 0 else None)
            used.append(row)
        ax.set_yticks(range(2), ['vs earlier residual' if r['b'] == 'previous_residual' else 'vs matched K20' for r in rows])
        ax.invert_yaxis()
        ax.axvline(0, color='#24384a', linestyle='--', linewidth=.8)
        ax.set_xlim(min(bounds) - .06 * span, max(bounds) + .06 * span)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel('Accuracy difference (pp)')
        ax.set_title('Maps after picks' if task == 'map_winner' else 'Series before play')
        ax.grid(axis='x', color='#dce1e6', linewidth=.5)
        ax.set_axisbelow(True)
    fig.text(.015, .98, 'Online refresh: comparator and weighting', fontsize=9, va='top')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.55, .94), ncol=2, frameon=False)
    fig.text(.015, .04, 'Positive favors rich. Pointwise 95% intervals, unadjusted and conditional on saved fits.\nPost-result exploratory development evidence; no prospective claim.', fontsize=8, va='bottom')
    caption = 'Online performance refresh versus the earlier residual and feedback-matched K20, broad nonglobal development population: 55,943 maps / 1,049 events and 25,186 series / 1,047 events. Points and intervals are percentage-point accuracy differences, with row and event weighting distinguished by both color and marker. All four comparisons are retained on a common scale. Pointwise 95% complete-event bootstrap intervals condition on saved fits, omit refitting uncertainty and are unadjusted. The six-hour scheduled availability proxy and global-update exclusion remain explicit. These are post-result exploratory accuracy comparisons, not adjusted probability-loss gains or prospective validation.'
    save(fig, 'online-refresh-journal', caption, used, 'offset circle/square points and full intervals; common task scale; visible zero reference')

    fig, ax = plt.subplots(figsize=(7.16, 5))
    ax.set_xlim(-.012, 1.012)
    ax.set_ylim(0, 1)
    ax.axis('off')
    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.008', linewidth=.7, edgecolor='#486170', facecolor=color))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=8)
    def arrow(x1, x2, y):
        ax.annotate('', xy=(x2, y), xytext=(x1, y), arrowprops={'arrowstyle': '->', 'linewidth': .8, 'color': '#486170'})
    ax.text(0, .98, 'Prediction horizons and chronology', fontsize=10, va='top')
    stages = ['Fit\ntraining', 'Early stop\nseparate history', 'Select\nmodel / threshold', 'Calibrate\nseparate history', 'Evaluate\nfrozen forecasts']
    for i, label in enumerate(stages):
        x = i * .205
        box(x, .81, .17, .10, label, '#e8eef2')
        if i < 4:
            arrow(x + .177, x + .197, .86)
    ax.text(0, .765, 'Retained later continuation reuses fitted models, thresholds and calibrators.', fontsize=8)
    for y, heading, middle in ((.59, 'Map winner after picks', 'Known map + observed lineup\nand agent composition'), (.37, 'Series winner before play', 'Team pair + earlier roster\nand agent-usage proxies')):
        ax.text(0, y + .15, heading, fontsize=9)
        box(0, y, .29, .12, 'Strictly prior admitted\nperformance / strength history', '#edf0f2')
        box(.36, y, .29, .12, middle, '#e4f1ed')
        box(.73, y, .26, .12, 'Freeze probability\nthen later score target', '#f5ece1')
        arrow(.30, .35, y + .06)
        arrow(.66, .72, y + .06)
    ax.text(0, .285, 'Prediction cutoff: scheduled start − 15 minutes; actual input publication unverified.', fontsize=8)
    ax.text(0, .215, 'Core historical release: schedule + 2 hours. Online release: schedule + 6 hours.\nOnly strictly earlier releases enter history; equal-time expectations remain frozen.\nGlobals do not update history. The new unreviewed cohort is score-only.\nRoster and availability clocks are proxies; this diagram does not establish live availability.', fontsize=8, va='top', linespacing=1.35)
    ax.text(0, .035, 'Schematic, not to scale. Before-play series never uses actual future picks or performance.', fontsize=8)
    fig.subplots_adjust(left=.03, right=.98, bottom=.02, top=.98)
    caption = 'Schematic of the two target horizons and permitted chronological stages. Map forecasts assume the observed map and lineups/compositions are known before round one; series forecasts use earlier-appearance/usage proxies. Both use the declared scheduled-start lead, which does not verify historical publication timing. The core block-local source release proxy is two hours; online state uses six hours with strictly prior, equal-time frozen feedback and global exclusion. New unreviewed later-cohort rows are score-only. The five fitting/selection/calibration/evaluation stages describe the original protocol; retained later inference performs no new fit, threshold selection or calibration. This diagram is conceptual, not a measured timeline or proof of prospective operational availability.'
    save(fig, 'horizons-chronology-journal', caption, [], 'schematic directed boxes; explicit source/horizon assumptions; not to scale')
    (output / 'captions.md').write_text('\n'.join(captions) + '\n')
    (output / 'code').mkdir()
    (output / 'code/journal_figures.py').write_bytes(Path(__file__).read_bytes())
    spec['outputs'] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'specification.json').write_text(json.dumps(spec, indent=2) + '\n')
    print(json.dumps({'figures': len(spec['figures']), 'rows': {k: len(v['plotted_rows']) for k, v in spec['figures'].items()}, 'output': str(output)}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for name in ('core', 'online', 'output', 'repository'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    main(args.core, args.online, args.output, args.repository)
