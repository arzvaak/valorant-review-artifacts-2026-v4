"""Portable, identity-bound Colab context; importing this module runs no fits."""
import hashlib
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import sys

WORK = Path(os.environ.get('VALORANT_COLAB_WORK', '/content/valorant-final-round'))
INPUT = WORK / 'input'
CORE = INPUT / 'core'
REGISTRY = INPUT / 'registry'
OUTPUT = WORK / 'results'
CONTROL = Path(os.environ.get('VALORANT_COLAB_CONTROL', str(OUTPUT)))
PERSIST = Path(os.environ.get('VALORANT_COLAB_PERSIST', '/content/drive/MyDrive/valorant-final-round-2026-10-24'))
# Keep this round's run/report modules ahead of the immutable adapters with
# the same filenames; resolve common/fit/models from the appended source path.
sys.path.append(str(CORE / 'code'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    with pending.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    pending.replace(path)


def retained_control(identity,manifest):
    """Bind an unchanged completed control to a separately versioned continuation."""
    if CONTROL==OUTPUT:return None
    previous=read(CONTROL/'identity.json')
    for name in ('device','capability','versions','python','torch_cuda','neural_precision','core_seeds','residual_saved_seeds'):
        if identity[name]!=previous[name]:raise RuntimeError('Retained-control environment changed: '+name)
    if read(CONTROL.parent/'input/manifest.json')['files']!=manifest['files']:
        raise RuntimeError('Retained-control numerical inputs changed')
    # The original control used saved inputs and this exact fitting wrapper;
    # the repair changes only the history builder used by the remaining arms.
    if digest(Path(__file__).parent/'run.py')!=digest(CONTROL.parent/'code/run.py'):
        raise RuntimeError('Retained-control fitting wrapper changed')
    def precision_body(path):
        tree=ast.parse(path.read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='configure_adapters'),include_attributes=False)
    if precision_body(Path(__file__))!=precision_body(CONTROL.parent/'code/context.py'):
        raise RuntimeError('Retained-control numerical precision changed')
    forecasts={}
    for fold in read(CORE/'config.json')['folds']:
        block=fold['name'];base=CONTROL/'frozen2h15m/blocks'/block;path=base/'forecasts-frozen.json';frozen=read(path)
        if frozen['identity']!=previous:raise RuntimeError('Retained-control forecast identity changed')
        for relative,expected in frozen['files'].items():
            rel=Path(relative)
            if rel.is_absolute() or '..' in rel.parts or digest(base/rel)!=expected:raise RuntimeError('Retained-control forecast file changed')
        forecasts[block]=digest(path)
    reports={}
    for name,folder in (('refresh8',CONTROL/'reports/refresh8'),('audits',CONTROL/'audits')):
        completion=read(folder/'completion.json')
        if digest(folder/'results.json')!=completion['results']:raise RuntimeError('Retained completed report changed: '+name)
        if read(folder/'results.json')['identity']!=previous:raise RuntimeError('Retained report identity changed: '+name)
        reports[name]=dict(results=completion['results'],completion=digest(folder/'completion.json'))
    return dict(root=str(CONTROL),identity=previous,identity_sha256=digest(CONTROL/'identity.json'),forecasts=forecasts,reports=reports,reason='feedback-v2 repair does not change completed frozen-history fits or saved-weight refresh/audits; original execution identities retained')


def bind():
    import torch
    if not os.environ.get('COLAB_RELEASE_TAG') and not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('This experiment is authorized for Colab only')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory; no CPU fallback')
    if not PERSIST.is_dir():
        raise RuntimeError('Persistent private study folder is not mounted')
    manifest = read(INPUT / 'manifest.json')
    for relative, expected in manifest['files'].items():
        p = Path(relative)
        if p.is_absolute() or '..' in p.parts:
            raise RuntimeError('Invalid input manifest path')
        if digest(INPUT / p) != expected:
            raise RuntimeError('Input changed: ' + relative)
    versions = {}
    for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'xgboost', 'catboost', 'duckdb', 'lightning', 'torchmetrics', 'tabm', 'rtdl_revisiting_models', 'rtdl_num_embeddings', 'cupy-cuda13x'):
        versions[name] = importlib.metadata.version(name)
    identity = dict(input_manifest=digest(INPUT / 'manifest.json'),
                    methods=digest(WORK / 'methods.md'),
                    code={p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                    device=torch.cuda.get_device_name(0),
                    capability=list(torch.cuda.get_device_capability(0)),
                    versions=versions, python=sys.version,torch_cuda=torch.version.cuda,neural_precision='16-mixed',core_seeds=read(CORE/'config.json')['seeds'],residual_saved_seeds=read(INPUT/'dynamic/config.json')['seeds'])
    retained=retained_control(identity,manifest)
    if retained is not None:identity['retained_control']=retained
    OUTPUT.mkdir(parents=True, exist_ok=True)
    previous = PERSIST / 'identity.json'
    if previous.exists() and read(previous) != identity:
        raise RuntimeError('Run identity changed; preserve prior run and version the repair')
    atomic(previous, identity)
    atomic(OUTPUT / 'identity.json', identity)
    return identity


def configure_adapters():
    """Explicit T4 precision amendment, bound by the new wrapper identity."""
    import models
    if not getattr(models.pl.Trainer, '_valorant_colab_precision', False):
        trainer = models.pl.Trainer
        class ColabTrainer(trainer):
            _valorant_colab_precision = True
            def __init__(self, *args, **kwargs):
                kwargs['precision'] = '16-mixed'
                super().__init__(*args, **kwargs)
        models.pl.Trainer = ColabTrainer
    import torch
    if models.torch is torch:
        class ColabTorch:
            """Scope the original neural inference autocast amendment to models."""
            def __getattr__(self,name):return getattr(torch,name)
            def autocast(self,device_type,*args,**kwargs):
                if device_type=='cuda' and kwargs.get('dtype') is torch.bfloat16:
                    kwargs['dtype']=torch.float16
                return torch.autocast(device_type,*args,**kwargs)
        models.torch=ColabTorch()
    torch.set_num_threads(2)
