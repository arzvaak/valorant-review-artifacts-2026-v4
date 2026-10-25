"""Owner-authorized local continuation; retained cloud units keep their identities."""
import importlib.metadata
import os
from pathlib import Path
import sys
import tarfile

import context as ctx


def local_bind():
    import torch
    if os.environ.get('VALORANT_LOCAL_CONTINUATION') != 'owner-20261025-v3':
        raise RuntimeError('Explicit local continuation authorization required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory; no CPU fallback')
    source = Path(os.environ['VALORANT_RETAINED_CLOUD'])
    cloud = ctx.read(source/'identity.json')
    for name, expected in cloud['code'].items():
        if Path(name).name != name or ctx.digest(Path(__file__).parent/name) != expected:
            raise RuntimeError('Qualified cloud numerical source changed: '+name)
    manifest = ctx.read(ctx.INPUT/'manifest.json')
    if cloud['input_manifest'] != ctx.digest(ctx.INPUT/'manifest.json'):
        raise RuntimeError('Retained cloud input manifest differs')
    for relative, expected in manifest['files'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or ctx.digest(ctx.INPUT/path) != expected:
            raise RuntimeError('Input manifest mismatch: '+relative)
    # The original verifier checks the cloud control's full files and numerical
    # wrapper against the original environment, rather than asserting that the
    # new local environment is identical to it.
    retained = ctx.retained_control(cloud, manifest)
    if retained is None:
        raise RuntimeError('Separate retained cloud control required')
    versions = {}
    for name in ('torch','numpy','scipy','scikit-learn','xgboost','catboost','duckdb','lightning','torchmetrics','tabm','rtdl_revisiting_models','rtdl_num_embeddings'):
        versions[name] = importlib.metadata.version(name)
    for name in ('cupy-cuda12x','cupy-cuda13x'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    identity = dict(version='owner-local-continuation-v3',input_manifest=cloud['input_manifest'],
        methods=ctx.digest(ctx.WORK/'methods.md'),code={p.name:ctx.digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        device=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),
        versions=versions,python=sys.version,torch_cuda=torch.version.cuda,neural_precision='16-mixed',
        core_seeds=ctx.read(ctx.CORE/'config.json')['seeds'],residual_saved_seeds=ctx.read(ctx.INPUT/'dynamic/config.json')['seeds'],
        retained_control=retained,retained_cloud_identity=cloud,retained_cloud_identity_sha256=ctx.digest(source/'identity.json'),
        environment_interpretation='Mixed cloud/local saved-fit continuation; unfinished local fits are not hardware-matched to retained T4 controls. B/BN unit environments must be disclosed.')
    ctx.OUTPUT.mkdir(parents=True,exist_ok=True)
    ctx.PERSIST.mkdir(parents=True,exist_ok=True)
    previous=ctx.PERSIST/'identity.json'
    if previous.exists() and ctx.read(previous)!=identity:
        raise RuntimeError('Local continuation identity changed; version repair required')
    ctx.atomic(previous,identity);ctx.atomic(ctx.OUTPUT/'identity.json',identity)
    return identity


def install_recovery():
    import storage
    original=storage.run_unit
    source=Path(os.environ['VALORANT_RETAINED_CLOUD'])
    cloud=ctx.read(source/'identity.json')
    def recover(key,directory,identity,action,verify):
        record=source/'completed-units'/(key+'.json')
        archive=source/'completed-units'/(key+'.tar.gz')
        if not record.exists():
            return original(key,directory,identity,action,verify)
        receipt=ctx.read(record)
        expected=dict(identity)
        if expected.get('colab_identity') is None:
            raise RuntimeError('Unexpected retained unit binding')
        expected['colab_identity']=cloud
        if receipt['identity']!=expected or ctx.digest(archive)!=receipt['archive_sha256']:
            raise RuntimeError('Retained cloud unit identity/archive mismatch: '+key)
        directory=Path(directory)
        relative=directory.relative_to(ctx.WORK)
        if receipt['directory']!=str(relative):
            raise RuntimeError('Retained cloud unit directory mismatch: '+key)
        if not directory.exists():
            storage.validate_archive(archive)
            with tarfile.open(archive,'r:gz') as tar:
                if any(not Path(m.name).is_relative_to(relative) for m in tar.getmembers()):
                    raise RuntimeError('Retained archive exceeds declared unit directory')
                tar.extractall(ctx.WORK,filter='data')
        verify()
        # Separate provenance receipt: never reissue an old fit as a local fit.
        ctx.atomic(ctx.PERSIST/'retained-units'/(key+'.json'),dict(key=key,origin='retained_colab',
            receipt_sha256=ctx.digest(record),archive_sha256=receipt['archive_sha256'],
            original_identity=receipt['identity'],source_receipt=str(record),directory=str(relative)))
        print('REUSE VERIFIED COLAB UNIT',key,flush=True)
        return receipt
    storage.run_unit=recover


def main():
    ctx.bind=local_bind
    install_recovery()
    import campaign
    import run as runner
    from common import verify_files, check
    def saved_path(value):
        path=Path(value)
        old=Path('/content/valorant-final-round-feedback-v2')
        if path.is_relative_to(old):
            return ctx.WORK/path.relative_to(old)
        if path.is_relative_to(ctx.WORK):
            return path
        raise RuntimeError('Saved model path is outside the declared work roots')
    def verify_portable(directory):
        receipt=ctx.read(directory/'frozen.json')
        verify_files(directory,receipt['files'])
        for item in receipt['model_records']:
            record=item['record']
            if record['status']!='fitted':continue
            candidate=saved_path(item['path'])
            check(saved_path(record['model']),record['model_sha256'])
            check(candidate/'predictions.npz',record['prediction_sha256'])
            check(candidate/'processor.json',record['processor_sha256'])
        return receipt
    # Verification-only relocation: records and saved predictions stay byte
    # identical. No retained estimator is loaded or executed by this mapping.
    runner.verify_package=verify_portable
    campaign.main()


if __name__=='__main__': main()
