"""Completed real units persist as checksum-bound archives in private Drive."""
import os
from pathlib import Path
import tarfile
import time
import traceback
from context import WORK, OUTPUT, PERSIST, digest, read, atomic


def validate_archive(archive):
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
                raise RuntimeError('Unsafe saved unit archive')


def run_unit(key, directory, identity, action, verify):
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.' for c in key):
        raise ValueError('Invalid unit key')
    directory = Path(directory)
    relative = directory.relative_to(WORK)
    units = PERSIST / 'completed-units'
    units.mkdir(parents=True, exist_ok=True)
    record = units / (key + '.json')
    archive = units / (key + '.tar.gz')
    if record.exists():
        receipt = read(record)
        if receipt['identity'] != identity or digest(archive) != receipt['archive_sha256']:
            raise RuntimeError('Completed unit identity/archive changed: ' + key)
        if not directory.exists():
            validate_archive(archive)
            with tarfile.open(archive, 'r:gz') as tar:
                tar.extractall(WORK, filter='data')
        verify()
        return receipt
    attempts = PERSIST / 'attempts'
    attempts.mkdir(parents=True, exist_ok=True)
    number = 1
    while (attempts / f'{key}-{number}.json').exists():
        number += 1
    attempt = attempts / f'{key}-{number}.json'
    status = dict(identity=identity, key=key, started=time.time(), status='running')
    atomic(attempt, status)
    try:
        action()
        verify()
        pending = archive.with_name(archive.name + '.pending')
        with tarfile.open(pending, 'w:gz', compresslevel=3) as tar:
            tar.add(directory, arcname=str(relative), recursive=True)
        # Ensure archive data is acknowledged before publishing its success record.
        with pending.open('rb') as f:
            os.fsync(f.fileno())
        checksum = digest(pending)
        pending.replace(archive)
        receipt = dict(identity=identity, archive_sha256=checksum, directory=str(relative), finished=time.time())
        atomic(record, receipt)
        atomic(attempt, {**status, 'status':'complete', 'finished':time.time(), 'receipt':str(record.name)})
        return receipt
    except BaseException as error:
        failure_archive=None
        if directory.exists():
            failed=PERSIST/'failed-units'
            failed.mkdir(parents=True,exist_ok=True)
            failure_archive=failed/f'{key}-{number}.tar.gz'
            with tarfile.open(failure_archive,'w:gz',compresslevel=3) as tar:
                tar.add(directory,arcname=str(relative),recursive=True)
        atomic(attempt, {**status, 'status':'failed', 'finished':time.time(), 'error':repr(error), 'traceback':traceback.format_exc(), 'preserved_partial':str(failure_archive) if failure_archive else None})
        raise
