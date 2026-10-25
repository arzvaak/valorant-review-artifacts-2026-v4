"""Finite owner-approved real campaign, recoverable at each saved unit."""
import time
import traceback
import fcntl
from context import CORE, WORK, OUTPUT, PERSIST, read, atomic, bind, configure_adapters
from run import run
from refresh import run as refresh
from report import report
from audits import main as audits


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    lock=(WORK/'campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=bind();configure_adapters()
    atomic(PERSIST/'campaign-status.json',dict(status='running',started=time.time(),identity=identity))
    try:
        # Saved-weight control and descriptive audits finish first; no new fitting.
        retained=identity.get('retained_control')
        if retained:
            print('RETAIN VERIFIED ORIGINAL CONTROL, REFRESH8 AND AUDITS',retained['identity_sha256'],flush=True)
        else:
            refresh(identity);report('refresh8',identity);audits(identity)
        policies=('fresh2h15m','frozen6h30m','frozen24h60m') if retained else ('frozen2h15m','fresh2h15m','frozen6h30m','frozen24h60m')
        for policy in policies:
            for fold in read(CORE/'config.json')['folds']:
                run(policy,fold['name'],identity)
            if policy=='fresh2h15m':
                report('fresh20',identity);report('change20',identity)
            if policy=='frozen24h60m':report('clocks40',identity)
        atomic(PERSIST/'campaign-status.json',dict(status='complete',finished=time.time(),identity=identity))
        print('DECLARED COLAB ROUND COMPLETE',flush=True)
    except BaseException as error:
        atomic(PERSIST/'campaign-status.json',dict(status='failed',finished=time.time(),identity=identity,error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
