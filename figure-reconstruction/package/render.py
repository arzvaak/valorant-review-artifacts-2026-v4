"""Regenerate figure deliverables from the included aggregate evidence."""
import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
subprocess.run([sys.executable,'code/core_journal_figures.py','--source','data/core/report/core/results.json','--output','rendered/core'],cwd=root,check=True)
subprocess.run([sys.executable,'code/journal_figures.py','--core','data/core','--online','data/online','--repository','.','--output','rendered/secondary'],cwd=root,check=True)
subprocess.run([sys.executable,'code/later_journal_figure.py','--source','data/later/results.json','--output','rendered/later'],cwd=root,check=True)
