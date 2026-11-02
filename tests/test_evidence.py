"""Verify score-file digests and the arithmetic behind the headline comparisons."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_claims import calculate, verify

def test_saved_claims():
    verify(calculate())
