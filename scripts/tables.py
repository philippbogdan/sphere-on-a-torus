"""Write the claim tables after a fresh numerical run."""
import sys
from check_claims import main
if __name__ == '__main__':
    sys.argv = [sys.argv[0], '--rerun', '--out', 'build/claims.json']
    main()
