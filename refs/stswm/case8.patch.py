"""Add file-based initial conditions to a modified derivative of NCAR STSWM.

This patch is supplied by sphere-on-a-torus. At TIME=0, test case 8 reads
u, v, h, divergence and vorticity from init_grid.bin as float64 values,
with longitude varying fastest. UCAR's terms still apply to the NCAR code."""
import sys
d = sys.argv[1]
s = open(f"{d}/init.f").read()
s = s.replace("""      ELSE IF (ICOND .EQ. 7) THEN""", """      ELSE IF (ICOND .EQ. 8) THEN
      WRITE(6,*) 'TEST CASE #8: INITIAL STATE FROM FILE init_grid.bin'
      RETURN
      ELSE IF (ICOND .EQ. 7) THEN""", 1)
open(f"{d}/init.f", "w").write(s)
a = open(f"{d}/anlytc.f").read()
a = a.replace("""      ELSEIF ((CASE .EQ. 7) .OR. (TIME .GT. 0.0)) THEN""", """      ELSEIF ((CASE .EQ. 8) .AND. (TIME .EQ. 0.0)) THEN
      OPEN(78,FILE='init_grid.bin',ACCESS='STREAM',FORM='UNFORMATTED',
     $     STATUS='OLD')
      READ(78) ((UICLL(I,J), I=1,NLON), J=1,NLAT)
      READ(78) ((VICLL(I,J), I=1,NLON), J=1,NLAT)
      READ(78) ((PICLL(I,J), I=1,NLON), J=1,NLAT)
      READ(78) ((DICLL(I,J,LN), I=1,NLON), J=1,NLAT)
      READ(78) ((EICLL(I,J,LN), I=1,NLON), J=1,NLAT)
      CLOSE(78)
      ELSEIF ((CASE .EQ. 7) .OR. (TIME .GT. 0.0)) THEN""", 1)
open(f"{d}/anlytc.f", "w").write(a)
print("Applied the initial-field patch to", d)
