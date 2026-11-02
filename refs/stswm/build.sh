#!/bin/sh
# Build a modified derivative of NCAR STSWM 2.0 at truncation T$1 in 64-bit precision.
# Changes: truncation in params.i, grid-field output through grdout.f,
# and stub libraries for graphics and the unused spectral netCDF output.
set -e
[ -d src ] || tar xzf stswm.tgz
T=$1; NLAT=$2; NLON=$3
D=build_T$T${4:-}
rm -rf $D; mkdir -p $D
cp src/*.f src/*.i lib/adquad.f lib/fft99f.f lib/netcdf.f lib/ncarg.f grdout.f $D/
cp src/netcdf.inc $D/
# Keep the unused netCDF include compatible with 64-bit real arithmetic.
sed -i.bak -e "s/1.797693134862315900e+308/1.0e+30/" -e "s/3.402823466e+38/1.0e+30/" $D/netcdf.inc
cd $D
# Set spectral truncation and Gaussian-grid dimensions.
sed -i.bak -e "s/^      PARAMETER (MM=42, NN=42, KK=42, NLAT=64, NLON=128)/      PARAMETER (MM=$T, NN=$T, KK=$T, NLAT=$NLAT, NLON=$NLON)/" \
    -e "s/^      PARAMETER (MAXH=42)/      PARAMETER (MAXH=$T)/" -e "s/^      PARAMETER (RTRUNC=42)/      PARAMETER (RTRUNC=$T)/" params.i
grep -n "^      PARAMETER (MM=\|^      PARAMETER (MAXH\|^      PARAMETER (RTRUNC" params.i
# Write grid fields at the configured TAUO output times.
sed -i.bak "s/^         CALL OUTPTP (ZETASC,DIVSC,PHISC,PHIBAR,LDIAG,.FALSE.,OCTR)/         CALL GRDOUT(WS1,WS2,WS3,TAU)/" stswm.f
# Save the initial velocity before ZD overwrites WS1 and WS2.
# Replace the initial spectral output with grid output at initialisation.
sed -i.bak2 "s/^         CALL OUTPTP (ZETASC,DIVSC,PHISC,PHIBAR,LDIAG,.TRUE.,OCTR)/         CONTINUE/" stswm.f
python3 - <<'PY'
s = open("stswm.f").read()
key = "      CALL ZD(WS1,WS2,DIVSC,ZETASC)"
i = s.index(key)
s = s[:i] + "      IF (TAUO .LE. TAUE) CALL GRDOUT(WS1,WS2,WS3,0.0)\n" + s[i:]
open("stswm.f", "w").write(s)
PY
grep -n "GRDOUT" stswm.f
python3 ../case8.patch.py .
FFLAGS="-O2 -std=legacy -fdefault-real-8 -fdefault-double-8 -w -fallow-argument-mismatch -ffixed-line-length-none"
gfortran $FFLAGS -o stswm *.f
echo built $D
