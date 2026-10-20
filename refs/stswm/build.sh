#!/bin/sh
# Build NCAR STSWM 2.0 (netlib chammp/stswm.tgz) at truncation T$1 in 64-bit precision.
# Changes to the original: params.i truncation, grid-point output (grdout.f) called at
# every output event instead of the netCDF spectral output, no graphics (stub library).
set -e
[ -d src ] || tar xzf stswm.tgz
T=$1; NLAT=$2; NLON=$3
D=build_T$T${4:-}
rm -rf $D; mkdir -p $D
cp src/*.f src/*.i lib/adquad.f lib/fft99f.f lib/netcdf.f lib/ncarg.f grdout.f $D/
cp src/netcdf.inc $D/
# netCDF is not used (stub library); keep the include compilable in 64-bit reals
sed -i.bak -e "s/1.797693134862315900e+308/1.0e+30/" -e "s/3.402823466e+38/1.0e+30/" $D/netcdf.inc
cd $D
# truncation
sed -i.bak -e "s/^      PARAMETER (MM=42, NN=42, KK=42, NLAT=64, NLON=128)/      PARAMETER (MM=$T, NN=$T, KK=$T, NLAT=$NLAT, NLON=$NLON)/" \
    -e "s/^      PARAMETER (MAXH=42)/      PARAMETER (MAXH=$T)/" -e "s/^      PARAMETER (RTRUNC=42)/      PARAMETER (RTRUNC=$T)/" params.i
grep -n "^      PARAMETER (MM=\|^      PARAMETER (MAXH\|^      PARAMETER (RTRUNC" params.i
# grid output at TAUO events
sed -i.bak "s/^         CALL OUTPTP (ZETASC,DIVSC,PHISC,PHIBAR,LDIAG,.FALSE.,OCTR)/         CALL GRDOUT(WS1,WS2,WS3,TAU)/" stswm.f
# day-0 output must be written before ZD overwrites WS1, WS2 (u, v): drop the day-0 OUTPTP call and
# write the initial state right after it is set up
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
