"""Compare a modified FP64 STSWM build with the supplied FP32 TC5 diagnostics."""


import sys, os, re, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "refs/stswm")
num = re.compile(r"-?\d+\.\d+E[+-]\d+")


def main(out):
    if not os.path.isdir(os.path.join(D, "test5")):
        subprocess.run(["tar", "xzf", "stswm.tgz"], cwd=D, check=True)
    subprocess.run(["./build.sh", "42", "64", "128", "_val"], cwd=D, check=True, capture_output=True)
    b = os.path.join(D, "build_T42_val")
    subprocess.run(f"cp ../test5/exp.0005 . && ./stswm < exp.0005 > my_stdout.0005 2> my_stderr.0005", shell=True, cwd=b, check=True)
    orig = open(os.path.join(D, "test5/stdout.0005")).read().split("\n")
    mine = open(os.path.join(b, "my_stdout.0005")).read().split("\n")
    keys = ("MASS", "TOTAL ENERGY", "POTENTIAL ENSTROPHY", "GRIDPOINT HEIGHT")
    worst = {k: 0.0 for k in keys}; n = 0
    for lo, lm in zip(orig, mine):
        for k in keys:
            if lo.strip().startswith(k + " ") or lo.strip().startswith(k + "="):
                a, c = float(num.findall(lo)[0]), float(num.findall(lm)[0])
                worst[k] = max(worst[k], abs(a - c) / abs(a)); n += 1
    res = dict(compared_values=n, max_relative_difference=worst,
               note="NCAR STSWM 2.0 test case 5 at T42: supplied 32-bit diagnostics compared with the modified 64-bit gfortran build")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1])
