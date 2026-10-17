"""Report how much code the two games share.

Two measures:
1. Static: non-blank, non-comment lines in the shared library vs each game's own modules.
2. Dynamic (if `coverage` is installed): lines actually executed when running each game's test-sized
   episode, split into shared vs game-specific, and the fraction of executed lines that are shared.

  python scripts/shared_code.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def count_lines(path: str) -> int:
    n = 0
    with open(path) as fh:
        in_doc = False
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith('"""') or s.startswith("'''"):
                if s.count('"""') == 2 or s.count("'''") == 2:
                    continue
                in_doc = not in_doc
                continue
            if in_doc:
                continue
            n += 1
    return n


def walk(dirpath: str) -> dict[str, int]:
    out = {}
    for d, _, files in os.walk(dirpath):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(d, f)
                out[os.path.relpath(p, ROOT)] = count_lines(p)
    return out


def main():
    shared = {k: v for k, v in walk(os.path.join(ROOT, "simspec")).items() if "/games/" not in k and "/adapters/" not in k and "/export/" not in k}
    a = walk(os.path.join(ROOT, "simspec", "games", "contractsim"))
    b = walk(os.path.join(ROOT, "simspec", "games", "norms"))
    S, A, B = sum(shared.values()), sum(a.values()), sum(b.values())
    print("Static line counts (code lines, excluding blanks/comments/docstrings)")
    print(f"  shared library (core + agents): {S}")
    print(f"  ContractSim-specific:           {A}   -> shared fraction for A: {S/(S+A):.2f}")
    print(f"  norms-specific:                 {B}   -> shared fraction for B: {S/(S+B):.2f}")
    print(f"  shared / (shared + A + B):      {S/(S+A+B):.2f}")
    try:
        import coverage  # noqa: F401
    except ImportError:
        print("\n(pip install coverage for the dynamic measure)")
        return
    for game, cmd in [("contractsim", "import tests.test_contractsim as t, tests.test_contractsim_baselines as b; t.test_rc_vs_rc_deterministic_env_full_compliance(); b.test_memo_supplier_complies_when_payments_are_enforceable()"),
                      ("norms", "import tests.test_norms as t; t.test_learner_updates_beliefs_and_logs_them()")]:
        datafile = f"/tmp/.cov_{game}"
        snippet = f"/tmp/.cov_{game}_snippet.py"
        with open(snippet, "w") as fh:
            fh.write("import sys; sys.path.insert(0, %r)\n%s\n" % (ROOT, cmd))
        subprocess.run([sys.executable, "-m", "coverage", "run", f"--data-file={datafile}", "--source=simspec", snippet], cwd=ROOT, check=True, capture_output=True)
        rep = subprocess.run([sys.executable, "-m", "coverage", "report", f"--data-file={datafile}"], cwd=ROOT, capture_output=True, text=True).stdout
        exec_shared = exec_game = 0
        for line in rep.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[0].endswith(".py"):
                stmts, miss = int(parts[1]), int(parts[2])
                executed = stmts - miss
                if "/games/" in parts[0]:
                    if f"/games/{game}/" in parts[0]:
                        exec_game += executed
                elif "/adapters/" in parts[0] or "/export/" in parts[0]:
                    continue
                else:
                    exec_shared += executed
        print(f"\nDynamic ({game} episode): executed shared lines {exec_shared}, game-specific {exec_game}, shared fraction {exec_shared/(exec_shared+exec_game):.2f}")


if __name__ == "__main__":
    main()
