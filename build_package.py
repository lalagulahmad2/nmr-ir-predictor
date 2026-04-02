"""
Build a distributable ZIP of the NMR & IR Spectrum Predictor.
Run:  python build_package.py
Output: nmr_ir_predictor.zip  (in the parent directory)
"""
import os, zipfile, pathlib, datetime

ROOT = pathlib.Path(__file__).parent
OUT  = ROOT.parent / "nmr_ir_predictor.zip"

# Files/dirs to include (relative to ROOT)
INCLUDE = [
    "app.py",
    "requirements.txt",
    "run.bat",
    "setup.bat",
    "templates/index.html",
    "static/remote",          # entire Ketcher bundle
]

# Patterns to always skip
SKIP_SUFFIXES = {".pyc", ".pyo"}
SKIP_DIRS     = {"__pycache__", ".claude", ".git"}

def should_skip(path: pathlib.Path) -> bool:
    if path.suffix in SKIP_SUFFIXES:
        return True
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    return False

collected: list[tuple[pathlib.Path, str]] = []

for entry in INCLUDE:
    full = ROOT / entry
    if full.is_file():
        if not should_skip(full):
            arc = str(pathlib.Path("nmr_ir_predictor") / full.relative_to(ROOT))
            collected.append((full, arc))
    elif full.is_dir():
        for f in full.rglob("*"):
            if f.is_file() and not should_skip(f):
                arc = str(pathlib.Path("nmr_ir_predictor") / f.relative_to(ROOT))
                collected.append((f, arc))

with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for src, arcname in collected:
        zf.write(src, arcname)
        print(f"  + {arcname}")

size_mb = OUT.stat().st_size / 1_048_576
print(f"\nDone — {OUT.name}  ({size_mb:.1f} MB)  [{len(collected)} files]")
print(f"Saved to: {OUT}")
