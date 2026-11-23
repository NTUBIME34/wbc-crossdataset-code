#!/usr/bin/env python3
"""Supplementary Fig. S1 (domain shift): the same WBC types across four
reproducible-license source datasets, illustrating staining/scanner domain shift.
Real dataset images, each reproduced under its own public license (PBC CC BY 4.0,
HRLS CC0 1.0, AML-LMU CC BY 3.0, BMCMHF CC BY 4.0; cite each in the caption). No
baked-in title/watermark. The images are the dataset-original files as distributed
(no detector cropping), shown at a common display size.

Raabin-WBC is deliberately not shown: it is a model-training source (Table 1), but its
images are available under the Raabin Health Database terms (free access with
citation) rather than a Creative Commons license, and the paper does not reproduce
them (Data availability). Do not add a Raabin column.

Image paths are PINNED for reproducibility. The defaults below are relative to
--sources-root, the folder holding the datasets as downloaded (the SOURCES_ROOT of
data_prep/crop_training_sources.sh, not the prepared-data DATA_ROOT), and follow the
authors' folder names (PBC is the folder dataset_DIB). The authors' PBC and AML-LMU
copies are PNG files, and their BMCMHF class folders are renamed (Band_neutrophil,
Lymphocyte, Monocyte for the release's NGB, LYT, MON). When a pinned file is missing,
the same file is looked up in the release's BMCMHF class-code folder and with another
image extension (.jpg, .tif, ...), so the BMCMHF release, the JPEG files of the PBC
release and the TIFF files of the AML-LMU release are found as well. Your copy may use
other folder names, so point any panel at your own file with --image ROW/COLUMN=PATH,
for example

    --image Neutrophil/PBC=/path/to/PBC/neutrophil/BNE_495973.jpg

A relative PATH is resolved against --sources-root. Run with --list to print the
pinned paths. A missing file is drawn as a "missing" panel and the script exits
with status 1.

Output (in --out-dir): fig_domainshift.pdf / .png
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
COLUMNS = ["PBC", "HRLS", "AML-LMU", "BMCMHF"]   # column labels as printed in the paper
ROWS = ["Neutrophil", "Lymphocyte", "Monocyte"]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Pinned image files (relative to --sources-root). Swap any single entry to change that cell.
IMG = {
 ("Neutrophil", "PBC"): "dataset_DIB/neutrophil/BNE_495973.png",
 ("Neutrophil", "HRLS"): "HRLS_dataset/Neutrophil_seg/ 405.png",
 ("Neutrophil", "AML-LMU"): "PKG_AML_Cytomorphology_LMU/NGS/NGS_4916.png",
 ("Neutrophil", "BMCMHF"): "BM_cytomorphology_data/Band_neutrophil/0001-1000/NGB_00004.jpg",
 ("Lymphocyte", "PBC"): "dataset_DIB/lymphocyte/LY_785608.png",
 ("Lymphocyte", "HRLS"): "HRLS_dataset/Lymphocyte/2338.png",
 ("Lymphocyte", "AML-LMU"): "PKG_AML_Cytomorphology_LMU/LYA/LYA_0006.png",
 ("Lymphocyte", "BMCMHF"): "BM_cytomorphology_data/Lymphocyte/10001-11000/LYT_10941.jpg",
 ("Monocyte", "PBC"): "dataset_DIB/monocyte/MO_307572.png",
 ("Monocyte", "HRLS"): "HRLS_dataset/Monocyte/1213.png",
 ("Monocyte", "AML-LMU"): "PKG_AML_Cytomorphology_LMU/MON/MON_0361.png",
 ("Monocyte", "BMCMHF"): "BM_cytomorphology_data/Monocyte/2001-3000/MON_02824.jpg",
}
# The authors renamed the class-code folders of the BMCMHF release; the public download keeps
# the codes, with the same numbered range sub-folders and file names.
RELEASE_DIRS = {
    "BM_cytomorphology_data/Band_neutrophil/": "BM_cytomorphology_data/NGB/",
    "BM_cytomorphology_data/Lymphocyte/": "BM_cytomorphology_data/LYT/",
    "BM_cytomorphology_data/Monocyte/": "BM_cytomorphology_data/MON/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Domain-shift montage: three cell types across four source datasets "
                    "(Supplementary Fig. S1).",
    )
    parser.add_argument("--sources-root", "--data-root", dest="sources_root",
                        default=os.environ.get("SOURCES_ROOT"),
                        help="Folder holding the four datasets as downloaded (the SOURCES_ROOT of "
                             "data_prep/crop_training_sources.sh, not the prepared-data "
                             "DATA_ROOT); pinned and relative image paths are resolved against "
                             "it (default: $SOURCES_ROOT; --data-root is accepted as an alias).")
    parser.add_argument("--image", action="append", default=[], metavar="ROW/COLUMN=PATH",
                        help=f"Use PATH for one panel; ROW is one of {', '.join(ROWS)} and "
                             f"COLUMN one of {', '.join(COLUMNS)}. Repeatable.")
    parser.add_argument("--list", action="store_true",
                        help="Print the pinned (default) panel paths and exit.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: $OUT_ROOT/figures, else ./outputs/figures).")
    return parser.parse_args()


def out_dir_from(arg) -> Path:
    """--out-dir, else $OUT_ROOT/figures, else ./outputs/figures; never inside the repository."""
    out = Path(arg) if arg else Path(os.environ.get("OUT_ROOT", "outputs")) / "figures"
    out = out.expanduser().resolve()
    if out == REPO or REPO in out.parents:
        sys.exit(f"[ERROR] output folder {out} is inside the repository; "
                 "pass --out-dir or set OUT_ROOT to a folder outside it")
    return out


def release_candidates(rel: str) -> list[str]:
    """The pinned path, then the same file under the folder names of the public release."""
    return [rel] + [release + rel[len(authors):]
                    for authors, release in RELEASE_DIRS.items() if rel.startswith(authors)]


def find_release_file(root: Path, rel: str) -> Path:
    """root/rel if it exists, else the same file under the release's folder names (BMCMHF
    class codes) and/or with another image extension (the public releases distribute PBC as
    JPEG and AML-LMU as TIFF). Returns root/rel when none of them exists."""
    for cand in release_candidates(rel):
        path = root / cand
        for alt in (path, *(path.with_suffix(e) for e in IMG_EXTS + tuple(x.upper() for x in IMG_EXTS))):
            if alt.exists():
                if cand != rel or alt != path:
                    print(f"[INFO] {rel} not found; using {alt.relative_to(root)}")
                return alt
    return root / rel


def panel_paths(args):
    """(row, column) -> path, after applying the --image overrides."""
    chosen = dict(IMG)
    overridden = set()
    rows = {r.lower(): r for r in ROWS}
    cols = {c.lower(): c for c in COLUMNS}
    for spec in args.image:
        cell, sep, path = spec.partition("=")
        row, sep2, col = cell.partition("/")
        if not sep or not sep2 or row.lower() not in rows or col.lower() not in cols or not path:
            sys.exit(f"[ERROR] --image expects ROW/COLUMN=PATH with ROW in {ROWS} and "
                     f"COLUMN in {COLUMNS}; got: {spec}")
        key = (rows[row.lower()], cols[col.lower()])
        chosen[key] = path
        overridden.add(key)
    root = Path(args.sources_root).expanduser() if args.sources_root else None
    resolved = {}
    for key, path in chosen.items():
        p = Path(path).expanduser()
        if not p.is_absolute():
            if root is None:
                sys.exit(f"[ERROR] {key[0]}/{key[1]} uses the relative path '{path}'; "
                         "set --sources-root (or SOURCES_ROOT) or pass an absolute path with --image")
            p = root / p
        resolved[key] = p if key in overridden else find_release_file(root, path)
    return resolved


def square(path, size=1000):
    return ImageOps.fit(Image.open(path).convert("RGB"), (size, size), Image.LANCZOS)


def main() -> None:
    args = parse_args()
    if args.list:
        for (cls, name), rel in IMG.items():
            print(f"  {cls}/{name}: {rel}" + "".join(f"  (release: {alt})" for alt in release_candidates(rel)[1:]))
        return
    paths = panel_paths(args)
    out = out_dir_from(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    missing = []
    nR, nC = len(ROWS), len(COLUMNS)
    fig, axes = plt.subplots(nR, nC, figsize=(nC*1.7, nR*1.8))
    for ri, cls in enumerate(ROWS):
        for ci, name in enumerate(COLUMNS):
            ax = axes[ri][ci]; ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_edgecolor("#888"); s.set_linewidth(0.8)
            f = paths[(cls, name)]
            if f.exists(): ax.imshow(square(f))
            else:
                missing.append(f"{cls}/{name}: {f}")
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            if ri == 0: ax.set_title(name, fontsize=10, weight="bold", pad=4)
            if ci == 0: ax.set_ylabel(cls, fontsize=10, weight="bold")
    plt.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.01, wspace=0.06, hspace=0.06)
    fig.savefig(out/"fig_domainshift.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(out/"fig_domainshift.png", dpi=600, bbox_inches="tight")
    print(f"saved {out / 'fig_domainshift.pdf'} / .png")
    if missing:
        print(f"[WARN] {len(missing)} panel(s) drawn as 'missing':", *missing, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
