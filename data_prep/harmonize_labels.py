#!/usr/bin/env python3
"""Label harmonization of the five detector-cropped training sources (step 2 of
data_prep/README.md; Supplementary Table S3), and the merged internal corpus.

1. Per-source pools (always written). Each source's native class folders are mapped to
   the nine internal labels (Basophil, Eosinophil, Erythroblast, Lymphoblast, Lymphocyte,
   Monocyte, Myeloblast, Neutrophil, Promyelocyte); only explicitly mapped native classes
   are kept and all others are dropped. Each pool is laid out as
   <pool>/<internal label>/<image> with the original file names. The five pools are the
   data of the leave-one-dataset-out (LODO) analysis (Supplementary Table S15).

   Crosswalk (native class -> internal label; everything else is dropped):
     AML-LMU     BAS->Basophil, EOS->Eosinophil, EBO->Erythroblast, LYT+LYA->Lymphocyte,
                 MON->Monocyte, MYO->Myeloblast, NGS->Neutrophil, PMO->Promyelocyte
                 (dropped: KSC, MMZ, MYB, MOB, NGB, PMB)
     BMCMHF      Basophil, Eosinophil, Erythroblast, Lymphocyte, Monocyte, Promyelocyte -> same;
                 Segmented + Band neutrophil -> Neutrophil
                 (dropped: Blast, Abnormal eosinophil, Faggot cell, Hairy cell, Immature
                 lymphocyte, Metamyelocyte, Myelocyte, Plasma cell, Proerythroblast)
     PBC         basophil, eosinophil, erythroblast, lymphocyte, monocyte, neutrophil -> same
                 (ig and platelet are not passed to the detector)
     Raabin-WBC  Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil -> same
     HRLS        Basophil, Eosinophil, Lymphoblast, Lymphocyte, Monocyte, Myeloblast -> same;
                 Neutrophil_seg -> Neutrophil (dropped: Normoblast, Neutrophil_band)
   Native folders are matched case-insensitively, by the release's class code (AML-LMU and
   BMCMHF: BAS, EOS, LYT, ...) or by the long folder names of the authors' copies.

   AML-LMU typical lymphocytes (LYT). The paper's AML-LMU split was built in two steps:
   the first nine-class pool omitted the typical lymphocytes, and the corrected split and
   held-out set restore them (Supplementary Note 1; build_corrected_lmu_split.py and
   build_lmu_lodo_test.py). To reproduce those splits exactly, the default
   --lmu-typical defer leaves LYT out of the AML-LMU pool written here (its Lymphocyte
   folder then holds the atypical LYA cells only); the two later scripts add LYT from the
   crop folder. --lmu-typical include writes the full Table S3 pool in one step instead;
   then skip build_corrected_lmu_split.py (the resulting AML-LMU split differs from the
   paper's).

2. Merged internal corpus (--merged-dir). The corpus behind the main results (127,938
   cells, Supplementary Table S2) was not built from these pools: it was cropped at 0.70
   for every source, and its merge kept a few native classes outside the crosswalk
   (Supplementary S1: 103 + 99 AML-LMU/HRLS band neutrophils under Neutrophil, 50 BMCMHF
   immature lymphocytes under Lymphocyte, 24 AML-LMU monoblasts under Myeloblast), plus
   Metamyelocyte and Myelocyte folders that were present when the split was drawn and
   removed after it. --merged-dir rebuilds that composition from the crop folders:
     * a crop is kept only if its best-box confidence, read from the crop folder's
       crop_log.csv (written by detector_crop.py), is at least --merged-min-conf (0.70);
     * the Table S3 crosswalk, typical lymphocytes included, plus the merge extras
       AML-LMU NGB->Neutrophil, MOB->Myeloblast, MMZ->Metamyelocyte, MYB->Myelocyte;
       BMCMHF Immature lymphocyte->Lymphocyte, Metamyelocyte, Myelocyte;
       HRLS Neutrophil_band->Neutrophil;
     * files are named <source code>__<native class>__<original name>.
   Split it with split_dataset.py ... --drop-after-split Metamyelocyte Myelocyte (see
   data_prep/README.md). --merged-crosswalk-only writes the pools' crosswalk instead
   (no merge extras, no Metamyelocyte/Myelocyte), a cleaner corpus that differs from
   the paper's. The rebuilt corpus has the paper's class counts but not its file names,
   so the seed-42 split draws other cells than the authors' split (data_prep/README.md).

Example:
  python data_prep/harmonize_labels.py --data-root "$DATA_ROOT" \\
      --merged-dir "$DATA_ROOT/Mixed_data_mix_mono"
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
INTERNAL_LABELS = ["Basophil", "Eosinophil", "Erythroblast", "Lymphoblast", "Lymphocyte",
                   "Monocyte", "Myeloblast", "Neutrophil", "Promyelocyte"]
SPLIT_ONLY_LABELS = ["Metamyelocyte", "Myelocyte"]  # merged corpus: split, then removed
TYPICAL = "Lymphocyte (typical, LYT)"  # AML-LMU marker: handled by --lmu-typical
DROP = None

# source -> crop folder (input), pool folder (output), fold code, crosswalk, merge extras
SOURCES = {
    "AML-LMU": {
        "input": "PKG_AML_Cytomorphology_LMU_single_wbc",
        "output": "PKG_AML_Cytomorphology_LMU_single_wbc_for_train_9class",
        "code": "LMU",
        "map": {
            "BAS": "Basophil", "Basophil": "Basophil",
            "EOS": "Eosinophil", "Eosinophil": "Eosinophil",
            "EBO": "Erythroblast", "Erythroblast": "Erythroblast",
            "LYA": "Lymphocyte", "Lymphocyte": "Lymphocyte",  # authors' copy: atypical = "Lymphocyte "
            "LYT": TYPICAL, "Lymphocyte_typical": TYPICAL,
            "MON": "Monocyte", "Monocyte": "Monocyte",
            "MYO": "Myeloblast", "Myeloblast": "Myeloblast",
            "NGS": "Neutrophil", "Neutrophil_seg": "Neutrophil",
            "PMO": "Promyelocyte", "Promyelocyte": "Promyelocyte",
            "KSC": DROP, "Smudge_cell": DROP,
            "MMZ": DROP, "Metamyelocyte": DROP,
            "MYB": DROP, "Myelocyte": DROP,
            "MOB": DROP, "Monoblast": DROP,
            "NGB": DROP, "Neutrophil_band": DROP,
            "PMB": DROP, "Promyelocyte_bi": DROP,
        },
        "merge_extra": {
            "NGB": "Neutrophil", "Neutrophil_band": "Neutrophil",
            "MOB": "Myeloblast", "Monoblast": "Myeloblast",
            "MMZ": "Metamyelocyte", "Metamyelocyte": "Metamyelocyte",
            "MYB": "Myelocyte", "Myelocyte": "Myelocyte",
        },
    },
    "BMCMHF": {
        "input": "BM_cytomorphology_data_single_wbc_07",
        "output": "BM_cytomorphology_data_single_wbc_07_for_train_9class",
        "code": "BM",
        "map": {
            "BAS": "Basophil", "Basophil": "Basophil",
            "EOS": "Eosinophil", "Eosinophil": "Eosinophil",
            "EBO": "Erythroblast", "Erythroblast": "Erythroblast",
            "LYT": "Lymphocyte", "Lymphocyte": "Lymphocyte",
            "MON": "Monocyte", "Monocyte": "Monocyte",
            "PMO": "Promyelocyte", "Promyelocyte": "Promyelocyte",
            "NGS": "Neutrophil", "Segmented_neutrophil": "Neutrophil",
            "NGB": "Neutrophil", "Band_neutrophil": "Neutrophil",
            "BLA": DROP, "Blast": DROP,
            "ABE": DROP, "Abnormal_eosinophil": DROP,
            "FGC": DROP, "Faggot_cell": DROP, "Faggott_cell": DROP,
            "HAC": DROP, "Hairy_cell": DROP,
            "LYI": DROP, "Immature_lymphocyte": DROP,
            "MMZ": DROP, "Metamyelocyte": DROP,
            "MYB": DROP, "Myelocyte": DROP,
            "PLM": DROP, "Plasma_cell": DROP,
            "PEB": DROP, "Proerythroblast": DROP,
            "ART": DROP, "Artefact": DROP, "Artifact": DROP,
            "NIF": DROP, "Not_identifiable": DROP,
            "OTH": DROP, "Other_cell": DROP,
            "KSC": DROP, "Smudge_cell": DROP,
        },
        "merge_extra": {
            "LYI": "Lymphocyte", "Immature_lymphocyte": "Lymphocyte",
            "MMZ": "Metamyelocyte", "Metamyelocyte": "Metamyelocyte",
            "MYB": "Myelocyte", "Myelocyte": "Myelocyte",
        },
    },
    "PBC": {
        "input": "dataset_DIB_single_wbc",
        "output": "dataset_DIB_single_wbc_copy",
        "code": "DIB",
        "map": {
            "basophil": "Basophil", "eosinophil": "Eosinophil", "erythroblast": "Erythroblast",
            "lymphocyte": "Lymphocyte", "monocyte": "Monocyte", "neutrophil": "Neutrophil",
            "ig": DROP, "platelet": DROP,
        },
        "merge_extra": {},
    },
    "Raabin-WBC": {
        "input": "Raabin_WBC_single_wbc",
        "output": "Raabin_WBC_single_wbc_copy",
        "code": "Raabin",
        "map": {
            "Basophil": "Basophil", "Eosinophil": "Eosinophil", "Lymphocyte": "Lymphocyte",
            "Monocyte": "Monocyte", "Neutrophil": "Neutrophil",
        },
        "merge_extra": {},
    },
    "HRLS": {
        "input": "HRLS_dataset_single_wbc",
        "output": "HRLS_dataset_single_wbc_copy",
        "code": "HRLS",
        "map": {
            "Basophil": "Basophil", "Eosinophil": "Eosinophil", "Lymphoblast": "Lymphoblast",
            "Lymphocyte": "Lymphocyte", "Monocyte": "Monocyte", "Myeloblast": "Myeloblast",
            "Neutrophil_seg": "Neutrophil",
            "Neutrophil_band": DROP, "Normoblast": DROP,
        },
        "merge_extra": {"Neutrophil_band": "Neutrophil"},
    },
}

# Harmonized pool sizes (Supplementary Table S15).
EXPECTED = {
    "AML-LMU": {"Neutrophil": 8483, "Lymphocyte": 3947, "Monocyte": 1779, "Eosinophil": 424,
                "Basophil": 79, "Erythroblast": 78, "Promyelocyte": 70, "Myeloblast": 3268},
    "PBC": {"Neutrophil": 3329, "Lymphocyte": 1214, "Monocyte": 1420, "Eosinophil": 3117,
            "Basophil": 1218, "Erythroblast": 1547},
    "Raabin-WBC": {"Neutrophil": 6217, "Lymphocyte": 2420, "Monocyte": 561, "Eosinophil": 744,
                   "Basophil": 212},
    "HRLS": {"Neutrophil": 3198, "Lymphocyte": 3037, "Monocyte": 2039, "Eosinophil": 1017,
             "Basophil": 1023, "Myeloblast": 2534, "Lymphoblast": 2557},
    "BMCMHF": {"Neutrophil": 26536, "Lymphocyte": 20284, "Monocyte": 3049, "Eosinophil": 3484,
               "Basophil": 355, "Erythroblast": 13470, "Promyelocyte": 5918},
}
# Merged internal corpus in the seven-class evaluation space (Supplementary Table S2).
EXPECTED_MERGED_7 = {"Neutrophil": 47551, "Lymphocyte": 30811, "Blast group": 23422,
                     "Eosinophil": 8766, "Monocyte": 8526, "Promyelocyte": 5988, "Basophil": 2874}
BLAST_GROUP = {"Erythroblast", "Lymphoblast", "Myeloblast"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map the native classes of the five cropped training sources to the nine "
                    "internal labels (Supplementary Table S3), write one pool per source and "
                    "optionally the merged internal corpus.")
    parser.add_argument("--data-root", type=Path, default=os.environ.get("DATA_ROOT"),
                        help="Default for --crops-root and --out-root (default: $DATA_ROOT).")
    parser.add_argument("--crops-root", type=Path, default=None,
                        help="Folder holding the crop folders of crop_training_sources.sh.")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Folder that receives the harmonized pool folders.")
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES),
                        help="Sources to harmonize (default: all five).")
    parser.add_argument("--mode", choices=["copy", "link", "symlink"], default="copy",
                        help="copy files (default), hard-link them (falls back to copy) or symlink them.")
    parser.add_argument("--lmu-typical", choices=["defer", "include"], default="defer",
                        help="defer (default): leave the AML-LMU typical lymphocytes out of the pool "
                             "so that the later scripts reproduce the paper's splits; include: put "
                             "them in the pool now.")
    parser.add_argument("--merged-dir", type=Path, default=None,
                        help="Also write the merged internal corpus here (class folders; the authors' "
                             "folder name is Mixed_data_mix_mono, see data_prep/README.md).")
    parser.add_argument("--merged-min-conf", type=float, default=0.70,
                        help="Minimum best-box confidence of a crop in the merged corpus, read from "
                             "crop_log.csv (default 0.70, as run; 0 merges every crop and needs no log).")
    parser.add_argument("--merged-crosswalk-only", action="store_true",
                        help="Merged corpus with the Table S3 crosswalk only (no merge extras, no "
                             "Metamyelocyte/Myelocyte folders); differs from the paper's corpus.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing output folders (default: refuse).")
    return parser.parse_args()


def lookup(mapping: dict[str, str | None], folder: str) -> tuple[bool, str | None]:
    key = folder.strip().lower()
    for native, label in mapping.items():
        if native.lower() == key:
            return True, label
    return False, None


def list_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXTS
                  and not any(part.startswith(".") for part in p.relative_to(class_dir).parts))


def place(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        raise SystemExit(f"Name clash: {dst} already exists (two native classes give the same "
                         "file name).")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "link":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise SystemExit(f"Refusing to write inside the repository: {path}")
    if path.exists():
        if not overwrite:
            raise SystemExit(f"{path} exists; rerun with --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def read_confidences(crop_dir: Path) -> dict[str, float]:
    """crop_file -> best-box confidence, from the crop_log.csv of detector_crop.py."""
    log = crop_dir / "crop_log.csv"
    if not log.exists():
        raise SystemExit(f"{log} not found. The merged corpus keeps crops by their best-box "
                         "confidence, which detector_crop.py logs; re-crop with "
                         "crop_training_sources.sh or pass --merged-min-conf 0 to merge every crop.")
    out: dict[str, float] = {}
    with log.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("crop_file") and row.get("conf"):
                out[row["crop_file"]] = float(row["conf"])
    return out


def merged_label(spec: dict, folder: str, crosswalk_only: bool) -> str | None:
    if not crosswalk_only:
        found, label = lookup(spec["merge_extra"], folder)
        if found:
            return label
    _, label = lookup(spec["map"], folder)
    return "Lymphocyte" if label == TYPICAL else label


def main() -> None:
    args = parse_args()
    crops_root = (args.crops_root or args.data_root)
    out_root = (args.out_root or args.data_root)
    if crops_root is None or out_root is None:
        raise SystemExit("Set DATA_ROOT or pass --data-root (or --crops-root and --out-root).")
    crops_root, out_root = Path(crops_root).expanduser().resolve(), Path(out_root).expanduser().resolve()
    merged = args.merged_dir.expanduser().resolve() if args.merged_dir else None

    # Check every input before writing anything.
    plan = []
    for source in args.sources:
        spec = SOURCES[source]
        in_dir = crops_root / spec["input"]
        if not in_dir.is_dir():
            raise SystemExit(f"{source}: crop folder not found: {in_dir}")
        class_dirs = sorted(p for p in in_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
        unknown = [d.name for d in class_dirs if not lookup(spec["map"], d.name)[0]]
        if unknown:
            raise SystemExit(f"{source}: class folders not in the crosswalk: {unknown}. "
                             "Rename them to the release's class codes or extend SOURCES.")
        conf = read_confidences(in_dir) if merged is not None and args.merged_min_conf > 0 else None
        plan.append((source, spec, in_dir, class_dirs, conf))
    if merged is not None:
        prepare_dir(merged, args.overwrite)

    grand_total = 0
    all_ok = True
    merged_counts: Counter[str] = Counter()
    for source, spec, in_dir, class_dirs, conf in plan:
        out_dir = out_root / spec["output"]
        prepare_dir(out_dir, args.overwrite)

        kept: Counter[str] = Counter()
        dropped: Counter[str] = Counter()
        typical: list[Path] = []
        typical_dir = None
        merged_skipped = merged_unlogged = 0
        for class_dir in class_dirs:
            _, label = lookup(spec["map"], class_dir.name)
            images = list_images(class_dir)
            m_label = None
            if merged is not None:
                m_label = merged_label(spec, class_dir.name, args.merged_crosswalk_only)
            if m_label is not None:
                for src in images:
                    if conf is not None:
                        c = conf.get(src.relative_to(in_dir).as_posix())
                        if c is None:
                            merged_unlogged += 1
                            continue
                        if c < args.merged_min_conf:
                            merged_skipped += 1
                            continue
                    name = f"{spec['code']}__{class_dir.name.strip()}__{src.name}"
                    place(src, merged / m_label / name, args.mode)
                    merged_counts[m_label] += 1
            if label is DROP:
                dropped[class_dir.name.strip()] += len(images)
                continue
            if label == TYPICAL:
                typical_dir = class_dir
                typical.extend(images)
                if args.lmu_typical == "defer":
                    continue
                label = "Lymphocyte"
            for src in images:
                place(src, out_dir / label / src.name, args.mode)
                kept[label] += 1

        deferred = len(typical) if args.lmu_typical == "defer" else 0
        written = sum(kept.values())
        print(f"\n== {source}: {in_dir.name} -> {out_dir.name}  ({written} images written)")
        expected = EXPECTED[source]
        for label in INTERNAL_LABELS:
            n = kept.get(label, 0) + (deferred if label == "Lymphocyte" else 0)
            exp = expected.get(label, 0)
            if n == 0 and exp == 0:
                continue
            extra = f"  (written {kept.get(label, 0)} + {deferred} typical deferred)" \
                if label == "Lymphocyte" and deferred else ""
            status = "ok" if n == exp else "DIFFERS"
            all_ok &= n == exp
            print(f"   {label:13s} {n:7d}   expected {exp:7d}  {status}{extra}")
        pool_total = written + deferred
        exp_total = sum(expected.values())
        all_ok &= pool_total == exp_total
        print(f"   {'pool total':13s} {pool_total:7d}   expected {exp_total:7d}  "
              f"{'ok' if pool_total == exp_total else 'DIFFERS'}")
        if dropped:
            print("   dropped native classes: " + ", ".join(f"{k} {v}" for k, v in sorted(dropped.items())))
        if deferred:
            print(f"   typical lymphocytes left in {typical_dir} for build_corrected_lmu_split.py "
                  "and build_lmu_lodo_test.py")
        if merged is not None and conf is not None:
            print(f"   merged corpus: {merged_skipped} crops below confidence {args.merged_min_conf:.2f} left out"
                  + (f"; {merged_unlogged} crops without a crop_log.csv entry left out" if merged_unlogged else ""))
        grand_total += pool_total

    if set(args.sources) == set(SOURCES):
        print(f"\nAll five pools: {grand_total} cells (Supplementary S1 and Table S15: 128,628)")
    if merged is not None:
        print(f"\n== merged internal corpus: {merged}")
        for label in sorted(merged_counts):
            note = "  (split, then removed: --drop-after-split)" if label in SPLIT_ONLY_LABELS else ""
            print(f"   {label:13s} {merged_counts[label]:7d}{note}")
        seven: Counter[str] = Counter()
        for label, n in merged_counts.items():
            if label not in SPLIT_ONLY_LABELS:
                seven["Blast group" if label in BLAST_GROUP else label] += n
        if set(args.sources) == set(SOURCES) and not args.merged_crosswalk_only:
            print("   seven-class view versus Supplementary Table S2:")
            for label, exp in EXPECTED_MERGED_7.items():
                ok = seven.get(label, 0) == exp
                all_ok &= ok
                print(f"   {label:13s} {seven.get(label, 0):7d}   expected {exp:7d}  {'ok' if ok else 'DIFFERS'}")
            total, exp_total = sum(seven.values()), sum(EXPECTED_MERGED_7.values())
            all_ok &= total == exp_total
            print(f"   {'total':13s} {total:7d}   expected {exp_total:7d}  {'ok' if total == exp_total else 'DIFFERS'}")
        drop = " --drop-after-split " + " ".join(SPLIT_ONLY_LABELS) if not args.merged_crosswalk_only else ""
        print(f"Next: python data_prep/split_dataset.py --src \"{merged}\" "
              f"--dst \"{merged.parent / 'Mixed_data_mix_mono_9_split_copy'}\"{drop}")
    if not all_ok:
        print("\nWARNING: some counts differ from Supplementary Tables S15/S2 (different dataset "
              "version, different crop thresholds, a different detector run near the threshold, "
              "or a missing class folder).")


if __name__ == "__main__":
    main()
