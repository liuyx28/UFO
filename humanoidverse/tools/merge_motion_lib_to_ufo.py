"""Merge SONIC/gear_sonic motion_lib PKLs into UFO ``ufo_pkl`` + optional clips/manifest.

Typical use after Bones-SEED CSV → motion_lib conversion
(``gear_sonic/data_process/convert_soma_csv_to_motion_lib.py --individual``):

.. code-block:: bash

    cd /path/to/UFO
    conda activate ufo
    python -m humanoidverse.tools.merge_motion_lib_to_ufo \\
      --input /path/to/gear_sonic/data/motion_lib_bones_seed/robot/220713 \\
      --out-dir humanoidverse/data/bones_seed_g1_220713 \\
      --name bones_seed_g1_220713 \\
      --clip-seconds 10 \\
      --write-manifest configs/data/bones_seed_g1_220713.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
from omegaconf import OmegaConf
from tqdm import tqdm

from humanoidverse.utils.motion_data.adapters import dump_ufo_pkl
from humanoidverse.utils.motion_data.clip import clip_ufo_motion_dict
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict


def _strip_clip_suffix(motion_key: str) -> str:
    marker = "__clip"
    if marker in motion_key:
        head, tail = motion_key.rsplit(marker, 1)
        if tail.isdigit():
            return head
    return motion_key


def _load_wanted_keys(keys_from: Path) -> set[str]:
    data = joblib.load(keys_from)
    if not isinstance(data, dict):
        raise TypeError(f"--keys-from must be a motion dict pkl, got {type(data)} in {keys_from}")
    return {_strip_clip_suffix(str(key)) for key in data.keys()}


def _iter_input_pkls(input_path: Path, *, recursive: bool = False) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix != ".pkl":
            raise ValueError(f"--input file must be a .pkl, got {input_path}")
        return [input_path]
    if input_path.is_dir():
        pkls = sorted(input_path.rglob("*.pkl") if recursive else input_path.glob("*.pkl"))
        if not pkls:
            raise FileNotFoundError(f"No .pkl files under {input_path}")
        return pkls
    raise FileNotFoundError(f"--input does not exist: {input_path}")


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep UFO-required fields; add ``dof_pos`` alias when only ``dof`` is present."""
    out = dict(record)
    if "dof_pos" not in out and "dof" in out:
        out["dof_pos"] = out["dof"]
    return out


def merge_motion_lib_pkls(
    input_path: Path,
    *,
    source_name: str,
    allow_key_collision: bool = True,
    recursive: bool = False,
    wanted_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Load one multi-motion pkl or many single-motion pkls into one UFO motion dict."""
    pkls = _iter_input_pkls(input_path, recursive=recursive)
    if wanted_keys is not None:
        pkls = [path for path in pkls if path.stem in wanted_keys]
        if not pkls:
            raise FileNotFoundError(f"No input pkls matched --keys-from under {input_path}")
    merged: dict[str, Any] = {}
    for path in tqdm(pkls, desc="merge pkls"):
        data = joblib.load(path)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict in {path}, got {type(data)}")
        # Single-motion files from --individual are usually {name: record};
        # a combined file is also {name: record, ...}.
        for raw_key, value in data.items():
            if not isinstance(value, dict):
                raise TypeError(f"Expected motion record dict for key={raw_key!r} in {path}")
            key = str(raw_key)
            if wanted_keys is not None and key not in wanted_keys and path.stem not in wanted_keys:
                continue
            if key in merged:
                if not allow_key_collision:
                    raise ValueError(f"Duplicate motion key={key!r} from {path}")
                key = f"{path.stem}__{raw_key}"
                if key in merged:
                    raise ValueError(f"Duplicate motion key after disambiguation: {key!r}")
            merged[key] = _normalize_record(value)
    if wanted_keys is not None:
        missing = sorted(wanted_keys - set(merged.keys()))
        if missing:
            preview = ", ".join(missing[:8])
            extra = f" ... (+{len(missing) - 8})" if len(missing) > 8 else ""
            raise FileNotFoundError(f"{len(missing)} keys from --keys-from missing in input: {preview}{extra}")
    return validate_ufo_motion_dict(merged, source_name)


def write_manifest(
    *,
    name: str,
    train_path: Path,
    inference_path: Path | None,
    out_path: Path,
    force: bool = False,
    robot_config: Path | None = None,
) -> Path:
    if out_path.exists() and not force:
        raise FileExistsError(f"Manifest exists: {out_path}. Pass --force to overwrite.")
    item: dict[str, Any] = {
        "name": name,
        "format": "ufo_pkl",
        "train_path": str(train_path),
        "weight": 1.0,
    }
    if inference_path is not None:
        item["inference_path"] = str(inference_path)
    config: dict[str, Any] = {"datasets": [item]}
    if robot_config is not None:
        config["robot_config"] = str(robot_config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), out_path)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory of individual motion_lib .pkl files, or one multi-motion .pkl",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for merged UFO pkls")
    parser.add_argument("--name", required=True, help="Dataset / source name (also used in filenames)")
    parser.add_argument(
        "--full-name",
        default="full_ufo.pkl",
        help="Filename for the unclipped merged pkl (default: full_ufo.pkl)",
    )
    parser.add_argument(
        "--train-name",
        default="train_near10s_ufo.pkl",
        help="Filename for the clipped training pkl (default: train_near10s_ufo.pkl)",
    )
    parser.add_argument("--clip-seconds", type=float, default=10.0, help="Clip length in seconds (0 = skip clipping)")
    parser.add_argument("--stride-seconds", type=float, default=None, help="Clip stride; default = clip-seconds")
    parser.add_argument("--keep-short", dest="keep_short", action="store_true", default=True)
    parser.add_argument("--drop-short", dest="keep_short", action="store_false")
    parser.add_argument("--min-clip-seconds", type=float, default=1.0)
    parser.add_argument(
        "--max-clips-per-motion",
        type=int,
        default=None,
        help="Keep only the first N clips per source motion (e.g. 1 for exact leading 10s)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search --input directories recursively for .pkl files",
    )
    parser.add_argument(
        "--keys-from",
        type=Path,
        default=None,
        help="Only keep motions whose keys appear in this UFO/motion_lib pkl",
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        help="Optional path to write a UFO data manifest YAML",
    )
    parser.add_argument(
        "--manifest-robot-config",
        type=Path,
        default=None,
        help="Optional robot_config path to embed in the written manifest",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs / manifest")
    parser.add_argument(
        "--no-key-collision-rename",
        action="store_true",
        help="Fail on duplicate motion keys instead of renaming with file stem",
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / args.full_name
    train_path = out_dir / args.train_name

    for path in (full_path, train_path):
        if path.exists() and not args.force:
            raise FileExistsError(f"Output exists: {path}. Pass --force to overwrite.")

    print(f"[merge] input={args.input}")
    wanted_keys = None
    if args.keys_from is not None:
        wanted_keys = _load_wanted_keys(args.keys_from.expanduser().resolve())
        print(f"[merge] keys_from={args.keys_from} wanted={len(wanted_keys)}")
    merged = merge_motion_lib_pkls(
        args.input.expanduser().resolve(),
        source_name=args.name,
        allow_key_collision=not args.no_key_collision_rename,
        recursive=args.recursive,
        wanted_keys=wanted_keys,
    )
    print(f"[merge] motions={len(merged)}")
    dump_ufo_pkl(merged, full_path, f"{args.name}:full")
    print(f"[merge] wrote {full_path} ({full_path.stat().st_size / 1e9:.2f} GB)")

    train_out = full_path
    if args.clip_seconds > 0.0:
        stride = args.stride_seconds if args.stride_seconds is not None else args.clip_seconds
        print(
            f"[clip] clip_seconds={args.clip_seconds} stride_seconds={stride} "
            f"max_clips_per_motion={args.max_clips_per_motion}"
        )
        train = clip_ufo_motion_dict(
            merged,
            clip_seconds=args.clip_seconds,
            stride_seconds=stride,
            keep_short=args.keep_short,
            min_clip_seconds=args.min_clip_seconds,
            source_name=f"{args.name}:train",
            max_clips_per_motion=args.max_clips_per_motion,
        )
        dump_ufo_pkl(train, train_path, f"{args.name}:train")
        print(f"[clip] clips={len(train)}")
        print(f"[clip] wrote {train_path} ({train_path.stat().st_size / 1e9:.2f} GB)")
        train_out = train_path
    else:
        print("[clip] skipped (clip-seconds=0); train_path will point at full pkl")

    if args.write_manifest is not None:
        manifest_path = write_manifest(
            name=args.name,
            train_path=train_out,
            inference_path=full_path,
            out_path=args.write_manifest.expanduser().resolve(),
            force=args.force,
            robot_config=args.manifest_robot_config,
        )
        print(f"[manifest] wrote {manifest_path}")


if __name__ == "__main__":
    main()
