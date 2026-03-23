import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "compare_geometry_branches_zju_report.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a ZJU geometry branch sweep across multiple frames and target cameras."
    )
    parser.add_argument(
        "--template_reports",
        nargs="+",
        required=True,
        help="Template report.json files that define source-view profiles.",
    )
    parser.add_argument(
        "--local_zju_root",
        type=str,
        required=True,
        help="Local ZJU_MoCap root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint used for all runs.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="output/geometry_view_sweep_zju",
        help="Root directory for sweep outputs.",
    )
    parser.add_argument(
        "--frame_ids",
        type=str,
        required=True,
        help="Comma-separated frame ids to evaluate, e.g. 0,150,300.",
    )
    parser.add_argument(
        "--target_cameras",
        type=str,
        default="all",
        help="Comma-separated target cameras, or 'all'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--export_max_points",
        type=int,
        default=100000,
        help="Reduce sweep disk usage while keeping render metrics unchanged.",
    )
    parser.add_argument(
        "--render_max_points",
        type=int,
        default=750000,
    )
    parser.add_argument(
        "--conf_percentile",
        type=float,
        default=25.0,
    )
    parser.add_argument(
        "--z_tolerance",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip cases whose summary.json already exists.",
    )
    parser.add_argument(
        "--source_policy",
        type=str,
        default="fixed_template",
        choices=["fixed_template", "rotate_template_offsets", "nearest_ring", "uniform_ring"],
        help="How to derive sparse source subsets for each target camera.",
    )
    parser.add_argument(
        "--limit_cases",
        type=int,
        default=0,
        help="Optional debugging cap on the number of cases to run.",
    )
    return parser.parse_args()


def parse_csv_list(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_frame_ids(text):
    return [int(item) for item in parse_csv_list(text)]


def load_template_profile(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = payload["meta"]
    return {
        "template_report": str(Path(path).resolve()),
        "seq_name": str(meta["seq_name"]),
        "view_profile": str(meta["view_profile"]),
        "profile_kind": str(meta.get("profile_kind", "")),
        "source_cameras": list(meta["src_cameras"]),
        "template_target_camera": str(meta["tgt_camera"]),
        "template_frame_id": int(meta["frame_id"]),
    }


def discover_all_cameras(seq_dir):
    return sorted(path.name for path in Path(seq_dir).glob("Camera_*") if path.is_dir())


def resolve_target_cameras(seq_dir, target_cameras_arg):
    all_cameras = discover_all_cameras(seq_dir)
    if target_cameras_arg.strip().lower() == "all":
        return all_cameras
    requested = parse_csv_list(target_cameras_arg)
    missing = [camera for camera in requested if camera not in all_cameras]
    if missing:
        raise ValueError(f"Requested target cameras are missing under {seq_dir}: {missing}")
    return requested


def discover_camera_ring_order(seq_dir):
    annots_path = Path(seq_dir) / "annots.npy"
    annots = np.load(annots_path, allow_pickle=True).item()
    rotations = annots["cams"]["R"]
    translations = annots["cams"]["T"]
    if len(rotations) != len(translations):
        raise ValueError(f"Camera calibration length mismatch under {annots_path}")

    rows = []
    for index, (rotation, translation) in enumerate(zip(rotations, translations), start=1):
        rotation = np.asarray(rotation, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
        center = (-rotation.T @ translation).reshape(3)
        azimuth = float(np.degrees(np.arctan2(center[0], center[2])))
        rows.append((f"Camera_B{index}", azimuth))

    rows.sort(key=lambda item: item[1])
    return [name for name, _ in rows]


def attach_template_offsets(profile, ring_order):
    if profile["profile_kind"] == "full_rig_excluding_target":
        profile["template_offsets"] = []
        return profile

    camera_to_index = {camera: idx for idx, camera in enumerate(ring_order)}
    template_target = profile["template_target_camera"]
    if template_target not in camera_to_index:
        raise ValueError(f"Template target camera {template_target} missing from ring order")

    template_target_idx = camera_to_index[template_target]
    offsets = []
    for camera in profile["source_cameras"]:
        if camera not in camera_to_index:
            raise ValueError(f"Template source camera {camera} missing from ring order")
        offset = (camera_to_index[camera] - template_target_idx) % len(ring_order)
        if offset == 0:
            raise ValueError(f"Template source camera {camera} collapsed onto target {template_target}")
        offsets.append(int(offset))

    profile["template_offsets"] = offsets
    return profile


def resolve_source_cameras(profile, target_camera, all_cameras, source_policy, ring_order):
    if profile["profile_kind"] == "full_rig_excluding_target":
        return [camera for camera in all_cameras if camera != target_camera]

    if source_policy == "fixed_template":
        source_cameras = list(profile["source_cameras"])
        if target_camera in source_cameras:
            return None
        return source_cameras

    if source_policy == "rotate_template_offsets":
        camera_to_index = {camera: idx for idx, camera in enumerate(ring_order)}
        if target_camera not in camera_to_index:
            raise ValueError(f"Target camera {target_camera} missing from ring order")
        target_idx = camera_to_index[target_camera]
        selected = [ring_order[(target_idx + offset) % len(ring_order)] for offset in profile["template_offsets"]]
        selected = list(dict.fromkeys(selected))
        if target_camera in selected:
            raise ValueError(f"Target-aware selection accidentally included target camera {target_camera}")
        if len(selected) != len(profile["template_offsets"]):
            raise ValueError(
                f"Target-aware selection changed source count for {profile['view_profile']} / {target_camera}: "
                f"expected {len(profile['template_offsets'])}, got {len(selected)}"
            )
        return selected

    if source_policy == "nearest_ring":
        camera_to_index = {camera: idx for idx, camera in enumerate(ring_order)}
        if target_camera not in camera_to_index:
            raise ValueError(f"Target camera {target_camera} missing from ring order")
        target_idx = camera_to_index[target_camera]
        selected = []
        for ring_step in range(1, len(ring_order)):
            for offset in (ring_step, -ring_step):
                camera = ring_order[(target_idx + offset) % len(ring_order)]
                if camera == target_camera or camera in selected:
                    continue
                selected.append(camera)
                if len(selected) == len(profile["source_cameras"]):
                    return selected
        raise ValueError(
            f"Unable to build nearest_ring selection for {profile['view_profile']} / {target_camera}"
        )

    if source_policy == "uniform_ring":
        camera_to_index = {camera: idx for idx, camera in enumerate(ring_order)}
        if target_camera not in camera_to_index:
            raise ValueError(f"Target camera {target_camera} missing from ring order")
        target_idx = camera_to_index[target_camera]
        source_count = len(profile["source_cameras"])
        total_offsets = len(ring_order) - 1
        raw_offsets = []
        for position in range(source_count):
            offset = int(np.floor((position + 0.5) * total_offsets / source_count)) + 1
            offset = max(1, min(total_offsets, offset))
            raw_offsets.append(offset)
        selected = []
        used = set()
        for offset in raw_offsets:
            camera = ring_order[(target_idx + offset) % len(ring_order)]
            if camera == target_camera or camera in used:
                continue
            selected.append(camera)
            used.add(camera)
        if len(selected) < source_count:
            for offset in range(1, len(ring_order)):
                camera = ring_order[(target_idx + offset) % len(ring_order)]
                if camera == target_camera or camera in used:
                    continue
                selected.append(camera)
                used.add(camera)
                if len(selected) == source_count:
                    break
        if len(selected) != source_count:
            raise ValueError(
                f"Unable to build uniform_ring selection for {profile['view_profile']} / {target_camera}: "
                f"expected {source_count}, got {len(selected)}"
            )
        return selected

    raise ValueError(f"Unsupported source policy: {source_policy}")


def build_case_meta(profile, frame_id, target_camera, seq_dir, all_cameras, source_policy, ring_order):
    if profile["profile_kind"] == "full_rig_excluding_target":
        source_cameras = [camera for camera in all_cameras if camera != target_camera]
    else:
        source_cameras = resolve_source_cameras(profile, target_camera, all_cameras, source_policy, ring_order)
        if source_cameras is None:
            return None

    if not source_cameras:
        return None

    frame_stem = f"{int(frame_id):06d}"
    return {
        "time": "",
        "zju_root": str(seq_dir.parent),
        "seq_name": profile["seq_name"],
        "frame_id": int(frame_id),
        "view_profile": profile["view_profile"],
        "profile_kind": profile["profile_kind"],
        "source_policy": source_policy,
        "num_total_cams": int(len(source_cameras) + (0 if target_camera in source_cameras else 1)),
        "num_src_views_actual": int(len(source_cameras)),
        "src_cameras": source_cameras,
        "tgt_camera": str(target_camera),
        "src_image_paths": [f"{seq_dir.as_posix()}/{camera}/{frame_stem}.jpg" for camera in source_cameras],
        "tgt_image_path": f"{seq_dir.as_posix()}/{target_camera}/{frame_stem}.jpg",
        "tgt_mask_path": f"{seq_dir.as_posix()}/mask/{target_camera}/{frame_stem}.png",
        "template_report": profile["template_report"],
        "template_target_camera": profile["template_target_camera"],
        "template_frame_id": profile["template_frame_id"],
    }


def build_case_list(profiles, seq_dir, frame_ids, target_cameras, source_policy, ring_order):
    all_cameras = discover_all_cameras(seq_dir)
    cases = []
    for profile in profiles:
        for frame_id in frame_ids:
            for target_camera in target_cameras:
                meta = build_case_meta(profile, frame_id, target_camera, seq_dir, all_cameras, source_policy, ring_order)
                if meta is None:
                    continue
                cases.append(
                    {
                        "profile": profile,
                        "frame_id": int(frame_id),
                        "target_camera": str(target_camera),
                        "meta": meta,
                    }
                )
    return cases


def case_output_dir(output_root, case):
    return (
        Path(output_root)
        / case["profile"]["view_profile"]
        / f"frame_{case['frame_id']:06d}_{case['target_camera']}"
    )


def write_synthetic_report(path, meta):
    payload = {
        "meta": meta,
        "metrics": {
            "native": {},
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_case(args, case):
    out_dir = case_output_dir(args.output_root, case)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if args.skip_existing and summary_path.exists():
        return {
            "status": "cached",
            "case_dir": str(out_dir),
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        }

    synthetic_report = out_dir / "synthetic_report.json"
    write_synthetic_report(synthetic_report, case["meta"])

    cmd = [
        sys.executable,
        str(RUNNER),
        "--report_json",
        str(synthetic_report),
        "--local_zju_root",
        str(args.local_zju_root),
        "--checkpoint",
        str(args.checkpoint),
        "--output_dir",
        str(out_dir),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
        "--conf_percentile",
        str(args.conf_percentile),
        "--export_max_points",
        str(args.export_max_points),
        "--render_max_points",
        str(args.render_max_points),
        "--z_tolerance",
        str(args.z_tolerance),
        "--min_conf",
        str(args.min_conf),
        "--skip_save_predictions",
    ]

    print(
        f"[zju-view-sweep] profile={case['profile']['view_profile']} frame={case['frame_id']:06d} target={case['target_camera']}",
        flush=True,
    )
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    (out_dir / "batch_run.log").write_text(
        (proc.stdout or "") + ("\n" if proc.stdout else "") + (proc.stderr or ""),
        encoding="utf-8",
    )

    result = {
        "status": "ok" if proc.returncode == 0 and summary_path.exists() else "failed",
        "case_dir": str(out_dir),
        "returncode": int(proc.returncode),
        "profile": case["profile"]["view_profile"],
        "frame_id": int(case["frame_id"]),
        "target_camera": str(case["target_camera"]),
        "source_count": int(len(case["meta"]["src_cameras"])),
        "error": "",
    }
    if result["status"] != "ok":
        result["error"] = (proc.stderr or proc.stdout or "").strip().splitlines()[-1] if (proc.stderr or proc.stdout) else ""
        return result
    result["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    return result


def profile_group(rows, profile_name):
    return [row for row in rows if row["view_profile"] == profile_name]


def count_decisions(rows):
    return {
        "depth_unproject": sum(1 for row in rows if row["decision"] == "depth_unproject"),
        "point_map": sum(1 for row in rows if row["decision"] == "point_map"),
        "tie": sum(1 for row in rows if row["decision"] == "tie"),
    }


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def write_summary_markdown(path, rows, failures):
    overall = count_decisions(rows)
    profiles = sorted({row["view_profile"] for row in rows})
    source_policy = rows[0]["source_policy"] if rows else "n/a"
    lines = [
        "# ZJU Geometry View Sweep",
        "",
        f"- source_policy: `{source_policy}`",
        f"- runs: `{len(rows)}`",
        f"- depth_unproject_wins: `{overall['depth_unproject']}`",
        f"- point_map_wins: `{overall['point_map']}`",
        f"- ties: `{overall['tie']}`",
        "",
        "## By Profile",
        "",
        "| Profile | Runs | Depth Wins | Point Wins | Ties | Avg Depth-Point MAE Gain | Avg Depth-Point Coverage Gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile_name in profiles:
        subset = profile_group(rows, profile_name)
        counts = count_decisions(subset)
        lines.append(
            "| {profile} | {runs} | {depth_wins} | {point_wins} | {ties} | {mae_gain:.4f} | {cov_gain:.4f} |".format(
                profile=profile_name,
                runs=len(subset),
                depth_wins=counts["depth_unproject"],
                point_wins=counts["point_map"],
                ties=counts["tie"],
                mae_gain=mean(row["point_mae"] - row["depth_mae"] for row in subset),
                cov_gain=mean(row["depth_cov"] - row["point_cov"] for row in subset),
            )
        )

    lines.extend(
        [
            "",
            "## Case Table",
            "",
            "| Profile | Frame | Target | Sources | Decision | Point MAE | Depth MAE | Point Cov | Depth Cov | Summary |",
            "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {view_profile} | {frame_id} | {target_camera} | {source_count} | {decision} | {point_mae:.4f} | {depth_mae:.4f} | {point_cov:.4f} | {depth_cov:.4f} | `{summary_md}` |".format(
                **row
            )
        )

    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(
                f"- `{failure['profile']} / frame_{failure['frame_id']:06d} / {failure['target_camera']}`: {failure['error'] or 'unknown error'}"
            )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- The sweep uses the original VGGT model and compares only the geometry source branch at render time.",
            f"- Sparse source views are derived with `{source_policy}`.",
            "- `depth + camera` is counted as the decision winner when it has lower MAE and no worse coverage.",
            "- A `tie` means one branch won MAE while the other won coverage.",
            "- This is the first large per-view sweep for the mentor's geometry-chain question.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_csv(path, rows):
    fieldnames = [
        "view_profile",
        "source_policy",
        "frame_id",
        "target_camera",
        "source_count",
        "decision",
        "mae_winner",
        "coverage_winner",
        "point_mae",
        "depth_mae",
        "point_cov",
        "depth_cov",
        "point_psnr",
        "depth_psnr",
        "summary_md",
        "case_dir",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    local_zju_root = Path(args.local_zju_root).resolve()
    profiles = [load_template_profile(path) for path in args.template_reports]
    seq_names = sorted({profile["seq_name"] for profile in profiles})
    if len(seq_names) != 1:
        raise ValueError(f"All template reports must belong to the same sequence. Got: {seq_names}")
    seq_dir = local_zju_root / seq_names[0]
    ring_order = discover_camera_ring_order(seq_dir)
    profiles = [attach_template_offsets(profile, ring_order) for profile in profiles]
    frame_ids = parse_frame_ids(args.frame_ids)
    target_cameras = resolve_target_cameras(seq_dir, args.target_cameras)
    cases = build_case_list(profiles, seq_dir, frame_ids, target_cameras, args.source_policy, ring_order)
    if args.limit_cases > 0:
        cases = cases[: args.limit_cases]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "local_zju_root": str(local_zju_root),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "frame_ids": frame_ids,
        "target_cameras": target_cameras,
        "source_policy": args.source_policy,
        "camera_ring_order": ring_order,
        "profiles": profiles,
        "case_count": len(cases),
    }
    (output_root / "sweep_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[zju-view-sweep] cases={len(cases)}", flush=True)
    results = [run_case(args, case) for case in cases]

    failures = [result for result in results if result["status"] == "failed"]
    rows = []
    for result in results:
        if result["status"] not in ("ok", "cached"):
            continue
        summary = result["summary"]
        rows.append(
            {
                "view_profile": summary["case"]["view_profile"],
                "source_policy": summary["case"].get("source_policy", args.source_policy),
                "frame_id": summary["case"]["frame_id"],
                "target_camera": summary["case"]["target_camera"],
                "source_count": summary["case"]["source_count"],
                "decision": summary["decision"]["decision"],
                "mae_winner": summary["decision"]["mae_winner"],
                "coverage_winner": summary["decision"]["coverage_winner"],
                "point_mae": summary["branches"]["point_map"]["metrics"]["mae"],
                "depth_mae": summary["branches"]["depth_unproject"]["metrics"]["mae"],
                "point_cov": summary["branches"]["point_map"]["render"]["coverage_ratio"],
                "depth_cov": summary["branches"]["depth_unproject"]["render"]["coverage_ratio"],
                "point_psnr": summary["branches"]["point_map"]["metrics"]["psnr"],
                "depth_psnr": summary["branches"]["depth_unproject"]["metrics"]["psnr"],
                "summary_md": str(Path(result["case_dir"]) / "summary.md"),
                "case_dir": str(result["case_dir"]),
            }
        )

    rows.sort(key=lambda row: (row["view_profile"], int(row["frame_id"]), row["target_camera"]))
    summary_json = output_root / "summary.json"
    summary_md = output_root / "summary.md"
    summary_csv = output_root / "summary.csv"
    summary_json.write_text(
        json.dumps({"rows": rows, "failures": failures, "manifest": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_markdown(summary_md, rows, failures)
    write_summary_csv(summary_csv, rows)
    print(summary_md, flush=True)


if __name__ == "__main__":
    main()
