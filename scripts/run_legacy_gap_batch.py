from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_REPO_ROOT = Path(r"G:\项目备份\vggt_小感度不起作用\vggt")
DEFAULT_SEQ_NAME = "CoreView_390"
DEFAULT_FRAME_ID = 1080
MODAL_OUTPUT_VOLUME = "vggt-out"
CURRENT_COMPARE_PS1 = REPO_ROOT / "scripts" / "run_modal_zju_geometry_branch_compare.ps1"
CURRENT_MODAL_APP_NAME = "vggt-zju-geometry-branch-compare"
LEGACY_MODAL_APP_NAME = "vggt-zju-runner"
TRANSIENT_PATTERNS = (
    "Connection lost",
    "WinError 10053",
    "WinError 10054",
    "SSL shutdown timed out",
    "Deadline exceeded",
    "heartbeat failed",
    "modal.exception.ConnectionError",
    "timed out waiting for final app logs",
    "Could not connect to the Modal server",
    "Cannot connect to host",
)


@dataclass(frozen=True)
class CaseSpec:
    profile: str
    target_camera: str
    camera_bucket: str

    @property
    def case_id(self) -> str:
        return f"{DEFAULT_SEQ_NAME}_frame_{DEFAULT_FRAME_ID:06d}_{self.target_camera}_{self.profile}"


CASE_MATRIX: tuple[CaseSpec, ...] = (
    CaseSpec("6src_hist", "Camera_B5", "control"),
    CaseSpec("6src_hist", "Camera_B17", "depth_win"),
    CaseSpec("6src_hist", "Camera_B4", "point_hard"),
    CaseSpec("6src_hist", "Camera_B15", "point_hard"),
    CaseSpec("12src_nested", "Camera_B5", "control"),
    CaseSpec("12src_nested", "Camera_B3", "depth_win"),
    CaseSpec("12src_nested", "Camera_B8", "point_hard"),
    CaseSpec("12src_nested", "Camera_B19", "point_hard"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the mentor-aligned legacy-gap overnight batch: "
            "reuse legacy B5 controls, backfill missing legacy hard targets, "
            "run current branch compare on matching cases, and summarize the batch."
        )
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--powershell", default="powershell")
    parser.add_argument("--modal", default="")
    parser.add_argument("--old_repo_root", type=Path, default=OLD_REPO_ROOT)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "output" / f"legacy_gap_batch_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--seq_name", default=DEFAULT_SEQ_NAME)
    parser.add_argument("--frame_id", type=int, default=DEFAULT_FRAME_ID)
    parser.add_argument("--zju_subdir", default="zju_mocap")
    parser.add_argument("--legacy_modal_gpu", default="A100-40GB")
    parser.add_argument("--legacy_modal_cpu", type=float, default=8.0)
    parser.add_argument("--legacy_modal_memory_mb", type=int, default=49152)
    parser.add_argument("--current_modal_gpu", default="A100-40GB")
    parser.add_argument("--current_modal_cpu", type=float, default=8.0)
    parser.add_argument("--current_modal_memory_mb", type=int, default=49152)
    parser.add_argument("--modal_timeout_sec", type=int, default=8 * 60 * 60)
    parser.add_argument("--conf_percentile", type=float, default=25.0)
    parser.add_argument("--export_max_points", type=int, default=100000)
    parser.add_argument("--render_max_points", type=int, default=500000)
    parser.add_argument("--z_tolerance", type=float, default=0.02)
    parser.add_argument("--min_conf", type=float, default=1e-6)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep_sec", type=int, default=15)
    parser.add_argument("--skip_modal_stop", action="store_true")
    parser.add_argument("--skip_legacy_modal", action="store_true")
    parser.add_argument("--skip_current_modal", action="store_true")
    parser.add_argument("--force_recopy_b5", action="store_true")
    parser.add_argument("--dry_run_only", action="store_true")
    return parser.parse_args()


def resolve_modal_exe(raw: str) -> str:
    if raw:
        return str(Path(raw))
    candidates = [
        REPO_ROOT / ".venv5080" / "Scripts" / "modal.exe",
        REPO_ROOT / ".venv" / "Scripts" / "modal.exe",
        REPO_ROOT / "venv" / "Scripts" / "modal.exe",
        Path(r"D:\anaconda\Scripts\modal.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return "modal"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    retries: int = 1,
    retry_sleep_sec: int = 0,
) -> tuple[int, str]:
    attempt = 0
    last_code = -1
    last_text = ""
    while attempt < max(1, retries):
        attempt += 1
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            shell=False,
        )
        last_code = int(proc.returncode)
        last_text = proc.stdout.decode("utf-8", errors="replace")
        if last_code == 0:
            return last_code, last_text
        is_transient = any(pattern in last_text for pattern in TRANSIENT_PATTERNS)
        if is_transient and attempt < retries:
            time.sleep(max(0, retry_sleep_sec))
            continue
        break
    if check and last_code != 0:
        raise RuntimeError(
            "Command failed.\n"
            f"cwd={cwd}\n"
            f"cmd={' '.join(cmd)}\n"
            f"exit_code={last_code}\n"
            f"output:\n{last_text}"
        )
    return last_code, last_text


def modal_volume_get(modal_exe: str, remote_path: str, local_dir: Path) -> None:
    ensure_dir(local_dir)
    proc = subprocess.run(
        [modal_exe, "volume", "get", "--force", MODAL_OUTPUT_VOLUME, remote_path, str(local_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=False,
        shell=False,
    )
    remote_leaf = Path(remote_path.rstrip("/")).name
    downloaded_leaf = local_dir / remote_leaf
    if proc.returncode == 0:
        return
    if downloaded_leaf.exists():
        return
    if proc.returncode != 0:
        raise RuntimeError(
            "modal volume get failed.\n"
            f"remote_path={remote_path}\n"
            f"local_dir={local_dir}\n"
            f"exit_code={proc.returncode}\n"
            f"stdout={proc.stdout.decode('utf-8', errors='replace')}"
        )


def modal_volume_ls_json(modal_exe: str, remote_path: str) -> list[dict]:
    proc = subprocess.run(
        [modal_exe, "volume", "ls", "--json", MODAL_OUTPUT_VOLUME, remote_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=False,
        shell=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"modal volume ls failed for {remote_path}")
    text = proc.stdout.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    return list(json.loads(text))


def modal_volume_put(modal_exe: str, local_path: Path, remote_subpath: str) -> None:
    proc = subprocess.run(
        [modal_exe, "volume", "put", MODAL_OUTPUT_VOLUME, str(local_path), remote_subpath, "--force"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=False,
        shell=False,
    )
    if proc.returncode == 0:
        return
    remote_parent = "/" + str(Path(remote_subpath).parent).replace("\\", "/").strip("/")
    remote_name = Path(remote_subpath).name
    try:
        items = modal_volume_ls_json(modal_exe, remote_parent)
    except Exception as exc:
        raise RuntimeError(
            "modal volume put failed and parent recheck also failed.\n"
            f"local_path={local_path}\nremote_subpath={remote_subpath}\nerror={exc}"
        ) from exc
    for item in items:
        if str(item.get("Type", "")).lower() != "file":
            continue
        filename = str(item.get("Filename", ""))
        if Path(filename).name == remote_name:
            return
    raise RuntimeError(
        "modal volume put failed.\n"
        f"local_path={local_path}\n"
        f"remote_subpath={remote_subpath}\n"
        f"stdout={proc.stdout.decode('utf-8', errors='replace')}"
    )


def remote_batch_completed(modal_exe: str, remote_root: str) -> bool:
    try:
        items = modal_volume_ls_json(modal_exe, remote_root)
    except Exception:
        return False
    wanted = {"batch_summary.json", "batch_summary.md", "batch_status.json"}
    names = {Path(str(item.get("Filename", ""))).name for item in items}
    return wanted.issubset(names)


def modal_app_list(modal_exe: str) -> list[dict]:
    _, text = run_command([modal_exe, "app", "list", "--json"], check=True)
    data = json.loads(text or "[]")
    return list(data)


def stop_modal_apps(modal_exe: str, descriptions: Iterable[str]) -> list[dict]:
    wanted = {item.strip() for item in descriptions if item.strip()}
    stopped: list[dict] = []
    for app in modal_app_list(modal_exe):
        desc = str(app.get("Description") or app.get("description") or app.get("AppName") or app.get("app_name") or "")
        app_id = str(app.get("App ID") or app.get("app_id") or app.get("id") or "")
        state = str(app.get("State") or app.get("state") or "")
        if not app_id or not desc:
            continue
        if desc not in wanted:
            continue
        if state.lower() == "stopped":
            continue
        run_command([modal_exe, "app", "stop", app_id], check=True)
        stopped.append({"app_id": app_id, "description": desc, "state": state})
    return stopped


def find_latest_report_json(root: Path) -> Path:
    matches = sorted(root.rglob("report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No report.json under {root}")
    return matches[0]


def find_current_summary_json(root: Path, case_id: str) -> Path:
    path = root / "current_cases" / case_id / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing current summary.json for {case_id}: {path}")
    return path


def infer_local_b5_legacy_report(old_repo_root: Path, profile: str) -> Path:
    frame_dir = old_repo_root / "infer_out" / "vggt_raw_viewcount" / profile / DEFAULT_SEQ_NAME / f"frame_{DEFAULT_FRAME_ID:06d}_Camera_B5"
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Missing legacy B5 frame dir: {frame_dir}")
    return find_latest_report_json(frame_dir)


def copy_existing_legacy_run(report_json: Path, dst_case_root: Path, *, overwrite: bool) -> Path:
    src_run_dir = report_json.parent
    src_frame_dir = src_run_dir.parent
    dst_frame_dir = dst_case_root / src_frame_dir.parent.name / src_frame_dir.name
    dst_run_dir = dst_frame_dir / src_run_dir.name
    if dst_run_dir.exists() and overwrite:
        shutil.rmtree(dst_run_dir)
    if not dst_run_dir.exists():
        ensure_dir(dst_frame_dir)
        shutil.copytree(src_run_dir, dst_run_dir, dirs_exist_ok=True)
    dst_report = dst_run_dir / "report.json"
    if not dst_report.is_file():
        raise FileNotFoundError(f"Copied legacy report missing: {dst_report}")
    return dst_report


def parse_last_tagged_line(text: str, prefix: str) -> str:
    found = ""
    for line in text.splitlines():
        if line.startswith(prefix):
            found = line[len(prefix) :].strip()
    return found


def infer_remote_legacy_case_root(batch_tag: str, case: CaseSpec) -> str:
    return f"/legacy_gap_batch/{batch_tag}/legacy_cases/{case.case_id}"


def infer_remote_current_root(batch_tag: str) -> str:
    return f"/legacy_gap_batch/{batch_tag}/current_cases"


def infer_remote_dry_run_root(batch_tag: str) -> str:
    return f"/legacy_gap_batch/{batch_tag}/dry_run_current"


def run_legacy_case_modal(
    *,
    args: argparse.Namespace,
    modal_exe: str,
    batch_tag: str,
    case: CaseSpec,
    legacy_cases_root: Path,
) -> Path:
    remote_case_root = infer_remote_legacy_case_root(batch_tag, case)
    case_local_root = legacy_cases_root / case.case_id
    existing = sorted(case_local_root.rglob("report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if existing:
        return existing[0]

    env = os.environ.copy()
    env["VGGT_CODE_DIR"] = str(args.old_repo_root)
    env["VGGT_MODE"] = "precompute"
    env["VGGT_PRECOMPUTE_SCRIPT"] = "scripts/orig_vggt_viewcount/render_raw_compare.py"
    env["VGGT_PRECOMPUTE_CKPT"] = "model.pt"
    env["VGGT_ZJU_ROOT"] = f"/mnt/data/{args.zju_subdir}"
    env["VGGT_SEQ_NAMES"] = args.seq_name
    env["VGGT_GEOM_SUBDIR"] = f"/mnt/out{remote_case_root}"
    env["VGGT_POINTMAP_SOURCE"] = "point_head"
    env["VGGT_POINT_HEAD_FRAME"] = "world"
    env["VGGT_GPU_SPEC_PRECOMPUTE"] = str(args.legacy_modal_gpu)
    env["VGGT_PRECOMPUTE_CPU"] = str(args.legacy_modal_cpu)
    env["VGGT_PRECOMPUTE_MEMORY_MB"] = str(args.legacy_modal_memory_mb)
    env["VGGT_TIMEOUT_SEC"] = str(args.modal_timeout_sec)
    extra = [
        "--seq_name",
        args.seq_name,
        "--frame_id",
        str(args.frame_id),
        "--tgt_camera",
        case.target_camera,
        "--view_profile",
        case.profile,
    ]
    env["VGGT_PRECOMPUTE_ARGS_EXTRA"] = " ".join(extra)
    cmd = [modal_exe, "run", "-q", "modal_run_train.py"]
    _, output = run_command(
        cmd,
        cwd=args.old_repo_root,
        env=env,
        check=True,
        retries=args.max_retries,
        retry_sleep_sec=args.retry_sleep_sec,
    )
    remote_run_dir = parse_last_tagged_line(output, "RUN_DIR:")
    if not remote_run_dir:
        remote_frame_root = f"{remote_case_root}/{args.seq_name}/frame_{args.frame_id:06d}_{case.target_camera}"
        items = modal_volume_ls_json(modal_exe, remote_frame_root)
        run_dirs = [str(item.get("Filename", "")) for item in items if str(item.get("Type", "")).lower() == "dir"]
        if not run_dirs:
            raise RuntimeError(
                f"Could not infer remote legacy run dir for {case.case_id}.\nOutput:\n{output}"
            )
    modal_volume_get(modal_exe, remote_case_root, legacy_cases_root)
    return find_latest_report_json(case_local_root)


def run_current_compare_batch(
    *,
    args: argparse.Namespace,
    modal_exe: str,
    output_subdir: str,
    report_jsons: list[Path],
    exp_name: str,
) -> None:
    case_payloads = []
    for report_json in report_jsons:
        report_text = report_json.read_text(encoding="utf-8-sig")
        report = json.loads(report_text)
        meta = report["meta"]
        case_id = "{seq}_frame_{frame:06d}_{tgt}_{profile}".format(
            seq=str(meta["seq_name"]),
            frame=int(meta["frame_id"]),
            tgt=str(meta["tgt_camera"]),
            profile=str(meta["view_profile"]),
        )
        case_payloads.append(
            {
                "case_id": case_id,
                "report_json_b64": base64.b64encode(report_text.encode("utf-8")).decode("ascii"),
            }
        )

    cfg = {
        "cases": case_payloads,
        "zju_subdir": args.zju_subdir,
        "checkpoint_subpath": "checkpoints/model.pt",
        "exp_name": exp_name,
        "output_subdir": output_subdir,
        "device": "cuda",
        "dtype": "bfloat16",
        "conf_percentile": float(args.conf_percentile),
        "export_max_points": int(args.export_max_points),
        "render_max_points": int(args.render_max_points),
        "z_tolerance": float(args.z_tolerance),
        "min_conf": float(args.min_conf),
        "primary_branch": "depth_unproject",
        "skip_save_predictions": True,
    }
    cfg_dir = ensure_dir(REPO_ROOT / "output" / "modal_geom_compare_cfg")
    cfg_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{exp_name}.json"
    cfg_local_path = cfg_dir / cfg_name
    cfg_local_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    cfg_subpath = f"geometry_compare_cfg/{cfg_name}"

    env = os.environ.copy()
    env["VGGT_ZJU_GEOM_COMPARE_MODAL_APP_NAME"] = CURRENT_MODAL_APP_NAME
    env["VGGT_ZJU_GEOM_COMPARE_DATA_VOLUME"] = "vggt-zju-data"
    env["VGGT_ZJU_GEOM_COMPARE_OUTPUT_VOLUME"] = MODAL_OUTPUT_VOLUME
    env["VGGT_ZJU_GEOM_COMPARE_GPU"] = str(args.current_modal_gpu)
    env["VGGT_ZJU_GEOM_COMPARE_CPU"] = str(args.current_modal_cpu)
    env["VGGT_ZJU_GEOM_COMPARE_MEMORY_MB"] = str(args.current_modal_memory_mb)
    env["VGGT_ZJU_GEOM_COMPARE_TIMEOUT_SEC"] = str(args.modal_timeout_sec)

    stop_modal_apps(modal_exe, [CURRENT_MODAL_APP_NAME])

    modal_volume_put(modal_exe, cfg_local_path, cfg_subpath)
    cmd = [
        modal_exe,
        "run",
        "-q",
        "modal_zju_geometry_branch_compare.py::run_remote_zju_geometry_branch_compare_batch_from_cfg_path",
        "--cfg-subpath",
        cfg_subpath,
    ]
    code, output = run_command(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        retries=args.max_retries,
        retry_sleep_sec=args.retry_sleep_sec,
    )
    if code == 0:
        return
    remote_root = "/" + output_subdir.strip("/").replace("\\", "/")
    if remote_batch_completed(modal_exe, remote_root):
        return
    raise RuntimeError(
        "Current compare batch failed.\n"
        f"output_subdir={output_subdir}\n"
        f"exit_code={code}\n"
        f"output:\n{output}"
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def legacy_case_key(report: dict) -> dict:
    meta = report["meta"]
    return {
        "seq_name": str(meta["seq_name"]),
        "frame_id": int(meta["frame_id"]),
        "view_profile": str(meta["view_profile"]),
        "source_cameras": [str(item) for item in meta["src_cameras"]],
        "target_camera": str(meta["tgt_camera"]),
    }


def current_case_key(summary: dict) -> dict:
    case = summary["case"]
    return {
        "seq_name": str(case["seq_name"]),
        "frame_id": int(case["frame_id"]),
        "view_profile": str(case["view_profile"]),
        "source_cameras": [str(item) for item in case["source_cameras"]],
        "target_camera": str(case["target_camera"]),
    }


def validate_case_alignment(legacy_report: dict, current_summary: dict) -> dict:
    lhs = legacy_case_key(legacy_report)
    rhs = current_case_key(current_summary)
    if lhs != rhs:
        raise RuntimeError(
            "Legacy/current case mismatch.\n"
            f"legacy={json.dumps(lhs, ensure_ascii=False)}\n"
            f"current={json.dumps(rhs, ensure_ascii=False)}"
        )
    return lhs


def resolve_legacy_images(report_path: Path, report: dict) -> dict[str, Path]:
    paths = report.get("paths", {})
    report_dir = report_path.parent

    def choose(*keys: str) -> Path:
        for key in keys:
            value = str(paths.get(key, "")).strip()
            if not value:
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = (report_dir / value).resolve()
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Could not resolve any of {keys} for {report_path}")

    return {
        "legacy_pred": choose("pred_native"),
        "legacy_target": choose("tgt_native"),
        "legacy_triplet": choose("cat_weight_pred_tgt_native", "pred_tgt_pair"),
    }


def resolve_current_images(summary_path: Path, summary: dict) -> dict[str, Path]:
    files = summary.get("files", {})
    summary_dir = summary_path.parent

    def resolve(key: str) -> Path:
        value = str(files.get(key, "")).strip()
        if not value:
            raise FileNotFoundError(f"Missing current file key {key} in {summary_path}")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (summary_dir / value).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing current asset {candidate}")
        return candidate

    return {
        "target": resolve("target_png"),
        "point": resolve("point_map_render_png"),
        "depth": resolve("depth_unproject_render_png"),
        "compare": resolve("comparison_png"),
    }


def fit_images_row(paths: list[Path], panel_size: tuple[int, int]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image = image.resize(panel_size, Image.Resampling.BILINEAR)
        images.append(image)
    return images


def save_case_mosaic(case_root: Path, row: dict, legacy_report: dict, current_summary: dict) -> Path:
    ensure_dir(case_root)
    legacy_assets = resolve_legacy_images(Path(row["legacy_report_json"]), legacy_report)
    current_assets = resolve_current_images(Path(row["current_summary_json"]), current_summary)
    panel_size = (420, 420)
    panels = fit_images_row(
        [
            legacy_assets["legacy_pred"],
            current_assets["point"],
            current_assets["depth"],
            current_assets["target"],
        ],
        panel_size,
    )
    labels = [
        f"Legacy Native\nMAE {row['legacy_native_mae']:.4f}",
        f"Current Point\nMAE {row['current_point_mae']:.4f}\nCov {row['current_point_cov']:.4f}",
        f"Current Depth\nMAE {row['current_depth_mae']:.4f}\nCov {row['current_depth_cov']:.4f}",
        "Target",
    ]
    width = panel_size[0] * len(panels)
    header_h = 110
    label_h = 68
    height = header_h + panel_size[1] + label_h
    canvas = Image.new("RGB", (width, height), color=(14, 16, 18))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    header = (
        f"{row['case_id']} | bucket={row['camera_bucket']} | decision={row['branch_decision']} | "
        f"depth_vs_point_gain={row['depth_vs_point_gain']:.4f} | legacy_gap_depth={row['legacy_gap_depth']:.4f}"
    )
    draw.text((16, 16), header, fill=(240, 240, 240), font=font)
    draw.text(
        (16, 40),
        (
            f"legacy psnr/ssim={row['legacy_native_psnr']:.4f}/{row['legacy_native_ssim']:.4f} | "
            f"point gap={row['legacy_gap_point']:.4f} | depth gap={row['legacy_gap_depth']:.4f}"
        ),
        fill=(185, 185, 185),
        font=font,
    )
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * panel_size[0]
        canvas.paste(panel, (x, header_h))
        draw.rectangle((x, header_h, x + panel_size[0] - 1, header_h + panel_size[1] - 1), outline=(52, 56, 60))
        draw.multiline_text((x + 12, header_h + panel_size[1] + 10), label, fill=(235, 235, 235), font=font, spacing=2)
    out_path = case_root / "legacy_vs_current_mosaic.png"
    canvas.save(out_path)
    return out_path


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict]) -> dict:
    by_bucket: dict[str, list[dict]] = {}
    by_decision: dict[str, int] = {}
    for row in rows:
        by_bucket.setdefault(str(row["camera_bucket"]), []).append(row)
        by_decision[str(row["branch_decision"])] = by_decision.get(str(row["branch_decision"]), 0) + 1
    depth_better = sum(1 for row in rows if row["depth_vs_point_gain"] > 0.0)
    return {
        "cases": len(rows),
        "depth_better_than_point": depth_better,
        "decision_counts": by_decision,
        "bucket_counts": {bucket: len(items) for bucket, items in by_bucket.items()},
        "avg_legacy_gap_depth": (
            sum(float(row["legacy_gap_depth"]) for row in rows) / len(rows) if rows else None
        ),
        "avg_legacy_gap_point": (
            sum(float(row["legacy_gap_point"]) for row in rows) / len(rows) if rows else None
        ),
    }


def write_summary_md(path: Path, rows: list[dict], batch_meta: dict) -> None:
    ensure_dir(path.parent)
    lines = [
        "# Legacy Gap Batch Summary",
        "",
        f"- batch_tag: `{batch_meta['batch_tag']}`",
        f"- cases: `{batch_meta['aggregate']['cases']}`",
        f"- depth_better_than_point: `{batch_meta['aggregate']['depth_better_than_point']}`",
        f"- decision_counts: `{json.dumps(batch_meta['aggregate']['decision_counts'], ensure_ascii=False)}`",
        f"- bucket_counts: `{json.dumps(batch_meta['aggregate']['bucket_counts'], ensure_ascii=False)}`",
        f"- avg_legacy_gap_depth: `{batch_meta['aggregate']['avg_legacy_gap_depth']:.6f}`" if batch_meta["aggregate"]["avg_legacy_gap_depth"] is not None else "- avg_legacy_gap_depth: `n/a`",
        f"- avg_legacy_gap_point: `{batch_meta['aggregate']['avg_legacy_gap_point']:.6f}`" if batch_meta["aggregate"]["avg_legacy_gap_point"] is not None else "- avg_legacy_gap_point: `n/a`",
        "",
    ]
    for bucket in ("control", "depth_win", "point_hard"):
        bucket_rows = [row for row in rows if row["camera_bucket"] == bucket]
        if not bucket_rows:
            continue
        lines.extend(
            [
                f"## {bucket}",
                "",
                "| Case | Legacy MAE | Point MAE | Depth MAE | Depth Gain | Gap Depth | Gap Point | Decision | Legacy Report | Current Summary | Mosaic |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        for row in bucket_rows:
            lines.append(
                "| {case_id} | {legacy_native_mae:.4f} | {current_point_mae:.4f} | {current_depth_mae:.4f} | "
                "{depth_vs_point_gain:.4f} | {legacy_gap_depth:.4f} | {legacy_gap_point:.4f} | {branch_decision} | "
                "`{legacy_report_json}` | `{current_summary_json}` | `{mosaic_png}` |".format(**row)
            )
        lines.append("")
    lines.extend(
        [
            "## Readout",
            "",
            "- This batch keeps the mentor-approved scope: legacy native is the formal baseline; current side only compares `point_map` vs `depth_unproject`.",
            "- No new gate tuning, no threshold/pow follow-up, and no old ghost-stack training logic is reintroduced here.",
            "- The first acceptance question is whether `depth_unproject` still beats current `point_map` on matched hard/control cases.",
            "- The second acceptance question is the size and pattern of the remaining gap to legacy native.",
            "- The third acceptance question is whether the gap concentrates on hard cameras instead of controls.",
            "- The fourth acceptance question is whether the per-case pattern looks more like a branch issue, a legacy-render gap, or source-policy sensitivity.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest_rows(
    *,
    cases: list[CaseSpec],
    batch_root: Path,
    legacy_cases_root: Path,
) -> list[dict]:
    rows: list[dict] = []
    for case in cases:
        legacy_report_json = find_latest_report_json(legacy_cases_root / case.case_id)
        current_summary_json = find_current_summary_json(batch_root, case.case_id)
        legacy_report = load_json(legacy_report_json)
        current_summary = load_json(current_summary_json)
        validate_case_alignment(legacy_report, current_summary)

        legacy_native = legacy_report["metrics"]["native"]
        point_metrics = current_summary["branches"]["point_map"]["metrics"]
        depth_metrics = current_summary["branches"]["depth_unproject"]["metrics"]
        point_cov = current_summary["branches"]["point_map"]["render"]["coverage_ratio"]
        depth_cov = current_summary["branches"]["depth_unproject"]["render"]["coverage_ratio"]

        row = {
            "profile": case.profile,
            "frame_id": DEFAULT_FRAME_ID,
            "target_camera": case.target_camera,
            "case_id": case.case_id,
            "legacy_report_json": str(legacy_report_json),
            "current_summary_json": str(current_summary_json),
            "legacy_native_mae": float(legacy_native["mae"]),
            "legacy_native_psnr": float(legacy_native["psnr"]),
            "legacy_native_ssim": float(legacy_native["ssim"]),
            "current_point_mae": float(point_metrics["mae"]),
            "current_point_cov": float(point_cov),
            "current_depth_mae": float(depth_metrics["mae"]),
            "current_depth_cov": float(depth_cov),
            "branch_decision": str(current_summary["decision"]["decision"]),
            "depth_vs_point_gain": float(point_metrics["mae"] - depth_metrics["mae"]),
            "legacy_gap_depth": float(depth_metrics["mae"] - legacy_native["mae"]),
            "legacy_gap_point": float(point_metrics["mae"] - legacy_native["mae"]),
            "camera_bucket": case.camera_bucket,
            "primary_branch": str(current_summary["primary"]["selected_branch"]),
            "current_target_compare_png": str((current_summary_json.parent / current_summary["files"]["comparison_png"]).resolve()),
        }
        case_root = batch_root / "cases" / case.case_id
        row["mosaic_png"] = str(save_case_mosaic(case_root, row, legacy_report, current_summary))
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    modal_exe = resolve_modal_exe(args.modal)
    batch_root = ensure_dir(args.output_root.resolve())
    legacy_cases_root = ensure_dir(batch_root / "legacy_cases")
    dry_run_root = batch_root / "dry_run_current"
    current_cases_root = batch_root / "current_cases"
    batch_tag = batch_root.name

    status_path = batch_root / "batch_status.json"
    status = {
        "batch_root": str(batch_root),
        "batch_tag": batch_tag,
        "seq_name": args.seq_name,
        "frame_id": args.frame_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "state": "running",
        "steps": [],
    }
    write_json(status_path, status)

    if not args.skip_modal_stop:
        stopped = stop_modal_apps(modal_exe, [LEGACY_MODAL_APP_NAME, CURRENT_MODAL_APP_NAME])
        status["steps"].append({"step": "stop_existing_modal_apps", "stopped": stopped})
        write_json(status_path, status)

    cases = list(CASE_MATRIX)
    manifest_payload = {
        "batch_tag": batch_tag,
        "seq_name": args.seq_name,
        "frame_id": args.frame_id,
        "cases": [asdict(case) | {"case_id": case.case_id} for case in cases],
    }
    write_json(batch_root / "case_manifest.json", manifest_payload)

    legacy_reports: dict[str, Path] = {}
    for case in cases:
        case_root = legacy_cases_root / case.case_id
        if case.target_camera == "Camera_B5":
            source_report = infer_local_b5_legacy_report(args.old_repo_root, case.profile)
            report_path = copy_existing_legacy_run(source_report, case_root, overwrite=args.force_recopy_b5)
            legacy_reports[case.case_id] = report_path
            status["steps"].append(
                {
                    "step": "reuse_legacy_b5",
                    "case_id": case.case_id,
                    "source_report": str(source_report),
                    "copied_report": str(report_path),
                }
            )
            write_json(status_path, status)
            continue
        if args.skip_legacy_modal:
            raise RuntimeError(f"Missing legacy report for {case.case_id} but --skip_legacy_modal was set.")
        print(f"[legacy-gap] run legacy modal case {case.case_id}", flush=True)
        report_path = run_legacy_case_modal(
            args=args,
            modal_exe=modal_exe,
            batch_tag=batch_tag,
            case=case,
            legacy_cases_root=legacy_cases_root,
        )
        legacy_reports[case.case_id] = report_path
        status["steps"].append({"step": "legacy_case_completed", "case_id": case.case_id, "report_json": str(report_path)})
        write_json(status_path, status)

    dry_run_case = next(case for case in cases if case.profile == "12src_nested" and case.target_camera == "Camera_B5")
    dry_run_report = legacy_reports[dry_run_case.case_id]
    if not args.skip_current_modal:
        print(f"[legacy-gap] run current dry-run {dry_run_case.case_id}", flush=True)
        run_current_compare_batch(
            args=args,
            modal_exe=modal_exe,
            output_subdir=infer_remote_dry_run_root(batch_tag).lstrip("/"),
            report_jsons=[dry_run_report],
            exp_name=f"legacy_gap_dry_run_{batch_tag}",
        )
        modal_volume_get(modal_exe, infer_remote_dry_run_root(batch_tag), batch_root)
        dry_summary_path = dry_run_root / dry_run_case.case_id / "summary.json"
        dry_summary = load_json(dry_summary_path)
        dry_legacy = load_json(dry_run_report)
        dry_key = validate_case_alignment(dry_legacy, dry_summary)
        status["steps"].append({"step": "dry_run_validated", "case_id": dry_run_case.case_id, "case_key": dry_key})
        write_json(status_path, status)

    if args.dry_run_only:
        status["state"] = "completed"
        status["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_json(status_path, status)
        print(f"[legacy-gap] dry run completed: {dry_run_root}", flush=True)
        return

    if not args.skip_current_modal:
        print(f"[legacy-gap] run current full batch ({len(cases)} cases)", flush=True)
        ordered_reports = [legacy_reports[case.case_id] for case in cases]
        run_current_compare_batch(
            args=args,
            modal_exe=modal_exe,
            output_subdir=infer_remote_current_root(batch_tag).lstrip("/"),
            report_jsons=ordered_reports,
            exp_name=f"legacy_gap_full_{batch_tag}",
        )
        modal_volume_get(modal_exe, infer_remote_current_root(batch_tag), batch_root)
        status["steps"].append(
            {
                "step": "current_batch_downloaded",
                "remote_root": infer_remote_current_root(batch_tag),
                "local_root": str(current_cases_root),
            }
        )
        write_json(status_path, status)
    else:
        for case in cases:
            if not find_current_summary_json(batch_root, case.case_id).is_file():
                raise RuntimeError(f"Missing current summary for {case.case_id} and --skip_current_modal was set.")

    rows = build_manifest_rows(cases=cases, batch_root=batch_root, legacy_cases_root=legacy_cases_root)
    rows.sort(key=lambda item: (item["profile"], item["target_camera"]))
    aggregate = summarize_rows(rows)
    summary_json = {
        "batch_tag": batch_tag,
        "batch_root": str(batch_root),
        "aggregate": aggregate,
        "rows": rows,
    }
    write_json(batch_root / "summary.json", summary_json)
    write_summary_csv(batch_root / "summary.csv", rows)
    write_summary_md(batch_root / "summary.md", rows, summary_json)

    if not args.skip_modal_stop:
        stopped = stop_modal_apps(modal_exe, [LEGACY_MODAL_APP_NAME, CURRENT_MODAL_APP_NAME])
        status["steps"].append({"step": "stop_modal_apps_final", "stopped": stopped})
        status["modal_apps_remaining"] = modal_app_list(modal_exe)
    else:
        status["modal_apps_remaining"] = modal_app_list(modal_exe)
    status["state"] = "completed"
    status["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    status["summary_json"] = str(batch_root / "summary.json")
    status["summary_md"] = str(batch_root / "summary.md")
    write_json(status_path, status)
    print(f"[legacy-gap] completed batch: {batch_root}", flush=True)


if __name__ == "__main__":
    main()
