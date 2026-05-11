# V9 external backend install attempt summary

Date: 2026-05-07, Asia/Shanghai.
Workspace: `D:\vggt\vggt-main`.

Scope honored: no mainline training/report files were edited. This attempt only
wrote logs/README under `tools/v9_external_backend_install_attempts/` and touched
external checkouts under `external/`.

No synthetic pointcloud, teacher bundle, candidate bundle, predictions, or
strict-pass artifact was generated.

## Official sources checked

- MUSt3R: `https://github.com/naver/must3r`
- 2DGS: `https://github.com/hbb1/2d-gaussian-splatting`
- MASt3R-SLAM: `https://github.com/rmurai0610/MASt3R-SLAM`

## Host environment observed

- Default Python: `D:\anaconda\python.exe`, Python 3.13.5.
- Existing conda probes:
  - `D:\anaconda\envs\vit-torch\python.exe`, Python 3.10.19.
  - `D:\anaconda\envs\g3splat\python.exe`, Python 3.10.19.
- `vit-torch` has `torch`, `torchvision`, `numpy`, `cv2`; it lacks `open3d`,
  `viser`, `gradio`, `dust3r`, `mast3r`, `faiss`.

Full environment log:
`tools/v9_external_backend_install_attempts/environment_20260507_203537.log`.

## MUSt3R result

Status: partially cloned, not runnable yet.

Local source:

```powershell
git -C external\must3r remote -v
git -C external\must3r rev-parse HEAD
git -C external\must3r submodule status --recursive
```

Observed:

- Remote is `https://github.com/naver/must3r.git`.
- HEAD is `0c0d76b76b68993e380752d56dd3e04979cbbeef`.
- Required submodule `dust3r` is not initialized:
  `-bb9f9f54826b4077ed9fd075308eb2fe2d4666f9 dust3r`.

Clone/submodule command attempted:

```powershell
git -C external\must3r submodule update --init --recursive
git -C external\must3r -c http.version=HTTP/1.1 submodule update --init --recursive --depth 1
```

Failure:

```text
fatal: unable to access 'https://github.com/naver/dust3r/': Recv failure: Connection was reset
fatal: unable to access 'https://github.com/naver/dust3r/': Failed to connect to github.com port 443 after 21091 ms: Could not connect to server
```

Install/import/entrypoint checks:

```powershell
D:\anaconda\envs\vit-torch\python.exe - <stdin>
D:\anaconda\envs\vit-torch\python.exe external\must3r\demo.py --help
D:\anaconda\envs\vit-torch\python.exe external\must3r\slam.py --help
```

Observed:

- Top-level package import from source succeeds:
  `must3r package import ok D:\vggt\vggt-main\external\must3r\must3r\__init__.py`.
- `from must3r.model import load_model` fails because `dust3r` is missing:

```text
ImportError: dust3r is not initialized, could not find:
D:\vggt\vggt-main\external\must3r\dust3r\dust3r.
Did you forget to run 'git submodule update --init --recursive' ?
```

- `demo.py --help` fails before argparse due missing UI dependency:

```text
ModuleNotFoundError: No module named 'gradio'
```

- `slam.py --help` fails before argparse due missing SLAM dependency:

```text
ModuleNotFoundError: No module named 'open3d'
```

- Python 3.13 isolated dry-run of `external\must3r\requirements.txt` fails on
  Open3D wheel availability:

```text
ERROR: Could not find a version that satisfies the requirement open3d
ERROR: No matching distribution found for open3d
```

- `pip install --dry-run 'dust3r @ git+https://github.com/naver/dust3r.git@dust3r_setup#egg=dust3r'`
  fails at GitHub clone:

```text
fatal: unable to access 'https://github.com/naver/dust3r.git/':
Failed to connect to github.com port 443 after 21057 ms: Could not connect to server
```

Checkpoint check:

```powershell
curl.exe -I -L --connect-timeout 15 --max-time 45 https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_224_cvpr.pth
```

Observed:

```text
HTTP/1.1 200 OK
Content-Length: 1693926903
```

Minimum next command once GitHub and dependencies work:

```powershell
git -C external\must3r submodule update --init --recursive
D:\anaconda\envs\<py311-must3r>\python.exe -m pip install -r external\must3r\dust3r\requirements.txt
D:\anaconda\envs\<py311-must3r>\python.exe -m pip install -r external\must3r\requirements.txt
D:\anaconda\envs\<py311-must3r>\python.exe external\must3r\demo.py --help
```

## 2DGS result

Status: unavailable; source clone did not land locally.

Official clone command attempted:

```powershell
git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git external\2d-gaussian-splatting_official
git -c http.version=HTTP/1.1 clone --depth=1 --recurse-submodules --shallow-submodules https://github.com/hbb1/2d-gaussian-splatting.git external\2d-gaussian-splatting_official
```

Failure:

```text
fatal: unable to access 'https://github.com/hbb1/2d-gaussian-splatting.git/':
Failed to connect to github.com port 443 after 21058 ms: Could not connect to server
```

Official codeload archive probe also failed:

```powershell
curl.exe -I -L --connect-timeout 15 --max-time 45 https://github.com/hbb1/2d-gaussian-splatting/archive/refs/heads/main.zip
```

```text
curl: (28) Connection timed out after 15007 milliseconds
```

Because no complete checkout exists, install/import/entrypoint checks could not
be run. Based on the official README, the expected next checks after clone are:

```powershell
conda env create --file external\2d-gaussian-splatting_official\environment.yml
conda activate surfel_splatting
python external\2d-gaussian-splatting_official\train.py --help
python external\2d-gaussian-splatting_official\render.py --help
```

Expected dependency risk after clone: CUDA/PyTorch environment and recursive
submodules for the surfel rasterizer stack.

## MASt3R-SLAM result

Status: unavailable; source clone did not land locally.

Official clone command attempted:

```powershell
git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git external\MASt3R-SLAM
git -c http.version=HTTP/1.1 clone --depth=1 --recurse-submodules --shallow-submodules https://github.com/rmurai0610/MASt3R-SLAM.git external\MASt3R-SLAM
```

Failure:

```text
fatal: unable to access 'https://github.com/rmurai0610/MASt3R-SLAM.git/':
Recv failure: Connection was reset
fatal: unable to access 'https://github.com/rmurai0610/MASt3R-SLAM.git/':
Failed to connect to github.com port 443 after 21107 ms: Could not connect to server
```

Official codeload archive probe also failed:

```powershell
curl.exe -I -L --connect-timeout 15 --max-time 45 https://github.com/rmurai0610/MASt3R-SLAM/archive/refs/heads/main.zip
```

```text
curl: (56) Recv failure: Connection was reset
```

Checkpoint host check:

```powershell
curl.exe -I -L --connect-timeout 15 --max-time 45 https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
```

Observed:

```text
HTTP/1.1 200 OK
Content-Length: 2754910614
```

Because no complete checkout exists, install/import/entrypoint checks could not
be run. Based on the official README, the expected next checks after clone are:

```powershell
conda create -n mast3r-slam python=3.11
conda activate mast3r-slam
pip install -e external\MASt3R-SLAM\thirdparty\mast3r
pip install -e external\MASt3R-SLAM\thirdparty\in3d
pip install --no-build-isolation -e external\MASt3R-SLAM
python external\MASt3R-SLAM\main.py --help
```

Expected dependency risk after clone: PyTorch 2.5.1 with matching CUDA,
recursive third-party submodules, custom CUDA/C++ dependencies, and Windows/WSL
branch choice. The official README says WSL users should check out the
`windows` branch.

## Repro logs

- `tools/v9_external_backend_install_attempts/environment_20260507_203537.log`
- `tools/v9_external_backend_install_attempts/clone_attempts_20260507_203825.log`
- `tools/v9_external_backend_install_attempts/network_retry_20260507_204127.log`
- `tools/v9_external_backend_install_attempts/must3r_install_import_entrypoint_20260507_204738.log`
- `tools/v9_external_backend_install_attempts/must3r_existing_env_stdin_20260507_210230.log`

Some earlier bounded-run logs in this folder record quoting/process-wrapper
mistakes during probing and are not used for the final feasibility conclusion.

## Final availability

- Usable now: none as a runnable backend.
- Closest to usable: MUSt3R, because the main repo is present and checkpoints
  are reachable. It is blocked by GitHub access to `dust3r` plus missing local
  deps (`gradio`, `open3d`, `viser`, `faiss`, and likely Python 3.11/CUDA wheel
  alignment).
- Not usable now: 2DGS and MASt3R-SLAM, because official source checkout from
  GitHub did not complete on this machine during the attempt.
