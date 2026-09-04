"""Say whether the model scripts will run on the gpu of this machine, and if not, why.

Run it with the same interpreter you run the models with, from the repo root:

    .venv\\Scripts\\python gpu_check.py        (windows)
    .venv/bin/python gpu_check.py             (linux / mac)

It never changes anything. Paste the output into the chat if the answer is not
obvious from it.
"""

import os
import subprocess
import sys


def nvidia_smi(fields):
	exes = ["nvidia-smi"]
	if os.name == "nt":
		exes += [os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"),
		         r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"]
	for exe in exes:
		try:
			out = subprocess.run([exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
			                     capture_output=True, text=True, timeout=20)
		except Exception:
			continue
		if out.returncode == 0 and out.stdout.strip():
			return [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
	return None


def main():
	print("python  ", sys.version.split()[0], sys.executable)
	in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
	print("venv    ", "yes" if in_venv else "NO, this is a system or conda python; use the repo's .venv interpreter")
	print("cpu     ", os.cpu_count(), "logical cores")
	smi = nvidia_smi("name,driver_version,memory.total")
	if smi:
		print("driver  ", smi[0], "driver", smi[1], f"{smi[2]} MiB")
	else:
		print("driver  ", "nvidia-smi not found: no nvidia driver installed, or no nvidia gpu")

	try:
		import torch
	except ImportError as e:
		print("torch    NOT INSTALLED in this interpreter:", e)
		print("fix      python -m pip install \"torch==2.11.*\" --index-url https://download.pytorch.org/whl/cu128")
		return 1
	print("torch   ", torch.__version__, "cuda build", torch.version.cuda)

	if torch.cuda.is_available():
		prop = torch.cuda.get_device_properties(0)
		print("cuda     OK:", prop.name, f"{prop.total_memory / 2 ** 30:.1f} GB, capability {prop.major}.{prop.minor}")
		x = torch.randn(2048, 2048, device="cuda")
		torch.cuda.synchronize()
		print("matmul   ", "OK" if torch.isfinite(x @ x).all().item() else "FAILED")
		print()
		print("verdict  the models will run on the gpu with this interpreter")
		return 0

	print("cuda     NOT AVAILABLE to torch")
	print()
	if smi is None:
		print("verdict  no nvidia gpu or no driver on this machine; the models would run on the cpu")
	elif torch.version.cuda is None:
		print("verdict  cpu-only torch build (what `pip install torch` gives on windows). fix, in this venv:")
		print("           python -m pip uninstall -y torch")
		print("           python -m pip install \"torch==2.11.*\" --index-url https://download.pytorch.org/whl/cu128")
		print("         or run `uv sync` from the repo root, whose pyproject points at that index")
	else:
		print(f"verdict  torch is built for cuda {torch.version.cuda} but the driver ({smi[1]}) does not accept it.")
		print("         cuda 12 wheels need driver 528+ on windows / 525+ on linux: update from nvidia.com/drivers,")
		print("         then reboot. also check CUDA_VISIBLE_DEVICES is not set to hide the gpu.")
	return 1


if __name__ == "__main__":
	sys.exit(main())
