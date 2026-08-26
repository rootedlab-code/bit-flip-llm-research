# %% [markdown]
# # Probe: can this Kaggle GPU be used at all?
#
# The image's torch is built for `sm_70` and above, while the P100 Kaggle assigns is
# `sm_60`. That is a property of the build, not of the card. PyTorch dropped Pascal in
# the CUDA 12.6+ wheels; the older CUDA 12.4 wheels still carry `sm_60`.
#
# This notebook establishes, rather than assumes, whether installing one of those makes
# the card usable — and whether the rest of the stack survives the downgrade.

# %%
import subprocess
import sys

print(
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout
)

import torch  # noqa: E402

print(f"preinstalled torch : {torch.__version__}")
print(f"arch list          : {torch.cuda.get_arch_list()}")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"device capability  : sm_{major}{minor}")
    print(f"usable as shipped  : {f'sm_{major}{minor}' in torch.cuda.get_arch_list()}")

# %% [markdown]
# ## Install a build that still carries Pascal
#
# Done in a subprocess and verified in a fresh interpreter: torch cannot be swapped
# under a process that has already imported it.

# %%
CANDIDATES = ["2.6.0", "2.5.1", "2.4.1"]
INDEX = "https://download.pytorch.org/whl/cu124"

CHECK = """
import torch
print("version", torch.__version__)
print("arch", torch.cuda.get_arch_list())
cap = torch.cuda.get_device_capability()
print("capability", cap)
a = torch.randn(512, 512, device="cuda")
b = torch.randn(512, 512, device="cuda")
print("matmul ok", float((a @ b).sum()) == float((a @ b).sum()))
print("bf16 supported", torch.cuda.is_bf16_supported())
"""

for version in CANDIDATES:
    print(f"\n=== trying torch=={version}+cu124 ===")
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            f"torch=={version}",
            "--index-url",
            INDEX,
        ],
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:
        print("install failed:", install.stderr.strip()[-500:])
        continue
    check = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True)
    print(check.stdout.strip() or check.stderr.strip()[-800:])
    if "matmul ok True" in check.stdout:
        print(f"\n>>> {version} WORKS on this card")
        break

# %% [markdown]
# ## Does the rest of the stack survive?

# %%
STACK = """
import time
import torch, transformers
print("torch", torch.__version__, "transformers", transformers.__version__, flush=True)
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    torch_dtype=torch.bfloat16,
).to(torch.float32).cuda().eval()
print("model loaded", flush=True)
ids = torch.randint(0, 1000, (1, 512), device="cuda")
with torch.no_grad():
    m(ids); torch.cuda.synchronize()
    t0 = time.time(); m(ids); torch.cuda.synchronize()
    forward = time.time() - t0
    t0 = time.time()
    out = m.generate(ids[:, :32], max_new_tokens=64, do_sample=False)
    torch.cuda.synchronize()
    gen = time.time() - t0
print("forward 512 tokens:", round(forward, 3), "s")
print("generate 64 tokens:", round(gen, 2), "s ->", round(64 / gen, 1), "tok/s")
"""


def run_stack(label: str) -> bool:
    result = subprocess.run([sys.executable, "-c", STACK], capture_output=True, text=True)
    print(f"--- stack check: {label} ---")
    print(result.stdout.strip())
    if result.returncode != 0:
        print("STDERR:", result.stderr.strip()[-1200:])
    return result.returncode == 0


def pip(*arguments: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", *arguments, "-q"], check=False)


if not run_stack("as installed"):
    # torchvision on the image is compiled against the torch that was replaced, so its
    # operators vanish and transformers fails on import. Pair it with the torch that is
    # now installed rather than leaving a mismatched build in place.
    print("\n=== pairing torchvision with torch 2.6 ===")
    pip(
        "install",
        "--no-cache-dir",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "--index-url",
        INDEX,
    )
    if not run_stack("with matched torchvision"):
        # If pairing fails, the lighter route: a text-only model needs no torchvision
        # at all, and transformers skips what is not installed.
        print("\n=== removing torchvision entirely ===")
        pip("uninstall", "-y", "torchvision", "torchaudio")
        run_stack("without torchvision")
