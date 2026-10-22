# Build the given AITER JIT modules ahead of time (no GPU needed; set GPU_ARCHS).
# aiter/jit/core.py is imported directly, as aiter's setup.py does: importing the
# aiter package needs a live GPU (Triton driver probe).
import importlib.util
import os
import sys

_spec = importlib.util.find_spec("aiter")
sys.path.insert(0, os.path.join(os.path.dirname(_spec.origin), "jit"))

from core import build_module, get_args_of_build  # noqa: E402

for md in sys.argv[1:]:
    d = get_args_of_build(md)
    print("building", md, flush=True)
    build_module(
        md,
        d["srcs"],
        d["flags_extra_cc"],
        d["flags_extra_hip"],
        d["blob_gen_cmd"],
        d["extra_include"],
        d["extra_ldflags"],
        d["verbose"],
        d["is_python_module"],
        d["is_standalone"],
        d["torch_exclude"],
        d["third_party"],
        flags_extra_hip_per_source=d.get("flags_extra_hip_per_source"),
    )
    print("built", md, flush=True)
