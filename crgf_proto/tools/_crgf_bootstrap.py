"""Import bootstrap for CRGF tools.

Call ``install()`` BEFORE importing ``supermimic``.  Two responsibilities:

1. If ``isaaclab.sim`` cannot be imported (CPU-only environments without the
   Isaac Sim runtime), install a stub package so that ``supermimic.robot``
   imports cleanly.  The stub only feeds the asset-config definitions that
   robot.py builds at import time; it is never used by any math path, and
   simulation tools (which need the real isaaclab) are unaffected when the
   real package is present.
2. Nothing else: sys.path handling stays in the calling tool.
"""
from __future__ import annotations

import sys
import types


class _AnyMeta(type):
    def __getattr__(cls, name):
        return _AnyCfg


class _AnyCfg(metaclass=_AnyMeta):
    """Permissive placeholder for asset config objects; carries no behavior."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _AnyCfg

    def __call__(self, *args, **kwargs):
        return _AnyCfg()


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        sub = sys.modules.get(f"{self.__name__}.{name}")
        if sub is not None:
            return sub
        return _AnyCfg


def install():
    try:
        import isaaclab.sim  # noqa: F401
        return False
    except Exception:
        pass
    if "isaaclab" in sys.modules and getattr(sys.modules["isaaclab"], "__crgf_stub__", False):
        return True
    stub = _StubModule("isaaclab")
    stub.__crgf_stub__ = True
    sys.modules["isaaclab"] = stub
    for name in ("isaaclab.sim", "isaaclab.actuators", "isaaclab.assets"):
        sys.modules[name] = _StubModule(name)
    return True


def fix_data_paths(repo_root):
    """Point the active kinematics module at this repo's MJCF if the variant's
    own data directory is absent (the OPCL proto ran inside a container and
    its data/ directory is empty)."""
    from pathlib import Path

    try:
        import supermimic.kinematics as K
    except Exception:
        return False
    mjcf = Path(K.MJCF_PATH)
    if mjcf.exists():
        return False
    candidate = Path(repo_root) / "data" / "g1_description" / "g1_29dof.xml"
    if candidate.exists():
        K.MJCF_PATH = str(candidate)
        return True
    raise FileNotFoundError(f"neither {mjcf} nor {candidate} exists")
