"""Shim around setuptools to (a) stage native artifacts into SuperKittens/_libs/
before sdist/wheel build, and (b) mark the wheel as platform-specific since it
bundles a Mach-O dylib + metallib."""

from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except Exception:  # wheel not installed at sdist time
    _bdist_wheel = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build"
LIBS_DIR = ROOT / "SuperKittens" / "_libs"
ARTIFACTS = ("libsk.dylib", "libsk.metallib")


def _stage_native() -> None:
    LIBS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        src = BUILD_DIR / name
        if not src.exists():
            # Fail loudly: a wheel without the dylib is broken at runtime.
            raise SystemExit(
                f"setup.py: missing required artifact {src}. "
                f"Run ./build.sh before building the wheel."
            )
        shutil.copy2(src, LIBS_DIR / name)


class BuildPy(_build_py):
    def run(self):  # type: ignore[override]
        _stage_native()
        super().run()


cmdclass = {"build_py": BuildPy}

if _bdist_wheel is not None:
    class BdistWheel(_bdist_wheel):
        def finalize_options(self):  # type: ignore[override]
            super().finalize_options()
            # Bundled Mach-O dylib pins us to macOS/arm64; cannot be pure-Python.
            self.root_is_pure = False

        def get_tag(self):  # type: ignore[override]
            python, abi, _plat = super().get_tag()
            # macOS arm64 wheel; deployment target inherited from the build host.
            import platform
            mac_ver = platform.mac_ver()[0].split(".")
            major = mac_ver[0] if mac_ver and mac_ver[0] else "11"
            minor = mac_ver[1] if len(mac_ver) > 1 and mac_ver[1] else "0"
            plat = f"macosx_{major}_{minor}_arm64"
            return python, abi, plat

    cmdclass["bdist_wheel"] = BdistWheel


setup(cmdclass=cmdclass)
