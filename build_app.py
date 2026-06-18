"""Build the standalone desktop app with PyInstaller."""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path


def _add_data_argument(source: Path, destination: str) -> str:
    separator = ";" if sys.platform.startswith("win") else ":"
    return f"{source}{separator}{destination}"


def _handle_remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, 0o700)
        func(path)
    except OSError:
        raise exc_info[1]


def _safe_rmtree(path: Path, attempts: int = 3, delay: float = 0.5) -> None:
    if not path.exists():
        return
    last_error = None
    for _ in range(attempts):
        try:
            shutil.rmtree(path, onerror=_handle_remove_readonly)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(delay)
    if path.exists():
        raise last_error  # pragma: no cover - depends on local filesystem state


def main() -> int:
    os.environ["QT_API"] = "pyside6"
    try:
        import PyInstaller.__main__
    except ImportError as exc:  # pragma: no cover - depends on local build tooling
        raise SystemExit(
            "PyInstaller is not installed. Install the build dependencies first."
        ) from exc

    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    build_dir = project_root / "build"
    dist_dir = build_dir / "dist"
    work_dir = build_dir / "work"
    spec_dir = build_dir / "spec"

    # PyInstaller sometimes struggles to clean previously-opened macOS app bundles.
    # Remove stale output folders ourselves before invoking it.
    for stale_path in (
        dist_dir / "chromatoPy-desktop",
        dist_dir / "chromatoPy-desktop.app",
        work_dir / "chromatoPy-desktop",
    ):
        _safe_rmtree(stale_path)

    data_files = [
        (project_root / "src" / "chromatopy" / "FID" / "peak_labels.json", "chromatopy/FID"),
        (project_root / "src" / "chromatopy" / "utils" / "gdgt_meta.json", "chromatopy/utils"),
        (project_root / "src" / "chromatopy" / "utils" / "gdgt_meta_default.json", "chromatopy/utils"),
        (project_root / "src" / "chromatopy" / "nice.wav", "chromatopy"),
        (project_root / "src" / "chromatopy" / "oof.wav", "chromatopy"),
        (project_root / "Data Input Templates", "Data Input Templates"),
    ]
    icon_args = []
    if sys.platform == "darwin":
        icon_path = project_root / "misc" / "chromatopy_icon.icns"
        if icon_path.exists():
            icon_args.append(f"--icon={icon_path}")
    elif sys.platform.startswith("win"):
        icon_path = project_root / "misc" / "chromatopy_icon.ico"
        if icon_path.exists():
            icon_args.append(f"--icon={icon_path}")

    pyinstaller_args = [
        str(project_root / "app.py"),
        "--noconfirm",
        "--clean",
        "--windowed",
        "--onedir",
        "--name=chromatoPy-desktop",
        f"--paths={src_dir}",
        # f"--icon={project_root / 'misc' / 'chromatopy_icon.icns'}",
        *icon_args,
        "--collect-submodules=chromatopy.gui",
        "--collect-submodules=chromatopy.utils",
        "--copy-metadata=chromatopy",
        #"--collect-all=PySide6",
        # "--collect-all=matplotlib",
        "--collect-all=rainbow",
        "--hidden-import=chromatopy.hplc_integration",
        "--hidden-import=chromatopy.hplc_to_csv",
        "--hidden-import=chromatopy.FID.FID_integration",
        "--hidden-import=chromatopy.FID.FID_Integration_functions",
        "--hidden-import=chromatopy.FID.import_data",
        "--hidden-import=chromatopy.FID.manual_peak_integration",
        "--hidden-import=chromatopy.FID.peak_labels_editor",
        "--hidden-import=chromatopy.FID.Tools",
        "--hidden-import=matplotlib.backends.backend_qtagg",
        "--exclude-module=chromatopy.IRMS",
        "--exclude-module=chromatopy.FID.FID_General",
        "--exclude-module=chromatopy.FID.bouqueter",
        "--exclude-module=ipywidgets",
        "--exclude-module=matplotlib.tests",
        "--exclude-module=numpy.tests",
        "--exclude-module=pandas.tests",
        "--exclude-module=scipy.tests",
        "--exclude-module=PySide6.examples",
        f"--distpath={dist_dir}",
        f"--workpath={work_dir}",
        f"--specpath={spec_dir}",
        "--exclude-module=tqdm.notebook",
        "--exclude-module=IPython",
        "--exclude-module=jedi",
        "--exclude-module=parso",
        "--exclude-module=prompt_toolkit",
        "--exclude-module=traitlets",
        "--exclude-module=matplotlib_inline",
        "--exclude-module=hdbscan",
        "--exclude-module=sklearn",
        "--exclude-module=imagehash",
        "--exclude-module=pywt",
        "--exclude-module=PySide6.QtPdf",
        "--exclude-module=PySide6.QtQml",
        "--exclude-module=PySide6.QtQuick",
        "--exclude-module=PySide6.QtVirtualKeyboard",
        "--exclude-module=PySide6.QtVirtualKeyboardQml",
        "--exclude-module=PySide6.QtDBus",
    ]

    if sys.platform.startswith("win"):
        pyinstaller_args.append("--noupx")
    else:
        pyinstaller_args.append("--strip")

    for source, destination in data_files:
        pyinstaller_args.append(
            f"--add-data={_add_data_argument(source, destination)}"
        )

    PyInstaller.__main__.run(pyinstaller_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
