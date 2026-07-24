"""PyInstaller build definition for the Windows SSPython application."""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


PROJECT_ROOT = Path(SPECPATH).parent
HOOKS_DIR = PROJECT_ROOT / "packaging" / "hooks"
BUILD_WITH_CONSOLE = os.environ.get("SSPYTHON_BUILD_CONSOLE") == "1"


def without_tests(module_name: str) -> bool:
    parts = set(module_name.split("."))
    return not parts.intersection({"tests", "testing", "test", "conftest"})


def required_pyvista_module(module_name: str) -> bool:
    if not without_tests(module_name):
        return False
    return not module_name.startswith(
        (
            "pyvista.examples",
            "pyvista.ext",
            "pyvista.trame",
        )
    )


# These libraries are loaded lazily by MNE/PyVista, so static import analysis
# cannot reliably discover them from the application entry point.
hidden_imports = [
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_qt5agg",
    "mne.viz.backends._pyvista",
    "mne.viz.backends._qt",
    "mne_qt_browser",
    "pyvista",
    "pyvistaqt",
]
hidden_imports += collect_submodules("pyvista", filter=required_pyvista_module)
hidden_imports += collect_submodules("pyvistaqt", filter=without_tests)


analysis = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        (str(PROJECT_ROOT / "assets"), "assets"),
        (str(PROJECT_ROOT / "style"), "style"),
    ],
    hiddenimports=sorted(set(hidden_imports)),
    hookspath=[str(HOOKS_DIR)],
    hooksconfig={
        "matplotlib": {
            "backends": ["Agg", "QtAgg"],
        },
    },
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jupyter",
        "jupyterlab",
        "notebook",
        "pytest",
        "sphinx",
        "sympy.testing",
        "tkinter",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="SSPython",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=BUILD_WITH_CONSOLE,
    icon=str(PROJECT_ROOT / "assets" / "icon.ico"),
    contents_directory="_internal",
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="SSPython",
)
