"""Collect MNE's lazily exposed modules and non-Python resources."""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


def without_tests(module_name: str) -> bool:
    return not any(part in {"tests", "testing", "commands"} for part in module_name.split("."))


datas = collect_data_files(
    "mne",
    excludes=["**/tests/**", "**/commands/**"],
)
hiddenimports = collect_submodules("mne", filter=without_tests)

# MNE's interactive 3D viewers use inspect.getsource() when they create
# weak-reference callbacks. Keep source next to bytecode for those modules;
# otherwise PyInstaller's PYZ-only modules fail with "could not get source
# code" when an STC brain window is opened.
module_collection_mode = {
    "mne.viz._brain._brain": "pyz+py",
    "mne.viz.backends.renderer": "pyz+py",
    "mne.viz.evoked_field": "pyz+py",
    "mne.viz.utils": "pyz+py",
}
