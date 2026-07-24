"""Collect MNE-LSL's dynamic modules, settings, and bundled liblsl binary."""

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


datas = collect_data_files("mne_lsl")
binaries = collect_dynamic_libs("mne_lsl")
hiddenimports = collect_submodules(
    "mne_lsl",
    filter=lambda name: ".tests" not in name and ".testing" not in name,
)
