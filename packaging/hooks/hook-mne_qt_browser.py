"""Collect the MNE Qt browser, which MNE loads through its backend selector."""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


datas = collect_data_files("mne_qt_browser")
hiddenimports = collect_submodules(
    "mne_qt_browser",
    filter=lambda name: ".tests" not in name and ".testing" not in name,
)
