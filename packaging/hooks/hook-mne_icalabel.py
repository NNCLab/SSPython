"""Collect ICLabel code, metadata, and bundled neural-network weights."""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


datas = collect_data_files("mne_icalabel")
datas += copy_metadata("mne-icalabel")
hiddenimports = collect_submodules(
    "mne_icalabel",
    filter=lambda name: ".tests" not in name and ".testing" not in name,
)
