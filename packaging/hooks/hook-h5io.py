from PyInstaller.utils.hooks import collect_submodules, copy_metadata


hiddenimports = collect_submodules(
    "h5io",
    filter=lambda module_name: not module_name.startswith("h5io.tests"),
)
datas = copy_metadata("h5io")
