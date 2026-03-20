# %% Imports
import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import ast

# epochs.proj -> True if project
# epochs.info['projs'] -> list with current projs

# %% Real Time Montage
data_path = Path("/Users/brunoandrynascimentocouto/Desktop/Test Dataset")
files = sorted(list(data_path.rglob("sub*baseline*M1*raw.fif")))
file = files[0]
raw = mne.io.read_raw_fif(file, preload=True)
info = raw.info.copy()


data = np.random.rand(66, 1000)
ch_names = [f"{i}" for i in range(len(data))]
ch_types = ["eeg"] * len(data)
curr_info = mne.create_info(ch_names, 4800, ch_types)

raw = mne.io.RawArray(data, info)

# %% REAL TIME
import mne_lsl

data_path = Path("/Users/brunoandrynascimentocouto/Desktop/Test Dataset")
files = sorted(list(data_path.rglob("sub*baseline*M1*raw.fif")))
file = files[0]
raw = mne.io.read_raw_fif(file, preload=True)
raw.crop(30)
raw.apply_function(lambda x: np.round(x), picks="TMS")
ch_names = np.array(raw.ch_names)
player = mne_lsl.player.PlayerLSL(
    raw, chunk_size=4800, n_repeat=np.inf, name="test_stream"
)
player.start()

# %% RAW
data_path = Path("/Users/brunoandrynascimentocouto/Desktop/Test Dataset")
files = sorted(list(data_path.rglob("s*task-M1_raw.fif")))

raw = mne.io.read_raw_fif(files[0], preload=True)
raw.filter(0.5, None)
raw.set_eeg_reference("average")

ica = mne.preprocessing.ICA()
ica.fit(raw.copy().resample(300))

ica.plot_components(inst=raw)

ica.plot_overlay(inst=raw)

# %% ICA Label
import mne_icalabel
from core.Preprocessing import ERPpreprocessor

data_path = Path("/Users/brunoandrynascimentocouto/Desktop/Test Dataset")
files = sorted(list(data_path.rglob("*-M1_raw.fif")))

preprocessor = ERPpreprocessor(files[0], data_path / "derivatives")

preprocessor.run_artifact_removal(window=(0, 0.015))
epochs = mne.Epochs(
    preprocessor.raw,
    events=mne.events_from_annotations(preprocessor.raw)[0],
    event_id=mne.events_from_annotations(preprocessor.raw)[1],
)
epochs.load_data()
epochs.resample(1200)

ica = mne.preprocessing.ICA()
ica.fit(epochs)

ic_labels = mne_icalabel.label_components(epochs, ica, method="iclabel")

# labels can be:
# brain
# muscle artifact
# eye blink
# heart beat
# line noise
# channel noise
# other

labels = ic_labels["labels"]
exclude_idx = [
    idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
]
print(f"Excluding these ICA components: {exclude_idx}")

# %% Description
from datetime import datetime
import json

data_path = Path("C:/Users/Bruno Couto/Desktop/Test Dataset")
files = sorted(list(data_path.rglob("*_epo.fif")))

epochs = mne.read_epochs(files[0], preload=True)
log_entries = []
log_entry_1 = {
    "function": "Epoching",
    "params": {"param1": 1, "param2": "value2"},
    "timestamp": datetime.now().isoformat(),
}
log_entries.append(log_entry_1)
log_entry_2 = {
    "function": "Filtering",
    "params": {"low_freq": 0.5, "high_freq": 40},
    "timestamp": datetime.now().isoformat(),
}
log_entries.append(log_entry_2)
log_string = json.dumps(log_entries, indent=4)

print("--- Serialized String ---")
print(log_string)

retrieved_logs = json.loads(log_string)

print("\n--- Retrieved Logs (as Python list) ---")
print(retrieved_logs)

# %% Average Reference
nepochs, nchs, ntimes = epochs.get_data().shape
sfreq = epochs.info["sfreq"]
xtime = lambda x: 1e-3 * x * ntimes - (1 / sfreq)

bads = [1, 2, 3]

fig = epochs.plot()
for bad in bads:
    fig._toggle_bad_epoch(xtime(bad))

evoked = epochs.average()
fig, axs = plt.subplots(1, 2, sharey=True)
evoked.plot(axes=axs[0], zorder="std", selectable=False)
evoked.plot(axes=axs[1], zorder="std", selectable=False)
plt.show()

for l, line in enumerate(axs[0].lines):
    line.set_ydata(evoked.data[l, :])


filtered_epochs = epochs.filter(1, 80)
filtered_epochs.plot(use_opengl=True, theme="light", events=True)
epochs.plot()

fig = epochs.average().apply_baseline().plot_topo()
for line in fig.axes[0].lines:
    line.set_linewidth(2)

ica = mne.preprocessing.ICA(5)

print(f"""
Average reference: {epochs.proj}
Baseline correction: {epochs.baseline}
""")

ica.fit(epochs.copy().apply_baseline())

fig = ica.plot_properties(
    epochs, 0, topomap_args=dict(cmap="turbo"), image_args=dict(cmap="turbo")
)

main_color = "C0"

# EVK
fig[0].axes[2].lines[0].set_color(main_color)
fig[0].axes[2].collections[0].set_color(main_color)

# PSD
fig[0].axes[3].lines[0].set_color(main_color)
fig[0].axes[3].collections[0].set_color(main_color)

# Variance
fig[0].axes[4].collections[0].set_color(main_color)
fig[0].axes[4].collections[1].set_color(main_color)

fig[0].axes[5].lines[0].set_color(main_color)

for patch in fig[0].axes[5].patches:
    patch.set_color(main_color)


ica.plot_components(inst=epochs.average())

# # -*- mode: python ; coding: utf-8 -*-

# # This is a PyInstaller spec file.
# # To build the executable, run: pyinstaller build.spec
# from PyInstaller.utils.hooks import collect_all
# block_cipher = None

# datas = []
# binaries = []
# hiddenimports = []
# libraries_list = [
#     'mne', 'mne_qt_browser', 'h5io',
#     'scipy', 'matplotlib', 'h5py', 'eeglabio', 'edfio']
# for library in libraries_list:
#     d, b, hi = collect_all(library)
#     datas+=d;binaries+=b;hiddenimports+=hi

# a = Analysis(
#     ['main.py'],
#     pathex=[],
#     binaries=binaries,
#     datas=[('assets', 'assets'),]+datas,
#     hiddenimports=hiddenimports,
#     hookspath=[],
#     hooksconfig={},
#     runtime_hooks=[],
#     excludes=[],
#     win_no_prefer_redirects=False,
#     win_private_assemblies=False,
#     cipher=block_cipher,
#     noarchive=False,
# )
# pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# exe = EXE(
#     pyz,
#     a.scripts,
#     [],
#     icon='C:/Users/Bruno Couto/Documents/PhD-Studies/LabApps/SSPy/assets/icon.ico',
#     exclude_binaries=True,
#     name='SSPython',
#     debug=False,
#     bootloader_ignore_signals=False,
#     strip=False,
#     upx=True,
#     console=False,  # Set to False to create a windowed app without a console
#     disable_windowed_traceback=False,
#     argv_emulation=False,
#     target_arch=None,
#     codesign_identity=None,
#     entitlements_file=None,
# )
# coll = COLLECT(
#     exe,
#     a.binaries,
#     a.zipfiles,
#     a.datas,
#     strip=False,
#     upx=True,
#     upx_exclude=[],
#     name='SSPython',
# )
