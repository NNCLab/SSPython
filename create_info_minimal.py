# %% Imports
import mne
from pathlib import Path

# %% 
data_path = Path('C:/Users/DW60ZF/Desktop/sspy_test/')
files = sorted(data_path.rglob('*_raw.fif'))
files = [f for f in files if 'desc' not in f.name]

file = files[0]
info = mne.io.read_info(file)

info.rename_channels({'Fp1': 'M1', 'PO8': 'M2'})
info.set_channel_types({'M1': 'emg', 'M2': 'emg'})
info.save(file.parent/'mep_info.fif', overwrite=True)
