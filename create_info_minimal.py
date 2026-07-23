# %% Imports
import mne
from pathlib import Path

# %% 
data_path = Path('C:/Users/DW60ZF/Desktop/sspy_test/')
files = sorted(data_path.rglob('*_raw.fif'))
files = [f for f in files if 'desc' not in f.name]

file = files[0]
info = mne.io.read_info(file)
info.set_montage('easycap-M1')
info.save(file.parent / 'sspy_4800_info.fif', overwrite=True)

ch_names = info['ch_names']
ch_types = info.get_channel_types()
sfreq = info['sfreq']

high_sampling_sfreq = 19200
high_sampling_info = mne.create_info(ch_names, high_sampling_sfreq, ch_types)
high_sampling_info.set_montage('easycap-M1')
high_sampling_info.save(file.parent / f'sspy_{int(high_sampling_info["sfreq"])}_info.fif', overwrite=True)

ch_names = ch_names[1:]
ch_types = ch_types[1:]
new_info = mne.create_info(ch_names, info['sfreq'], ch_types)
new_info.set_montage('easycap-M1')
new_info.save(file.parent / 'sspy_4800_notimes_info.fif', overwrite=True)

high_sampling_notimes = mne.create_info(ch_names, high_sampling_sfreq, ch_types)
high_sampling_notimes.set_montage('easycap-M1')
high_sampling_notimes.save(file.parent / f'sspy_{int(high_sampling_notimes["sfreq"])}_notimes_info.fif', overwrite=True)
