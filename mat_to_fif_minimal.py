
with open('channels.json', 'r') as f:
    channels_info = json.load(f)
    
files = list(sorted(data_path.rglob('*.mat')))
files = [f for f in files if 'TMS' in f.name]

file = files[0]
pbar = tqdm(files, desc="Processing .mat files")

for file in pbar:
    mne.set_log_level('WARNING')
    mat = loadmat(file, variable_names=['MetaData'])

    has_metadata = 'MetaData' in mat and mat['MetaData'].size > 0
    if not has_metadata or mat.get('MetaData', None) is None:
        pbar.set_description(f"Skipping {file.name} (no metadata)")
        continue

    metadata = str(mat['MetaData'][0][0][-1][0])
    
    if metadata.lower() in ['delete', '', 'del']:
        pbar.set_description(f"Skipping {file.name} (marked for deletion)")
        continue
    pbar.set_description(f"Processing {file.name} (metadata: {metadata})")
    
    subject = file.parent.name.split('_')[1]
    task, session = metadata.split('_')[:2]
    label = '_'.join(file.stem.split('_')[:-1])
    existing_files = list(sorted(file.parent.glob(f"{label}*.mat")))

    new_path = raw_path / f"{subject}_{task}_{session}_raw.fif"
    if new_path.exists():
        pbar.set_description(f"Skipping {file.name} (already processed as {new_path.name})")
        continue

    mats = [loadmat(f, variable_names=['y']) for f in existing_files]
    sfreq = loadmat(existing_files[-1], variable_names=['SR'])['SR'][0][0]
    data = np.hstack([m['y'] for m in mats])
    eeg_idx = [i for i, ch in enumerate(channels_info['ch_types']) if ch == 'eeg']
    data[eeg_idx, :] *= 1e-6  # Convert from V to µV
    info = mne.create_info(channels_info['ch_names'], sfreq, channels_info['ch_types'])
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage('easycap-M1')
    events = mne.find_events(raw, verbose=False)
    # check if has events
    if len(events) > 0:
        raw.set_annotations(mne.annotations_from_events(events, sfreq, event_desc=lambda x: 'TMS'))

    raw.save(new_path, overwrite=True, verbose=False)