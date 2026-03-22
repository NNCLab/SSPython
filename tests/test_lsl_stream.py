# %% Imports
import mne_lsl
from pathlib import Path

DATA_PATH = Path('...')
files = list(sorted(DATA_PATH.rglob('*raw.fif')))
print(files)

file = files[2]

player = mne_lsl.player.PlayerLSL(
    file,
    chunk_size=200,
    name="SSPy-Player"
).start()
player.info