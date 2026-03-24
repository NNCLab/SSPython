import time
from mne_lsl.stream import StreamLSL, EpochsStream
import matplotlib.pyplot as plt
import numpy as np

# 1. Connect Continuous Stream
stream = StreamLSL(bufsize=10, name='SSPy-Player')
stream.connect(acquisition_delay=0.1, processing_flags='all')

# 2. Connect Epochs Stream
epochs_stream = EpochsStream(
    stream=stream,
    bufsize=20,
    event_id=dict(stimulus=1),
    event_channels=['TMS'],
    tmin=-0.2,
    tmax=0.5
)
epochs_stream.connect(acquisition_delay=0.1)

count=0
try:
    print("Listening to streams... Press Ctrl+C to stop.")
    while True:
        count+=1
        # --- Read Continuous Data ---
        if stream.n_new_samples > 0:
            # Convert the number of new samples into seconds for 'winsize'
            new_time_window = stream.n_new_samples / stream.info["sfreq"]
            
            # get_data() returns both the array and the LSL timestamps
            continuous_data, continuous_ts = stream.get_data(winsize=new_time_window)
            
        # start looking for epochs
        if epochs_stream.n_new_epochs == 0:
            continue
        print(f"New Epochs: {epochs_stream.n_new_epochs}")
        data = epochs_stream.get_data(n_epochs=epochs_stream.n_new_epochs)
        new_evoked = np.mean(data, 0)
        time.sleep(1/24)

except KeyboardInterrupt:
    print("\nDisconnecting streams...")
    epochs_stream.disconnect()
    stream.disconnect()