Currently, mne-qt-browser is not updating properly, decided to fix it like the following:

```python
fig = epochs.plot()
fig.mne.view.scene().sigMouseClicked.connect(fig._redraw) # Important fix for MNE-QT-Browser
```

# TODO:

- [ ] `Check why some pulses are missing (sometimes). Might be device related (trigger to lsl not capping properly).`
- [ ] Check muilti-event data visualization
- [ ] Finish convert and merge utils
- [ ] Add TFR module