Currently, mne-qt-browser is not updating properly, decided to fix it like the following:

```python
fig = epochs.plot()
fig.mne.view.scene().sigMouseClicked.connect(fig._redraw) # Important fix for MNE-QT-Browser
```

# TODO:

- [ ] `ICA Sources Plot` as option of `Plot ICA`
- [ ] `Better handling of multi-event data: Evoked plot, topoplot and such`
- [ ] `TFR vizualization module`
- [ ] `Move average reference to epoching dialog`