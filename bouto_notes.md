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

- [ ] `Real-time trigger issue`
- [ ] `Real-time optimization`
- [ ] `Real-time Filter issue (nan instead of interp)`
- [ ] `Real-time clear option`
- [ ] `Real-time channel remove`
- [ ] `Real-time single/fixed trials option`
- [ ] `Real-time docking`