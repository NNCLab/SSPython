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
- [ ] `Real-time Filter issue (nan instead of interp)` 
- [ ] `Real-time single/fixed trials option`
- [ ] `Real-time docking`
- [ ] `Real time error when closing thread:`

```plaintext
RuntimeError: Error calling Python override of QDialog::closeEvent(): Internal C++ object (PySide6.QtCore.QThread) already deleted.
QObject::killTimer: Timers cannot be stopped from another thread
QObject::~QObject: Timers cannot be stopped from another thread
```