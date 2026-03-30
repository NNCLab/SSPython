Currently, mne-qt-browser is not updating properly, decided to fix it like the following:

```python
fig = epochs.plot()
fig.mne.view.scene().sigMouseClicked.connect(fig._redraw) # Important fix for MNE-QT-Browser
```

# Deploy

```plaintext
uv add setuptools wheel nuitka ordered-set zstandard imageio
```

```plaintext
uv run python -m nuitka --mode=onefile --enable-plugin=pyside6 --windows-console-mode=disable --windows-icon-from-ico=assets/icon.png --include-data-dir=assets=assets --include-package=mne --report=build-report.xml --output-filename=SSPython.exe --main=main.py
```

# TODO:

- [ ] `Check why some pulses are missing (sometimes). Might be device related (trigger to lsl not capping properly).`
- [ ] Check muilti-event data visualization
- [ ] Finish convert and merge utils
- [ ] Add TFR module