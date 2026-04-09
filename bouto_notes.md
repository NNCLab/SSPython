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
uv run python -m nuitka --mode=standalone --enable-plugin=pyside6 --windows-console-mode=disable --windows-icon-from-ico=assets/icon.ico --include-data-dir=assets=assets --include-data-dir=style=style --include-package=mne --output-filename=SSPython.exe main.py

```

# TODO:

- [ ] Check muilti-event data visualization
- [ ] Check real-time decays (filter related or sampling rate related?)
- [ ] Add TFR module