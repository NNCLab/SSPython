Need to update
On line 5623 in /opt/anaconda3/envs/mne/lib/python3.13/site-packages/mne_qt_browser/_pg_figure.py DataTrace toggle_bar
 
 -> Now on line 628 on ...same/_graphics_items.py 

'''python 
# Update overview-bar
self.mne.overview_bar.update_bad_epochs()

# Update other traces inlcuding self
# for trace in self.mne.traces:
#     trace.update_color()
#     # Update data is necessary because colored segments will vary
#     trace.update_data()

# Force a full redraw to re-fetch data from the updated Epochs object
self.weakmain()._redraw()
'''

new fix:
```python
fig = epochs.plot()
fig.mne.view.scene().sigMouseClicked.connect(fig._redraw) # Important fix for MNE-QT-Browser
```