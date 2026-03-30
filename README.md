# SSPython

SSPython is a desktop EEG analysis application built with PySide6 and MNE-Python. It is designed to take a recording from raw import through preprocessing, inspection, real-time visualization, and downstream analysis inside a single workspace-oriented interface.

The current application ships two analysis pipelines:

- `TMS-EEG`: artifact-aware preprocessing and ERP/TEP analysis.
- `Continuous EEG`: continuous inspection, filtering, PSD analysis, and fixed-length epoch workflows.

The software is currently in a pre-release stage. Validate all outputs before using them for publication, regulatory work, or clinical interpretation.

> Figure Placeholder 1. Main application window showing the navigation sidebar, workspace panel, central analysis page, and derivative inspector.

## Table of Contents

- [1. What SSPython Does](#1-what-sspython-does)
- [2. Key Features](#2-key-features)
- [3. System Requirements](#3-system-requirements)
- [4. Installation](#4-installation)
- [5. Launching the Application](#5-launching-the-application)
- [6. First-Time Setup](#6-first-time-setup)
- [7. Workspace Model](#7-workspace-model)
- [8. Data Conversion and Merging](#8-data-conversion-and-merging)
- [9. Main Interface Overview](#9-main-interface-overview)
- [10. Preprocessing Workflow](#10-preprocessing-workflow)
- [11. Continuous Analysis](#11-continuous-analysis)
- [12. ERP and TEP Analysis](#12-erp-and-tep-analysis)
- [13. Real-Time Visualization](#13-real-time-visualization)
- [14. Preferences and Persistent Settings](#14-preferences-and-persistent-settings)
- [15. Output Files and Naming](#15-output-files-and-naming)
- [16. Repository Structure](#16-repository-structure)
- [17. Testing](#17-testing)
- [18. Packaging and Distribution](#18-packaging-and-distribution)
- [19. Known Limitations and Assumptions](#19-known-limitations-and-assumptions)
- [20. License](#20-license)

## 1. What SSPython Does

SSPython provides an end-to-end EEG workflow centered around a workspace folder:

- Convert supported raw recordings into standardized `*_raw.fif` files.
- Track each dataset through pipeline stages.
- Inspect raw recordings, epochs, ICA results, and final outputs.
- Apply artifact interpolation, filtering, rereferencing, epoch extraction, and ICA.
- Visualize PSD, evoked responses, topographic plots, and time-frequency results.
- Run ERP/TEP-oriented analyses such as response amplitude and natural frequency.
- Launch a live visualizer for streaming or file-backed real-time inspection.

SSPython uses MNE-Python as its signal-processing foundation and recommends the uses a BIDS-friendly file-naming convention.

## 2. Key Features

- Batch conversion of supported EEG source files through `mne.io.read_raw()`.
- Shared conversion dialog for folder scanning, output renaming, montage selection, and optional file merging.
- Special-case `.mat` import for datasets that are not directly readable by MNE.
- Workspace browser that discovers all `*_raw.fif` files under the selected workspace.
- Pipeline-specific derivative tracking with progress indicators and stage inspection.
- Interactive evoked plotting with single-click channel selection, event-condition dropdowns, and drag-to-topomap inspection.
- PSD visualization for continuous data.
- ERP/TEP analysis tools including topoplot, response amplitude analysis, natural frequency analysis, and time-frequency analysis.
- Real-time connection and plotting settings saved in application preferences.
- Theme support and global plotting defaults.

## 3. System Requirements

The project metadata currently specifies:

- Python `>= 3.14`
- PySide6
- MNE-Python `>= 1.11.0`
- `mne-lsl`
- `mne-qt-browser`
- `mne-icalabel`
- SciPy
- scikit-learn
- statsmodels
- PyOpenGL
- pyqtgraph
- torch

The repository is currently Windows-oriented:

- the installer script is `inno_setup.iss`
- existing build artifacts are stored in `main.build/` and `main.dist/`
- many usage and packaging assumptions in the repo are Windows-first

Other operating systems may work, but the documented and tested path is Windows.

## 4. Installation

### Recommended: `uv`

```powershell
uv sync
```

This creates the local virtual environment defined by `pyproject.toml` and `uv.lock`.

### Notes

- OpenGL-backed viewers may require a functional graphics driver.
- Some interactive MNE viewers rely on Qt-compatible rendering support.
- If you plan to use real-time streaming, install and validate any required stream-side dependencies before launching SSPython (Lab Streaming Layer).

## 5. Launching the Application

Run the app from the project root:

```powershell
.venv\Scripts\python.exe main.py
or
uv run main.py
```

The main entry point is [main.py](/C:/Users/DW60ZF/Documents/SSPython/main.py).

## 6. First-Time Setup

Recommended first steps:

1. Launch the app.
2. Open the `Preferences` page.
3. Set the active pipeline.
4. Set the derivative root folder name if you do not want the default `derivatives`.
5. Review preprocessing defaults, ICA defaults, and real-time settings.
6. Open a workspace folder.
7. Convert data to `*_raw.fif` if your workspace does not already contain raw FIF inputs.

## 7. Workspace Model

SSPython is workspace-driven. A workspace is simply a directory on disk that contains:

- source recordings converted to `*_raw.fif`
- optionally, any original vendor files
- derivatives written by SSPython under the configured derivative root

Dataset discovery is based on recursive search for `*_raw.fif` files under the selected workspace. Files under the derivative root are excluded from discovery so processed outputs are not mistaken for new source datasets.

### Workspace Example

```text
workspace/
|- sub-01/
|  \- eeg/
|     |- sub-01_task-rest_raw.fif
|     \- original_vendor_files...
|- sub-02/
|  \- eeg/
|     \- sub-02_task-tms_raw.fif
\- derivatives/
   \- tms_eeg/
      \- sub-02/
         \- eeg/
            |- sub-02_task-tms_desc-filtered_raw.fif
            |- sub-02_task-tms_desc-epoched_epo.fif
            \- sub-02_task-tms_desc-preprocessed_epo.fif
```

### Pipeline Awareness

Each discovered dataset is interpreted in the context of the currently active pipeline. The same raw source file can therefore appear under different derivative trees depending on the selected pipeline.

## 8. Data Conversion and Merging

SSPython now uses one shared conversion tool instead of separate BrainAmp and gTEC dialogs.

Open it from the menu:

- `File > Convert / Merge EEG data`

The dialog is implemented in [conversion_tool.py](/C:/Users/DW60ZF/Documents/SSPython/ui/widgets/tools/conversion_tool.py).

> Figure Placeholder 2. Conversion dialog showing source folder selection, montage configuration, per-file output names, and merge selection columns.

### What the Conversion Tool Does

- Scans a folder recursively.
- Includes files that can be loaded by `mne.io.read_raw()`.
- Also includes `.mat` files handled by SSPython's custom MAT loader.
- Lets you edit the output filename for each selected file.
- Normalizes every output name so it ends in `_raw.fif`.
- Lets you mark any compatible subset of files for concatenation into a merged output.
- Requires the user to provide a montage for all converted files.

### Output Location Rules

- Individual converted outputs are written next to the original source file.
- The optional merged output is written to the selected source folder root.

### Supported Input Paths

There are two conversion paths:

1. Generic MNE-supported raw files

SSPython uses `mne.io.read_raw()` for any format MNE can load directly.

1. `.mat` files

SSPython uses a custom loader in [conversion.py](/C:/Users/DW60ZF/Documents/SSPython/core/conversion.py) for `.mat` files.

### MAT Conversion Requirements

For `.mat` files, montage information is not enough by itself. SSPython also needs a shared channel-info JSON file that contains:

- `ch_names`
- `ch_types`

The MAT loader currently expects:

- a data matrix under `y` or `data`
- a sampling frequency under `SR`, `sfreq`, or `Fs`

If the data matrix is transposed relative to the channel-info file, SSPython will transpose it automatically when the dimensions make that interpretation unambiguous.

### Montage Options

The conversion dialog supports:

- built-in MNE standard montages
- custom montage files

Custom montage files currently accept the file types exposed in the dialog:

- `.fif`
- `.loc`
- `.locs`
- `.elc`
- `.csd`
- `.sfp`
- `.elp`
- `.hpts`
- `.txt`

Montage application is strict. If the selected montage does not match the converted recording's channel names, the conversion will fail instead of silently dropping the montage requirement.

### Merging Behavior

Merged outputs are created by concatenating the selected converted raws with MNE. This requires the selected recordings to be compatible in practice:

- same channel layout
- same sampling frequency
- same general raw structure

If the merge selection is incompatible, conversion or concatenation will fail with an error.

## 9. Main Interface Overview

The main UI is composed of four primary regions:

- navigation sidebar
- workspace panel
- central page area
- derivative inspector

### Navigation Sidebar

The sidebar provides access to:

- Home
- Real-Time
- Preprocessing
- Continuous Analysis
- ERP Analysis
- Preferences

### Workspace Panel

The workspace panel:

- opens a workspace folder
- refreshes dataset discovery
- filters visible datasets
- shows per-dataset progress
- emits the currently selected dataset to the active analysis page

### Derivative Inspector

The derivative inspector shows the current dataset's stage-by-stage status and allows stage objects to be inspected.

## 10. Preprocessing Workflow

The preprocessing page is implemented in [preprocessing_page.py](/C:/Users/DW60ZF/Documents/SSPython/ui/pages/preprocessing_page.py).

It organizes work into three sections:

- Continuous Processing
- Epoching and ICA
- Preprocessed Output

> Figure Placeholder 3. Preprocessing page showing workflow stages, object summaries, and the evoked preview widget for the current dataset.

### Available Actions

Continuous processing:

- Inspect Raw Data
- Remove Stimulation Artifact
- Filter and Resample
- Run Continuous ICA
- Inspect Continuous ICA
- Segment into Epochs

Epoching and ICA:

- Plot Raw Evoked
- Inspect Epochs
- Re-reference Epochs
- Run Epoch ICA
- Inspect Epoch ICA

Final output:

- Apply ICA and Final Filters

### Pipeline Stages

The current pipeline model tracks the following stages:


| Stage ID         | Meaning                                                        |
| ---------------- | -------------------------------------------------------------- |
| `raw`            | Source recording available                                     |
| `filtered_raw`   | Continuous cleanup, filtering, or artifact interpolation saved |
| `continuous_ica` | Continuous ICA fitted                                          |
| `epochs`         | Epoched data created                                           |
| `epochs_ica`     | ICA fitted on epoched data                                     |
| `preprocessed`   | Final cleaned output available                                 |


Some stages are optional depending on the workflow.

### Processing Operations in `Preprocessor`

The core preprocessing logic lives in [processing.py](/C:/Users/DW60ZF/Documents/SSPython/core/processing.py) and includes:

- TMS pulse interpolation
- continuous filtering and resampling
- continuous ICA
- event-based and fixed-length epoching
- epoch updates after inspection
- rereferencing
- epoch ICA
- final epoch filtering and output generation

## 11. Continuous Analysis

The continuous analysis page is implemented in [continuous_analysis_page.py](/C:/Users/DW60ZF/Documents/SSPython/ui/pages/continuous_analysis_page.py).

Current functionality:

- load the most recent continuous representation of a dataset
- show a recording summary
- display PSD through the PSD plot widget

This page is intended for spectral inspection of raw or filtered continuous data, depending on what has already been generated for the selected dataset.

## 12. ERP and TEP Analysis

The ERP/TEP page is implemented in [erp_analysis_page.py](/C:/Users/DW60ZF/Documents/SSPython/ui/pages/erp_analysis_page.py).

It is enabled when a dataset has reached the `preprocessed` stage.

### Visualization Tools

- evoked plot
- topoplot button

### Analysis Tools

- Response Amplitude Analysis
- Natural Frequency Analysis
- Run Time-Frequency Analysis

### Evoked Plot Interactions

The shared evoked plot widget supports:

- event-condition selection through a dropdown
- single-click channel selection
- drag-range topomap creation
- average-reference toggling

This widget is reused across preprocessing and ERP/TEP analysis pages.

## 13. Real-Time Visualization

The real-time page wraps [RealTimeMainWidget](/C:/Users/DW60ZF/Documents/SSPython/ui/widgets/real_time_widget.py), which provides:

- connection settings
- channel configuration
- saved and loaded stream configurations
- plotting settings
- a launch path into the live visualizer

Based on the current UI, the real-time stack includes:

- stream name and duration controls
- epoching window controls
- fallback montage selection
- channel discovery from stream
- channel table with per-channel type assignment
- save/load config actions
- launch button for the live visualizer

The live visualizer contains grouped views for:

- raw data monitor
- channel layout
- evoked potentials

Real-time settings are persisted through the preferences system.

## 14. Preferences and Persistent Settings

The preferences page is implemented in [preferences_page.py](/C:/Users/DW60ZF/Documents/SSPython/ui/pages/preferences_page.py).

It exposes four main categories:

- Workspace and Appearance
- Preprocessing Defaults
- ICA Defaults
- Real-Time

### Global Settings

Current global settings include:

- active pipeline
- application theme
- derivative root folder name
- parallel jobs (`n_jobs`)
- global plotting defaults

### Persistent Storage

Settings are stored through `QSettings` using the schema-aware wrapper in [app_settings.py](/C:/Users/DW60ZF/Documents/SSPython/core/app_settings.py).

The settings layer supports:

- migration from legacy keys
- default values for theme, output root, and active pipeline
- per-pipeline settings paths

## 15. Output Files and Naming

Derivative naming is defined in [pipelines.py](/C:/Users/DW60ZF/Documents/SSPython/core/pipelines.py).

### Source Naming Rule

Raw source files must end with:

```text
*_raw.fif
```

This naming rule is important because SSPython discovers source datasets by recursively searching for `*_raw.fif`.

### Derivative Naming Rule

Given a source stem such as:

```text
sub-01_task-rest
```

SSPython generates derivative names such as:

```text
sub-01_task-rest_desc-filtered_raw.fif
sub-01_task-rest_desc-continuousica_ica.fif
sub-01_task-rest_desc-epoched_epo.fif
sub-01_task-rest_desc-epochsica_ica.fif
sub-01_task-rest_desc-preprocessed_epo.fif
```

### Derivative Location Rule

Derivatives are saved under:

```text
<workspace>/<output_root>/<pipeline_id>/<relative_dataset_dir>/
```

The default output root is `derivatives`.

## 16. Repository Structure

```text
SSPython/
|- assets/                  UI assets and icons
|- core/                    Processing, conversion, pipelines, settings
|- style/                   Qt stylesheet themes
|- tests/                   Unit tests
|- ui/
|  |- pages/                Top-level application pages
|  \- widgets/              Reusable UI widgets and dialogs
|- main.py                  Application entry point
|- pyproject.toml           Project metadata and dependencies
|- utils.py                 Shared helpers, threading, and theme support
\- README.md                Project documentation
```

### Core Modules

- [processing.py](/C:/Users/DW60ZF/Documents/SSPython/core/processing.py): preprocessing and downstream EEG analysis logic
- [pipelines.py](/C:/Users/DW60ZF/Documents/SSPython/core/pipelines.py): pipeline definitions, naming, discovery, and derivative paths
- [conversion.py](/C:/Users/DW60ZF/Documents/SSPython/core/conversion.py): generic import, MAT loading, montage application, and merging
- [app_settings.py](/C:/Users/DW60ZF/Documents/SSPython/core/app_settings.py): persistent settings wrapper

## 17. Testing

Run the full test suite with:

```powershell
.venv\Scripts\python.exe -m unittest discover tests
```

Targeted test files include:

- `tests/test_core.py`
- `tests/test_ui.py`
- `tests/test_conversion.py`

The test suite covers selected preprocessing logic, UI initialization, event handling, and conversion behavior. It is not yet a complete end-to-end validation harness.

## 18. Packaging and Distribution

The repository includes packaging-related artifacts for Windows:

- `inno_setup.iss`

These indicate a Windows packaging flow, creating Nuitka-built executables plus an Inno Setup installer.

## 19. Known Limitations and Assumptions

- SSPython discovers source datasets only from files ending in `*_raw.fif`.
- `.mat` inputs require an external channel-info JSON file. Channel names and types are not inferred automatically.
- The MAT loader currently assumes electrophysiology-like channels are stored in microvolts and scales them to volts for MNE.
- Merging requires compatible selected recordings. SSPython does not currently auto-reconcile channel mismatches or sampling-rate differences.
- Montage application is strict and will fail if the selected montage does not match the recording.
- The project is pre-release and should not be treated as a validated clinical or publication pipeline without independent verification.
- The documented path is Windows-first.

## 20. License

See [LICENSE](/C:/Users/DW60ZF/Documents/SSPython/LICENSE).