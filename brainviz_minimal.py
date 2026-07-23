import sys
import os
from pathlib import Path

import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
    QInputDialog,
)
from PySide6.QtCore import Qt

# Set the MNE_QT_BACKEND environment variable to 'PySide6'
# This helps MNE integrate its Qt-based plots with the application's event loop.
os.environ['MNE_QT_BACKEND'] = 'PySide6'


class MainWindow(QMainWindow):
    """
    Main application window for the MNE source modeling GUI.
    """

    def __init__(self):
        super().__init__()

        # --- Window Properties ---
        self.setWindowTitle("MNE Source Modeling Tool")
        self.setGeometry(100, 100, 500, 250)

        # --- MNE Data Attributes ---
        # These will store the MNE objects as they are created.
        self.epochs = None
        self.stc = None
        self.brain = None
        self.subjects_dir = None
        self.fs_dir = None
        self.src = None
        self.bem = None
        self.trans = "fsaverage"  # MNE has a built-in fsaverage transformation

        # --- UI Initialization ---
        self.init_ui()

        # --- Initial button states ---
        self.update_button_states()

    def init_ui(self):
        """
        Sets up the user interface widgets and layouts.
        """
        # --- Main Layout ---
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- File Loading Section ---
        self.file_label = QLabel("No file loaded.")
        self.file_label.setStyleSheet("font-style: italic; color: #555;")
        self.file_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        load_button = QPushButton("Load Epochs File (.fif)")
        load_button.clicked.connect(self.load_file)
        
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(load_button)
        main_layout.addLayout(file_layout)

        # --- Action Buttons ---
        self.align_button = QPushButton("1. Check Coregistration Alignment")
        self.align_button.clicked.connect(self.check_alignment)
        main_layout.addWidget(self.align_button)

        self.run_button = QPushButton("2. Run Source Modeling & Plot")
        self.run_button.clicked.connect(self.run_source_modeling)
        main_layout.addWidget(self.run_button)

        self.movie_button = QPushButton("3. Generate & Save Movie")
        self.movie_button.clicked.connect(self.generate_movie)
        main_layout.addWidget(self.movie_button)

        # --- Central Widget ---
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def update_button_states(self):
        """
        Enables or disables buttons based on the application state
        (e.g., if data has been loaded).
        """
        is_epochs_loaded = self.epochs is not None
        # The brain object might be closed by the user manually,
        # so we check if its underlying plotter still exists.
        is_stc_plotted = self.brain is not None and self.brain.plotter is not None

        self.align_button.setEnabled(is_epochs_loaded)
        self.run_button.setEnabled(is_epochs_loaded)
        self.movie_button.setEnabled(is_stc_plotted)

    def load_file(self):
        """
        Opens a file dialog to select an MNE epochs file.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Epochs File", "", "FIF Files (*-epo.fif *.fif)"
        )
        if file_path:
            try:
                self.epochs = mne.read_epochs(file_path, preload=True)
                self.file_label.setText(f"Loaded: {Path(file_path).name}")
                self.file_label.setStyleSheet("color: #005500;") # Green text for success
                print(f"Successfully loaded epochs from: {file_path}")
                print(self.epochs)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {e}")
                self.file_label.setText("Failed to load file.")
                self.file_label.setStyleSheet("color: #990000;") # Red text for error
                return
            
            # Reset downstream data and update UI
            self.stc = None
            self.brain = None
            self.update_button_states()

    def _prepare_fs_files(self):
        """
        Downloads fsaverage files if not already present and sets paths.
        This is a helper function to avoid redundant downloads.
        """
        if self.fs_dir is None:
            try:
                print("Fetching fsaverage files...")
                self.fs_dir = Path(fetch_fsaverage(verbose=True))
                self.subjects_dir = self.fs_dir.parent
                self.src = self.fs_dir / "bem" / "fsaverage-ico-5-src.fif"
                self.bem = self.fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
                print("fsaverage files are ready.")
                return True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not fetch fsaverage files: {e}")
                return False
        return True

    def check_alignment(self):
        """
        Plots the coregistration alignment.
        """
        if not self.epochs:
            QMessageBox.warning(self, "Warning", "Please load an epochs file first.")
            return

        if not self._prepare_fs_files():
            return
            
        print("Plotting alignment. This may open a new window...")
        try:
            # The MNE plot opens its own interactive window.
            mne.viz.plot_alignment(
                self.epochs.info,
                src=self.src,
                eeg=["original", "projected"],
                trans=self.trans,
                show_axes=True,
                mri_fiducials=True,
                dig="fiducials",
                subjects_dir=self.subjects_dir,
            )
        except Exception as e:
            QMessageBox.critical(self, "Plotting Error", f"An error occurred during plotting: {e}")

    def run_source_modeling(self):
        """
        Computes and plots the source time course (STC).
        """
        if not self.epochs:
            QMessageBox.warning(self, "Warning", "Please load an epochs file first.")
            return

        if not self._prepare_fs_files():
            return

        try:
            print("Starting source modeling process...")
            # 1. Set EEG reference
            self.epochs.set_eeg_reference(projection=True)
            print("EEG reference set.")

            # 2. Compute noise covariance
            print("Computing noise covariance...")
            noise_cov = mne.compute_covariance(
                self.epochs, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=True
            )
            print("Noise covariance computed.")

            # 3. Create average evoked response
            evoked = self.epochs.average().pick("eeg")
            print("Evoked response created.")

            # 4. Make forward solution
            print("Making forward solution...")
            fwd = mne.make_forward_solution(
                self.epochs.info, trans=self.trans, src=self.src, bem=self.bem, eeg=True, mindist=5.0, n_jobs=-1
            )
            print("Forward solution created.")

            # 5. Make inverse operator
            print("Making inverse operator...")
            inverse_operator = make_inverse_operator(
                self.epochs.info, fwd, noise_cov, loose=0.2, depth=0.8
            )
            print("Inverse operator created.")

            # 6. Apply inverse solution (dSPM)
            print("Applying inverse solution...")
            method = "dSPM"
            snr = 3.0
            lambda2 = 1.0 / snr**2
            self.stc = apply_inverse(
                evoked,
                inverse_operator,
                lambda2,
                method=method,
                pick_ori=None,
                verbose=True,
            )
            print("STC computed.")

            # 7. Plot the result
            print("Plotting STC. This may open a new window...")
            vertno_max, time_max = self.stc.get_peak(hemi="rh")
            
            surfer_kwargs = dict(
                hemi="both",
                subjects_dir=self.subjects_dir,
                colormap='turbo',
                views="dorsal",
                initial_time=time_max,
                time_unit="s",
                size=(800, 800),
                smoothing_steps=5,
                time_viewer=True, # Important for movie generation
                brain_kwargs=dict(surf='pial')
            )
            # The plot function returns a Brain object, which we need for the movie
            self.brain = self.stc.plot(**surfer_kwargs)
            print("Plotting complete.")
            self.update_button_states()

        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"An error occurred: {e}")
            self.stc = None
            self.brain = None
        
        self.update_button_states()

    def generate_movie(self):
        """
        Prompts user for tmin/tmax and saves the brain animation to a file.
        Does not close the brain plot afterwards.
        """
        self.update_button_states() # Check if plot window was closed manually
        if not self.movie_button.isEnabled():
            QMessageBox.warning(self, "Warning", "Please run source modeling first to generate a plot.")
            return

        # Prompt for tmin
        tmin, ok1 = QInputDialog.getDouble(self, "Movie Time Range", "Enter start time (tmin, in seconds):", 0.0, -10.0, 10.0, 3)
        if not ok1:
            return # User cancelled

        # Prompt for tmax
        tmax, ok2 = QInputDialog.getDouble(self, "Movie Time Range", "Enter end time (tmax, in seconds):", 0.15, -10.0, 10.0, 3)
        if not ok2:
            return # User cancelled

        # Get save file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Movie", "", "Movie Files (*.mov *.mp4 *.gif)"
        )

        if file_path:
            try:
                print(f"Generating movie, please wait... Saving to: {file_path}")
                self.brain.save_movie(
                    file_path,
                    tmin=tmin,
                    tmax=tmax,
                    interpolation='linear',
                    time_dilation=20,
                    framerate=10,
                    time_viewer=True
                )
                QMessageBox.information(self, "Success", f"Movie saved successfully to {file_path}")
                print("Movie generation complete.")
            except Exception as e:
                QMessageBox.critical(self, "Movie Error", f"Failed to save movie: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
