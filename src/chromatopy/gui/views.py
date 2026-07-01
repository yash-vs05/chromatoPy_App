"""PySide6 desktop views for chromatoPy."""

from __future__ import annotations
from .settings_memory import load_theme, save_theme
from .themes import LIGHT_APP_STYLE, DARK_APP_STYLE

import copy

from .. import __version__
from ..qt_compat import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QVBoxLayout,
    WaitCursor,
    QWidget,
    exec_dialog,
)
from ..utils.GDGT_compounds import edit_gdgt_meta_qt
from .logic import (
    IntegrationConfiguration,
    available_compound_histories,
    build_general_summary,
    calculate_hplc_fractional_abundance,
    calculate_hplc_indices,
    calculate_hplc_peak_area_confidence_intervals,
    collect_general_header_options,
    detect_general_window_bounds,
    integration_file_status,
    load_persisted_integration_configuration,
    remember_integration_configuration,
    refresh_integration_config,
    remove_compound_history,
    run_data_conversion,
    run_peak_integration,
    summarize_gdgt_meta,
    summarize_integration_configuration,
)


class IntegrationConfigurationDialog(QDialog):
    """Collect parameters for HPLC, FID, or General peak integration."""

    def __init__(self, config: IntegrationConfiguration, parent=None, on_file_count=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Peak Integration")
        self.resize(820, 780)
        self._config = copy.deepcopy(config)
        self._on_file_count = on_file_count
        self._last_counted_folder = ""

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Load data folder containing converted data and verify the settings before launching integration"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.mode_label = QLabel(f"Processing Mode: {self._config.mode}")
        layout.addWidget(self.mode_label)

        folder_row = QHBoxLayout()
        self.input_folder_edit = QLineEdit(self._config.input_folder)
        self.input_folder_edit.setPlaceholderText("Select or enter folder location of samples")
        self.input_folder_edit.textChanged.connect(self._folder_text_changed)
        self.input_folder_edit.editingFinished.connect(self._refresh_from_folder)
        folder_row.addWidget(self.input_folder_edit, 1)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse_input_folder)
        folder_row.addWidget(browse_button)
        layout.addLayout(folder_row)

        self.settings_title = QLabel("Mode Settings")
        layout.addWidget(self.settings_title)

        self.summary_box = QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(170)
        layout.addWidget(self.summary_box)

        self.general_section = QWidget()
        general_layout = QVBoxLayout(self.general_section)
        self.general_header_options: list[str] = []

        headers_layout = QGridLayout()
        headers_layout.addWidget(QLabel("Time Column Header"), 0, 0)
        self.general_time_header_combo = QComboBox()
        self.general_time_header_combo.setEditable(True)
        self.general_time_header_combo.currentTextChanged.connect(self._general_time_header_changed)
        headers_layout.addWidget(self.general_time_header_combo, 0, 1)
        headers_layout.addWidget(QLabel("Signal Column Header"), 1, 0)
        self.general_signal_header_combo = QComboBox()
        self.general_signal_header_combo.setEditable(True)
        headers_layout.addWidget(self.general_signal_header_combo, 1, 1)
        general_layout.addLayout(headers_layout)

        history_row = QHBoxLayout()
        history_row.addWidget(QLabel("Saved Compound Lists"))
        self.compound_history_combo = QComboBox()
        history_row.addWidget(self.compound_history_combo, 1)
        load_history_button = QPushButton("Load")
        load_history_button.clicked.connect(self._load_selected_history)
        history_row.addWidget(load_history_button)
        delete_history_button = QPushButton("Delete")
        delete_history_button.clicked.connect(self._delete_selected_history)
        history_row.addWidget(delete_history_button)
        general_layout.addLayout(history_row)

        self.general_compounds_edit = QLineEdit(", ".join(self._config.general_compounds))
        self.general_compounds_edit.setPlaceholderText("Compound Names (comma-separated)")
        general_layout.addWidget(self.general_compounds_edit)

        general_window_row = QHBoxLayout()
        general_window_row.addWidget(QLabel("Window Start"))
        self.general_window_start_spin = QDoubleSpinBox()
        self.general_window_start_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.general_window_start_spin.setDecimals(4)
        self.general_window_start_spin.setValue(self._config.general_window[0])
        general_window_row.addWidget(self.general_window_start_spin)
        general_window_row.addWidget(QLabel("Window End"))
        self.general_window_end_spin = QDoubleSpinBox()
        self.general_window_end_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.general_window_end_spin.setDecimals(4)
        self.general_window_end_spin.setValue(self._config.general_window[1])
        general_window_row.addWidget(self.general_window_end_spin)
        general_layout.addLayout(general_window_row)

        general_toggles = QHBoxLayout()
        self.asymmetric_checkbox = QCheckBox("Asymmetric Peak Fitting")
        self.asymmetric_checkbox.setChecked(self._config.use_asymmetric_peak_integration)
        general_toggles.addWidget(self.asymmetric_checkbox)
        self.deconvolution_checkbox = QCheckBox("Peak Deconvolution")
        self.deconvolution_checkbox.setChecked(self._config.enable_peak_deconvolution)
        self.deconvolution_checkbox.toggled.connect(self._sync_general_fit_controls)
        general_toggles.addWidget(self.deconvolution_checkbox)
        general_layout.addLayout(general_toggles)
        layout.addWidget(self.general_section)

        self.hplc_section = QWidget()
        hplc_layout = QVBoxLayout(self.hplc_section)
        edit_meta_row = QHBoxLayout()
        edit_meta_row.addWidget(QLabel("Multi-channel compound mapping"))
        edit_meta_row.addStretch(1)
        self.edit_hplc_button = QPushButton("Edit Sample Groups")
        self.edit_hplc_button.clicked.connect(self._edit_hplc_meta)
        edit_meta_row.addWidget(self.edit_hplc_button)
        hplc_layout.addLayout(edit_meta_row)
        self.hplc_normalization_checkbox = QCheckBox("Enable Time Normalization")
        self.hplc_normalization_checkbox.setChecked(self._config.normalize_by_standard)
        hplc_layout.addWidget(self.hplc_normalization_checkbox)
        layout.addWidget(self.hplc_section)

        self.fid_section = QWidget()
        fid_layout = QFormLayout(self.fid_section)
        self.fid_peak_method_combo = QComboBox()
        self.fid_peak_method_combo.addItem("Asymmetric", "asymmetric")
        self.fid_peak_method_combo.addItem("Multi-Gaussian", "multi")
        self.fid_peak_method_combo.addItem("Asymmetric or MultiGaussian", "asymmetric_or_multi")
        fid_method_index = self.fid_peak_method_combo.findData(self._config.fid_peak_integration_method)
        if fid_method_index >= 0:
            self.fid_peak_method_combo.setCurrentIndex(fid_method_index)
        self.fid_peak_method_combo.currentIndexChanged.connect(self._refresh_summary)
        fid_layout.addRow("Peak Integration Method", self.fid_peak_method_combo)
        fid_window_grid = QGridLayout()
        self.fid_window_xmin_edit = QLineEdit("" if self._config.fid_window_xmin is None else str(self._config.fid_window_xmin))
        self.fid_window_xmax_edit = QLineEdit("" if self._config.fid_window_xmax is None else str(self._config.fid_window_xmax))
        self.fid_window_ymin_edit = QLineEdit("" if self._config.fid_window_ymin is None else str(self._config.fid_window_ymin))
        self.fid_window_ymax_edit = QLineEdit("" if self._config.fid_window_ymax is None else str(self._config.fid_window_ymax))
        for edit in [self.fid_window_xmin_edit, self.fid_window_xmax_edit, self.fid_window_ymin_edit, self.fid_window_ymax_edit]:
            edit.setPlaceholderText("Auto")
            edit.textChanged.connect(self._refresh_summary)
        fid_window_grid.addWidget(QLabel("X Min"), 0, 0)
        fid_window_grid.addWidget(self.fid_window_xmin_edit, 0, 1)
        fid_window_grid.addWidget(QLabel("X Max"), 0, 2)
        fid_window_grid.addWidget(self.fid_window_xmax_edit, 0, 3)
        fid_window_grid.addWidget(QLabel("Y Min"), 1, 0)
        fid_window_grid.addWidget(self.fid_window_ymin_edit, 1, 1)
        fid_window_grid.addWidget(QLabel("Y Max"), 1, 2)
        fid_window_grid.addWidget(self.fid_window_ymax_edit, 1, 3)
        fid_layout.addRow("Peak Selection Window", fid_window_grid)
        layout.addWidget(self.fid_section)

        shared_grid = QGridLayout()
        self.peak_neighborhood_spin = QSpinBox()
        self.peak_neighborhood_spin.setRange(1, 50)
        self.peak_neighborhood_spin.setValue(self._config.peak_neighborhood_n)
        shared_grid.addWidget(QLabel("Peak Neighborhood"), 0, 0)
        shared_grid.addWidget(self.peak_neighborhood_spin, 0, 1)

        self.smoothing_window_spin = QSpinBox()
        self.smoothing_window_spin.setRange(3, 99)
        self.smoothing_window_spin.setSingleStep(2)
        self.smoothing_window_spin.setValue(self._config.smoothing_window)
        shared_grid.addWidget(QLabel("Smoothing Window"), 1, 0)
        shared_grid.addWidget(self.smoothing_window_spin, 1, 1)

        self.smoothing_factor_spin = QSpinBox()
        self.smoothing_factor_spin.setRange(1, 12)
        self.smoothing_factor_spin.setValue(self._config.smoothing_factor)
        shared_grid.addWidget(QLabel("Smoothing Factor"), 2, 0)
        shared_grid.addWidget(self.smoothing_factor_spin, 2, 1)

        self.gaussian_iterations_spin = QSpinBox()
        self.gaussian_iterations_spin.setRange(100, 50000)
        self.gaussian_iterations_spin.setSingleStep(500)
        self.gaussian_iterations_spin.setValue(self._config.gaus_iterations)
        shared_grid.addWidget(QLabel("Gaussian Iterations"), 0, 2)
        shared_grid.addWidget(self.gaussian_iterations_spin, 0, 3)

        self.minimum_peak_spin = QDoubleSpinBox()
        self.minimum_peak_spin.setRange(-1.0, 1_000_000_000.0)
        self.minimum_peak_spin.setDecimals(6)
        self.minimum_peak_spin.setSpecialValueText("Auto (legacy)")
        self.minimum_peak_spin.setValue(self._config.minimum_peak_amplitude if self._config.minimum_peak_amplitude is not None else -1.0)
        shared_grid.addWidget(QLabel("Minimum Peak Amplitude"), 1, 2)
        shared_grid.addWidget(self.minimum_peak_spin, 1, 3)

        self.maximum_peak_spin = QDoubleSpinBox()
        self.maximum_peak_spin.setRange(0.0, 1_000_000_000.0)
        self.maximum_peak_spin.setDecimals(3)
        self.maximum_peak_spin.setSpecialValueText("Auto")
        self.maximum_peak_spin.setValue(self._config.maximum_peak_amplitude or 0.0)
        shared_grid.addWidget(QLabel("Maximum Peak Amplitude"), 2, 2)
        shared_grid.addWidget(self.maximum_peak_spin, 2, 3)

        self.peak_boundary_spin = QDoubleSpinBox()
        self.peak_boundary_spin.setRange(0.000001, 10.0)
        self.peak_boundary_spin.setDecimals(6)
        self.peak_boundary_spin.setValue(self._config.peak_boundary_derivative_sensitivity)
        shared_grid.addWidget(QLabel("Boundary Sensitivity"), 3, 2)
        shared_grid.addWidget(self.peak_boundary_spin, 3, 3)

        self.peak_prominence_spin = QDoubleSpinBox()
        self.peak_prominence_spin.setRange(0.000001, 1000.0)
        self.peak_prominence_spin.setDecimals(6)
        self.peak_prominence_spin.setValue(self._config.peak_prominence)
        shared_grid.addWidget(QLabel("Peak Prominence"), 4, 0)
        shared_grid.addWidget(self.peak_prominence_spin, 4, 1)

        self.clip_negative_checkbox = QCheckBox("Clip Negative Amplitudes to Zero")
        self.clip_negative_checkbox.setChecked(self._config.clip_negative_amplitudes)
        general_layout.addWidget(self.clip_negative_checkbox)

        self.shared_settings_widget = QWidget()
        self.shared_settings_widget.setLayout(shared_grid)
        layout.addWidget(self.shared_settings_widget)

        opt_button_row = QHBoxLayout()
        default_button = QPushButton("Revert to Default")
        default_button.clicked.connect(self._revert_to_default)
        opt_button_row.addWidget(default_button)
        opt_button_row.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        opt_button_row.addWidget(buttons)
        layout.addLayout(opt_button_row)


        self._reload_histories()
        self._update_folder_placeholder_style(self.input_folder_edit.text())
        self._sync_general_fit_controls(self.deconvolution_checkbox.isChecked())
        self._refresh_from_folder()

    def _browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Integration Input Folder", self.input_folder_edit.text().strip())
        if folder:
            self.input_folder_edit.setText(folder)
            self._refresh_from_folder()

    def _folder_text_changed(self, text: str):
        self._update_folder_placeholder_style(text)

    def _update_folder_placeholder_style(self, text: str):
        font = self.input_folder_edit.font()
        font.setItalic(not bool(text.strip()))
        self.input_folder_edit.setFont(font)

    def _update_window_bounds_from_header(self, header_text: str):
        if self._config.mode != "General":
            return
        folder_path = self.input_folder_edit.text().strip()
        header_text = header_text.strip()
        if not folder_path or not header_text:
            return
        try:
            window_start, window_end = detect_general_window_bounds(folder_path, header_text)
        except Exception:
            return
        self.general_window_start_spin.setValue(window_start)
        self.general_window_end_spin.setValue(window_end)

    def _general_time_header_changed(self, text: str):
        self._update_window_bounds_from_header(text)
        self._refresh_summary()

    def _reload_histories(self):
        self.compound_history_combo.clear()
        histories = available_compound_histories()
        for history in histories:
            self.compound_history_combo.addItem(", ".join(history), history)

    def _load_selected_history(self):
        history = self.compound_history_combo.currentData()
        if history:
            self.general_compounds_edit.setText(", ".join(history))

    def _delete_selected_history(self):
        history = self.compound_history_combo.currentData()
        if not history:
            return
        remove_compound_history(history)
        self._reload_histories()

    def _set_general_header_options(self, options: list[str], time_header: str, signal_header: str):
        self.general_header_options = options
        current_time = self.general_time_header_combo.currentText().strip()
        current_signal = self.general_signal_header_combo.currentText().strip()

        self.general_time_header_combo.clear()
        self.general_signal_header_combo.clear()
        self.general_time_header_combo.addItems(options)
        self.general_signal_header_combo.addItems(options)

        selected_time = self._config.general_time_header or current_time or time_header
        selected_signal = self._config.general_signal_header or current_signal or signal_header
        self.general_time_header_combo.setCurrentText(selected_time)
        self.general_signal_header_combo.setCurrentText(selected_signal)
        self._update_window_bounds_from_header(selected_time)

    def _edit_hplc_meta(self):
        if self._config.mode != "HPLC":
            return
        try:
            updated = edit_gdgt_meta_qt(self._config.gdgt_meta_set, parent=self)
        except Exception as exc:
            QMessageBox.critical(self, "HPLC settings failed", str(exc))
            return
        self._config.gdgt_meta_set = updated
        self._refresh_summary()

    def _refresh_from_folder(self):
        self._config.input_folder = self.input_folder_edit.text().strip()
        self.mode_label.setText(f"Processing Mode: {self._config.mode}")
        if not self._config.input_folder:
            self._update_mode_sections()
            self._refresh_summary()
            return
        try:
            refresh_integration_config(self._config)
            if self._config.mode == "General":
                options, detected_time, detected_signal = collect_general_header_options(self._config.input_folder)
                self._set_general_header_options(options, detected_time, detected_signal)
        except Exception:
            pass
        self._update_mode_sections()
        self._refresh_summary()
        self._report_file_count()

    def _report_file_count(self):
        if self._on_file_count is None or not self._config.input_folder:
            return
        count_key = f"{self._config.mode}:{self._config.input_folder}"
        if count_key == self._last_counted_folder:
            return
        try:
            status = integration_file_status(self._config)
        except Exception:
            return
        self._last_counted_folder = count_key
        self._on_file_count(status)

    def _update_mode_sections(self):
        is_general = self._config.mode == "General"
        is_hplc = self._config.mode == "HPLC"
        is_fid = self._config.mode == "FID"
        #has_folder = bool(self.input_folder_edit.text().strip())
        self.general_section.setVisible(is_general)
        self.hplc_section.setVisible(is_hplc)
        self.fid_section.setVisible(is_fid)
        self.general_section.setEnabled(True)
        self.hplc_section.setEnabled(True)
        self.fid_section.setEnabled(True)
        self.shared_settings_widget.setEnabled(True)
        self.summary_box.setEnabled(True)
        self._sync_general_fit_controls(self.deconvolution_checkbox.isChecked())

    def _sync_general_fit_controls(self, deconvolution_enabled: bool):
        if deconvolution_enabled:
            self.asymmetric_checkbox.setChecked(False)
        self.asymmetric_checkbox.setEnabled((not deconvolution_enabled) and self._config.mode == "General")

    def _optional_float_from_edit(self, edit: QLineEdit, label: str) -> float | None:
        text = edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{label} must be blank or a number.") from exc

    def _refresh_summary(self):
        if self._config.mode == "General":
            compounds = [name.strip() for name in self.general_compounds_edit.text().split(",") if name.strip()]
            preview = copy.deepcopy(self._config)
            preview.general_compounds = compounds
            preview.general_time_header = self.general_time_header_combo.currentText().strip()
            preview.general_signal_header = self.general_signal_header_combo.currentText().strip()
            preview.general_window = [self.general_window_start_spin.value(), self.general_window_end_spin.value()]
            preview.minimum_peak_amplitude = None if self.minimum_peak_spin.value() < 0.0 else self.minimum_peak_spin.value()
            preview.maximum_peak_amplitude = None if self.maximum_peak_spin.value() == 0.0 else self.maximum_peak_spin.value()
            preview.peak_prominence = self.peak_prominence_spin.value()
            preview.enable_peak_deconvolution = self.deconvolution_checkbox.isChecked()
            preview.use_asymmetric_peak_integration = (not preview.enable_peak_deconvolution) and self.asymmetric_checkbox.isChecked()
            preview.clip_negative_amplitudes = self.clip_negative_checkbox.isChecked()
            self.summary_box.setPlainText(
                summarize_integration_configuration(preview) + "\n\nConfigured channels and compounds:\n" + build_general_summary(preview)
            )
            return
        if self._config.mode == "FID":
            preview = copy.deepcopy(self._config)
            preview.fid_peak_integration_method = self.fid_peak_method_combo.currentData()
            try:
                preview.fid_window_xmin = self._optional_float_from_edit(self.fid_window_xmin_edit, "FID X Min")
                preview.fid_window_xmax = self._optional_float_from_edit(self.fid_window_xmax_edit, "FID X Max")
                preview.fid_window_ymin = self._optional_float_from_edit(self.fid_window_ymin_edit, "FID Y Min")
                preview.fid_window_ymax = self._optional_float_from_edit(self.fid_window_ymax_edit, "FID Y Max")
            except ValueError:
                pass
            self.summary_box.setPlainText(summarize_integration_configuration(preview))
            return
        self.summary_box.setPlainText(summarize_integration_configuration(self._config))

    def _config_helper(self):
        self.mode_label.setText(f"Processing Mode: {self._config.mode}")
        self.input_folder_edit.setText(self._config.input_folder)

        self.general_compounds_edit.setText(", ".join(self._config.general_compounds))
        self.general_window_start_spin.setValue(self._config.general_window[0])
        self.general_window_end_spin.setValue(self._config.general_window[1])

        self.asymmetric_checkbox.setChecked(self._config.use_asymmetric_peak_integration)
        self.deconvolution_checkbox.setChecked(self._config.enable_peak_deconvolution)
        self.hplc_normalization_checkbox.setChecked(self._config.normalize_by_standard)

        fid_method_index = self.fid_peak_method_combo.findData(self._config.fid_peak_integration_method)
        if fid_method_index >= 0:
            self.fid_peak_method_combo.setCurrentIndex(fid_method_index)
        self.fid_window_xmin_edit.setText("" if self._config.fid_window_xmin is None else str(self._config.fid_window_xmin))
        self.fid_window_xmax_edit.setText("" if self._config.fid_window_xmax is None else str(self._config.fid_window_xmax))
        self.fid_window_ymin_edit.setText("" if self._config.fid_window_ymin is None else str(self._config.fid_window_ymin))
        self.fid_window_ymax_edit.setText("" if self._config.fid_window_ymax is None else str(self._config.fid_window_ymax))

        self.peak_neighborhood_spin.setValue(self._config.peak_neighborhood_n)
        self.smoothing_window_spin.setValue(self._config.smoothing_window)
        self.smoothing_factor_spin.setValue(self._config.smoothing_factor)
        self.gaussian_iterations_spin.setValue(self._config.gaus_iterations)
        self.minimum_peak_spin.setValue(self._config.minimum_peak_amplitude if self._config.minimum_peak_amplitude is not None else -1.0)
        self.maximum_peak_spin.setValue(self._config.maximum_peak_amplitude or 0.0)
        self.peak_boundary_spin.setValue(self._config.peak_boundary_derivative_sensitivity)
        self.peak_prominence_spin.setValue(self._config.peak_prominence)
        self.clip_negative_checkbox.setChecked(self._config.clip_negative_amplitudes)
        self._update_folder_placeholder_style(self.input_folder_edit.text())
        self._sync_general_fit_controls(self.deconvolution_checkbox.isChecked())
        self._refresh_from_folder()

    def _accept(self):
        self._config.input_folder = self.input_folder_edit.text().strip()
        # if not self._config.input_folder:
        #     QMessageBox.warning(self, "Missing folder", "Select an input folder before continuing.")
        #     return

        self._config.peak_neighborhood_n = self.peak_neighborhood_spin.value()
        self._config.smoothing_window = self.smoothing_window_spin.value()
        self._config.smoothing_factor = self.smoothing_factor_spin.value()
        self._config.gaus_iterations = self.gaussian_iterations_spin.value()
        self._config.minimum_peak_amplitude = None if self.minimum_peak_spin.value() < 0.0 else self.minimum_peak_spin.value()
        self._config.maximum_peak_amplitude = None if self.maximum_peak_spin.value() == 0.0 else self.maximum_peak_spin.value()
        self._config.peak_boundary_derivative_sensitivity = self.peak_boundary_spin.value()
        self._config.peak_prominence = self.peak_prominence_spin.value()
        self._config.normalize_by_standard = self.hplc_normalization_checkbox.isChecked()

        if self._config.mode == "FID":
            self._config.fid_peak_integration_method = self.fid_peak_method_combo.currentData()
            try:
                self._config.fid_window_xmin = self._optional_float_from_edit(self.fid_window_xmin_edit, "FID X Min")
                self._config.fid_window_xmax = self._optional_float_from_edit(self.fid_window_xmax_edit, "FID X Max")
                self._config.fid_window_ymin = self._optional_float_from_edit(self.fid_window_ymin_edit, "FID Y Min")
                self._config.fid_window_ymax = self._optional_float_from_edit(self.fid_window_ymax_edit, "FID Y Max")
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid FID window", str(exc))
                return
            if (
                self._config.fid_window_xmin is not None
                and self._config.fid_window_xmax is not None
                and self._config.fid_window_xmin >= self._config.fid_window_xmax
            ):
                QMessageBox.warning(self, "Invalid FID window", "FID X Min must be less than FID X Max.")
                return
            if (
                self._config.fid_window_ymin is not None
                and self._config.fid_window_ymax is not None
                and self._config.fid_window_ymin >= self._config.fid_window_ymax
            ):
                QMessageBox.warning(self, "Invalid FID window", "FID Y Min must be less than FID Y Max.")
                return

        if self._config.mode == "General":
            self._config.general_time_header = self.general_time_header_combo.currentText().strip()
            self._config.general_signal_header = self.general_signal_header_combo.currentText().strip()
            self._config.general_compounds = [name.strip() for name in self.general_compounds_edit.text().split(",") if name.strip()]
            self._config.general_window = [self.general_window_start_spin.value(), self.general_window_end_spin.value()]
            self._config.enable_peak_deconvolution = self.deconvolution_checkbox.isChecked()
            self._config.use_asymmetric_peak_integration = (not self._config.enable_peak_deconvolution) and self.asymmetric_checkbox.isChecked()
            self._config.clip_negative_amplitudes = self.clip_negative_checkbox.isChecked()
            self._config.normalize_by_standard = False
            if not self._config.general_time_header or not self._config.general_signal_header:
                QMessageBox.warning(self, "Missing headers", "General mode requires both time and signal headers.")
                return
            if not self._config.general_compounds:
                QMessageBox.warning(self, "Missing compounds", "Provide at least one compound name for General mode.")
                return

        refresh_integration_config(self._config)
        self.accept()

    def _revert_to_default(self):
        current_mode = self._config.mode
        current_folder = self.input_folder_edit.text()
        self._config = IntegrationConfiguration()
        self._config.mode = current_mode
        self._config.input_folder = current_folder
        self._config_helper()

    def configuration(self) -> IntegrationConfiguration:
        return copy.deepcopy(self._config)


class ModulePage(QWidget):
    def __init__(self, title: str, description: str, on_back, parent=None):
        super().__init__(parent)
        self._root_layout = QVBoxLayout(self)
        header = QHBoxLayout()
        back_button = QPushButton("Back to Dashboard")
        back_button.clicked.connect(on_back)
        header.addWidget(back_button)
        header.addStretch(1)
        self._root_layout.addLayout(header)

        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        self._root_layout.addWidget(title_label)

        description_label = QLabel(description)
        description_label.setWordWrap(True)
        description_label.setObjectName("pageDescription")
        self._root_layout.addWidget(description_label)


class DataConversionPage(ModulePage):
    def __init__(self, on_back, parent=None):
        super().__init__(
            title="Data Conversion",
            description="Convert raw instrument directories into chromatoPy-ready CSV files.",
            on_back=on_back,
            parent=parent,
        )

        form = QFormLayout()
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(["HPLC", "IC MS"])
        form.addRow("Data type", self.data_type_combo)

        input_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        input_row.addWidget(self.input_edit, 1)
        browse_input = QPushButton("Browse")
        browse_input.clicked.connect(self._browse_input)
        input_row.addWidget(browse_input)
        form.addRow("Raw data folder", input_row)
        self._root_layout.addLayout(form)

        run_button = QPushButton("Run Data Conversion")
        run_button.clicked.connect(self._run_conversion)
        self._root_layout.addWidget(run_button)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self._root_layout.addWidget(self.log, 1)

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Raw HPLC Folder")
        if folder:
            self.input_edit.setText(folder)

    def _run_conversion(self):
        try:
            QApplication.setOverrideCursor(WaitCursor)
            result = run_data_conversion(
                self.input_edit.text().strip(),
                data_type=self.data_type_combo.currentText(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Data conversion failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        summary = [f"{self.data_type_combo.currentText()} conversion exported {len(result['files_exported'])} file(s) to {result['output_path']}."]
        skipped = result.get("files_skipped", [])
        if skipped:
            summary.append(f"Skipped {len(skipped)} file(s).")
        self.log.appendPlainText("\n".join(summary))


class PeakIntegrationPage(ModulePage):
    def __init__(self, on_back, parent=None):
        super().__init__(
            title="Peak Integration",
            description="Use one unified entry point for HPLC, FID, or General single-channel integration.",
            on_back=on_back,
            parent=parent,
        )
        self.current_config = load_persisted_integration_configuration()

        controls = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["HPLC", "FID", "General"])
        self.mode_combo.setCurrentText(self.current_config.mode)
        self.mode_combo.currentTextChanged.connect(self._mode_changed)
        controls.addWidget(QLabel("Processing Mode"))
        controls.addWidget(self.mode_combo)

        configure_button = QPushButton("Configure")
        configure_button.clicked.connect(self.open_configuration)
        controls.addWidget(configure_button)

        self.run_button = QPushButton("Run Peak Integration")
        self.run_button.clicked.connect(self._run_workflow)
        controls.addWidget(self.run_button)

        self.manual_run_button = QPushButton("Manual Peak Integration")
        self.manual_run_button.clicked.connect(self._run_manual_workflow)
        controls.addWidget(self.manual_run_button)
        controls.addStretch(1)
        self._root_layout.addLayout(controls)

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(280)
        self._root_layout.addWidget(self.summary)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self._root_layout.addWidget(self.log, 1)
        self._update_mode_controls()
        self._refresh_summary()

    def _append_log_message(self, message: str):
        self.log.appendPlainText(str(message))
        QApplication.processEvents()

    def _mode_changed(self, mode: str):
        self.current_config.mode = mode
        self.current_config.input_folder = ""
        self._update_mode_controls()
        self._refresh_summary()

    def _update_mode_controls(self):
        self.manual_run_button.setVisible(self.current_config.mode == "FID")

    def _refresh_summary(self):
        self.summary.setPlainText(summarize_integration_configuration(self.current_config))

    def _log_identified_files(self, status: dict):
        total_files = status["total_files"]
        processed_files = status["processed_files"]
        if str(status.get("results_file_path", "")).endswith("results_peak_area.csv"):
            self.log.appendPlainText(
                f"Identified {total_files} files. {processed_files} files are already processed. "
                "To re-integrate already processed files, please delete the row containing the integrated peaks "
                "in 'results_peak_area.csv'."
            )
            return
        if str(status.get("results_file_path", "")).endswith("Sample Data"):
            self.log.appendPlainText(
                f"Identified {total_files} files. {processed_files} files are already processed. "
                "To re-integrate an FID sample, delete that sample's JSON file from 'chromatoPy output/Sample Data'."
            )
            return
        self.log.appendPlainText(
            f"Identified {total_files} files. {processed_files} files are already processed."
        )

    def open_configuration(self):
        self.current_config.mode = self.mode_combo.currentText()
        dialog = IntegrationConfigurationDialog(
            self.current_config,
            self,
            on_file_count=self._log_identified_files,
        )
        if exec_dialog(dialog) == QDialog.Accepted:
            self.current_config = dialog.configuration()
            remember_integration_configuration(self.current_config)
            self.mode_combo.setCurrentText(self.current_config.mode)
            self._refresh_summary()
            self.log.appendPlainText(f"{self.current_config.mode} configuration updated.")
            return True
        return False

    def _run_manual_workflow(self):
        self._run_workflow(manual_peak_integration=True)

    def _run_workflow(self, manual_peak_integration: bool = False):
        if not self.current_config.input_folder and not self.open_configuration():
            return
        try:
            result = run_peak_integration(
                self.current_config,
                message_callback=self._append_log_message,
                manual_peak_integration=manual_peak_integration,
            )
        except SystemExit:
            self.log.appendPlainText("Peak integration was cancelled before completion.")
            return
        except Exception as exc:
            QMessageBox.critical(self, "Peak integration failed", str(exc))
            return

        if isinstance(result, dict):
            details = []
            if "results_file_path" in result:
                details.append(f"Results: {result['results_file_path']}")
            if "figures_folder" in result:
                details.append(f"Figures: {result['figures_folder']}")
            workflow_name = "manual integration" if manual_peak_integration else "integration"
            self.log.appendPlainText("\n".join([f"{self.current_config.mode} {workflow_name} finished."] + details))
        else:
            workflow_name = "manual integration" if manual_peak_integration else "integration"
            self.log.appendPlainText(f"{self.current_config.mode} {workflow_name} finished.")


class PostProcessingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Post Processing")
        self.resize(720, 420)

        layout = QVBoxLayout(self)

        title = QLabel("Post Processing")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        form = QFormLayout()
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(["HPLC"])
        form.addRow("Data type", self.data_type_combo)

        output_row = QHBoxLayout()
        self.output_location_edit = QLineEdit()
        self.output_location_edit.setPlaceholderText("Select Output_chromatoPy folder or raw data folder")
        output_row.addWidget(self.output_location_edit, 1)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse_output_location)
        output_row.addWidget(browse_button)
        form.addRow("Output data location", output_row)
        layout.addLayout(form)

        action_row = QHBoxLayout()
        fractional_button = QPushButton("Calculate Fractional Abundance")
        fractional_button.clicked.connect(self._calculate_fractional_abundance)
        action_row.addWidget(fractional_button)
        indices_button = QPushButton("Calculate Indices")
        indices_button.clicked.connect(self._calculate_indices)
        action_row.addWidget(indices_button)
        peak_ci_button = QPushButton("Calculate Peak Area 95% CI")
        peak_ci_button.clicked.connect(self._calculate_peak_area_ci)
        action_row.addWidget(peak_ci_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_output_location(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select HPLC Output Data Folder",
            self.output_location_edit.text().strip(),
        )
        if folder:
            self.output_location_edit.setText(folder)

    def _calculate_fractional_abundance(self):
        try:
            QApplication.setOverrideCursor(WaitCursor)
            result = calculate_hplc_fractional_abundance(self.output_location_edit.text().strip())
        except Exception as exc:
            QMessageBox.critical(self, "Fractional abundance failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.log.appendPlainText(
            f"Calculated fractional abundance for {result['rows']} sample(s).\n"
            f"Saved: {result['fractional_abundance_path']}"
        )

    def _calculate_indices(self):
        try:
            QApplication.setOverrideCursor(WaitCursor)
            result = calculate_hplc_indices(self.output_location_edit.text().strip())
        except Exception as exc:
            QMessageBox.critical(self, "Index calculation failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.log.appendPlainText(
            f"Calculated indices for {result['rows']} sample(s).\n"
            f"Saved: {result['indices_path']}\n"
            f"Saved: {result['meth_set_path']}\n"
            f"Saved: {result['cyc_set_path']}"
        )

    def _calculate_peak_area_ci(self):
        try:
            QApplication.setOverrideCursor(WaitCursor)
            result = calculate_hplc_peak_area_confidence_intervals(self.output_location_edit.text().strip())
        except Exception as exc:
            QMessageBox.critical(self, "Peak area CI calculation failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.log.appendPlainText(
            f"Calculated peak area 95% CI for {result['rows']} sample(s) and {result['peaks']} peak(s).\n"
            f"Saved: {result['peak_area_ci_path']}"
        )


class ChromatoPyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("chromatoPy Desktop")
        self.resize(1280, 860)

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(18)
        self.setCentralWidget(central)

        title_row = QHBoxLayout()
        title = QLabel("chromatoPy Desktop")
        title.setObjectName("appTitle")
        title_row.addWidget(title)
        version = QLabel(f"(v{__version__})")
        version.setObjectName("appVersion")
        title_row.addWidget(version)
        title_row.addStretch(1)

        self.dark_mode = load_theme() == "dark"
        self.theme_button = QPushButton()
        self.theme_button.setFixedSize(42, 42)
        self.theme_button.clicked.connect(self.toggle_theme)
        self.update_theme_button()

        title_row.addWidget(self.theme_button)

        root_layout.addLayout(title_row)

        subtitle = QLabel(
            "A simplified entry point for data conversion and unified peak integration."
        )
        subtitle.setObjectName("appSubtitle")
        subtitle.setWordWrap(True)
        root_layout.addWidget(subtitle)

        self.dashboard = QWidget()
        dashboard_layout = QHBoxLayout(self.dashboard)
        dashboard_layout.setSpacing(18)
        dashboard_layout.addWidget(
            self._build_module_card(
                "Data Conversion",
                "Convert raw instrument directories into chromatoPy-ready CSV files.",
                self.show_data_conversion,
            )
        )
        dashboard_layout.addWidget(
            self._build_module_card(
                "Peak Integration",
                "Use one integration screen with HPLC, FID, and General modes.",
                self.show_peak_integration,
            )
        )
        dashboard_layout.addWidget(
            self._build_module_card(
                "Post Processing",
                "Calculate fractional abundances and common HPLC indices from integrated output.",
                self.open_post_processing,
            )
        )
        root_layout.addWidget(self.dashboard)

        self.pages = QWidget()
        pages_layout = QHBoxLayout(self.pages)
        pages_layout.setContentsMargins(0, 0, 0, 0)
        self.data_conversion_page = DataConversionPage(self.show_dashboard)
        self.peak_integration_page = PeakIntegrationPage(self.show_dashboard)
        pages_layout.addWidget(self.data_conversion_page)
        pages_layout.addWidget(self.peak_integration_page)
        root_layout.addWidget(self.pages, 1)

        self.show_dashboard()

    def _build_module_card(self, title: str, description: str, on_click):
        card = QFrame()
        card.setObjectName("moduleCard")
        layout = QVBoxLayout(card)
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        layout.addWidget(title_label)
        body = QLabel(description)
        body.setWordWrap(True)
        body.setObjectName("cardBody")
        layout.addWidget(body, 1)
        launch_button = QPushButton(f"Open {title}")
        launch_button.clicked.connect(on_click)
        layout.addWidget(launch_button)
        return card

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode

        theme = "dark" if self.dark_mode else "light"
        save_theme(theme)

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(DARK_APP_STYLE if self.dark_mode else LIGHT_APP_STYLE)

        self.update_theme_button()

    def update_theme_button(self):
        self.theme_button.setText("☀️" if self.dark_mode else "🌙")
        self.theme_button.setToolTip(
            "Switch to light mode" if self.dark_mode else "Switch to dark mode"
        )

    def show_dashboard(self):
        self.dashboard.show()
        self.data_conversion_page.hide()
        self.peak_integration_page.hide()

    def show_data_conversion(self):
        self.dashboard.hide()
        self.data_conversion_page.show()
        self.peak_integration_page.hide()

    def show_peak_integration(self):
        self.dashboard.hide()
        self.data_conversion_page.hide()
        self.peak_integration_page.show()

    def open_post_processing(self):
        dialog = PostProcessingDialog(self)
        exec_dialog(dialog)
