"""Non-visual desktop application logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import simpson

from ..FID.FID_integration import integration as fid_integration
from ..hplc_integration import hplc_integration
from ..hplc_to_csv import hplc_to_csv
from ..ic_ms_to_csv import ic_ms_to_csv
from ..utils.GDGT_compounds import get_gdgt, json_to_gdgt_meta, load_gdgt_meta_json
from ..utils.calculate_indices import calculate_fa, calculate_indices, calculate_raberg2021
from ..utils.data_schema import (
    EXCLUDED_COLUMNS,
    build_single_channel_meta,
    detect_data_schema,
    list_csv_files,
)
from .settings_memory import (
    delete_compound_history,
    list_compound_histories,
    load_integration_configuration_memory,
    remember_compound_history,
    save_integration_configuration_memory,
)


def load_saved_gdgt_meta():
    try:
        return json_to_gdgt_meta(load_gdgt_meta_json())
    except FileNotFoundError:
        return get_gdgt("4")


def _clean_header_name(column: str) -> str:
    return column[:-2] if str(column).endswith(".0") else str(column)


def _looks_like_time_header(column: str) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in ("time", "second", "seconds", "minute", "minutes"))


def collect_general_header_options(folder_path: str) -> tuple[list[str], str, str]:
    csv_files = list_csv_files(folder_path)
    if not csv_files:
        raise ValueError("No CSV files were found in the selected folder.")

    unique_headers: list[str] = []
    seen: set[str] = set()
    for sample_file in csv_files:
        preview = pd.read_csv(os.path.join(folder_path, sample_file), nrows=0)
        for raw_column in preview.columns:
            column = _clean_header_name(raw_column)
            if column not in seen:
                unique_headers.append(column)
                seen.add(column)

    if len(unique_headers) < 2:
        raise ValueError("General mode requires at least two columns: time and signal.")

    time_candidates = [column for column in unique_headers if _looks_like_time_header(column)]
    time_header = time_candidates[0] if time_candidates else unique_headers[0]

    signal_candidates = [
        column
        for column in unique_headers
        if column not in EXCLUDED_COLUMNS
        and not str(column).startswith("Unnamed:")
        and column != time_header
    ]
    if not signal_candidates:
        signal_candidates = [column for column in unique_headers if column != time_header]
    if not signal_candidates:
        raise ValueError("General mode requires at least one signal column.")

    signal_header = signal_candidates[0]
    return unique_headers, time_header, signal_header


def detect_general_headers(folder_path: str) -> tuple[str, str]:
    _, time_header, signal_header = collect_general_header_options(folder_path)
    return time_header, signal_header


def detect_general_window_bounds(folder_path: str, time_header: str) -> tuple[float, float]:
    csv_files = list_csv_files(folder_path)
    if not csv_files:
        raise ValueError("No CSV files were found in the selected folder.")

    min_value = None
    max_value = None
    matched_header = None

    for sample_file in csv_files:
        sample_path = os.path.join(folder_path, sample_file)
        preview = pd.read_csv(sample_path)
        cleaned_columns = {_clean_header_name(column): column for column in preview.columns}
        if time_header not in cleaned_columns:
            continue

        source_header = cleaned_columns[time_header]
        series = pd.to_numeric(preview[source_header], errors="coerce").dropna()
        if series.empty:
            continue

        matched_header = time_header
        series_min = float(series.min())
        series_max = float(series.max())
        min_value = series_min if min_value is None else min(min_value, series_min)
        max_value = series_max if max_value is None else max(max_value, series_max)

    if matched_header is None or min_value is None or max_value is None:
        raise ValueError(f"Could not calculate window bounds for time header '{time_header}'.")

    return min_value, max_value


@dataclass
class IntegrationConfiguration:
    mode: str = "HPLC"
    input_folder: str = ""
    schema_type: str = "multi_channel"
    fid_peak_integration_method: str = "asymmetric"
    fid_window_xmin: float | None = None
    fid_window_xmax: float | None = None
    fid_window_ymin: float | None = None
    fid_window_ymax: float | None = None
    time_column: str = "RT (min)"
    signal_columns: list[str] = field(default_factory=list)
    general_time_header: str = ""
    general_signal_header: str = ""
    gdgt_meta_set: dict = field(default_factory=load_saved_gdgt_meta)
    general_compounds: list[str] = field(default_factory=list)
    general_window: list[float] = field(default_factory=lambda: [0.0, 60.0])
    peak_neighborhood_n: int = 10
    smoothing_window: int = 9
    smoothing_factor: int = 3
    gaus_iterations: int = 4000
    minimum_peak_amplitude: float | None = None
    maximum_peak_amplitude: float | None = None
    peak_boundary_derivative_sensitivity: float = 0.01
    peak_prominence: float = 2.0
    cheers: bool = False
    debug: bool = False
    normalize_by_standard: bool = True
    reference_trace: str = ""
    reference_compound: str = ""
    reference_window: list[float] = field(default_factory=lambda: [10.0, 30.0])
    use_asymmetric_peak_integration: bool = False
    enable_peak_deconvolution: bool = True
    clip_negative_amplitudes: bool = True


def _coerce_optional_float(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_persisted_integration_configuration() -> IntegrationConfiguration:
    config = IntegrationConfiguration()
    saved = load_integration_configuration_memory()
    valid_fields = {item.name: item for item in fields(IntegrationConfiguration)}
    for name, value in saved.items():
        if name not in valid_fields:
            continue
        current = getattr(config, name)
        if name.startswith("fid_window_"):
            setattr(config, name, _coerce_optional_float(value))
        elif isinstance(current, bool):
            setattr(config, name, bool(value))
        elif isinstance(current, int) and not isinstance(current, bool):
            try:
                setattr(config, name, int(value))
            except (TypeError, ValueError):
                pass
        elif isinstance(current, float):
            try:
                setattr(config, name, float(value))
            except (TypeError, ValueError):
                pass
        elif isinstance(current, list):
            if isinstance(value, list):
                setattr(config, name, value)
        elif isinstance(current, dict):
            if isinstance(value, dict):
                setattr(config, name, value)
        elif isinstance(current, str):
            if isinstance(value, str):
                setattr(config, name, value)
        elif current is None:
            setattr(config, name, value)
    if config.fid_peak_integration_method not in {"asymmetric", "multi", "asymmetric_or_multi"}:
        config.fid_peak_integration_method = "asymmetric"
    return config


def remember_integration_configuration(config: IntegrationConfiguration) -> None:
    payload = asdict(config)
    save_integration_configuration_memory(payload)


def refresh_integration_config(config: IntegrationConfiguration) -> IntegrationConfiguration:
    if not config.input_folder:
        return config

    if config.mode == "FID":
        config.schema_type = "fid_text"
        return config

    schema = detect_data_schema(config.input_folder)
    config.schema_type = schema.schema_type
    config.time_column = schema.time_column
    config.signal_columns = schema.signal_columns

    if config.mode == "General":
        time_header, signal_header = detect_general_headers(config.input_folder)
        if not config.general_time_header:
            config.general_time_header = time_header
        if not config.general_signal_header:
            config.general_signal_header = signal_header
        if not config.general_compounds:
            config.general_compounds = [config.general_signal_header]
        if not config.reference_trace:
            config.reference_trace = config.general_signal_header
        if not config.reference_compound:
            config.reference_compound = config.general_compounds[0]
        if config.reference_window == [10.0, 30.0]:
            config.reference_window = list(config.general_window)
        return config

    if not config.reference_trace and config.gdgt_meta_set["Trace"]:
        config.reference_trace = config.gdgt_meta_set["Trace"][0][0]
    if not config.reference_compound:
        reference_mapping = config.gdgt_meta_set["GDGT_dict"][0].get(config.reference_trace, "")
        config.reference_compound = reference_mapping[0] if isinstance(reference_mapping, list) else str(reference_mapping)
    if config.reference_window == [10.0, 30.0] and config.gdgt_meta_set["window"]:
        config.reference_window = list(config.gdgt_meta_set["window"][0])
    return config


def summarize_gdgt_meta(gdgt_meta_set: dict) -> str:
    lines = []
    for name_group, traces, window in zip(
        gdgt_meta_set["names"], gdgt_meta_set["GDGT_dict"], gdgt_meta_set["window"]
    ):
        group_name = name_group[0] if isinstance(name_group, list) else str(name_group)
        traces_text = []
        for trace_id, compounds in traces.items():
            label = ", ".join(compounds) if isinstance(compounds, list) else str(compounds)
            traces_text.append(f"{trace_id}: {label}")
        lines.append(f"{group_name} [{window[0]}, {window[1]} min]")
        lines.append(f"  {' | '.join(traces_text)}")
    return "\n".join(lines)


def summarize_integration_configuration(config: IntegrationConfiguration) -> str:
    refresh_integration_config(config)
    lines = [
        f"Mode: {config.mode}",
        f"Input folder: {config.input_folder or 'Not set'}",
        f"Schema: {config.schema_type}",
        f"Peak neighborhood size: {config.peak_neighborhood_n}",
        f"Smoothing window: {config.smoothing_window}",
        f"Smoothing factor: {config.smoothing_factor}",
        f"Gaussian iterations: {config.gaus_iterations}",
        f"Minimum peak amplitude: {config.minimum_peak_amplitude if config.minimum_peak_amplitude is not None else 'Auto (legacy)'}",
        f"Peak prominence: {config.peak_prominence}",
    ]

    if config.mode == "General":
        asym_enabled = (not config.enable_peak_deconvolution) and config.use_asymmetric_peak_integration
        lines.extend(
            [
                f"Time header: {config.general_time_header or 'Not set'}",
                f"Signal header: {config.general_signal_header or 'Not set'}",
                f"Compound names: {', '.join(config.general_compounds) if config.general_compounds else 'Not set'}",
                f"Asymmetric peak fitting: {'On' if asym_enabled else 'Off'}",
                f"Peak deconvolution: {'On' if config.enable_peak_deconvolution else 'Off'}",
                f"Clip negatives to zero: {'On' if config.clip_negative_amplitudes else 'Off'}",
                "Time normalization: Disabled",
            ]
        )
    elif config.mode == "HPLC":
        lines.extend(
            [
                f"Time normalization: {'On' if config.normalize_by_standard else 'Off'}",
                "",
                "Configured channels and compounds:",
                summarize_gdgt_meta(config.gdgt_meta_set),
            ]
        )
    else:
        axis_values = [
            config.fid_window_xmin,
            config.fid_window_xmax,
            config.fid_window_ymin,
            config.fid_window_ymax,
        ]
        def axis_value_text(value):
            return "auto" if value is None else str(value)
        axis_text = (
            "Full chromatogram"
            if all(value is None for value in axis_values)
            else f"x=({axis_value_text(config.fid_window_xmin)}, {axis_value_text(config.fid_window_xmax)}), "
                 f"y=({axis_value_text(config.fid_window_ymin)}, {axis_value_text(config.fid_window_ymax)})"
        )
        lines.extend(
            [
                f"Peak integration method: {fid_peak_integration_method_label(config.fid_peak_integration_method)}",
                f"Peak selection window: {axis_text}",
                "Select the folder containing the exported .txt chromatograms.",
            ]
        )
    return "\n".join(lines)


def fid_peak_integration_method_label(method: str) -> str:
    labels = {
        "asymmetric": "Asymmetric",
        "multi": "Multi-Gaussian",
        "asymmetric_or_multi": "Asymmetric or MultiGaussian",
    }
    return labels.get(method, method)


def build_general_summary(config: IntegrationConfiguration) -> str:
    meta = build_single_channel_meta(
        signal_column=config.general_signal_header or "signal",
        compound_names=config.general_compounds,
        window_bounds=config.general_window,
    )
    return summarize_gdgt_meta(meta)


def run_data_conversion(input_folder: str, output_folder: str | None = None, data_type: str = "HPLC"):
    if not input_folder:
        raise ValueError("Select the folder that contains the raw data to convert.")
    input_path = Path(input_folder).expanduser()
    if not input_path.is_dir():
        raise ValueError(f"Provided path is not a valid directory: {input_path}")
    output_path = Path(output_folder).expanduser() if output_folder else input_path / "Converted Files"
    if data_type == "IC MS":
        return ic_ms_to_csv(base_path=str(input_path), output_base_path=str(output_path))
    return hplc_to_csv(base_path=str(input_path), output_base_path=str(output_path))


def _resolve_hplc_peak_area_file(output_location: str) -> Path:
    if not output_location:
        raise ValueError("Select the folder containing HPLC output data.")

    output_path = Path(output_location).expanduser()
    if output_path.is_file():
        if output_path.name != "results_peak_area.csv":
            raise ValueError("Select results_peak_area.csv or the folder containing it.")
        return output_path

    if not output_path.is_dir():
        raise ValueError(f"Provided path is not a valid file or directory: {output_path}")

    candidates = [
        output_path / "results_peak_area.csv",
        output_path / "Output_chromatoPy" / "results_peak_area.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise ValueError(
        "Could not find results_peak_area.csv in the selected folder or its Output_chromatoPy subfolder."
    )


def _resolve_hplc_individual_samples_folder(output_location: str) -> tuple[Path, Path]:
    if not output_location:
        raise ValueError("Select the folder containing HPLC output data.")

    output_path = Path(output_location).expanduser()
    if not output_path.is_dir():
        raise ValueError(f"Provided path is not a valid directory: {output_path}")

    candidates = [
        output_path if output_path.name == "Individual Samples" else None,
        output_path / "Individual Samples",
        output_path / "Output_chromatoPy" / "Individual Samples",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate, candidate.parent

    raise ValueError(
        "Could not find the HPLC Individual Samples folder in the selected location."
    )


def _first_number(value, default=np.nan) -> float:
    if isinstance(value, list):
        if not value:
            return float(default)
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _first_fit_axis(value) -> np.ndarray:
    if not isinstance(value, list) or not value:
        return np.asarray([], dtype=float)
    first = value[0]
    if isinstance(first, list):
        return np.asarray(first, dtype=float)
    return np.asarray(value, dtype=float)


def _gaussian_peak_area_distribution(peak: dict, n_draws: int = 1000) -> tuple[float, float, float]:
    area = _first_number(peak.get("Area"), default=0.0)
    params = peak.get("Model Parameters", {})
    amp = _first_number(params.get("Amplitude"))
    cen = _first_number(params.get("Center"))
    wid = _first_number(params.get("Width"))
    amp_unc = _first_number(params.get("Amplitude Unc"))
    cen_unc = _first_number(params.get("Center Unc"))
    wid_unc = _first_number(params.get("Width Unc"))
    x_values = _first_fit_axis(peak.get("Fit", {}).get("x"))

    values = [amp, cen, wid, amp_unc, cen_unc, wid_unc]
    if x_values.size < 2 or not all(np.isfinite(value) for value in values) or wid <= 0:
        return np.nan, area, np.nan

    rng = np.random.default_rng(0)
    draws = rng.normal(
        loc=np.asarray([amp, cen, wid], dtype=float),
        scale=np.asarray([amp_unc, cen_unc, wid_unc], dtype=float),
        size=(n_draws, 3),
    )
    draws[:, 0] = np.clip(draws[:, 0], 0.0, None)
    draws[:, 2] = np.clip(draws[:, 2], np.finfo(float).eps, None)

    areas = []
    for batch_start in range(0, n_draws, 100):
        batch = draws[batch_start : batch_start + 100]
        scaled_x = (x_values[None, :] - batch[:, 1, None]) / batch[:, 2, None]
        y_values = batch[:, 0, None] * np.exp(-0.5 * scaled_x * scaled_x)
        batch_areas = simpson(y_values, x=x_values, axis=1)
        areas.extend(np.maximum(batch_areas, 0.0))

    low, median, high = np.nanpercentile(np.asarray(areas, dtype=float), [2.5, 50.0, 97.5])
    return float(low), float(median), float(high)


def _iter_hplc_peak_entries(sample_data: dict):
    for group_name, group_data in sample_data.items():
        if group_name == "Sample Name" or not isinstance(group_data, dict):
            continue
        for peak_name, peak_data in group_data.items():
            if isinstance(peak_data, dict):
                yield str(peak_name), peak_data
            elif isinstance(peak_data, (int, float)) and float(peak_data) == 0.0:
                yield str(peak_name), None


def calculate_hplc_peak_area_confidence_intervals(output_location: str) -> dict:
    samples_folder, output_dir = _resolve_hplc_individual_samples_folder(output_location)
    sample_files = sorted(samples_folder.glob("*.json"))
    if not sample_files:
        raise ValueError(f"No sample JSON files were found in {samples_folder}.")

    rows = []
    peak_order = []
    seen_peaks = set()
    for sample_file in sample_files:
        with sample_file.open("r", encoding="utf-8") as handle:
            sample_data = json.load(handle)
        row = {"Sample Name": sample_data.get("Sample Name", sample_file.stem)}
        for peak_name, peak_data in _iter_hplc_peak_entries(sample_data):
            if peak_name not in seen_peaks:
                seen_peaks.add(peak_name)
                peak_order.append(peak_name)
            if peak_data is None:
                low, median, high = 0.0, 0.0, 0.0
            else:
                low, median, high = _gaussian_peak_area_distribution(peak_data)
            row[f"{peak_name} low"] = low
            row[f"{peak_name} median"] = median
            row[f"{peak_name} high"] = high
        rows.append(row)

    columns = ["Sample Name"]
    for peak_name in peak_order:
        columns.extend([f"{peak_name} low", f"{peak_name} median", f"{peak_name} high"])
    df = pd.DataFrame(rows).reindex(columns=columns)
    output_path = output_dir / "chromatopy_peak_area_95ci.csv"
    df.to_csv(output_path, index=False)
    return {
        "input_path": str(samples_folder),
        "peak_area_ci_path": str(output_path),
        "rows": len(df),
        "peaks": len(peak_order),
    }


def calculate_hplc_fractional_abundance(output_location: str) -> dict:
    results_path = _resolve_hplc_peak_area_file(output_location)
    df = pd.read_csv(results_path)
    df_fa = calculate_fa(df)
    output_path = results_path.parent / "chromatopy_fractional_abundance.csv"
    df_fa.to_csv(output_path, index=False)
    return {
        "input_path": str(results_path),
        "fractional_abundance_path": str(output_path),
        "rows": len(df_fa),
    }


def calculate_hplc_indices(output_location: str) -> dict:
    results_path = _resolve_hplc_peak_area_file(output_location)
    df = pd.read_csv(results_path)
    df_fa = calculate_fa(df)
    df_meth, df_cyc = calculate_raberg2021(df)
    df_indices = calculate_indices(df_fa, df_meth, df_cyc)

    output_dir = results_path.parent
    indices_path = output_dir / "chromatopy_indices.csv"
    meth_path = output_dir / "chromatopy_meth_set.csv"
    cyc_path = output_dir / "chromatopy_cyc_set.csv"
    df_indices.to_csv(indices_path, index=False)
    df_meth.to_csv(meth_path, index=False)
    df_cyc.to_csv(cyc_path, index=False)
    return {
        "input_path": str(results_path),
        "indices_path": str(indices_path),
        "meth_set_path": str(meth_path),
        "cyc_set_path": str(cyc_path),
        "rows": len(df_indices),
    }


def count_integration_files(config: IntegrationConfiguration) -> int:
    return integration_file_status(config)["total_files"]


def integration_file_status(config: IntegrationConfiguration) -> dict:
    if not config.input_folder:
        return {"total_files": 0, "processed_files": 0, "results_file_path": ""}
    input_path = Path(config.input_folder).expanduser()
    if not input_path.is_dir():
        raise ValueError(f"Provided path is not a valid directory: {input_path}")
    if config.mode == "FID":
        sample_names = {
            path.stem
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        }
        output_path = input_path / "chromatoPy output"
        sample_data_path = output_path / "Sample Data"
        processed_names = set()
        if sample_data_path.exists():
            for sample_file in sample_data_path.glob("*.json"):
                try:
                    with sample_file.open("r", encoding="utf-8") as handle:
                        sample_data = json.load(handle)
                except Exception:
                    sample_data = {}
                sample_name = sample_data.get("Sample Name", sample_file.stem)
                processed = sample_data.get("Processed Data") if isinstance(sample_data, dict) else None
                if isinstance(processed, dict) and bool(processed):
                    processed_names.add(str(sample_name))
        return {
            "total_files": len(sample_names),
            "processed_files": len(sample_names & processed_names),
            "results_file_path": str(sample_data_path),
        }

    csv_files = list_csv_files(str(input_path))
    sample_names = {Path(filename).stem for filename in csv_files}
    results_file_path = input_path / "Output_chromatoPy" / "results_peak_area.csv"
    processed_names = set()
    if results_file_path.exists():
        try:
            results_df = pd.read_csv(results_file_path)
        except Exception:
            results_df = pd.DataFrame()
        if "Sample Name" in results_df.columns:
            processed_names = {
                str(sample_name).strip()
                for sample_name in results_df["Sample Name"].dropna()
            }
    return {
        "total_files": len(sample_names),
        "processed_files": len(sample_names & processed_names),
        "results_file_path": str(results_file_path),
    }


def run_peak_integration(config: IntegrationConfiguration, message_callback=None, manual_peak_integration: bool = False):
    refresh_integration_config(config)
    if not config.input_folder:
        raise ValueError("Select an input folder for peak integration.")

    if config.mode == "FID":
        return fid_integration(
            folder_path=config.input_folder,
            gaussian_fit_mode=config.fid_peak_integration_method,
            fid_window_limits=(
                config.fid_window_xmin,
                config.fid_window_xmax,
                config.fid_window_ymin,
                config.fid_window_ymax,
            ),
            minimum_peak_amplitude=config.minimum_peak_amplitude,
            maximum_peak_amplitude=config.maximum_peak_amplitude,
            peak_boundary_derivative_sensitivity=config.peak_boundary_derivative_sensitivity,
            peak_prominence=config.peak_prominence,
            peak_neighborhood_n=config.peak_neighborhood_n,
            smoothing_window=config.smoothing_window,
            smoothing_factor=config.smoothing_factor,
            gaus_iterations=config.gaus_iterations,
            manual_peak_integration=manual_peak_integration,
        )

    if config.mode == "General":
        compounds = [name.strip() for name in config.general_compounds if name.strip()]
        if not compounds:
            raise ValueError("General mode requires at least one compound name.")
        remember_compound_history(compounds)
        use_asymmetric_fit = (not config.enable_peak_deconvolution) and config.use_asymmetric_peak_integration
        return hplc_integration(
            folder_path=config.input_folder,
            schema_type="single_channel",
            time_column=config.general_time_header,
            single_channel_compounds=compounds,
            single_channel_signal_column=config.general_signal_header,
            single_channel_window=config.general_window,
            peak_neighborhood_n=config.peak_neighborhood_n,
            smoothing_window=config.smoothing_window,
            smoothing_factor=config.smoothing_factor,
            gaus_iterations=config.gaus_iterations,
            minimum_peak_amplitude=config.minimum_peak_amplitude,
            maximum_peak_amplitude=config.maximum_peak_amplitude,
            peak_boundary_derivative_sensitivity=config.peak_boundary_derivative_sensitivity,
            peak_prominence=config.peak_prominence,
            cheers=config.cheers,
            debug=config.debug,
            normalize_by_standard=False,
            reference_trace=None,
            reference_compound=None,
            reference_window=None,
            use_asymmetric_peak_integration=use_asymmetric_fit,
            enable_peak_deconvolution=config.enable_peak_deconvolution,
            clip_negative_amplitudes=config.clip_negative_amplitudes,
            message_callback=message_callback,
            edit_metadata=False,
        )

    return hplc_integration(
        folder_path=config.input_folder,
        windows=config.gdgt_meta_set["window"],
        gdgt_meta_set=config.gdgt_meta_set,
        peak_neighborhood_n=config.peak_neighborhood_n,
        smoothing_window=config.smoothing_window,
        smoothing_factor=config.smoothing_factor,
        gaus_iterations=config.gaus_iterations,
        minimum_peak_amplitude=config.minimum_peak_amplitude,
        maximum_peak_amplitude=config.maximum_peak_amplitude,
        peak_boundary_derivative_sensitivity=config.peak_boundary_derivative_sensitivity,
        peak_prominence=config.peak_prominence,
        cheers=config.cheers,
        debug=config.debug,
        schema_type="multi_channel",
        time_column=config.time_column,
        normalize_by_standard=config.normalize_by_standard,
        reference_trace=config.reference_trace or None,
        reference_compound=config.reference_compound or None,
        reference_window=config.reference_window,
        message_callback=message_callback,
        edit_metadata=False,
    )


def available_compound_histories() -> list[list[str]]:
    return list_compound_histories()


def remove_compound_history(compounds: list[str]) -> None:
    delete_compound_history(compounds)
