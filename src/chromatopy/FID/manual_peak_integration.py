# ─── Standard Library ───────────────────────────────────────────────────────────
import os
import re
import sys
import shutil

# ─── Third-Party Libraries ─────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Qt5Agg')
from matplotlib.widgets import TextBox, Button
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
import json

# ─── Qt GUI Toolkit ────────────────────────────────────────────────────────────
from ..qt_compat import (
    QApplication,
    QEventLoop,
    QDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    Qt,
    QVBoxLayout,
    run_application,
)

# ─── Peak Integration ─────────────────────────────────────────────────────────
from .FID_Integration_functions import run_peak_integrator, smoother, hplc_style_baseline, find_valleys, find_peak_neighborhood_boundaries, fit_gaussians
    
def run_peak_integrator_manual(data, key, gi, pk_sns, smoothing_params, max_peaks_for_neighborhood, fp, gaussian_fit_mode, minimum_peak_amplitude=None, peak_prominence=0.001):
    # Setup data
    md = data['Integration Metadata']
    x = pd.Series(data['Samples'][key]['Raw Data'][md['time_column']])
    x = x.fillna(0)
    y = pd.Series(data['Samples'][key]['Raw Data'][md['signal_column']])
    y = y.fillna(0)
    # Subset to x limits: either deduce from a dict of times, or fall back to explicit x-limits
    pdict = md['peak dictionary']
    if isinstance(pdict, dict):
         peak_times = list(pdict.values())
         labels     = list(pdict.keys())
         xmin, xmax = min(peak_times) - 0.4, max(peak_times) + 0.4
    else:
        # we're given just a list of labels
        labels     = pdict
        x0, x1     = md['x limits']
        xmin, xmax = x0 - 0.4, x1 + 0.4
    mask = (x >= xmin) & (x <= xmax)
    xdata = x[mask].reset_index(drop=True)
    ydata = y[mask].reset_index(drop=True)
    ydata[ydata < 0] = 0
    base, min_peak_amp = hplc_style_baseline(xdata, ydata)
    y_bcorr = np.clip(ydata - base, 0, None)
    y_bcorr = smoother(pd.Series(y_bcorr, index=xdata.index), *smoothing_params)
    y_bcorr = pd.Series(y_bcorr, index=xdata.index)
    min_peak_amp = minimum_peak_amplitude if minimum_peak_amplitude is not None else min_peak_amp
    peak_indices, peak_properties = find_peaks(y_bcorr, height=min_peak_amp, prominence=peak_prominence)
    valleys = find_valleys(y_bcorr, peak_indices)
    peak_labels = labels

    # return data
    peak_selector = ManualPeakIntegrator(
        xdata,                   # as a numpy array
        y_bcorr,
        peak_indices,                       # pass the raw peak indices
        peak_properties,                    # needed for neighborhood fitting
        valleys,
        peak_labels,
        smoothing_params,
        pk_sns,
        gi,
        gaussian_fit_mode,
        max_peaks_for_neighborhood,
        sample_name=key,
        output_figure_path=str(fp) + f"/{key}.png")
    # app = QApplication.instance() or QApplication(sys.argv)
    # app.exec_()
    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True
    
    if owns_app:
        peak_selector.fig.show()
        run_application(app)
    else:
        peak_selector.fig.show()
        peak_selector.fig.canvas.draw_idle()
        QApplication.processEvents()
        loop = QEventLoop()
        peak_selector.on_done = loop.quit
        peak_selector.fig.canvas.mpl_connect("close_event", lambda event: loop.quit())
        loop.exec()
    
    
    if peak_selector.force_exit:
        tqdm.write("Manual integration was forcefully exited by the user.")
        raise SystemExit  # or return None, or raise a custom exception
    
    # Save output
    data['Samples'][key]['Processed Data'] = peak_selector.processed_data
    return data


class ManualPeakIntegrator:
    def __init__(self,
                 x, y,
                 peaks,           # <— list/array of peak indices
                 peak_properties,
                 valleys,
                 labels,
                 smoothing_params,
                 pk_sns,
                 gi,
                 gaussian_fit_mode,
                 max_peaks_for_neighborhood,
                 sample_name="",
                 output_figure_path=None,
                 owns_app=False):
        self._owns_app = owns_app
        self.x, self.y = pd.Series(x), pd.Series(y)
        self.valleys = valleys
        self.peaks = np.asarray(peaks)
        self.peak_properties = peak_properties
        self.labels = labels
        self.smoothing_params = smoothing_params
        self.pk_sns = pk_sns
        self.gi = gi
        self.gaussian_fit_mode = gaussian_fit_mode
        self.max_peaks_for_neighborhood = max_peaks_for_neighborhood
        self.sample_name = sample_name
        self.output_figure_path = output_figure_path

        self.index = 0
        self.processed_data = {}
        self.artists_stack = []
        self.click_tolerance = 10/60  # 10 seconds in minutes
        self.finished = False
        self.on_done = None

        # figure + data plot
        self.fig, self.ax = plt.subplots()
        self.ax.axhline(0, c='k')
        self.ax.plot(self.x, self.y, c='k', alpha = 0.6)
        if self.sample_name:
            self.ax.set_title(str(self.sample_name))
        self.text = self.ax.text(
            0.5, 0.95, f"Click peak for: {self.labels[self.index]}",
            transform=self.ax.transAxes, ha='center')
        self.ax.set_xlabel('Retention Time (min)')
        self.ax.set_ylabel("Value (pA)")
        # “Finished” button
        btn_ax = self.fig.add_axes([0.82, 0.02, 0.15, 0.05])
        self.finish_button = Button(btn_ax, "Finished")
        # Now finish accepts the event
        self.finish_button.on_clicked(self.finish)
        
        # Exit button
        exit_ax = self.fig.add_axes([0.82, 0.94, 0.15, 0.05])
        self.exit_button = Button(exit_ax, "Exit")
        self.exit_button.on_clicked(self.exit_program)
        
        # Flag for termination
        self.force_exit = False

        # connect events
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.cid_key   = self.fig.canvas.mpl_connect("key_press_event",   self.on_key)

    def exit_program(self, event=None):
        """Triggered when the Exit button is clicked."""
        self.force_exit = True
        self.text.set_text("Exiting...")
        self.fig.canvas.draw()
        # QApplication.quit()
        if getattr(self, "_owns_app", False):
            app = QApplication.instance()
            if app is not None:
                app.quit()
        plt.close(self.fig)
        
    # def onclick(self, event):
    #     if event.inaxes != self.ax:
    #         return
    #     # Check if peaks are selected
    #     if self.index >= len(self.labels):
    #        msg = "[Manual] All peaks have already been selected. No more selections expected."
    #        try:
    #            tqdm.write(msg)
    #        except Exception:
    #            print(msg)
    #        # (optional) stop processing further clicks
    #        try:
    #            self.fig.canvas.mpl_disconnect(self.cid_click)
    #        except Exception:
    #            pass
    #        return
        
    #     click_time = event.xdata
    #     peak_times = self.x.to_numpy()[self.peaks]
    #     dists = np.abs(peak_times - click_time)
    #     best = dists.argmin()
   
    #     # if the nearest real peak is > tolerance, treat as “no peak”
    #     if dists[best] <= self.click_tolerance:
    #         peak_idx = int(self.peaks[best])
    #     else:
    #         # no valid peak → grey dashed line & record NaN
    #         line = self.ax.axvline(click_time, color='grey', linestyle='--')
    #         self.artists_stack.append([line])
    #         self.processed_data[self.labels[self.index]] = {'Values': [np.nan]}
    #         self._advance_prompt()
    #         return
    #     drawn = []
    #     try:
    #         if self.gaussian_fit_mode in {"multi","both"}:
    #             _, _, neigh = find_peak_neighborhood_boundaries(
    #                 self.x, self.y, self.peaks, self.valleys,
    #                 peak_idx, self.pk_sns,
    #                 peak_properties=self.peak_properties,
    #                 gi=self.gi,
    #                 smoothing_params=self.smoothing_params,
    #                 pk_sns=self.pk_sns)
    #         else:
    #             neigh = [peak_idx]
    #         # print("debug 1")   
    #         x_fit, y_fit, _, area_ensemble, model_params = fit_gaussians(
    #             self.x, self.y, peak_idx, neigh,
    #             self.smoothing_params, self.pk_sns,
    #             gi=self.gi,
    #             mode=self.gaussian_fit_mode)
    #         # print("debug 2")
    #         poly = self.ax.fill_between(x_fit, 0, y_fit, color='red', alpha=0.4)
    #         drawn.append(poly)
    #         self.processed_data[self.labels[self.index]] = {
    #             'Peak Area - median': np.median(area_ensemble),
    #             'Peak Area - mean': np.mean(area_ensemble),
    #             'Peak Area - standard deviation': np.std(area_ensemble, ddof=1),
    #             'Peak Area - number of ensemble members': len(area_ensemble),
    #             'Model Parameters': model_params,
    #             'Retention Time': float(click_time)}
   
    #     except Exception as e:
    #         tqdm.write(f"[Manual Warning] Failed to fit {self.labels[self.index]}: {e}")
    #         line = self.ax.axvline(click_time, color='grey', linestyle='--')
    #         drawn.append(line)
    #         self.processed_data[self.labels[self.index]] = {'Values':[np.nan]}
   
    #     # save for undo, then advance
    #     self.artists_stack.append(drawn)
    #     self._advance_prompt()
    def _fit_color(self, model_params):
        model_name = ""
        if isinstance(model_params, dict):
            model_name = str(model_params.get("name", "")).lower()
        return "blue" if model_name == "asymmetric" else "red"

    def _add_peak_label(self, label, x_fit, y_fit):
        if len(x_fit) == 0 or len(y_fit) == 0:
            return None

        fit_peak_idx = int(np.argmax(y_fit))
        x_peak = float(x_fit[fit_peak_idx])
        y_peak = float(y_fit[fit_peak_idx])
        y_values = self.y.to_numpy(dtype=float)
        y_min, y_max = self.ax.get_ylim()
        y_range = max(y_max - y_min, np.nanmax(y_values) - np.nanmin(y_values), 1e-9)

        local_mask = np.abs(self.x.to_numpy(dtype=float) - x_peak) <= max(self.click_tolerance / 2, 0.03)
        if np.any(local_mask):
            y_peak = max(y_peak, float(np.nanmax(y_values[local_mask])))

        y_text = y_peak + 0.05 * y_range
        if y_text > y_max - 0.04 * y_range:
            self.ax.set_ylim(y_min, y_text + 0.08 * y_range)

        return self.ax.text(
            x_peak,
            y_text,
            str(label),
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5),
            zorder=3)

    def _record_no_peak(self, label, click_time):
        line = self.ax.axvline(click_time, color='grey', linestyle='--')
        self.artists_stack.append([line])
        self.processed_data[label] = {'Values': [np.nan]}
        self._advance_prompt()

    def onclick(self, event):
        if event.inaxes != self.ax:
            return
    
        if self.index >= len(self.labels):
            msg = "[Manual] All peaks have already been selected. No more selections expected."
            try:
                tqdm.write(msg)
            except Exception:
                print(msg)
            return
    
        # Safe current label (use after the guard above)
        current_label = self.labels[self.index]
    
        click_time = event.xdata
        if click_time is None:
            return

        if self.peaks.size == 0:
            self._record_no_peak(current_label, click_time)
            return

        peak_times = self.x.to_numpy()[self.peaks]
        dists = np.abs(peak_times - click_time)
        if dists.size == 0:
            self._record_no_peak(current_label, click_time)
            return
        best = dists.argmin()
    
        # if the nearest real peak is > tolerance, treat as “no peak”
        if dists[best] > self.click_tolerance:
            self._record_no_peak(current_label, click_time)
            return
    
        peak_idx = int(self.peaks[best])
    
        drawn = []
        try:
            if self.gaussian_fit_mode in {"multi", "both", "asymmetric_or_multi"}:
                _, _, neigh = find_peak_neighborhood_boundaries(
                    self.x, self.y, self.peaks, self.valleys,
                    peak_idx, self.max_peaks_for_neighborhood, #self.pk_sns,
                    peak_properties=self.peak_properties,
                    gi=self.gi,
                    smoothing_params=self.smoothing_params,
                    pk_sns=self.pk_sns)
                if not neigh:
                    neigh = [peak_idx]
            else:
                neigh = [peak_idx]
            x_fit, y_fit, area_smooth, area_ensemble, model_params = fit_gaussians(
                self.x, self.y, peak_idx, neigh,
                self.smoothing_params, self.pk_sns,
                gi=self.gi,
                mode=self.gaussian_fit_mode)
            poly = self.ax.fill_between(x_fit, 0, y_fit, color=self._fit_color(model_params), alpha=0.4)
            drawn.append(poly)
            label_artist = self._add_peak_label(current_label, x_fit, y_fit)
            if label_artist is not None:
                drawn.append(label_artist)
    
            self.processed_data[current_label] = {
                'Peak Area - best fit': float(area_smooth),
                'Peak Area - median': float(np.median(area_ensemble)),
                'Peak Area - mean': float(np.mean(area_ensemble)),
                'Peak Area - standard deviation': float(np.std(area_ensemble, ddof=1)),
                'Peak Area - number of ensemble members': int(len(area_ensemble)),
                'Model Parameters': model_params,
                'Retention Time': float(click_time),}
    
        except Exception as e:
            tqdm.write(f"[Manual Warning] Failed to fit {current_label}: {e}")
            line = self.ax.axvline(click_time, color='grey', linestyle='--')
            drawn.append(line)
            self.processed_data[current_label] = {'Values': [np.nan]}
    
        # save for undo, then advance
        self.artists_stack.append(drawn)
        self._advance_prompt()
    
        # If that was the last label, optionally inform user & disconnect clicks
        if self.index >= len(self.labels):
            done_msg = "[Manual] All peaks selected. You can press Finished."
            try:
                tqdm.write(done_msg)
            except Exception:
                print(done_msg)
    
    def _advance_prompt(self):
        """increment index, update the onscreen prompt, redraw."""
        self.index += 1
        if self.index < len(self.labels):
            self.text.set_text(f"Click peak for: {self.labels[self.index]}")
        else:
            self.text.set_text("All peaks selected. Click 'Finished' to proceed.")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        key = event.key.lower()
        if key in ('shift+delete','shift+del','shift+backspace') and self.index > 0:
            # undo
            self.index -= 1
            label = self.labels[self.index]
            self.processed_data.pop(label, None)
            last = self.artists_stack.pop()
            for art in last:
                try: art.remove()
                except: pass
            self.text.set_text(f"Click peak for: {self.labels[self.index]}")
            if self.cid_click is None:
                self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
            self.fig.canvas.draw()

    def finish(self, event=None):
        # disconnect callbacks and close GUI
        self.text.set_text("")
        self.fig.canvas.draw()
        if self.output_figure_path:
            self.fig.savefig(self.output_figure_path, dpi=300)
        if self.cid_click is not None:
            self.fig.canvas.mpl_disconnect(self.cid_click)
            self.cid_click = None
        self.fig.canvas.mpl_disconnect(self.cid_key)
        self.finished=True
        if callable(self.on_done):
            self.on_done()
        plt.close(self.fig)
        # QApplication.quit()
        if getattr(self, "_owns_app", False):
            app = QApplication.instance()
            if app is not None:
                app.quit()
