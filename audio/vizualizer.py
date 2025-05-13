import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from pprint import pprint
from typing import Callable

import librosa
import numpy as np
import scipy
import sounddevice as sd
import soundfile as sf
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from notes import get_notes, Note


def normalize(arr):
    peak = np.max(arr)
    if peak < 1e-9:
        return np.zeros_like(arr)
    return arr / peak


def clamp(val, min_val=0.0, max_val=1.0):
    return min(max(val, min_val), max_val)


def get_freq_bin_edges(freqs: list[float]):
    bin_freqs = np.array(freqs)
    mid_points = (bin_freqs[:-1] + bin_freqs[1:]) / 2

    left_edge = 2 * bin_freqs[0] - mid_points[0]
    right_edge = 2 * bin_freqs[-1] - mid_points[-1]

    bin_edges = np.r_[left_edge, mid_points, right_edge]

    return bin_edges


class Recorder:
    def __init__(
            self,
            sample_rate: int = 44100,
            buffer_dur_sec: float = 0.09,
            frame_dur_sec: float = 0.05
    ):
        self.sample_rate: int = sample_rate
        self.buffer_dur_sec: float = buffer_dur_sec
        self.frame_dur_sec: float = frame_dur_sec

        self.frame_size: int = int(self.sample_rate * self.frame_dur_sec)
        self.buffer_size: int = int(self.sample_rate * self.buffer_dur_sec)

        self.hanning_window = np.hanning(self.buffer_size)

        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)

    def record_samples_callback(self, indata, frames, time, status):
        self.buffer[:-frames] = self.buffer[frames:]
        self.buffer[-frames:] = indata[:, 0]

    def get_input_stream(self):
        return sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            callback=self.record_samples_callback
        )

    def get_buffer(self, windowed: bool = True):
        if windowed:
            return self.buffer * self.hanning_window

        return self.buffer


class Viz:
    def __init__(
            self,
            figsize=(14, 8),
            min_note: str = 'C1',
            max_note: str = 'C7',

    ):
        self.notes: list[Note] = get_notes(min_note=min_note, max_note=max_note)
        self.num_notes: int = len(self.notes)

        self.fig, (self.wave_ax, self.fft_ax) = plt.subplots(
            nrows=2,
            figsize=figsize
        )
        plt.tight_layout()

        self.recorder: Recorder = Recorder()

    def display(self, callback_fn):
        with self.recorder.get_input_stream():
            interval_ms = self.recorder.frame_dur_sec * 1000
            ani = FuncAnimation(
                self.fig,
                callback_fn,
                interval=interval_ms,
                blit=True,
                cache_frame_data=False
            )
            plt.show()

    def viz_wave(
            self,
            waveform_alpha: float = 0.35,
            fft_alpha: float = 0.35
    ):
        # Make room for sliders
        plt.subplots_adjust(right=0.75)
        wv_slider_ax = plt.axes([0.8, 0.25, 0.03, 0.65])  # [left, bottom, width, height]
        wv_slider = Slider(
            ax=wv_slider_ax,
            label='Wv alpha',
            valmin=0,
            valmax=1,
            valstep=0.01,
            valinit=waveform_alpha,
            orientation='vertical'
        )
        fft_slider_ax = plt.axes([0.85, 0.25, 0.03, 0.65])  # [left, bottom, width, height]
        fft_slider = Slider(
            ax=fft_slider_ax,
            label='FFT alpha',
            valmin=0,
            valmax=1,
            valstep=0.01,
            valinit=fft_alpha,
            orientation='vertical'
        )

        # Waveform
        t_x_axis = np.arange(self.recorder.buffer_size) / self.recorder.sample_rate
        prev_waveform = np.zeros_like(t_x_axis)

        waveform, = self.wave_ax.plot(
            t_x_axis,
            prev_waveform
        )
        self.wave_ax.set_ylim(-1, 1)
        self.wave_ax.set_xlim(0, self.recorder.buffer_dur_sec)

        # FFT
        note_freqs = [_.freq for _ in self.notes]
        bin_edges = get_freq_bin_edges(freqs=note_freqs)
        bin_widths = np.diff(bin_edges) * 0.7
        prev_fft = np.zeros_like(note_freqs)

        fft = self.fft_ax.bar(
            note_freqs,
            prev_fft,
            width=bin_widths,
            align='center',
            color='purple',
            alpha=1
        )
        self.fft_ax.set_xscale('log')
        self.fft_ax.set_xlim(bin_edges[0], bin_edges[-1])
        self.fft_ax.set_xticks(note_freqs)
        note_names = [_.note_name for _ in self.notes]
        self.fft_ax.set_xticklabels(note_names, rotation=90)
        self.fft_ax.set_ylim(-1, 1)

        self.fft_ax.set_facecolor('black')

        def update_waveform(_y, _wv_alpha):
            # Waveform
            y_norm = np.zeros_like(_y)

            peak = np.max(np.abs(_y))
            if peak > 1e-2:
                y_norm = _y / peak

            prev_waveform[:] = _wv_alpha * y_norm + (1 - _wv_alpha) * prev_waveform
            waveform.set_ydata(prev_waveform)

        def update_fft(
                _y,
                _fft_alpha,
        ):
            y_harm, y_perc = librosa.effects.hpss(_y)

            binned_cqt = np.abs(
                librosa.hybrid_cqt(
                    y_harm,
                    fmin=note_freqs[0],
                    n_bins=self.num_notes,
                    bins_per_octave=12,
                    pad_mode='reflect',
                    tuning=-12
                )
            )
            cqt = np.mean(
                binned_cqt,
                axis=1
            )
            # cqt = normalize((cqt+1) ** 10)

            # Thresh
            cqt[cqt < 1e-4] = 0

            strength = 5.0
            x = np.arange(self.num_notes)
            x_norm = (x - x.min()) / (x.max() - x.min())  # normalize to [0, 1]
            exp_curve = np.exp(strength * x_norm)
            expo = np.maximum(1e-9, (exp_curve - 1) / (np.exp(strength) - 1))

            # CQT db
            top_db = 80
            cqt_db = librosa.amplitude_to_db(cqt, ref=np.max, top_db=top_db) + top_db

            energy = np.sum(cqt_db ** 2)
            if energy > 1e-3:
                cqt_db /= np.max(cqt_db)

            # Peaks mask
            peaks, props = find_peaks(cqt, prominence=0.1)
            n_peaks = 6
            top_n_peaks = peaks[np.argsort(props["prominences"])[-n_peaks:]]

            peaks_mask = np.zeros(len(cqt))
            decay_rate = 0.5
            # print([note_names[i] for i in top_n_peaks])
            for p in top_n_peaks:
                peak_value = max(cqt[p], 0.3)

                peak_decay = peak_value * decay_rate ** np.abs(np.arange(self.num_notes) - p)
                peak_decay[p] = peak_value
                peaks_mask[:] = np.maximum(peaks_mask, peak_decay)

            final_cqt = np.log1p(peaks_mask * cqt_db)
            if np.max(final_cqt) > 1e-9:
                final_cqt /= np.max(final_cqt)

            prev_fft[:] = _fft_alpha * final_cqt + (1 - _fft_alpha) * prev_fft

            for i, (bar, height) in enumerate(zip(fft, prev_fft)):
                bar.set_y(-height)
                bar.set_height(2 * height)

                note = Note(note_names[i])
                bar.set_color(note.get_rgb(_255=False))
                # my_alpha = clamp(height * 5)
                # if isinstance(alpha, float):
                #     bar.set_alpha(alpha)

                w = clamp(bin_widths[i] * (1 - height) + 0.2, min_val=0.25 * bin_widths[i], max_val=bin_widths[i])
                bar.set_width(w)
                bar.set_x(note_freqs[i] - w / 2)

        def update_fig(frame):
            y = self.recorder.get_buffer(windowed=False)

            update_waveform(_y=y, _wv_alpha=wv_slider.val)
            update_fft(_y=y, _fft_alpha=fft_slider.val)

            return [waveform] + list(fft)

        self.display(update_fig)


def main():
    Viz().viz_wave()


if __name__ == "__main__":
    main()
