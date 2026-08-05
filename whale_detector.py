
# ================================================================================ #
# ================================================================================ #
#   Script: whale_detector.py                                                      #
#                                                                                  #
#   Author: Farid Jedari-Eyvazi                                                    #
#   Contact: farid.je@dal.ca                                                       #
#   Organization: Dalhousie University (https://dal.ca/)                           #
#   Department: Mathematics & Statistics                                           #
#   Version: 1.0.0                                                                 #
#   Date: August 2026                                                              #
#                                                                                  #
#   Description: Detect whale vocalizations, detect vessel noise, and compute      #
#                acoustic indices from underwater acoustic recordings.             #
#                                                                                  #
#   Status: Experimental / Under Active Development                                #
#                                                                                  #
#   Intended Use: Research, monitoring, and production applications.               #
#                 This software is distributed WITHOUT ANY WARRANTY,               #
#                 including the implied warranties of MERCHANTABILITY or           #
#                 FITNESS FOR A PARTICULAR PURPOSE.                                #
#                                                                                  #
#   License: GNU General Public License v3.0 or later (GPL-3.0-or-later).          #
#                                                                                  #
#       This program is free software: you can redistribute it and/or modify       #
#       it under the terms of the GNU General Public License as published by       #
#       the Free Software Foundation, either version 3 of the License, or          #
#       (at your option) any later version.                                        #
#                                                                                  #
#       This program is distributed in the hope that it will be useful,            #
#       but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               #
#       GNU General Public License for more details.                               #
#                                                                                  #
#       You should have received a copy of the GNU General Public License          #
#       along with this program. If not, see <https://www.gnu.org/licenses/>.      #
#                                                                                  #
# ================================================================================ #
# ================================================================================ #



# ===================================================================================
# ===================================================================================
# ============================== Import Python Packages =============================
# ===================================================================================
# ===================================================================================

# --- Standard libraries ---
import re
import os
import json
from datetime import datetime
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

# --- Third-party libraries ---
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
from multiprocessing import Pool

# --- MAAD features for acoustic analysis ---
from maad.features import spectral_entropy, frequency_entropy
from maad.sound import spectral_snr

# --- Ketos-specific imports for audio loading and neural networks ---
from ketos.audio.audio_loader import AudioFrameLoader
from ketos.data_handling.parsing import load_audio_representation
from ketos.neural_networks.resnet import ResNetInterface
from ketos.audio.waveform import Waveform
from ketos.neural_networks.dev_utils.detection import (batch_load_audio_file_data,  
    filter_by_threshold)


# ===================================================================================
# ===================================================================================
# ================================ Helper Classes ===================================
# ===================================================================================
# ===================================================================================

# ============================================================
# Class: Batch Object
# ============================================================

class batchObject:
    """
    Container holding all parameters required to process a
    batch of WAV files.

    Instances of this class are passed to worker processes
    during multiprocessing, avoiding long argument lists.
    """

    def __init__(
        self,
        wav_files_path,
        wav_files_list,
        spec_config,
        step,
        ch2use,
        cal_linear,
        notch_filters,
        sos_bandpass,
    ):

        self.wav_files_path = wav_files_path
        self.wav_files_list = wav_files_list
        self.spec_config = spec_config
        self.step = step
        self.ch2use = ch2use
        self.cal_linear = cal_linear
        self.notch_filters = notch_filters
        self.sos_bandpass = sos_bandpass


# ===================================================================================
# ===================================================================================
# =============================== Utility Functions =================================
# ===================================================================================
# ===================================================================================

# ============================================================
# Function: Load Configuration
# ============================================================

def load_config(path: Union[str, Path]) -> dict:
    """
    Load configuration parameters from a JSON file.

    Args:
        path:
            Path to the JSON configuration file.

    Returns:
        A dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError:
            If the configuration file does not exist.

        ValueError:
            If the configuration file contains invalid JSON.
    """
    path = Path(path)

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Configuration file '{}' not found.".format(path)
        ) from exc

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Invalid JSON in configuration file '{}': {}".format(
                path,
                exc,
            )
        ) from exc


# ============================================================
# Function: Extract Recording Date and Time from Filename
# ============================================================

def extract_date_time(
    filename: Union[str, Path],
    date_position: Tuple[int, int],
    time_position: Tuple[int, int],
) -> int:
    """
    Extract date and time information from a filename.

    The function expects the date and time to be encoded as substrings
    within the filename. Supported date formats are YYYYMMDD and YYMMDD.
    Supported time formats are HHMMSS and HHMM.

    Invalid date or time substrings are replaced with default values
    corresponding to 1970-01-01 00:00:00.

    Args:
        filename:
            Filename containing the encoded date and time.

        date_position:
            Inclusive start and end indices of the date substring.

        time_position:
            Inclusive start and end indices of the time substring.

    Returns:
        UNIX timestamp in seconds.

    Notes:
        The resulting timestamp is interpreted using the local timezone
        of the system running the code.
    """
    filename = str(filename)

    date_start, date_end = date_position
    time_start, time_end = time_position

    # The end indices are inclusive, hence the +1 in the slices.
    date_str = filename[date_start:date_end + 1]
    time_str = filename[time_start:time_end + 1]

    # Validate the date format.
    # Supported formats:
    #   YYYYMMDD
    #   YYMMDD
    if not re.fullmatch(r"\d{8}|\d{6}", date_str):
        date_str = "19700101"

    # Validate the time format.
    # Supported formats:
    #   HHMMSS
    #   HHMM
    if not re.fullmatch(r"\d{6}|\d{4}", time_str):
        time_str = "000000"

    date_format = (
        "%Y%m%d"
        if len(date_str) == 8
        else "%y%m%d"
    )

    time_format = (
        "%H%M%S"
        if len(time_str) == 6
        else "%H%M"
    )

    datetime_value = datetime.strptime(
        date_str + time_str,
        date_format + time_format,
    )

    return int(datetime_value.timestamp())


# ============================================================
# Function: Convert Timestamp to Datetime
# ============================================================

def timestamp2datetime(timestamp_: float) -> str:
    """
    Convert a UNIX timestamp to a formatted datetime string.

    Args:
        timestamp_:
            UNIX timestamp in seconds.

    Returns:
        Datetime formatted as "YYYY-MM-DD HH:MM:SS".

    Notes:
        The timestamp is converted using the local timezone of the
        system running the code.
    """
    return datetime.fromtimestamp(timestamp_).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# Function: Apply Score Threshold
# ============================================================

def apply_score_threshold(
    df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Apply a classification threshold to prediction scores.

    Scores associated with class 0 are converted to class-1
    probabilities using ``1 - score``. The label is then reassigned
    according to the specified classification threshold.

    Args:
        df:
            DataFrame containing ``label`` and ``score`` columns.

        threshold:
            Classification threshold. Must be between 0 and 1.

    Returns:
        A copy of the input DataFrame with updated ``score`` and
        ``label`` columns.

    Raises:
        ValueError:
            If ``threshold`` is outside the range [0, 1].

        KeyError:
            If ``label`` or ``score`` is missing from the DataFrame.
    """
    if not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1; got {}.".format(
                threshold
            )
        )

    # Return a copy to avoid modifying the original DataFrame.
    if df.empty:
        return df.copy()

    required_columns = {
        "label",
        "score",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required columns: {}".format(
                sorted(missing_columns)
            )
        )

    result = df.copy()

    # Convert class-0 scores to class-1 probabilities.
    result["score"] = np.where(
        result["label"] == 0,
        1 - result["score"],
        result["score"],
    )

    # Apply the classification threshold.
    result["label"] = (
        result["score"] >= threshold
    ).astype(int)

    return result


# ============================================================
# Function: Design Audio Filter
# ============================================================

def make_filter(
    nyq: float,
    fmin: float,
    fmax: float,
    order: int = 4,
) -> np.ndarray:
    """
    Design a Butterworth low-pass or band-pass filter.

    A low-pass filter is designed when ``fmin <= 0``.
    Otherwise, a band-pass filter is designed between ``fmin``
    and ``fmax``.

    Args:
        nyq:
            Nyquist frequency in Hz.

        fmin:
            Lower cutoff frequency in Hz. Values <= 0 indicate
            that a low-pass filter should be designed.

        fmax:
            Upper cutoff frequency in Hz.

        order:
            Butterworth filter order. Defaults to 4.

    Returns:
        Filter coefficients in second-order-sections (SOS) format.

    Raises:
        ValueError:
            If the cutoff frequencies are invalid.
    """
    if nyq <= 0:
        raise ValueError(
            "Nyquist frequency must be positive; got {}.".format(
                nyq
            )
        )

    if fmax <= 0 or fmax >= nyq:
        raise ValueError(
            "fmax must be greater than 0 and less than "
            "the Nyquist frequency ({}); got {}.".format(
                nyq,
                fmax,
            )
        )

    if fmin > 0 and fmin >= fmax:
        raise ValueError(
            "fmin ({}) must be less than fmax ({}).".format(
                fmin,
                fmax,
            )
        )

    # Design a low-pass filter when no lower cutoff is specified.
    if fmin <= 0:
        return signal.butter(
            order,
            fmax / nyq,
            btype="lowpass",
            output="sos",
        )

    # Otherwise, design a band-pass filter.
    return signal.butter(
        order,
        [
            fmin / nyq,
            fmax / nyq,
        ],
        btype="bandpass",
        output="sos",
    )


# ============================================================
# Function: Design Notch Filters
# ============================================================

def make_notch_filters(
    nyq: float,
    self_noise_freq: List[float],
    band_width: float = 2.0,
    order: int = 4,
) -> List[np.ndarray]:
    """
    Design Butterworth band-stop filters for self-noise frequencies.

    Each filter suppresses a frequency band centered on one of the
    specified self-noise frequencies.

    Args:
        nyq:
            Nyquist frequency in Hz.

        self_noise_freq:
            List of center frequencies in Hz to suppress.

        band_width:
            Width of each notch band in Hz. Defaults to 2 Hz.

        order:
            Butterworth filter order. Defaults to 4.

    Returns:
        A list of filter coefficients in SOS format.

    Raises:
        ValueError:
            If a notch frequency or filter bandwidth is invalid.
    """
    if nyq <= 0:
        raise ValueError(
            "Nyquist frequency must be positive; got {}.".format(
                nyq
            )
        )

    if band_width <= 0:
        raise ValueError(
            "Band width must be positive; got {}.".format(
                band_width
            )
        )

    notch_filters = []

    for freq in self_noise_freq:

        low_freq = (
            freq - band_width / 2
        )

        high_freq = (
            freq + band_width / 2
        )

        # Ensure that the complete notch band is within
        # the valid frequency range.
        if low_freq <= 0 or high_freq >= nyq:
            raise ValueError(
                "Notch band [{}, {}] Hz is outside the valid "
                "frequency range (0, {}) Hz.".format(
                    low_freq,
                    high_freq,
                    nyq,
                )
            )

        sos_notch = signal.butter(
            order,
            [
                low_freq / nyq,
                high_freq / nyq,
            ],
            btype="bandstop",
            output="sos",
        )

        notch_filters.append(
            sos_notch
        )

    return notch_filters


# ============================================================
# Function: Calculate SPL for a Frequency Band
# ============================================================

def get_spl(
    audio: np.ndarray,
    sos_bandpass: np.ndarray,
    notch_filters: Optional[List[np.ndarray]] = None,
) -> float:
    """
    Calculate sound pressure level (SPL) for a filtered audio signal.

    The audio is detrended and filtered using the supplied band-pass
    filter. Optional notch filters can then be applied to suppress
    known self-noise frequencies. A Tukey window is applied before
    calculating the RMS amplitude.

    Args:
        audio:
            One-dimensional audio waveform.

        sos_bandpass:
            Band-pass filter coefficients in SOS format.

        notch_filters:
            Optional list of notch filter coefficients in SOS format.

    Returns:
        SPL value in dB re 1 µPa.

    Notes:
        This function assumes that the audio amplitude is already
        calibrated in units compatible with the 1 µPa reference.
        No explicit calibration factor is applied here.
    """
    # Handle empty audio input.
    if audio.size == 0:
        return -np.inf

    # Remove the DC component from the audio signal.
    audio_detrended = signal.detrend(
        audio,
        type="constant",
    )

    # Apply the main frequency-band filter.
    # sosfiltfilt provides zero-phase filtering.
    filtered_audio = signal.sosfiltfilt(
        sos_bandpass,
        audio_detrended,
    )

    # Apply optional notch filters to suppress known
    # self-noise frequency bands.
    if notch_filters:
        for sos_notch in notch_filters:
            filtered_audio = signal.sosfiltfilt(
                sos_notch,
                filtered_audio,
            )

    # Apply a Tukey window to reduce edge effects and
    # spectral leakage before calculating RMS.
    window = signal.windows.tukey(
        len(filtered_audio),
        alpha=0.1,
    )

    windowed_audio = (
        filtered_audio * window
    )

    # Calculate RMS amplitude.
    rms = np.sqrt(
        np.mean(
            windowed_audio ** 2
        )
    )

    # Return -inf for zero or invalid RMS values.
    if rms <= 0 or not np.isfinite(rms):
        return -np.inf

    # Calculate SPL relative to a 1 µPa reference.
    return 20 * np.log10(
        rms / 1.0
    )


# ============================================================
# Function: Merge Detection Results
# ============================================================

def df_merge(
    df_bio: pd.DataFrame,
    df_vnd: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge biophony and vessel-noise detections based on time overlap.

    For each biophony detection, the vessel-noise detection with the
    largest positive temporal overlap is selected from the same audio
    file. The corresponding vessel label and score are appended to
    the biophony detection.

    Args:
        df_bio:
            DataFrame containing biophony detections.

            Required columns:
                - filename
                - start
                - end
                - label
                - score

        df_vnd:
            DataFrame containing vessel-noise detections.

            Required columns:
                - filename
                - start
                - end
                - label
                - score

    Returns:
        DataFrame containing biophony detections matched with the
        vessel-noise detection having the largest temporal overlap.

    Notes:
        A detection pair is considered overlapping when the overlap
        duration is strictly greater than zero.

        Biophony detections without an overlapping vessel detection
        are not included in the returned DataFrame.
    """
    # Work on copies so that the original DataFrames remain unchanged.
    df_bio = df_bio.copy()

    df_vnd = df_vnd.copy().rename(
        columns={
            "score": "vessel_score",
            "label": "vessel_label",
        }
    )

    # Normalize filenames to avoid mismatches caused by
    # leading or trailing whitespace.
    df_bio["filename"] = (
        df_bio["filename"]
        .astype(str)
        .str.strip()
    )

    df_vnd["filename"] = (
        df_vnd["filename"]
        .astype(str)
        .str.strip()
    )

    # Preserve the original vessel-detection order.
    # This provides a deterministic tie-breaker when two
    # detections have identical overlap durations.
    df_vnd["_coarse_order"] = np.arange(
        len(df_vnd)
    )

    # Join detections belonging to the same audio file.
    merged = df_bio.merge(
        df_vnd,
        on="filename",
        how="left",
        suffixes=(
            "",
            "_vnd",
        ),
    )

    # Calculate temporal intersection between each pair
    # of biophony and vessel detections.
    merged["overlap"] = (
        np.minimum(
            merged["end"],
            merged["end_vnd"],
        )
        -
        np.maximum(
            merged["start"],
            merged["start_vnd"],
        )
    )

    # Retain only detection pairs with positive overlap.
    merged = merged[
        merged["overlap"] > 0
    ]

    # Sort by biophony detection and then by decreasing
    # temporal overlap.
    merged = merged.sort_values(
        by=[
            "filename",
            "start",
            "end",
            "overlap",
            "_coarse_order",
        ],
        ascending=[
            True,
            True,
            True,
            False,
            True,
        ],
    )

    # Keep only the vessel detection with the largest overlap
    # for each biophony detection.
    merged_best = merged.drop_duplicates(
        subset=[
            "filename",
            "start",
            "end",
        ],
        keep="first",
    )

    # Remove temporary columns used during processing.
    result = merged_best.drop(
        columns=[
            "start_vnd",
            "end_vnd",
            "_coarse_order",
        ]
    )

    return result

# ============================================================
# Function: Compute Acoustic Indices
# ============================================================

def compute_acoustic_indices(
    wav_files_path,
    wav_files_list,
    spec_config,
    step,
    ch2use,
    cal_linear,
    notch_filters,
    sos_bandpass,
) -> pd.DataFrame:
    """
    Compute acoustic indices for a collection of WAV files.

    Each audio file is divided into analysis segments using
    ``AudioFrameLoader``. For every segment, the following
    acoustic indices are calculated:

        - Sound Pressure Level (SPL)
        - Signal-to-Noise Ratio (SNR)
        - Frequency Entropy (Hf)
        - Entropy of the Coefficient of Variation (ECV)

    Args:
        wav_files_path:
            Directory containing the WAV files.

        wav_files_list:
            List of WAV filenames to process.

        spec_config:
            Dictionary containing spectrogram parameters,
            including analysis duration and frequency limits.

        step:
            Time interval between consecutive analysis windows.

        ch2use:
            Audio channel to process.

        cal_linear:
            Linear calibration factor used to convert waveform
            amplitudes into calibrated pressure values.

        notch_filters:
            Notch filters applied during SPL computation.

        sos_bandpass:
            Second-order-section coefficients defining the
            band-pass filter used during SPL computation.

    Returns:
        DataFrame containing one row per analysis segment with
        the following columns:

            - filename
            - start
            - end
            - SPL
            - SNR
            - Hf
            - ECV

    Notes:
        The waveform and spectrogram are calibrated before
        computing any acoustic indices.

        Frequency-based indices are computed from the power
        spectrogram.
    """

    # Create an audio loader that sequentially reads audio
    # segments together with their spectrogram representation.
    loader = AudioFrameLoader(
        path=wav_files_path,
        filename=wav_files_list,
        channel=ch2use,
        duration=spec_config["duration"],
        step=step,
        stop=True,
        representation=[Waveform, spec_config["type"]],
        representation_params=[None, spec_config],
    )

    fmin = spec_config["freq_min"]
    fmax = spec_config["freq_max"]

    results = []

    # Process every analysis segment.
    for waveform, spec in tqdm(
        loader,
        desc="Processing audio indices",
        unit="segment",
    ):

        # Convert waveform and spectrogram into calibrated
        # pressure values.
        audio_calibrated = (
            waveform.get_data().astype(np.float32)
            * cal_linear
        )

        spec_calibrated = (
            spec.get_data().T
            * cal_linear
        )

        # Compute Sound Pressure Level.
        spl = get_spl(
            audio_calibrated,
            sos_bandpass,
            notch_filters,
        )

        # Compute Signal-to-Noise Ratio from the power
        # spectrogram.
        _, _, snr, _, _, _ = spectral_snr(
            spec_calibrated**2
        )

        # Compute Frequency Entropy and average over all
        # time frames.
        hf_val, _ = frequency_entropy(
            spec_calibrated**2
        )

        hf = (
            np.mean(hf_val)
            if isinstance(hf_val, np.ndarray)
            else hf_val
        )

        # Build the frequency vector required by the
        # spectral entropy calculation.
        n_freq_bins = spec_calibrated.shape[0]

        freqs = np.linspace(
            fmin,
            fmax,
            n_freq_bins,
        )

        # Compute Entropy of the Coefficient of Variation.
        _, _, ECV, _, _, _ = spectral_entropy(
            spec_calibrated**2,
            freqs,
            flim=(fmin, fmax),
        )

        # Store the computed indices.
        results.append(
            {
                "filename": waveform.filename,
                "start": waveform.offset,
                "end": (
                    waveform.offset
                    + float(spec_config["duration"])
                ),
                "SPL": spl,
                "SNR": snr,
                "Hf": hf,
                "ECV": np.mean(ECV),
            }
        )

    return pd.DataFrame(results).round(
        {
            "SPL": 1,
            "SNR": 2,
            "Hf": 7,
            "ECV": 3,
        }
    )


# ============================================================
# Function: Process One Batch
# ============================================================

def compute_indices_batch(batch_obj):
    """
    Compute acoustic indices for a single batch of WAV files.

    Args:
        batch_obj:
            BatchObject containing all parameters required
            for processing the batch.

    Returns:
        DataFrame containing the acoustic indices computed
        for all segments within the batch.
    """

    return compute_acoustic_indices(
        wav_files_path=batch_obj.wav_files_path,
        wav_files_list=batch_obj.wav_files_list,
        spec_config=batch_obj.spec_config,
        step=batch_obj.step,
        ch2use=batch_obj.ch2use,
        cal_linear=batch_obj.cal_linear,
        notch_filters=batch_obj.notch_filters,
        sos_bandpass=batch_obj.sos_bandpass,
    )


# ============================================================
# Function: Run Acoustic Indices on Batches
# ============================================================

def run_acoustic_indices_on_batch(
    wav_files_path,
    wav_files_list,
    spec_config,
    cal_linear,
    notch_filters,
    sos_bandpass,
    step,
    ch2use,
    batch_size,
):
    """
    Compute acoustic indices using multiprocessing.

    The input WAV files are divided into batches. Each batch
    is processed independently by a worker process, allowing
    multiple CPU cores to compute acoustic indices in
    parallel.

    Args:
        wav_files_path:
            Directory containing the WAV files.

        wav_files_list:
            List of WAV filenames to process.

        spec_config:
            Spectrogram configuration dictionary.

        cal_linear:
            Linear calibration factor.

        notch_filters:
            Notch filters applied during SPL computation.

        sos_bandpass:
            Second-order-section coefficients of the
            band-pass filter.

        step:
            Time interval between analysis windows.

        ch2use:
            Audio channel to process.

        batch_size:
            Number of WAV files assigned to each worker.

    Returns:
        DataFrame containing the acoustic indices for all
        processed audio segments.

    Notes:
        One CPU core is reserved for the operating system
        whenever more than one core is available.
    """

    # Create one BatchObject for each file batch.
    BO_list = []

    for i in range(
        0,
        len(wav_files_list),
        batch_size,
    ):

        batch_filelist = wav_files_list[
            i : i + batch_size
        ]

        BO_list.append(
            batchObject(
                wav_files_path=wav_files_path,
                wav_files_list=batch_filelist,
                spec_config=spec_config,
                step=step,
                ch2use=ch2use,
                cal_linear=cal_linear,
                notch_filters=notch_filters,
                sos_bandpass=sos_bandpass,
            )
        )

    # Determine the number of worker processes.
    n_cpu = os.cpu_count() or 1

    # Process batches in parallel while leaving one CPU
    # available for other system tasks.
    with Pool(
        processes=max(1, n_cpu - 1)
    ) as pool:

        mp_result = pool.map(
            compute_indices_batch,
            BO_list,
        )

    # Merge the results from all workers into a single
    # DataFrame.
    return pd.concat(
        mp_result,
        ignore_index=True,
    )


# ===================================================================================
# ===================================================================================
# ================================= Deep Learning ===================================
# ===================================================================================
# ===================================================================================

# ============================================================
# Function: Whale Detector
# ============================================================

def whale_detector(
    wav_files_path,
    wav_files_list,
    spec_config,
    model_path,
    score_thr,
    step,
    ch2use,
    date_position,
    time_position,
    temp_folder=Path("tmp_folder"),
    batch_size=4,
) -> pd.DataFrame:
    """
    Detect whale vocalizations in WAV files using a pre-trained
    ResNet classification model.

    Audio files are processed sequentially and divided into
    analysis segments. Segments are grouped into batches for
    efficient inference. Model predictions are converted into
    whale detections using a fixed classification threshold,
    after which the user-defined score threshold is applied to
    determine the final detection labels.

    Detection start and end timestamps are reconstructed from
    the filename and the segment offsets.

    Args:
        wav_files_path:
            Directory containing the WAV files.

        wav_files_list:
            List of WAV filenames to process.

        spec_config:
            Dictionary containing the spectrogram parameters.

        model_path:
            Path to the trained ResNet model.

        score_thr:
            Classification score threshold used to assign the
            final whale detection label.

        step:
            Time interval between consecutive analysis windows.

        ch2use:
            Audio channel to process.

        date_position:
            Position of the date field within the filename.

        time_position:
            Position of the time field within the filename.

        temp_folder:
            Temporary directory used while loading the model.

        batch_size:
            Number of audio segments processed in each
            inference batch.

    Returns:
        DataFrame containing one row per analysed audio segment
        with the following columns:

            - filename
            - start
            - end
            - start_time
            - end_time
            - whale_label
            - whale_score
    """

    # Load the trained whale-classification model.
    model = ResNetInterface.load(
        model_file=model_path,
        new_model_folder=temp_folder,
    )

    # Create an audio loader that sequentially generates
    # spectrograms for the input WAV files.
    audio_loader = AudioFrameLoader(
        path=wav_files_path,
        filename=wav_files_list,
        channel=ch2use,
        duration=spec_config["duration"],
        step=step,
        stop=True,
        representation=spec_config["type"],
        representation_params=spec_config,
    )

    print(f"Number of audio segments = {audio_loader.num()}")

    # Store detections generated from every batch.
    all_batches = []

    # Create a generator that loads spectrograms in batches.
    batch_generator = batch_load_audio_file_data(
        loader=audio_loader,
        batch_size=batch_size,
    )

    print(f"batch_generator = {batch_generator}")

    # Process one batch of spectrograms at a time.
    for batch_data in batch_generator:

        # Run model inference on the current batch.
        batch_predictions = model.run_on_batch(
            batch_data["data"],
            return_raw_output=True,
        )

        # Organize the model predictions together with the
        # corresponding segment metadata.
        raw_output = {
            "filename": batch_data["filename"],
            "start": batch_data["start"],
            "end": batch_data["end"],
            "score": batch_predictions,
        }

        # Convert prediction scores into detections using the
        # default model threshold.
        batch_detections = filter_by_threshold(
            raw_output,
            threshold=0.5,
        )

        all_batches.append(batch_detections)

    # Combine detections from all processed batches.
    detections = pd.concat(
        all_batches,
        ignore_index=True,
    )

    # Apply the user-defined score threshold to assign the
    # final whale labels.
    detections = apply_score_threshold(
        detections,
        score_thr,
    )

    # Round numeric outputs.
    detections = detections.round(
        {
            "label": 1,
            "score": 3,
        }
    )

    # Extract segment information required to build absolute timestamps.
    filenames = detections["filename"].to_numpy()
    starts = detections["start"].to_numpy()
    ends = detections["end"].to_numpy()

    start_times = []
    end_times = []

    # Convert relative segment offsets into absolute
    # timestamps using the recording start time encoded in
    # the filename.
    for filename, start, end in zip(
        filenames,
        starts,
        ends,
    ):

        base_ts = extract_date_time(
            filename,
            date_position,
            time_position,
        )

        start_times.append(
            timestamp2datetime(base_ts + start)
        )

        end_times.append(
            timestamp2datetime(base_ts + end)
        )

    detections["start_time"] = start_times
    detections["end_time"] = end_times

    # Keep only the required output columns and rename the
    # detection fields to distinguish them from other
    # classifiers.
    detections = (
        detections[
            [
                "filename",
                "start",
                "end",
                "start_time",
                "end_time",
                "label",
                "score",
            ]
        ]
        .rename(
            columns={
                "label": "whale_label",
                "score": "whale_score",
            }
        )
    )

    return detections


# ============================================================
# Function: Vessel Noise Detector
# ============================================================

def vnd_detector(
    wav_files_path,
    wav_files_list,
    vnd_spec_config,
    vnd_model_path,
    score_thr,
    temp_folder=Path("tmp_folder"),
    batch_size=4,
):
    """
    Detect vessel noise in WAV files using a pre-trained
    ResNet classification model.

    Audio files are converted into spectrograms and processed
    in batches. Prediction scores are converted into vessel
    detections using the default model threshold, after which
    the user-defined score threshold is applied to determine
    the final detection labels.

    Args:
        wav_files_path:
            Directory containing the WAV files.

        wav_files_list:
            List of WAV filenames to process.

        vnd_spec_config:
            Dictionary containing the spectrogram parameters.

        vnd_model_path:
            Path to the trained vessel-noise model.

        score_thr:
            Classification score threshold used to assign the
            final vessel-noise detection label.

        temp_folder:
            Temporary directory used while loading the model.

        batch_size:
            Number of audio segments processed in each
            inference batch.

    Returns:
        DataFrame containing one row per analysed audio segment
        with the following columns:

            - filename
            - start
            - end
            - label
            - score
    """

    # Load the trained vessel-noise classification model.
    model = ResNetInterface.load(
        model_file=vnd_model_path,
        new_model_folder=temp_folder,
    )

    # Create an audio loader that sequentially generates
    # spectrograms for the input WAV files.
    audio_loader = AudioFrameLoader(
        path=wav_files_path,
        filename=wav_files_list,
        duration=vnd_spec_config["duration"],
        step=None,
        stop=False,
        representation=vnd_spec_config["type"],
        representation_params=vnd_spec_config,
    )

    print("VND batch_size =", audio_loader.batch_size)

    # Store detections generated from every batch.
    all_batches = []

    # Create a generator that loads spectrograms in batches.
    batch_generator = batch_load_audio_file_data(
        loader=audio_loader,
        batch_size=batch_size,
    )

    # Process one batch of spectrograms at a time.
    for batch_data in batch_generator:

        # Run model inference on the current batch.
        batch_predictions = model.run_on_batch(
            batch_data["data"],
            return_raw_output=True,
        )

        # Organize the model predictions together with the
        # corresponding segment metadata.
        raw_output = {
            "filename": batch_data["filename"],
            "start": batch_data["start"],
            "end": batch_data["end"],
            "score": batch_predictions,
        }

        # Convert prediction scores into detections using the
        # default model threshold.
        batch_detections = filter_by_threshold(
            raw_output,
            threshold=0.5,
        )

        all_batches.append(batch_detections)

    # Combine detections from all processed batches.
    detections = pd.concat(
        all_batches,
        ignore_index=True,
    )

    # Apply the user-defined score threshold to assign the
    # final vessel-noise labels.
    detections = apply_score_threshold(
        detections,
        score_thr,
    )

    # Round numeric outputs.
    detections = detections.round(
        {
            "label": 1,
            "score": 3,
        }
    )

    return detections


# ================================================================
# ================================================================
# ======================= Main Function ==========================
# ================================================================
# ================================================================

# ============================================================
# Function: Main
# ============================================================

def main(args):
    """
    Run the complete whale-detection processing pipeline.

    The pipeline performs the following steps:

        1. Load project and spectrogram configuration files.
        2. Construct the required signal-processing filters.
        3. Detect whale vocalizations using the species-specific
           deep-learning model.
        4. Detect vessel noise using the vessel-noise model.
        5. Compute acoustic indices for every analysed segment.
        6. Merge all detection and acoustic-index results.
        7. Save the combined detections to a CSV file.

    Args:
        args:
            Parsed command-line arguments.

            Required arguments:
                - data_path
                - species

            Optional arguments:
                - filelist
                - model_path
                - vnd_model_path
                - project_config
                - spec_config
                - vnd_spec_config
                - score_thr
                - results_path
                - mode

    Returns:
        None.

    Notes:
        Default configuration and model files are used whenever
        corresponding command-line arguments are omitted.

        Results are written only when at least one detection is
        produced.
    """


    # --------------------------------------------------------
    # Read command-line arguments and determine the locations
    # of models, configuration files and input data.
    # --------------------------------------------------------

    species = args.species
    score_thr = args.score_thr
    mode_ = args.mode
    filelist = args.filelist

    # Wav files path
    wav_files_path = Path(args.data_path)

    # Whale Model path
    if args.model_path is None:
        model_path = Path(f'./model/{species}_model.kt')
    else:
        model_path = Path(args.model_path)

    # VND Model path
    if args.vnd_model_path is None:
        vnd_model_path = Path(f'./model/vnd_model.kt')
    else:
        vnd_model_path = Path(args.vnd_model_path)

    # Project config path
    if args.project_config is None:
        project_config_path = Path(f'./config/project_config.json')
    else:
        project_config_path = Path(args.project_config)

    # Whale detector config path
    if args.spec_config is None:
        spec_config_path = Path(f'./config/{species}_spec_config.json')
    else:
        spec_config_path = Path(args.spec_config) 

    # Vessel detector config path
    if args.spec_config is None:
        vnd_spec_config_path = Path(f'./config/vnd_spec_config.json')
    else:
        vnd_spec_config_path = Path(args.vnd_spec_config)

    # --------------------------------------------------------
    # Load project and detector configuration files.
    # --------------------------------------------------------
    
    # Project detector configuration parameters

    project_config = load_config(project_config_path)
    
    project_name = project_config.get("project_name", "unknown_project")
    step = project_config.get("detection_window_step", 1.5)
    date_position = project_config.get("date_position")
    time_position = project_config.get("time_position")
    ch2use = int(project_config.get("channel_number"))
    cal_dB = project_config.get("hydrophone sensitivity (dB)")
    self_noise_freq = project_config.get("system_noise_frequencies")
    batch_size = project_config.get("batch_size", 4)

    # Whale detector configuration parameters

    spec_config = load_audio_representation(spec_config_path)

    fs = spec_config['rate']
    fmin = spec_config['freq_min']
    fmax = spec_config['freq_max']
    nyq = 0.5 * fs
    if nyq == fmax: fmax -= 1
    cal_linear = 10**(abs(cal_dB) / 20)

    # Load vessel detector configuration parameters
    vnd_spec_config = load_audio_representation(vnd_spec_config_path)

    # --------------------------------------------------------
    # Pre-compute filters required during acoustic-index
    # calculations.
    # --------------------------------------------------------

    # Band/low-pass filter
    sos_bandpass = make_filter(nyq, fmin, fmax)

    # Notch Filters
    if self_noise_freq:
        notch_filters =  make_notch_filters(nyq, self_noise_freq)
    else:
        notch_filters = []

    # --------------------------------------------------------
    # Create the output directory if it does not already
    # exist.
    # --------------------------------------------------------
    
    if args.results_path is None:
        results_dir = Path(f'./output_{species}_{project_name}/')
    else:
        results_dir = Path(args.results_path)

    results_dir.mkdir(parents=True, exist_ok=True)  # ensures directory exists


    # --------------------------------------------------------
    # Display the selected processing configuration.
    # --------------------------------------------------------
    print('')
    print('###########################################################################################')
    print('################################# SPECIFIED CONFIGURATION #################################')
    print('###########################################################################################')

    print(f"Project: {project_name}")
    print(f"Species: {species}")
    print(f"Configuration file: {project_config_path}")
    print(f"Frequency band: {fmin} - {fmax} (Hz)")
    print(f"ML score threshold: {score_thr}")

    print('###########################################################################################')
    print('###########################################################################################')
    print('###########################################################################################')
    print('')


    # Run the deep learning model specified by the 'species' command-line argument 

    tqdm.write("") 
    tqdm.write(">>> Running ML model")

    # --------------------------------------------------------
    # Collect all WAV files that will be analysed.
    # --------------------------------------------------------

    if filelist == None:
        wav_files_list = [p.name for p in Path(wav_files_path).glob("*.wav")]
    else:
        df_wav = pd.read_csv(filelist)
        wav_files_list = [Path(file) for file in os.listdir(wav_files_path) if file in df_wav]
    
    # Call the deep learning model for detection
    print(f"\nProcessing {len(wav_files_list)} WAV files for project '{project_name}'...")

    # --------------------------------------------------------
    # Run the whale detector.
    # --------------------------------------------------------
    df_bio = whale_detector(wav_files_path, wav_files_list, spec_config, model_path, score_thr, step, ch2use, date_position, time_position, temp_folder = Path('tmp_folder'), batch_size = batch_size)
    
    # --------------------------------------------------------
    # Run the vessel-noise detector.
    # --------------------------------------------------------
    df_vnd = vnd_detector(wav_files_path, wav_files_list, vnd_spec_config, vnd_model_path, score_thr, temp_folder = Path('tmp_folder'), batch_size = batch_size)

    # --------------------------------------------------------
    # Compute acoustic indices for every analysed segment.
    # --------------------------------------------------------
    df_acoustic_indices = run_acoustic_indices_on_batch(wav_files_path, wav_files_list, spec_config, cal_linear, notch_filters, sos_bandpass, step, ch2use, batch_size = batch_size)

    # --------------------------------------------------------
    # Merge whale detections with the computed acoustic
    # indices.
    # --------------------------------------------------------
    df_bio_aug = df_bio.merge(
    df_acoustic_indices,
    on=["filename", "start", "end"],
    how="left",
    validate="one_to_one")

    # --------------------------------------------------------
    # Match whale detections with overlapping vessel-noise
    # detections.
    # --------------------------------------------------------
    df_comb = df_merge(df_bio_aug, df_vnd)

    # --------------------------------------------------------
    # Save the combined detections if any were found.
    # --------------------------------------------------------
    if not df_comb.empty:
        df_comb.to_csv(
            results_dir / f"{species}_{project_name}_detections.csv",
            mode=mode_,
            index=False,
        )

# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":

    # Create the command-line argument parser.
    parser = argparse.ArgumentParser(
        description="Process WAV files for audio analysis."
    )

    # --------------------------------------------------------
    # Input data
    # --------------------------------------------------------

    parser.add_argument(
        "data_path",
        type=str,
        help="Specify the path to the directory containing WAV files."
    )

    parser.add_argument(
        "species",
        type=str,
        choices=["bl", "nw", "bh"],
        help=(
            "Species to detect. Choose 'bl' for Beluga, "
            "'nw' for Narwhal, or 'bh' for Bowhead."
        )
    )
    
    parser.add_argument(
        "--filelist",
        type=str,
        default=None,
        help=(
            "Path to a text file containing the names of the "
            "audio files to process. If not provided, all WAV files "
            "in the input directory are processed."
        )
    )

    # --------------------------------------------------------
    # Deep-learning models
    # --------------------------------------------------------

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Path to the trained whale detection model. "
            "Defaults to './model/<species>_model.kt'."
        )
    )

    parser.add_argument(
        "--vnd_model_path",
        type=str,
        default=None,
        help=(
            "Path to the trained vessel-noise detector model. "
            "Defaults to './model/vnd_model.kt'."
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help=(
            "Directory for saving detection results. "
            "Defaults to './output_<species>_<project_name>/'."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="a",
        help=(
            "CSV writing mode: 'w' to overwrite or "
            "'a' to append (default)."
        )
    )

    # --------------------------------------------------------
    # Configuration files
    # --------------------------------------------------------

    parser.add_argument(
        "--project_config",
        type=str,
        default=None,
        help=(
            "Path to the project configuration JSON file. "
            "Defaults to './config/project_config.json'."
        )
    )

    parser.add_argument(
        "--spec_config",
        type=str,
        default=None,
        help=(
            "Path to the whale detector spectrogram "
            "configuration JSON file. Defaults to "
            "'./config/<species>_spec_config.json'."
        )
    )

    parser.add_argument(
        "--vnd_spec_config",
        type=str,
        default=None,
        help=(
            "Path to the vessel-noise detector "
            "spectrogram configuration JSON file. "
            "Defaults to './config/vnd_spec_config.json'."
        )
    )

    # --------------------------------------------------------
    # Detection parameters
    # --------------------------------------------------------

    parser.add_argument(
        "--score_thr",
        type=float,
        default=0.5,
        help=(
            "Minimum prediction score required to classify "
            "a segment as a whale detection "
            "(default: 0.50)."
        )
    )

    # Parse the command-line arguments.
    args = parser.parse_args()

    # Execute the processing pipeline and report any
    # argument or runtime errors.
    try:
        main(args)

    except argparse.ArgumentTypeError as e:
        print(f"Error: {e}")
        parser.print_help()

    except Exception as e:
        print(f"Unexpected error: {e}")
        parser.print_help()
