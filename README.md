# Whale Detector Tool

A Python-based acoustic analysis framework for detecting whale vocalizations, vessel noise, and computing acoustic indices from underwater acoustic recordings. The tool combines deep learning–based detection models with traditional acoustic signal-processing methods to support passive acoustic monitoring (PAM) applications.

---

## Features

The pipeline provides the following capabilities:

- 🐋 **Whale vocalization detection**
  - Deep learning–based classification using trained ResNet models.
  - Supports multiple species-specific detection models.

- 🚢 **Vessel noise detection**
  - Deep learning–based vessel noise classification.

- 📊 **Acoustic index computation**
  - Sound Pressure Level (SPL)
  - Signal-to-Noise Ratio (SNR)
  - Frequency Entropy (Hf)
  - Entropy of the Coefficient of Variation (ECV)

- 🔊 **Signal processing**
  - Band-pass filtering
  - Self-noise notch filtering
  - Acoustic calibration

- ⚡ **Parallel processing**
  - Multi-core CPU processing using Python multiprocessing.
  - Configurable batch processing for improved performance.

---

# Table of Contents

- [License and Disclaimer](#license-and-disclaimer)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Install Dependencies](#install-dependencies)
  - [Required Libraries](#required-libraries)
    - [Python Standard Libraries](#1-python-standard-libraries)
    - [Third-Party Libraries](#2-third-party-libraries)
    - [Acoustic Feature Extraction (MAAD)](#acoustic-feature-extraction-maad)
    - [Deep Learning and Audio Processing (Ketos)](#deep-learning-and-audio-processing-ketos)
- [Usage](#usage)
  - [Positional Arguments](#positional-arguments)
  - [Optional Arguments](#optional-arguments)
- [Configuration Files](#configuration-files)
  - [Project Configuration](#project-configuration)
  - [Spectrogram Configuration Files](#spectrogram-configuration-files)
  - [Batch Size Configuration](#batch-size-configuration)
  - [Configuration Notes](#configuration-notes)
- [Data Processing Workflow](#data-processing-workflow)
- [Repository Structure](#repository-structure)
  - [Directory Overview](#directory-overview)
- [Quick Start](#quick-start)
- [Output Format](#output-format)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

---

# License and Disclaimer

This tool is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html) (GPLv3).

This software is provided **"as-is"**, without warranty of any kind, either express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, or non-infringement.

By using this tool, you acknowledge and accept all risks associated with its use. Please refer to the full GPLv3 license text for additional details regarding usage, modification, and redistribution.

---

# Installation

## Requirements

The detector requires **Python 3.8** to ensure compatibility with the Ketos packages.

The software has been tested with:

- Python 3.8.0
- Ketos-compatible dependencies listed in `requirements.txt`

Python standard libraries are included with Python and do not require separate installation.

---

## Install Dependencies

It is recommended to create a virtual environment before installing the required packages.

### 1. Install Python 3.8

Download and install Python 3.8:

https://www.python.org/downloads/

---

### 2. Create a Virtual Environment

#### Linux/macOS

```bash
python3 -m venv myenv
source myenv/bin/activate
```

#### Windows

```bash
myenv\Scripts\activate
```

---

### 3. Install Required Packages

Install all dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### 4. Deactivate the Virtual Environment

```bash
deactivate
```

---

## Notes

- If a virtual environment is not required, install dependencies directly using:

```bash
pip install -r requirements.txt
```

- Python standard libraries are included with Python and do not require additional installation.

---

# Required Libraries

## 1. Python Standard Libraries

The detector uses the following built-in Python libraries:

- **`argparse`** – Parses command-line arguments.
- **`json`** – Reads and writes JSON configuration files.
- **`os`** – Performs file and directory operations.
- **`re`** – Supports regular expression pattern matching and string processing.
- **`datetime`** – Handles date and time formatting.
- **`multiprocessing` (`Pool`)** – Enables parallel processing to improve performance.
- **`pathlib` (`Path`)** – Provides an object-oriented interface for file system paths.
- **`typing`** – Provides type hints (`List`, `Optional`, `Tuple`, `Union`) for improved code readability and static analysis.

---

## 2. Third-Party Libraries

### Data Processing

- **NumPy (`numpy`)** – Performs numerical computations and array operations.
- **Pandas (`pandas`)** – Reads, writes, and manipulates tabular data such as CSV files.

### Signal Processing

- **SciPy (`scipy.signal`)** – Provides signal-processing functions, including filtering, spectral analysis, and windowing operations.

### Progress Monitoring

- **`tqdm`** – Displays progress bars during long-running processing tasks.

---

## Acoustic Feature Extraction (MAAD)

The detector uses MAAD for acoustic feature extraction:

- **`maad.features.spectral_entropy`**  
  Computes spectral entropy as an acoustic complexity metric.

- **`maad.features.frequency_entropy`**  
  Computes frequency entropy to characterize the distribution of spectral energy.

- **`maad.sound.spectral_snr`**  
  Estimates the spectral signal-to-noise ratio (SNR) of audio recordings.

---

## Deep Learning and Audio Processing (Ketos)

The detector uses Ketos libraries for audio representation, model inference, and detection processing.

### Audio Handling

- **`ketos.audio.audio_loader.AudioFrameLoader`**  
  Loads audio files as sequential frames for processing.

- **`ketos.audio.waveform.Waveform`**  
  Represents waveform data and provides waveform-processing utilities.

- **`ketos.data_handling.parsing.load_audio_representation`**  
  Loads audio representation configurations from JSON files.

### Neural Network Inference

- **`ketos.neural_networks.resnet.ResNetInterface`**  
  Loads and runs ResNet-based deep learning models for whale call detection.

### Detection Utilities

- **`ketos.neural_networks.dev_utils.detection.batch_load_audio_file_data`**  
  Efficiently loads batches of audio data for model inference.

- **`ketos.neural_networks.dev_utils.detection.filter_by_threshold`**  
  Filters model predictions using user-defined confidence thresholds.

  # Usage

Run the detector from the command line using:

```bash
python whale_detector.py <data_path> <species> [options]
```

---

## Positional Arguments

- **`<data_path>`**  
  Path to the directory containing the input WAV files.

- **`<species>`**  
  Whale species to detect.

  Supported species:

  | Code | Species |
  |------|---------|
  | `bl` | Beluga |
  | `nw` | Narwhal |
  | `bh` | Bowhead |

---

## Optional Arguments

### Input Data

- **`--filelist`**  
  Path to a text file containing the names of audio files to process.

  If this option is not specified, all WAV files in the input directory are processed.

---

### Deep Learning Models

- **`--model_path`**  
  Path to the trained whale detection model.

  Default:

  ```text
  ./model/<species>_model.kt
  ```

- **`--vnd_model_path`**  
  Path to the trained vessel noise detection model.

  Default:

  ```text
  ./model/vnd_model.kt
  ```

---

### Output

- **`--results_path`**  
  Directory where detection results are saved.

  Default:

  ```text
  ./output_<species>_<project_name>/
  ```

- **`--mode`**  
  CSV output writing mode:

  - `w` – Overwrite the existing output file.
  - `a` – Append results to the existing output file (default).

---

### Configuration Files

- **`--project_config`**  
  Path to the project configuration JSON file.

  Default:

  ```text
  ./config/project_config.json
  ```

- **`--spec_config`**  
  Path to the whale detector spectrogram configuration file.

  Default:

  ```text
  ./config/<species>_spec_config.json
  ```

- **`--vnd_spec_config`**  
  Path to the vessel noise detector spectrogram configuration file.

  Default:

  ```text
  ./config/vnd_spec_config.json
  ```

---

### Detection Parameters

- **`--score_thr`**  
  Minimum prediction score required to classify an audio segment as a detection.

  Range:

  ```text
  0.0 - 1.0
  ```

  Default:

  ```text
  0.5
  ```

---

# Configuration Files

The detector uses two types of configuration files located in the `config/` directory:

1. **Project configuration**
   - Defines project-specific processing parameters.

2. **Spectrogram configuration**
   - Defines the audio representation parameters required by the trained deep learning models.

The default configuration files are:

```text
config/
├── project_config.json
├── bl_spec_config.json
├── nw_spec_config.json
├── bh_spec_config.json
└── vnd_spec_config.json
```

---

# Project Configuration

The `project_config.json` file defines processing parameters, including:

- Project name
- Hydrophone channel number
- Hydrophone sensitivity
- Detection window step size
- Date and time positions in filenames
- System self-noise frequencies
- Multiprocessing batch size

Example:

```json
{
    "project_name": "Arctic_Project",
    "channel_number": 1,
    "hydrophone sensitivity (dB)": -149.7,
    "detection_window_step": 1.5,
    "date_position": [0, 7],
    "time_position": [9, 14],
    "system_noise_frequencies": [60, 120, 180],
    "batch_size": 8
}
```

By default, the detector loads:

```text
./config/project_config.json
```

A different project configuration can be specified using:

```bash
python whale_detector.py <data_path> <species> \
    --project_config path/to/project_config.json
```

---

# Spectrogram Configuration Files

The spectrogram configuration files define the audio representation used by the trained neural network models.

For whale detection, the appropriate configuration file is automatically selected based on the selected species.

| Species | Configuration File |
|---------|--------------------|
| `bl` (Beluga) | `bl_spec_config.json` |
| `nw` (Narwhal) | `nw_spec_config.json` |
| `bh` (Bowhead) | `bh_spec_config.json` |

The vessel noise detector uses:

```text
config/vnd_spec_config.json
```

---

## Spectrogram Parameters

The spectrogram configuration files contain parameters such as:

- Sampling rate
- Spectrogram window size
- Window overlap
- Frequency range
- Segment duration
- Spectrogram resolution

> **Important**
>
> These parameters must match the values used during model training. Modifying them without retraining the models may reduce detection accuracy or produce invalid results.

Alternative spectrogram configuration files can be supplied using:

```bash
python whale_detector.py <data_path> <species> \
    --spec_config path/to/spec_config.json \
    --vnd_spec_config path/to/vnd_spec_config.json
```

---

# Batch Size Configuration

The multiprocessing batch size is controlled through:

```text
project_config.json
```

Example:

```json
{
    "batch_size": 8
}
```

The batch size controls the number of WAV files assigned to each worker during:

- Acoustic index computation
- Detection processing

Increasing the batch size may improve processing speed on systems with sufficient memory.

Reducing the batch size can reduce memory usage on systems with limited resources.

> **Note**
>
> The `batch_size` parameter is no longer specified through the command line. It is managed exclusively through `project_config.json`.

---

# Configuration Notes

- If no custom configuration paths are provided, the detector automatically loads the default files from the `config/` directory.
- For most deployments, only `project_config.json` requires modification.
- Spectrogram configuration files should generally remain unchanged unless new models have been trained using different spectrogram parameters.

# Data Processing Workflow

The `whale_detector.py` script processes underwater acoustic recordings through a sequence of signal-processing and deep learning stages.

The overall workflow is illustrated below:

```text
                              Input WAV Files
                                    │
                                    ▼
                         Load Project Configuration
                         (project_config.json)
                                    │
                                    ▼
                    Load Spectrogram Configurations
             (species_spec_config.json & vnd_spec_config.json)
                                    │
                                    ▼
                       Build Signal-Processing Filters
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
                 ▼                                     ▼
          Band-pass / Low-pass              Optional notch filters
              filtering                    (system self-noise)
                 │                                     │
                 └──────────────────┬──────────────────┘
                                    │
                                    ▼
                         Select WAV Files to Process
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
          Whale Vocalization Detector       Vessel Noise Detector
              (ResNet Model)                    (ResNet Model)
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                       Compute Acoustic Indices
                                    │
                  ┌────────────────────────────────┐
                  │ • Sound Pressure Level (SPL)   │
                  │ • Signal-to-Noise Ratio (SNR)  │
                  │ • Frequency Entropy (Hf)        │
                  │ • ECV                          │
                  └────────────────────────────────┘
                                    │
                                    ▼
                         Merge Detection Results
                                    │
                                    ▼
                           Export CSV Results
```

---

## Processing Steps

### 1. Load Configuration Files

The detector:

- Loads the project configuration file (`project_config.json`).
- Loads the species-specific spectrogram configuration.
- Loads the vessel noise detector spectrogram configuration.

---

### 2. Prepare Signal Processing

The detector creates the required signal-processing filters:

- Band-pass or low-pass filters.
- Optional notch filters to suppress known system self-noise frequencies.

---

### 3. Select Input Recordings

Input WAV files are selected using one of the following methods:

- If `--filelist` is not provided:
  - All WAV files in the input directory are processed.

- If `--filelist` is provided:
  - Only the listed recordings are processed.

---

### 4. Run Whale Detection

The whale detection pipeline:

1. Generates spectrograms from audio recordings.
2. Performs batch inference using the trained ResNet model.
3. Applies the user-defined score threshold.
4. Generates whale detection events.
5. Reconstructs absolute timestamps from recording filenames.

---

### 5. Run Vessel Noise Detection

The vessel noise detection pipeline:

1. Processes recordings using the vessel noise ResNet model.
2. Applies the detection threshold.
3. Generates vessel noise detection events.

---

### 6. Compute Acoustic Indices

For each processed audio segment, the detector computes:

- Sound Pressure Level (SPL)
- Signal-to-Noise Ratio (SNR)
- Frequency Entropy (Hf)
- Entropy of the Coefficient of Variation (ECV)

Acoustic index computation is parallelized across multiple CPU cores.

---

### 7. Merge Results

The detector combines:

- Whale detections
- Vessel noise detections
- Acoustic indices

Vessel detections are matched with whale detections based on temporal overlap.

---

### 8. Export Results

The combined detection results and acoustic indices are exported as CSV files in the specified output directory.

---

# Repository Structure

The recommended project layout is:

```text
whale_detector/
│
├── whale_detector.py                  # Main detector script
│
├── model/
│   ├── vnd_model.kt                   # Vessel noise detection model
│   ├── bl_model.kt                    # Beluga model
│   ├── nw_model.kt                    # Narwhal model
│   ├── bh_model.kt                    # Bowhead model
│   └── *_model.kt                     # Additional whale models
│
├── config/
│   ├── project_config.json            # Global project configuration
│   ├── vnd_spec_config.json           # Vessel noise spectrogram configuration
│   ├── bl_spec_config.json            # Beluga spectrogram configuration
│   ├── nw_spec_config.json            # Narwhal spectrogram configuration
│   ├── bh_spec_config.json            # Bowhead spectrogram configuration
│   └── *_spec_config.json             # Additional spectrogram configurations
│
├── data/
│   ├── *.wav                          # Input audio files
│   └── *.txt                          # Optional list of audio files to process
│
├── output/                            # Detection results and generated files
│
└── README.md                          # Project documentation
```

---

## Directory Overview

- **`whale_detector.py`**
  - Main entry point for running whale detection.

- **`model/`**
  - Contains trained Ketos (`.kt`) neural network models.

- **`config/`**
  - Stores project and spectrogram configuration files.

- **`data/`**
  - Contains input WAV files.
  - Optionally contains text files listing selected recordings to process.

- **`output/`**
  - Stores detection results, logs, and generated files.

---

# Quick Start

## Process All WAV Files in a Directory

If no file list is provided, all WAV files in the input directory are processed.

Example:

```bash
python whale_detector.py ./data bh
```

This command will:

1. Load all WAV files from `./data`.
2. Run the Bowhead whale detector.
3. Run vessel noise detection.
4. Compute acoustic indices.
5. Save the combined results.

---

## Process Selected WAV Files

A subset of recordings can be processed using the `--filelist` option.

Example:

```bash
python whale_detector.py ./data bh --filelist ./wav_file_list.txt
```

The file list should contain one WAV filename per line:

```text
recording_001.wav
recording_002.wav
recording_003.wav
```

---

# Output Format

The final detection table contains the following fields:

| Column | Description |
|--------|-------------|
| `filename` | Audio filename |
| `start` | Segment start time (seconds) |
| `end` | Segment end time (seconds) |
| `start_time` | Absolute UTC/local timestamp |
| `end_time` | Absolute UTC/local timestamp |
| `whale_label` | Whale detection label |
| `whale_score` | Whale prediction score |
| `vessel_label` | Vessel noise detection label |
| `vessel_score` | Vessel prediction score |
| `SPL` | Sound Pressure Level |
| `SNR` | Signal-to-Noise Ratio |
| `Hf` | Frequency Entropy |
| `ECV` | Entropy of the Coefficient of Variation |

---

# Performance Optimization

The pipeline uses multiprocessing to accelerate acoustic index computation.

The number of workers is automatically determined as:

```text
number_of_workers = CPU_cores - 1
```

One CPU core is reserved for system operations.

---

## Recommendations for Large Datasets

- Increase `batch_size` if sufficient RAM is available.
- Reduce `batch_size` on systems with limited memory.
- Use SSD storage for faster WAV file loading.
- Use `--filelist` to avoid processing unnecessary recordings.

---

# Troubleshooting

## No Detections Produced

Check:

- Model file paths.
- Detection threshold (`--score_thr`).
- Spectrogram configuration files.
- Input audio quality.
- Audio sampling rate compatibility.

---

## Memory Errors

Reduce the batch size in:

```text
config/project_config.json
```

Example:

```json
{
    "batch_size": 4
}
```

A smaller batch size reduces memory consumption during multiprocessing.
