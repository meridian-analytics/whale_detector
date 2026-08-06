# Whale Detector Tool

A Python-based acoustic analysis framework for detecting whale vocalizations, vessel noise, and computing acoustic indices from underwater acoustic recordings. The tool combines deep learning–based detection models with traditional acoustic signal-processing methods to support passive acoustic monitoring (PAM) applications.

### Features

The detector provides the following capabilities:

* 🐋 **Whale vocalization detection** – Deep learning–based detection using species-specific ResNet models.
* 🚢 **Vessel noise detection** – Deep learning–based vessel noise detection.
* 📊 **Acoustic index computation** – Computes Sound Pressure Level (SPL), Signal-to-Noise Ratio (SNR), Frequency Entropy (Hf), and Entropy of the Coefficient of Variation (ECV).
* ⚡ **Parallel processing** – Multi-core processing with configurable batch sizes for improved performance.

---

# License and Disclaimer

This tool is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html) (GPLv3).

This software is provided **"as-is"**, without warranty of any kind, either express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, or non-infringement.

By using this tool, you acknowledge and accept all risks associated with its use. Please refer to the full GPLv3 license text for additional details regarding usage, modification, and redistribution.

---

# Table of Contents

- [Installation](#installation)
- [Recommended Project Layout](#recommended-project-layout)
- [Usage](#usage)
- [Output](#output)
- [A Note on Performance Optimization](#a-note-on-performance-optimization)
- [Required Libraries](#required-libraries)
- [Configuration Files](#configuration-files)
- [Data Processing Workflow](#data-processing-workflow)

---

# Installation

The detector requires **Python 3.8** to ensure compatibility with the Ketos packages. It is recommended to install the specified version of Python and create a virtual environment before installing the required packages listed in the requirements.txt file.

1. [Download](https://www.python.org/downloads/) and install `Python 3.8.0`
2.	Install virtualenv (if needed; on UNIX-based systems): `sudo apt install python3-venv` 
3.	Create a virtual environment: `python3 -m venv myenv` 
4.	Activate it: `source myenv/bin/activate` on UNIX-based systems OR `myenv\Scripts\activate` on Windows
5.	Install packages: `pip install -r requirements.txt`
6.	Deactivate: `deactivate`

#### Notes

- If a virtual environment is not required, install dependencies directly using:

```bash
pip install -r requirements.txt
```

- Python standard libraries are included with Python and do not require additional installation.

---

# Recommended Project Layout

The project layout is flexible and can be customized to suit specific workflows. However, the following structure is recommended to improve organization, maintain clarity, and ensure compatibility with the default configuration and file-loading behavior.

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
│   └── *.txt                          # Optional text file listing selected recordings to process
│
├── output/                            # Detection results and generated files
|
```

#### Note : The file list should contain one WAV filename per line

---

# Usage

Run the detector from the command line:

```bash
python whale_detector.py <data_path> <species> [options]
```

#### Positional Arguments

* **`<data_path>`** – Directory containing the input WAV files.
* **`<species>`** – Target whale species (`bl`: Beluga, `nw`: Narwhal, `bh`: Bowhead).

#### Optional Arguments

| Option                  | Description                                                                                                             | Default                               |
| :---------------------- | :---------------------------------------------------------------------------------------------------------------------- | :------------------------------------ |
| **`--filelist`**        | Text file containing the names of WAV files to process. If omitted, all WAV files in the input directory are processed. | All WAV files                         |
| **`--model_path`**      | Path to the trained whale detection model.                                                                              | `./model/<species>_model.kt`          |
| **`--vnd_model_path`**  | Path to the trained vessel noise detection model.                                                                       | `./model/vnd_model.kt`                |
| **`--results_path`**    | Directory for saving detection results.                                                                                 | `./output_<species>_<project_name>/`  |
| **`--mode`**            | CSV output mode (`w` = overwrite, `a` = append).                                                                        | `a`                                   |
| **`--project_config`**  | Path to the project configuration file.                                                                                 | `./config/project_config.json`        |
| **`--spec_config`**     | Path to the whale detector spectrogram configuration file.                                                              | `./config/<species>_spec_config.json` |
| **`--vnd_spec_config`** | Path to the vessel noise detector spectrogram configuration file.                                                       | `./config/vnd_spec_config.json`       |
| **`--score_thr`**       | Minimum prediction score required for a detection (0.0–1.0).                                                            | `0.5`                                 |


---

# Output

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
| `overlap` | Temporal overlap duration (seconds) between vessel noise and whale detection segments |


---

# A Note on Performance Optimization

The detector uses multiprocessing to accelerate acoustic index computation. By default, the number of worker processes is set to **CPU cores − 1**, reserving one CPU core for system operations.

The workload assigned to each worker is controlled by the `batch_size` parameter in `config/project_config.json`. Selecting an appropriate batch size can significantly affect both processing speed and memory usage.

**Recommendations:**

- Increase `batch_size` to improve throughput on systems with sufficient RAM.
- Reduce `batch_size` to lower memory usage on resource-constrained systems.
- Use SSD storage to improve WAV file loading performance.
- Use `--filelist` to process only the recordings of interest.

If the detector runs out of memory during processing, reduce the `batch_size` value. Smaller batch sizes reduce memory consumption but may also decrease processing throughput.

---

# Additional Information


## Required Libraries

### Python Standard Libraries
- **`argparse`** – Parses command-line arguments.
- **`json`** – Reads and writes JSON configuration files.
- **`os`** – Performs file and directory operations.
- **`re`** – Supports regular expression pattern matching and string processing.
- **`datetime`** – Handles date and time formatting.
- **`multiprocessing` (`Pool`)** – Enables parallel processing to improve performance.
- **`pathlib` (`Path`)** – Provides an object-oriented interface for file system paths.
- **`typing`** – Provides type hints (`List`, `Optional`, `Tuple`, `Union`) for improved code readability and static analysis.

### Third-Party Libraries

- **NumPy (`numpy`)** – Performs numerical computations and array operations.
- **Pandas (`pandas`)** – Reads, writes, and manipulates tabular data such as CSV files.
- **SciPy (`scipy.signal`)** – Provides signal-processing functions, including filtering, spectral analysis, and windowing operations.
- **`tqdm`** – Displays progress bars during long-running processing tasks.
- **`maad.features.spectral_entropy`** - Computes spectral entropy as an acoustic complexity metric.
- **`maad.features.frequency_entropy`** - Computes frequency entropy to characterize the distribution of spectral energy.
- **`maad.sound.spectral_snr`** - Estimates the spectral signal-to-noise ratio (SNR) of audio recordings.
- **`ketos.audio.audio_loader.AudioFrameLoader`** - Loads audio files as sequential frames for processing.
- **`ketos.audio.waveform.Waveform`** - Represents waveform data and provides waveform-processing utilities.
- **`ketos.data_handling.parsing.load_audio_representation`** - Loads audio representation configurations from JSON files.
- **`ketos.neural_networks.resnet.ResNetInterface`** - Loads and runs ResNet-based deep learning models for whale call detection.
- **`ketos.neural_networks.dev_utils.detection.batch_load_audio_file_data`** - Efficiently loads batches of audio data for model inference.
- **`ketos.neural_networks.dev_utils.detection.filter_by_threshold`** - Filters model predictions using user-defined confidence thresholds.

 
# Configuration Files

The detector uses two types of configuration files located in the `config/` directory:

1. **Project configuration** - Defines project-specific processing parameters.
2. **Spectrogram configuration** - Defines the audio representation parameters required by the trained deep learning models.

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

### Project Configuration

The default `project_config.json` file defines processing parameters, including:

- Project name
- Hydrophone channel number
- Hydrophone sensitivity
- Detection window step size
- Date and time positions in filenames
- System self-noise frequencies
- Multiprocessing batch size

### Spectrogram Configuration Files

The spectrogram configuration files define the audio representation used by the trained neural network models. For whale detection, the appropriate configuration file is automatically selected based on the selected species.

| Species | Configuration File |
|---------|--------------------|
| `bl` (Beluga) | `bl_spec_config.json` |
| `nw` (Narwhal) | `nw_spec_config.json` |
| `bh` (Bowhead) | `bh_spec_config.json` |

The vessel noise detector uses:

```text
config/vnd_spec_config.json
```

> **Important**
>
> These parameters must match the values used during model training. Modifying them without retraining the models may reduce detection accuracy or produce invalid results.

---

### Batch Size Configuration

The multiprocessing batch size is controlled through:

```text
project_config.json
```

The batch size controls the number of WAV files assigned to each worker during:

- Acoustic index computation
- Detection processing

Increasing the batch size may improve processing speed on systems with sufficient memory.

Reducing the batch size can reduce memory usage on systems with limited resources.

#### Notes

- If no custom configuration paths are provided, the detector automatically loads the default files from the `config/` directory.
- For most deployments, only `project_config.json` requires modification.
- Spectrogram configuration files should generally remain unchanged unless new models have been trained using different spectrogram parameters.

---

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
                    Band-pass / Low-pass                 Optional notch filters
                         filtering                         (system self-noise)
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
                                │ • Frequency Entropy (Hf)       │
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

### Processing Steps

#### 1. Load Configuration Files

The detector:

- Loads the project configuration file (`project_config.json`).
- Loads the species-specific spectrogram configuration.
- Loads the vessel noise detector spectrogram configuration.


#### 2. Prepare Signal Processing

The detector creates the required signal-processing filters:

- Band-pass or low-pass filters.
- Optional notch filters to suppress known system self-noise frequencies.


#### 3. Select Input Recordings

Input WAV files are selected using one of the following methods:

- If `--filelist` is not provided:
  - All WAV files in the input directory are processed.

- If `--filelist` is provided:
  - Only the listed recordings are processed.

#### 4. Run Whale Detection

The whale detection pipeline:

1. Generates spectrograms from audio recordings.
2. Performs batch inference using the trained ResNet model.
3. Applies the user-defined score threshold.
4. Generates whale detection events.
5. Reconstructs absolute timestamps from recording filenames.


#### 5. Run Vessel Noise Detection

The vessel noise detection pipeline:

1. Processes recordings using the vessel noise ResNet model.
2. Applies the detection threshold.
3. Generates vessel noise detection events.


#### 6. Compute Acoustic Indices

For each processed audio segment, the detector computes:

- Sound Pressure Level (SPL)
- Signal-to-Noise Ratio (SNR)
- Frequency Entropy (Hf)
- Entropy of the Coefficient of Variation (ECV)

Acoustic index computation is parallelized across multiple CPU cores.


#### 7. Merge Results

The detector combines:

- Whale detections
- Vessel noise detections
- Acoustic indices

Vessel detections are matched with whale detections based on temporal overlap.


#### 8. Export Results

The combined detection results and acoustic indices are exported as CSV files in the specified output directory.



