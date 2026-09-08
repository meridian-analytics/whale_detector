# Whale Detector Tool

A Python-based acoustic analysis framework for detecting whale vocalizations, vessel noise, and computing acoustic indices from underwater acoustic recordings. The tool combines deep learning–based detection models with traditional acoustic signal-processing methods to support passive acoustic monitoring (PAM) applications.

### Features

The detector provides the following capabilities:

* 🐋 **Whale vocalization detection** – Deep learning–based detection using species-specific ResNet models.
* 🚢 **Vessel noise detection** – Deep learning–based vessel noise detection.
* 📊 **Acoustic index computation** – Computes Sound Pressure Level (SPL), Signal-to-Noise Ratio (SNR), Frequency Entropy (Hf), and Entropy of the Coefficient of Variation (ECV).
* ⚡ **Parallel processing** – Multi-core processing with configurable batch sizes for improved performance.

---
> [!WARNING]
> **Provided Models Are for Demonstration Only**
>
> The included detection models are provided for demonstration and testing purposes only. They have not been fully optimized or validated for operational deployment, and detection performance may be limited. For research or production applications, users should retrain or fine-tune the models using appropriate datasets for their target environment and species.

---

## Limitation Notice and Practical Considerations!
1- Time Zone: This tool does not handle time zone changes, so users must account for any time zone differences on their own. 

2- The current implementation only supports the following date and time formats in a filename:
- `YYYYMMDD` or `YYMMDD` (Date)
- `HHMMSS` or `HHMM` (Time)
If the filename does not follow one of these formats or is missing, the timestamp will default to January 1st, 1970, at midnight UTC (Unix epoch zero time). On Windows, the timestamp may show an offset depending on the local time zone's difference from UTC (Coordinated Universal Time).

3- Hydrophone Data Format is assumed to be WAV with units in Volts. If the input data use a different unit and/or have been scaled or normalized, the resulting SPL values may not be accurate. Thus, users must account for any unit differences or scaling of the input data via preprocessing.

4- Signal Detrending: Before SPL calculation, a constant detrending (DC filter) is applied to the data. Nonlinear trends are left unchanged, and users are responsible for preprocessing such trends if needed. 

---

# Author

**Farid Jedari-Eyvazi, PhD**

Senior Data Scientist / Machine Learning Engineer

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
- [A Note on Performance Optimization](#a-note-on-performance-optimization)
- [Output](#output)
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

### - Positional Arguments

* **`<data_path>`** – Directory containing the input WAV files.
* **`<species>`** – Target whale species (`bl`: Beluga, `nw`: Narwhal, `bh`: Bowhead).

### - Optional Arguments

| Option                  | Description                                                                                                             | Default                               |
| :---------------------- | :---------------------------------------------------------------------------------------------------------------------- | :------------------------------------ |
| **`--filelist`**        | Text file containing the names of WAV files to process. If omitted, all WAV files in the input directory are processed. | All WAV files                         |
| **`--batch_size`**      | Number of WAV files assigned to each worker during multiprocessing.                                                     | `4`                                   |
| **`--model_path`**      | Path to the trained whale detection model.                                                                              | `./model/<species>_model.kt`          |
| **`--vnd_model_path`**  | Path to the trained vessel noise detection model.                                                                       | `./model/vnd_model.kt`                |
| **`--results_path`**    | Directory for saving detection results.                                                                                 | `./output_<species>_<project_name>/`  |
| **`--mode`**            | CSV output mode (`w` = overwrite, `a` = append).                                                                        | `a`                                   |
| **`--project_config`**  | Path to the project configuration file.                                                                                 | `./config/project_config.json`        |
| **`--spec_config`**     | Path to the whale detector spectrogram configuration file.                                                              | `./config/<species>_spec_config.json` |
| **`--vnd_spec_config`** | Path to the vessel noise detector spectrogram configuration file.                                                       | `./config/vnd_spec_config.json`       |
| **`--score_thr`**       | Minimum prediction score required for a detection (0.0–1.0).                                                            | `0.5`                                 |


The detector uses two types of configuration files located in the `config/` directory:

#### 1. Project Configuration

The **project configuration** (`project_config.json`) defines project-specific processing parameters, including:

- Project name
- Hydrophone channel number
- Hydrophone sensitivity
- Detection window step size
- Date and time positions in filenames
- System self-noise frequencies

> **Note:** To ensure accurate timestamp extraction from filenames, WAV files must include date and time information in one of the supported formats before running the detector.  
>
> The `"date_position"` and `"time_position"` values in `project_config.json` must be adjusted according to the filename format used in your dataset. To define these positions correctly:
>
> - Examine the filename format to identify the location of the date and time fields.
> - Use zero-based indexing to specify the start and end positions of each field.
> - The date field must be a 6- or 8-digit substring (`YYMMDD` or `YYYYMMDD`).
> - The time field must be a 4- or 6-digit substring (`HHMM` or `HHMMSS`).

#### 2. Spectrogram Configuration

The **spectrogram configuration** files define the audio representation parameters required by the trained deep learning models.

For whale detection, the appropriate configuration file is selected automatically based on the specified species:

| Species | Configuration File |
|---------|--------------------|
| `bl` (Beluga) | `bl_spec_config.json` |
| `nw` (Narwhal) | `nw_spec_config.json` |
| `bh` (Bowhead) | `bh_spec_config.json` |

The vessel noise detector uses the `vnd_spec_config.json` configuration file.
> **Important**
>
> The spectrogram configuration parameters must match the values used during model training. Modifying them without retraining the models may reduce detection accuracy or produce invalid results.
> If a frequency limit exceeds the Nyquist frequency, the data will be automatically filtered at the Nyquist frequency. Calculations, such as SPL, will then reflect values only up to the Nyquist frequency.

#### Notes

- If no custom configuration paths are provided, the detector automatically loads the default files from the `config/` directory.
- For most deployments, only `project_config.json` requires modification.
- Spectrogram configuration files should generally remain unchanged unless new models have been trained using different spectrogram parameters.

---

# A Note on Performance Optimization

The detector uses multiprocessing to accelerate acoustic index computation. By default, the number of worker processes is set to **CPU cores − 1**, reserving one CPU core for system operations.

The workload assigned to each worker is controlled by the `batch_size` command-line argument. Selecting an appropriate batch size can significantly affect both processing speed and memory usage.

**Recommendations:**

- Increase `batch_size` to improve throughput on systems with sufficient RAM.
- Reduce `batch_size` to lower memory usage on resource-constrained systems. (may decrease processing throughput!)


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

# Data Processing Workflow

The `whale_detector.py` script processes underwater acoustic recordings through a sequence of signal-processing and deep learning stages. The overall workflow is illustrated below:

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


Processing steps include:

1. Load Configuration Files
- Loads the project configuration file (`project_config.json`).
- Loads the species-specific spectrogram configuration.
- Loads the vessel noise detector spectrogram configuration.


2. Prepare Signal Processing
- Creates band-pass or low-pass filters.
- Creates optional notch filters to suppress known system self-noise frequencies.


3. Select Input Recordings
- If `--filelist` is not provided:
  - All WAV files in the input directory are processed.

- If `--filelist` is provided:
  - Only the listed recordings are processed.

4. Run Whale Detection
- Generates spectrograms from audio recordings.
- Performs batch inference using the trained ResNet model.
- Applies the user-defined score threshold.
- Generates whale detection events.
- Reconstructs absolute timestamps from recording filenames.

5. Run Vessel Noise Detection
- Processes recordings using the vessel noise ResNet model.
- Applies the detection threshold.
- Generates vessel noise detection events.

6. Compute Acoustic Indices (parallelized across multiple CPU cores)
- Sound Pressure Level (SPL)
- Signal-to-Noise Ratio (SNR)
- Frequency Entropy (Hf)
- Entropy of the Coefficient of Variation (ECV)


7. Merge Results (based on temporal overlap)
- Whale detections
- Vessel noise detections
- Acoustic indices


8. Export Results (as CSV files in the specified output directory)




