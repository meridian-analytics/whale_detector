# Whale Detector

A Python-based acoustic analysis framework for detecting whale vocalizations, vessel noise, and computing acoustic indices from underwater acoustic recordings.

The software combines deep-learning-based detection models with traditional acoustic signal-processing methods to support passive acoustic monitoring (PAM) applications.

---

## Features

The pipeline provides the following capabilities:

- 🐋 **Whale vocalization detection**
  - Deep-learning classification using trained ResNet models.
  - Supports multiple species-specific models.

- 🚢 **Vessel noise detection**
  - Deep-learning-based vessel-noise detection.

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
  - Multi-core CPU processing using multiprocessing.
  - Configurable batch processing.

---

# Installation

### Requirements

The detector requires Python 3.8 to ensure compatibility with the Ketos packages.

### Install dependencies

pip install -r requirements.txt

# Repository Structure

The recommended project layout is shown below:

```text
whale_detector/

  ├── whale_detector.py                   # Main detector script
  |
  ├── model/
  │   ├── vnd_model.kt                    # Vessel-noise model
  │   ├── bl_model.kt                     # Beluga whale model
  │   ├── nw_model.kt                     # Narwhal whale model
  │   ├── bh_model.kt                     # Bowhead whale model
  |   └── *_model.kt                      # Additional whale models
  |
  ├── config/
  │   ├── project_config.json             # Global project configuration
  │   ├── vnd_spec_config.json            # Vessel noise detector configuration
  │   ├── bl_spec_config.json             # Beluga whale spectrogram configuration
  │   ├── nw_spec_config.json             # Narwhal spectrogram configuration
  │   ├── bh_spec_config.json             # Bowhead whale spectrogram configuration
  |   └── *_spec_config.json              # Additional spectrogram configurations
  |         
  ├── data/
  │   └── *.wav                           # Input audio files
  │   └── *.txt                           # Text file including audio file list to process
  |
  ├── output/                             # Detection results and generated files

```

### Directory Overview

* **`whale_detector.py`** – Entry point for running whale detection.
* **`model/`** – Contains the trained Ketos (`.kt`) models.
* **`config/`** – Stores project and species-specific configuration files.
* **`data/`** – Place input WAV audio files for processing. Optionally, a list of files to be processed can also be placed in this directory.
* **`output/`** – Generated detection results, logs, and other output files.


# Quick Start
Process all WAV files in a directory

If no file list is provided, all WAV files in the input directory are processed.

Example:

python whale_detector.py ./data 'bh'

This will:
Load all WAV files in ./data
Run the Bowhead whale detector
Run vessel-noise detection
Compute acoustic indices
Save the combined results
Process Selected WAV Files

A subset of recordings can be processed using the --filelist option.

Example:

python whale_detector.py ./data 'bh' --filelist './wav_file_list.txt'

The file list should contain one WAV filename per line:

recording_001.wav
recording_002.wav
recording_003.wav
...



# Batch Processing Configuration

The processing batch size is controlled through:

project_config.json

Example:

{
    "batch_size": 8
}

The batch size controls:

Number of audio files assigned to CPU workers

# Memory usage
Multiprocessing workload



The final detection table contains:

Column	Description
filename	Audio filename
start	Segment start time (seconds)
end	Segment end time (seconds)
start_time	Absolute UTC/local timestamp
end_time	Absolute UTC/local timestamp
whale_label	Whale detection label
whale_score	Whale prediction score
vessel_label	Vessel detection label
vessel_score	Vessel prediction score
SPL	Sound pressure level
SNR	Signal-to-noise ratio
Hf	Frequency entropy
ECV	Entropy coefficient of variation

# Processing Workflow
```text
                 WAV Files
                     |
                     |
          +----------+----------+
          |                     |
          v                     v
   Whale Detector       Vessel Noise Detector
          |                     |
          +----------+----------+
                     |
                     v
          Acoustic Index Calculation
                     |
                     v
             Detection Merging
                     |
                     v
              CSV Output
```

# Performance Optimization

The pipeline uses multiprocessing for acoustic-index calculation.

The number of workers is automatically determined:

number_of_workers = CPU_cores - 1

One CPU core is reserved for system operations.

### For large datasets:

Increase batch_size if sufficient RAM is available.
Reduce batch_size for limited memory systems.
Use SSD storage for faster WAV loading.

# Troubleshooting
No detections produced

Check:

Model path
Detection threshold
Spectrogram configuration
Input audio quality

# Memory errors

Reduce:

{
    "batch_size": 4
}

in:

project_config.json

