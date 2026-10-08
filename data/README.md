# Dataset

This directory contains the data-generation notebook used to create the gravitational-wave dataset for this project.

## Data Generation

The dataset is generated using simulated binary black hole (BBH) gravitational-wave signals combined with real detector noise.

The waveform generation uses **PyCBC** and the `IMRPhenomD` waveform approximant. The generated signals are varied over different binary black hole parameters and signal-to-noise ratios (SNRs) to create a diverse labeled dataset.

### Main dataset characteristics

| Property            | Description                             |
| ------------------- | --------------------------------------- |
| Signal type         | Binary black hole (BBH) merger          |
| Waveform model      | IMRPhenomD                              |
| Sampling rate       | 4096 Hz                                 |
| Duration            | 1 second                                |
| Component masses    | Approximately 20–75 M☉                  |
| Detector            | LIGO Hanford (H1)                       |
| Positive class      | GW signal + detector noise              |
| Negative class      | Detector noise only                     |
| Main SNR ranges     | 8–14, 14–20, 20–30                      |
| Number of templates | 80                                      |
| Dataset split       | Template-disjoint train/validation/test |

## Dataset Structure

The final dataset is divided into three splits:

```text
Train       : 3920 samples
Validation  : 840 samples
Test        : 840 samples
```

The class distribution is:

| Split      | Noise-only | GW + Noise | Total |
| ---------- | ---------: | ---------: | ----: |
| Train      |       1120 |       2800 |  3920 |
| Validation |        240 |        600 |   840 |
| Test       |        240 |        600 |   840 |

Here:

* **0 → Noise only**
* **1 → Gravitational wave signal + detector noise**

## Template-Disjoint Splitting

The train, validation, and test sets are separated by waveform template rather than simply splitting individual samples.

The 80 waveform templates are divided approximately as:

```text
Training templates     : 56
Validation templates   : 12
Testing templates      : 12
```

This prevents samples generated from the same underlying waveform template from appearing in both training and testing data.

As a result, the models are evaluated on waveform configurations that they did not directly encounter during training.

## Metadata

Each generated sample is associated with metadata such as:

* `label` — class label
* `template_id` — waveform template identifier
* `mass1` — primary black-hole mass
* `mass2` — secondary black-hole mass
* `snr` — injected signal-to-noise ratio
* `noise_id` — detector-noise segment identifier
* `split` — train/validation/test assignment
* `peak_position` — approximate position of the merger peak

This metadata makes it possible to analyze model performance with respect to waveform parameters and SNR.

## Notebook

The data-generation process is implemented in:

```text
data/
└── data_synthesis.ipynb
```

The notebook covers the generation of BBH waveforms, selection of detector noise, signal injection, SNR control, labeling, and dataset organization.

## Large Data Files

The generated HDF5 datasets are **not included in this GitHub repository** because of their large size.

To reproduce the dataset:

1. Open `data_synthesis.ipynb`.
2. Install the dependencies listed in the root `requirements.txt`.
3. Obtain the required LIGO detector noise data.
4. Run the waveform-generation and injection cells.
5. Save the resulting datasets in HDF5 format.
6. Update the paths in the machine-learning notebooks to point to the generated files.

A typical local data directory can be organized as:

```text
data/
├── data_synthesis.ipynb
├── train.h5
├── validation.h5
└── test.h5
```

The exact filenames may be changed according to the output configuration used during dataset generation.

## Important Note

The dataset generation process uses **real detector noise** together with simulated gravitational-wave signals. Therefore, reproducing the exact dataset may require obtaining the same or equivalent H1 noise segments used during the original experiment.

The machine-learning notebooks should therefore be treated as experiments built on the generated HDF5 datasets rather than assuming that the large datasets are stored inside this repository.
