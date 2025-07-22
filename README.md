<h1 align="center"> PreMix: Addressing Label Scarcity in Whole Slide Image Classification with Pre-trained Multiple Instance Learning Aggregators </h1>

**Under Submission Review for Computer Methods and Programs in Biomedicine**

## Overview
- This study introduces *PreMix*, a framework that leverages the Barlow Twins method with a Slide Mixing approach to pre-train MIL aggregators, addressing the scarce labeled WSIs and the underexplored potential of utilizing unlabeled WSIs in weakly supervised classification.
- By integrating Mixup and Manifold Mixup during fine-tuning, *PreMix* effectively manages the gigapixel nature and diverse sizes of WSIs, significantly enhancing classification performance under conditions with limited labeled WSI data.
- Experimental results demonstrate that integrating HIPT into PreMix achieves a 4.7\% mean F1 improvement over the baseline HIPT, showcasing its scalability and effectiveness across varying WSI datasets and labeling scenarios.

<p align="center">
  <img src="figures/common_mil_vs_premix_framework.png" alt="Setting Image" width="1000">
</p>

## Overall Training Process

<p align="center">
  <img src="figures/overall_training_process.png" alt="Setting Image" width="1000">
</p>

To assess the robustness of both the original MIL framework and the proposed *PreMix* framework, we conducted experiments using random sampling and active learning settings with a limited WSI labeled dataset

## Environments
- Windows 10 Enterprise
- 2 NVIDIA RTX 2080 Ti GPUs (11GB each)
- CUDA version: 11.7
- Python version: 3.8.16

## Installation

Install [Anaconda](https://www.anaconda.com/download#)<br>
Create a new environment and activate it

```bash
conda create --name premix python=3.8.16
conda activate premix
````

Install all required packages
```bash
pip install -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Dataset Preparation

Extract square regions from each WSI to create a `pretrain` dataset using [HS2P](https://github.com/clemsgrs/hs2p), which tiles tissue and extracts relevant regions at a given pixel spacing

The results from HS2P should be structured as follows:

<details>
<summary>
Folder structure
</summary>

```bash
<data_dir>/
├── pretrain/
    ├── hs2p_20x_4096/
        ├── debug/
            ├── patches/
                ├── slide_1/
                    ├── 4096
                        ├── jpg
                            ├── slide_1_x1_y1.png
                            ├── slide_1_x2_y2.png
                            ├── ...
                    slide_2/
                    ├── 4096
                        ├── jpg
                            ├── slide_2_x1_y1.png
                            ├── slide_2_x2_y2.png
                            ├── ...
                    ├── ...
                
```
</details>

Download HIPT pre-trained weights using the following commands:

<details>
<summary>
Download commands
</summary>

```
mkdir checkpoints
cd checkpoints
gdown 1Qm-_XrTMYhu9Hl-4FClaOMuroyWlOAxw
gdown 1A2eHTT0dedHgdCvy6t3d9HwluF8p5yjz
```
</details>

## Feature Extraction

Create a configuration file under config/feature_extraction/ inspired by existing files <br>
To extract region-level features, set level: 'global' in the config (refer to `config/feature_extraction/global.yaml`)

Ensure that slides_list.txt contains a list of all slide names in a .txt file:
```
slide_1
slide_2
...
```

Run the following command to initiate feature extraction:

```bash
python extract_features.py --config-name global
```

The results should be structured as follow:

<details>
<summary>
Folder structure
</summary>

```bash
outputs/
├── pretrain/
    ├── features/
        ├── hipt/
            ├── global/
                ├── region/
                    ├── slide_1_x1_y1.pt
                    ├── slide_2_x2_y2.pt
                    ├── ...
                ├── slide/
                    ├── slide_1.pt
                    ├── slide_2.pt
                    ├── ...
```
</details>

## MIL Aggregator Initialization: Barlow Twins Slide Mixing

<p align="center">
  <img src="figures/premix_barlow_twins_slide_mixing_pre-training.png" alt="Setting Image" width="1000">
</p>

**1. Prepapre a csv file inside `data/pretrain/` (refer to `data/pretrain/camelyon16_cptac_ucec.csv`)**

This csv list all the slides for pretraining

```
slide_id
slide_1
slide_2
...
```

**2. Create a configuration file under `config/training/` inspired by existing files**

Refer to `config/training/pretrain.yaml` for inspiration

**3. Pretrain Barlow Twins Slide Mixing**

Run the following command to initiate Barlow Twins Slide Mixing:

```bash
python barlow_twins_slide_mixing.py --config-name pretrain
```

The results should be structured as follows:

<details>
<summary>
Folder structure
</summary>

```bash
outputs/
├── pretrain/
    ├── checkpoints/
        ├── global
            ├── <model_name>_<epoch>.pth
            ├── <model_name>_<epoch>.pth
            ├── ...
```
</details>

## MIL Aggregator Fine-tuning: Mixup and Manifold Mixup

<p align="center">
  <img src="figures/premix_mil_aggregator_architecture.png" alt="Setting Image" width="1000">
</p>

**1. Prepare pool and test csv files for downstream classification**

Refer to `data/camelyon16/pool.csv` and `data/camelyon16/test.csv`  for inspiration

These two csv files list all the slides for downstream classification

```
slide_id,label
slide_1,0
slide_2,1
...
```

**2. Create a configuration file under `config/training/` inspired by existing files**

Refer to `config/training/global.yaml` for inspiration <br>

Note that the `<model_name>` in the config file should be the full name `<model_name>_<epoch>` <br>

Make sure to include the following to integrate slide mixing strategies during fine-tuning and uncomment if they are not needed
```
mixing:
    mixup: True
    manifold_mixup: True
    manifold_mixup_transformer: True
    mixup_alpha: 1
    mixup_alpha_per_sample: False
    mixup_type: random # [random, cosine_sim, class_aware]
...
```

**3. Fine-tuning MIL aggregator with mixup and manifold mixup**

Run the following command to initiate the fine-tuning process:

```bash
python main.py --config-name global
```

The results should be structured as follows:

<details>
<summary>
Folder structure
</summary>

```bash
outputs/
├── <downstream_dataset>
    ├── checkpoints/
        ├── <all_settings_folder>
            ├── <AL_strategy>
                ├── best_model.pth
    ├── results/
        ├── <all_settings_folder>
            ├── <AL_strategy>
                ├── train_0.csv
                ├── train_1.csv
                ├── ...
                ├── test.csv
                ├── test_results.csv
                ├── roc_auc_curve.png
    ├── scripts/
        ├── <all_settings_folder>
            ├── log.txt
            
```
</details>

`best_model.pth` is the best model given the specified settings and AL strategy <br>
`train_0.csv`, `train_1.csv`, etc., contain the predicted probability over classes <br>
`log.txt` contains all the model's performance metrics (ACC, AUC, Precision, Recall) over all AL strategies and training labeled budgets

## Acknowledgement

This codebase builds upon [HIPT](https://github.com/mahmoodlab/HIPT) and [Re-Implementation HIPT](https://github.com/clemsgrs/hipt)
