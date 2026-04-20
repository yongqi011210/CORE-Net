# CORE-Net

**CORE-Net: A Collaborative Optimization Framework for Rotated Ship Detection in Complex SAR Scenes**

This repository contains the official implementation of **CORE-Net**, a collaborative optimization framework for **rotated ship detection in complex SAR scenes**. The project is built upon the **Ultralytics YOLO OBB framework** and extends it with dedicated modules for **multi-scale directional consistency modeling, progressive angle regression, orientation-aware regression enhancement, and training-stage sample reliability regulation**.

## Overview

Rotated ship detection in complex synthetic aperture radar (SAR) scenes remains challenging due to three major issues:

- inconsistent directional responses across multi-scale features,
- unstable angle regression in rotated bounding box prediction,
- non-uniform supervision quality of positive samples during training.

To address these problems, CORE-Net introduces a **multi-level collaborative optimization framework** that jointly improves the detection pipeline from three perspectives:

1. **Feature representation**
2. **Regression prediction**
3. **Training regulation**

The proposed framework consists of four key components:

- **RCFP**: Rotation-Consistent Feature Pyramid
- **PCR Head**: Progressive Cascade Rotation Head
- **OAREU**: Orientation-Aware Regression Enhancement Unit
- **UARS**: Uncertainty-Aware Sample Reliability Steering

Together, these modules improve robustness and fine-grained localization quality for rotated ship detection, especially in **complex inshore SAR environments**.

---

## Highlights

- Proposes a collaborative optimization framework for rotated ship detection in complex SAR scenes.
- Introduces **RCFP**, **PCR Head**, and **OAREU** in the forward detection stage.
- Introduces **UARS** as a training-stage reliability regulation strategy.
- Achieves consistent improvements on **RSDD-SAR**, **SSDD+**, and **RSAR**.
- Delivers more pronounced gains under **high-IoU thresholds** and in **inshore scenes**.

---

## Framework

### 1. RCFP: Rotation-Consistent Feature Pyramid
RCFP calibrates directional responses of multi-scale features **before feature fusion**.  
It alleviates cross-level directional conflicts between shallow texture-rich features and deep semantic features, improving directional consistency for subsequent rotated box modeling.

### 2. PCR Head: Progressive Cascade Rotation Head
PCR Head replaces one-shot angle prediction with a **two-stage progressive cascade design**.  
The first stage provides an initial angle estimate, while the second stage refines it through cross-stage attention (CSA), leading to more stable and accurate angle regression.

### 3. OAREU: Orientation-Aware Regression Enhancement Unit
OAREU strengthens directional geometric representation in the **regression branch**.  
It combines current-scale directional enhancement, adjacent-scale auxiliary fusion, and channel recalibration to improve rotated box localization quality.

### 4. UARS: Uncertainty-Aware Sample Reliability Steering
UARS is a **training-stage optimization module**.  
It identifies positive samples with **high classification confidence but low geometric consistency** and softly downweights their contribution **only in the regression branch**, thereby stabilizing training and improving AP50:95.

---

## Main Results

Experiments were conducted on three public SAR ship detection benchmarks:

- **RSDD-SAR**
- **SSDD+**
- **RSAR**

### Dataset Statistics

| Dataset   | Images | Ship Instances |
|-----------|--------|----------------|
| RSDD-SAR  | 7,000  | 10,263         |
| SSDD+     | 1,160  | 2,456          |
| RSAR      | 95,842 | 114,142        |

### Representative Results

#### CORE-Net module ablation
On **RSDD-SAR (near-shore)**:
- Baseline AP50:95: **0.5096**
- CORE-Net (PCR Head + RCFP + OAREU): **0.5353**

On **SSDD+ (near-shore)**:
- Baseline AP50:95: **0.6023**
- CORE-Net (PCR Head + RCFP + OAREU): **0.6364**

#### UARS effectiveness
On **RSAR**:
- YOLOv8-OBB AP50:95: **0.6080**
- CORE-Net AP50:95: **0.6114**
- YOLOv8-OBB + UARS AP50:95: **0.6109**
- CORE-Net + UARS AP50:95: **0.6215**

These results indicate that:
- the proposed forward modules provide complementary gains,
- the benefits are especially clear in **complex near-shore scenes**,
- UARS further improves **high-IoU localization quality**.

---

## Repository Structure

A typical project structure is organized as follows:

```text
CORE-Net/
├── train.py
├── ultralytics/
│   ├── nn/
│   │   ├── tasks.py
│   │   └── modules/
│   │       ├── PCR_Head.py
│   │       └── RCFP.py
├── configs/
│   └── Model.yaml
└── README.md