> **About this Repository**  
> This repository hosts the complete *Path-IO* computational pathology framework for predicting immunotherapy outcomes in metastatic non–small cell lung cancer (mNSCLC) using H&E whole-slide images (WSIs).  
> The pipeline integrates patch-level tissue classification, WSI habitat mapping, ROI preprocessing, survival modeling, and patient-level risk stratification.  
> It is designed for reproducible, multi-institutional research in digital pathology and translational oncology.
---

## 🌟 Key Features

- **Multi-stage modular design:** From patch extraction to patient-level stratification, each step is modular and reproducible.  
- **Tissue habitat mapping:** Generates interpretable WSI-level maps capturing tumor–stroma–immune microenvironment composition.  
- **ROI-based survival modeling:** Aggregates biologically relevant regions of interest to build robust slide-level and patient-level survival models.  
- **Fisher vector encoding:** Encodes multi-slide patient information into a unified representation for Random Survival Forest (RSF) analysis.  
- **Clinical stratification:** Provides Kaplan–Meier curves, hazard ratios, and subgroup analyses (PD-L1, IO strategy, etc.) for OS and PFS.  
- **Scalable & reproducible:** Designed for multi-institutional datasets with standardized outputs for downstream statistical analysis and publication.

---

# 🧬 Path-IO Pipeline

End-to-end computational pathology pipeline for predicting immunotherapy outcomes from H&E whole-slide images (WSIs).  
The workflow includes patch extraction, tissue habitat mapping, ROI preprocessing, survival modeling, patient-level risk scoring, and clinical stratification.

---

## 📦 Repository Structure

```
├── 1.Patch_extraction_WSI_habitat_map_generation/   # Patch extraction & habitat map generation  
├── 2.WSI_map_preprocessing/                         # ROI extraction & map preprocessing  
├── 3.Survival_prediction/                           # Slide-level & patient-level survival modeling  
├── 4.Stratification/                                # Risk stratification and KM plots  
├── requirements.txt                                 # Dependencies  
└── README.md                                        # Documentation  
```

---

## ⚙️ Environment Setup

Before running any step, create and activate an environment and install the required dependencies.

```bash
# (optional) create a new conda environment
conda create -n pathio_env python=3.10 -y
conda activate pathio_env

# install all dependencies
pip install -r requirements.txt
```

---

## 🧩 Step 1: Patch Extraction + WSI Habitat Map Generation

**Folder:** `1.Patch_extraction_WSI_habitat_map_generation/`

This stage prepares slide-level spatial context.

### 1.1 Patch Extraction
- Divide each WSI into fixed-size tiles/patches at the desired magnification.  
- Store patch coordinates for spatial reconstruction.

### 1.2 Tissue Classification Training
- Train a deep learning model to classify tissue patches (e.g., tumor, stroma, necrosis, immune, etc.).  
- The model learns to identify biologically meaningful tissue types.

### 1.3 Patch-Level Inference
- Predict tissue type for all patches using the trained model.  
- Output includes patch label and (x, y) spatial coordinates.

### 1.4 WSI Habitat Map Generation
- Stitch labeled patches to reconstruct the full WSI tissue map.  
- Each region encodes the predicted tissue type.

### 🧾 Outputs from Step 1
- Patch-level labels and features  
- Tissue habitat map per WSI  

---

## 🧠 Step 2: WSI Habitat Map Preprocessing

**Folder:** `2.WSI_map_preprocessing/`

This stage refines the raw tissue habitat maps into biologically meaningful **Regions of Interest (ROIs)** suitable for downstream survival modeling.

### 2.1 ROI Extraction & Cleaning
- Identify informative tissue regions (e.g., tumor-enriched or immune-infiltrated habitats).  
- Remove background, artifacts, or non-tissue areas.  
- Optionally smooth, merge, or threshold small regions.

### 2.2 ROI-Level Feature Representation
- Compute region-based quantitative features such as:  
  - Proportion of tissue types per ROI  
  - Spatial heterogeneity metrics  
  - Morphological or texture features  
- Convert cleaned maps into **slide-level feature matrices**.

### 🧾 Outputs from Step 2
- ROI-based feature tables (one per WSI)  
- Mapping file linking **WSIs → patients**  
- Preprocessed features ready for survival training  

---

## 🔬 Step 3: Survival Prediction

**Folder:** `3.Survival_prediction/`

This stage builds the **slide-level** and **patient-level survival models** to predict progression-free survival (PFS) and overall survival (OS).

### 3.1 Slide-Level Survival Model
- Train a survival model using **slide-level or ROI-derived features**.  
- Supported types: Cox PH, Random Survival Forest (RSF), or deep survival networks.  
- Identify morphological patterns associated with prognosis.

### 3.2 Patient-Level Risk Score Prediction
Aggregate multiple WSIs per patient to form a unified patient representation.

#### 🧮 Fisher Vector Encoding
- Encode all slide-level features for each patient into a single high-dimensional vector.  
- Captures the distribution of morphological patterns.

#### 🌲 Random Survival Forest (RSF) Training
- Train an RSF model on the Fisher-encoded patient vectors.  
- Output continuous **risk scores** per patient:  
  - High score → higher risk of progression/death  
  - Low score → lower risk and better prognosis

### 🧾 Outputs from Step 3
- Trained slide- and patient-level models  
- Fisher-encoded patient features  
- Predicted risk scores for OS and PFS  
- Evaluation metrics (C-index, log-rank p)  

---

## 📊 Step 4: Stratification

**Folder:** `4.Stratification/`

This stage interprets **patient-level risk scores** into clinically meaningful groups and visualizes survival outcomes.

### 4.1 Risk Group Assignment
- Stratify patients into **High** and **Low** risk groups using:  
  - Median risk score (default)  
  - Quantile cut-offs or optimal KM thresholds  

### 4.2 Survival Stratification
- Generate **Kaplan–Meier (KM)** survival curves for:  
  - **Overall Survival (OS)**  
  - **Progression-Free Survival (PFS)**  
- Compute and report: HR, 95 % CI, log-rank *p* value.  
- Optionally annotate HR/CI on the plots.

### 4.3 Clinical Subgroup Analysis (Optional)
- Evaluate model performance across subgroups:  
  - PD-L1 status (High / Low / Intermediate)  
  - IO strategy (ICI-mono vs ICI-chemo)  
  - ECOG score, Stage, Metastatic sites, etc.  
- Generate subgroup-wise KM or forest plots.

### 🧾 Outputs from Step 4
- Kaplan–Meier plots (OS & PFS)  
- Forest plots / summary tables (HR & CI)  
- CSVs with risk groups & statistics  
- Publication-ready figures  

---



## 🧪 Notes
- All paths and parameters (patch size, magnification, model architecture, etc.) are configurable in the corresponding folders.  
- Supports multiple WSIs per patient for Fisher encoding and RSF aggregation.  
- Stratification assumes available survival metadata (OS, PFS, status).  

---



## 🐞 Reporting Issues

Path-IO is under continuous development.  
If you encounter any issues while running the pipeline, please first ensure that all required packages listed in `requirements.txt` are properly installed and that your environment matches the recommended setup.  

If the problem persists, kindly open a new issue in the repository describing:
- The exact step or module where the error occurred  
- The error message or traceback (if any)  
- A minimal code snippet or demo example to help reproduce the issue  

We will review your report and provide a fix or workaround as soon as possible.  
Thank you for helping us improve Path-IO!

