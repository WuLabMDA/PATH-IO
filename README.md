> **About this Repository**  
> This repository hosts the complete *Path-IO* computational pathology framework for predicting immunotherapy outcomes in metastatic non–small cell lung cancer (mNSCLC) using H&E whole-slide images (WSIs).  
> The pipeline integrates patch-level tissue classification, WSI habitat mapping, ROI preprocessing, survival modeling, and patient-level risk stratification.  
> It is designed for reproducible, multi-institutional research in digital pathology and translational oncology.

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

## 🔁 End-to-End Workflow Summary

The complete Path-IO pipeline from patch extraction to patient stratification is shown below.

```mermaid
flowchart LR
    A[🧩 Step 1<br/>Patch Extraction & Habitat Map Generation] --> B[🧠 Step 2<br/>WSI Habitat Map Preprocessing]
    B --> C[🔬 Step 3<br/>Survival Prediction]
    C --> D[📊 Step 4<br/>Stratification]

    subgraph S1[ ]
        A1[Patch Extraction] --> A2[Tissue Classification Training]
        A2 --> A3[Patch-Level Prediction]
        A3 --> A4[WSI Habitat Map Reconstruction]
    end

    subgraph S2[ ]
        B1[ROI Extraction & Cleaning] --> B2[ROI-Level Feature Computation]
    end

    subgraph S3[ ]
        C1[Slide-Level Survival Modeling] --> C2[Fisher Encoding per Patient]
        C2 --> C3[Random Survival Forest (RSF)]
        C3 --> C4[Patient Risk Scores]
    end

    subgraph S4[ ]
        D1[High/Low Risk Grouping] --> D2[Kaplan–Meier (OS & PFS)]
        D2 --> D3[Hazard Ratio / CI / p-value]
    end

    style A fill:#e7f1ff,stroke:#1f5eff,stroke-width:1px
    style B fill:#e7ffe7,stroke:#22aa22,stroke-width:1px
    style C fill:#fff7e6,stroke:#ffaa22,stroke-width:1px
    style D fill:#ffe7ef,stroke:#ff2277,stroke-width:1px
```

---

## 🧪 Notes
- All paths and parameters (patch size, magnification, model architecture, etc.) are configurable in the corresponding folders.  
- Supports multiple WSIs per patient for Fisher encoding and RSF aggregation.  
- Stratification assumes available survival metadata (OS, PFS, status).  

---



## 🧑‍💻 Contact
**Rukhmini Bandyopadhyay, PhD**  
Postdoctoral Fellow, MD Anderson Cancer Center  
📧 [email protected]
