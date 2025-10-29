# 🧬 Path-IO Pipeline

Path-IO (Pathology-Driven Immunotherapy Optimization) is a biologically grounded, interpretable framework that predicts patient response to immunotherapy directly from routine H&E slides.

---

# 📦 Repository Structure

```
├── 1.Patch_extraction_WSI_habitat_map_generation/   # Patch extraction & habitat map generation  
├── 2.WSI_map_preprocessing/                         # ROI extraction & map preprocessing  
├── 3.Survival_prediction/                           # Slide-level & patient-level survival modeling  
├── 4.Stratification/                                # Risk stratification and KM plots  
├── requirements.txt                                 # Dependencies  
└── README.md                                        # Documentation  
```

## ⚙️ Environment Setup

Before running any step, create and activate an environment and install the required dependencies.

```bash
# (optional) create a new conda environment
conda create -n pathio_env python=3.10 -y
conda activate pathio_env

# install all dependencies
pip install -r requirements.txt
```
## 🧩 Step 1: Patch Extraction + WSI Habitat Map Generation

**Folder:** `1.Patch_extraction_WSI_habitat_map_generation/`

This stage prepares slide-level spatial context.

---

### 1.1 Patch Extraction
- Divide each WSI into fixed-size tiles/patches at the desired magnification.  
- Store patch coordinates for spatial reconstruction.

---

### 1.2 Tissue Classification Training
- Train a deep learning model to classify tissue patches (e.g., tumor, stroma, necrosis, immune, etc.).  
- The model learns to identify biologically meaningful tissue types.

---

### 1.3 Patch-Level Inference
- Predict tissue type for all patches using the trained model.  
- Output includes patch label and (x, y) spatial coordinates.

---

### 1.4 WSI Habitat Map Generation
- Stitch labeled patches to reconstruct the full WSI tissue map.  
- Each region encodes the predicted tissue type.

---

### 🧾 Outputs from Step 1
- Patch-level labels  
- Tissue habitat map per WSI
## 🧠 Step 2: WSI Habitat Map Preprocessing

**Folder:** `2.WSI_map_preprocessing/`

This stage refines the raw tissue habitat maps into biologically meaningful **Regions of Interest (ROIs)** suitable for downstream survival modeling.

---

### 2.1 ROI Extraction & Cleaning
- Identify and isolate informative tissue regions (e.g., tumor-enriched or immune-infiltrated habitats).  
- Remove background, artifacts, or non-tissue areas.  
- Optionally smooth, merge, or threshold small disconnected regions to ensure spatial consistency.

---

### 2.2 ROI-Level Feature Representation
- Compute region-based quantitative features such as:
  - Proportion of tissue types per ROI  
  - Spatial heterogeneity metrics  
  - Morphological or texture features  
- Convert cleaned habitat maps into **slide-level feature matrices** summarizing each WSI.

---

### 🧾 Outputs from Step 2
- ROI-based feature tables (one per WSI)  
- Mapping file linking **WSIs → patients**  
- Preprocessed features ready for survival model training
