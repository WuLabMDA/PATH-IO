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
# ⚙️ Environment Setup
# (optional) create a new conda environment
conda create -n pathio_env python=3.10 -y
conda activate pathio_env

# install all dependencies
pip install -r requirements.txt
