Path-IO/
│
├── 1.Patch_extraction_WSI_habitat_map_generation/
│   ├── Extracts tissue regions and tiles WSIs
│   ├── Generates patch-level habitat maps 
│   └── Outputs: Tissue habitat maps
│
├── 2.WSI_map_preprocessing/
│   ├── Preprocess WSI Habitat Maps 
│   ├── Performs tissue ROI generation
│   └── Outputs: .npy files for tissue ROIs
│
├── 3.Survival_prediction/
│   ├── Implements model training for risk prediction (Path-IO risk score)
│
│   └── Outputs: trained models, risk scores
└── 4.Stratification/
    ├── Performs patient grouping by predicted risk and clinical features
    ├── Generates Kaplan–Meier for OS/PFS analyses
    └── Outputs: stratification figures and summary statistics
    
git clone https://github.com/WuLabMDA/PATH-IO.git
cd Path-IO
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117

