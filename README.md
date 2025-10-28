Path-IO/
│
├── 1.Patch_extraction_WSI_habitat_map_generation/
│   ├── Extracts tissue regions and tiles WSIs
│   ├── Generates patch-level habitat maps using foundation model features
│   └── Outputs: patch features (.h5 / .csv) and tissue masks
│
├── 2.WSI_map_preprocessing/
│   ├── Aggregates patch-level features to slide-level representations
│   ├── Performs feature normalization, cleaning, and ROI filtering
│   └── Outputs: processed slide-level feature maps
│
├── 3.Survival_prediction/
│   ├── Implements model training for risk prediction (Path-IO risk score)
│   ├── Includes Cox, DeepSurv, and attention-based MIL modules
│   └── Outputs: trained models, risk scores, survival curves
│
└── 4.Stratification/
    ├── Performs patient grouping by predicted risk and clinical features
    ├── Generates Kaplan–Meier and forest plots for OS/PFS analyses
    └── Outputs: stratification figures and summary statistics

