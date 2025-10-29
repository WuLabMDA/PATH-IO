# 🧬 Path-IO Pipeline

End-to-end computational pathology pipeline for predicting immunotherapy outcomes from H&E whole-slide images (WSIs).  
The workflow includes patch extraction, tissue habitat mapping, ROI preprocessing, survival modeling, patient-level risk scoring, and clinical stratification.

---

## 📦 Repository Structure

---

## 🔁 High-Level Workflow

```mermaid
flowchart LR
    A[Step 1: Patch Extraction<br/>+ Tissue Habitat Mapping] --> B[Step 2: WSI Habitat Map Preprocessing]
    B --> C[Step 3: Survival Prediction]
    C --> D[Step 4: Stratification]

    subgraph A_block[Step 1 Details]
        A1[Extract patches from WSI] --> A2[Train tissue classifier]
        A2 --> A3[Predict patch-level tissue types]
        A3 --> A4[Reconstruct WSI tissue habitat map]
    end

    subgraph B_block[Step 2 Details]
        B1[Clean habitat map] --> B2[Select ROIs]
        B2 --> B3[Compute ROI / slide-level features]
    end

    subgraph C_block[Step 3 Details]
        C1[Train slide-level survival model] --> C2[Fisher encoding per patient]
        C2 --> C3[Random Survival Forest (RSF)]
        C3 --> C4[Patient-level risk score]
    end

    subgraph D_block[Step 4 Details]
        D1[High vs Low risk grouping] --> D2[Kaplan–Meier analysis (OS/PFS)]
        D2 --> D3[Hazard ratio, CI, p-value]
    end
