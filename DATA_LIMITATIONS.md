# Scientific Limitations & Hackathon Claims Boundary

This document outlines the **strict boundaries, scientific assumptions, and claims that team members MUST NOT make** when presenting this Flash Flood Prediction System to judges at the Smart India Hackathon (SIH).

---

## 1. Absolute Negative Claims (What We Must NEVER Claim)

### ❌ 1. Never Claim "Direct Satellite Telemetry"
- **The Reality:** The real-time weather integration uses the **Open-Meteo API**, which aggregates Numerical Weather Prediction (NWP) model outputs (ECMWF IFS, DWD ICON, GFS) blended with regional surface station observations.
- **The Violation:** Claiming we possess "real-time radar feeds" or "direct geostationary satellite radiometry (INSAT-3D/GPM)".
- **Truthful Pitch:** *"We consume real-time high-resolution Numerical Weather Prediction model data and regional station observations via Open-Meteo API."*

### ❌ 2. Never Claim State-Wide Microclimate Generalization from a Single Point
- **The Reality:** The historical ML training baseline is derived from NASA POWER MERRA-2 at coordinate **31.10°N, 77.17°E (Shimla / Sutlej Basin)**.
- **The Violation:** Claiming that this single-station model directly measures localized rainfall in Dharamshala or Chamba without spatial adjustment.
- **Truthful Pitch:** *"Our historical reanalysis model is calibrated on the fragile Sutlej River Basin / Shimla Hills terrain (2,200 m). For operational deployment across other Himalayan valleys, we architected a multi-station GIS telemetry grid covering 8 official river monitoring nodes."*

### ❌ 3. Never Claim 2025 Data is Historical Real-World Observation
- **The Reality:** Observational reanalysis records for late 2025 do not exist.
- **The Violation:** Claiming that 2025 records in any dataset represent retrospective satellite observations or that 2025 disaster dates are published government records.
- **Truthful Pitch:** *"We strictly bounded our observational training and testing baseline to 2015–2024 (3,653 verified days), encompassing the historic 2023 and 2024 deluge seasons."*

### ❌ 4. Never Claim Rule Heuristics are Ground-Truth Disasters
- **The Reality:** IMD rainfall threshold exceedances ($\ge 64.5\text{ mm/day}$) indicate flash flood hazard conditions, but do not guarantee that an actual destructive disaster occurred.
- **The Violation:** Blending rule alerts into the ground-truth target.
- **Truthful Pitch:** *"Our ML target (`DOCUMENTED_FLOOD_EVENT`) trains exclusively on 52 confirmed disaster events documented by HPSDMA and NDMA. We preserve the IMD hydrometeorological rule as an independent benchmark (`RULE_BASELINE_ALERT`)."*

---

## 2. Inherent Scientific & Technical Limitations

| Dimension | Limitation Description | Mitigation / System Disclosure |
| :--- | :--- | :--- |
| **Spatial Resolution** | NASA MERRA-2 reanalysis uses a $0.5^\circ \times 0.625^\circ$ grid cell (~50 km × 60 km). Hyper-localized mountain cloudbursts (e.g. within a 2 km valley) may be smoothed out. | Disclosed in UI: The grid elevation is 897.65 m (regional average), while Shimla ridge is at 2,200 m. |
| **Cloudburst Lead Time** | Sudden convective cloudbursts develop within 1 to 3 hours. Atmospheric reanalysis operates at daily granularity. | The model serves as an **Antecedent Catchment Saturation & Flash Flood Susceptibility System**, identifying slopes primed for catastrophic runoff when rain falls. |
| **CWC Telemetry Gaps** | Historically, streamflow telemetry at mountain river gauges in remote upper catchments (e.g. Khab, Parbati) experiences transmission outages during extreme deluges. | Our system combines antecedent rainfall accumulation with terrain vulnerability, functioning even if water-level sensors wash away. |
| **Class Imbalance** | Flash floods are rare, catastrophic events: 52 disaster days out of 3,653 days (1.42% positive class prevalence). | Handled via balanced class weights in `HistGradientBoostingClassifier`, optimizing for Precision (83.3%) and ROC-AUC (0.9275). |

---

## 3. Defense Script for Judge Inquiries

> **Judge:** *"Where did you get your training ground truth?"*  
> **Team Answer:** *"We cross-referenced 10 years of official Himachal Pradesh State Disaster Management Authority (HPSDMA) annual incident reports, NDMA Situation Reports, and IMD extreme weather logs to compile 52 verified historical flood and cloudburst events between 2015 and 2024. We strictly avoided mixing heuristic rainfall rules into our ground truth target."*

> **Judge:** *"Is your live data coming from ISRO or INSAT satellites?"*  
> **Team Answer:** *"No, we are scientifically honest: our live data is ingested via Open-Meteo, which synthesizes ECMWF IFS, DWD ICON, and national weather station grids into calibrated numerical weather prediction models. We do not claim raw direct satellite radiometry."*

> **Judge:** *"How did you prevent data leakage in your time-series model?"*  
> **Team Answer:** *"We performed a strict chronological time-series split: 2015 to 2022 was used strictly for training, and 2023 to 2024 was reserved as an unseen test set. This test set subjected our model to the two worst monsoon seasons in modern Himalayan history—the July–August 2023 deluge and the July 2024 Samej cloudburst."*
