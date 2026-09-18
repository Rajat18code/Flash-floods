# Data Provenance & Authoritative Sources Registry

This document establishes the scientific provenance, authoritative citations, geographical coverage, and validation classification for every dataset utilized in the **Flash Flood Prediction System (Himachal Pradesh)**.

---

## 1. Master Dataset Provenance Ledger

| Dataset Name | File Path | Authoritative Source / Provider | Geographic Coordinates & Coverage | Temporal Range | Verification Category | Parameters & Units |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Historical Weather Baseline** | `data/raw/himachal_weather.csv` | **NASA POWER Project (MERRA-2 Reanalysis)**<br>NASA Langley Research Center<br>Reference: `https://power.larc.nasa.gov` | Primary Historical Data Point: Shimla, Himachal Pradesh (31.1°N, 77.17°E)<br>MERRA-2 Grid Elevation: 897.65 m (Area-averaged 0.5° × 0.625° cell)<br>*(Note: 2,200 m is an approximate representative elevation of the Shimla Ridge, not a model feature)* | **2015-01-01 to 2024-12-31**<br>*(3,653 days verified observational baseline)* | **Verified Third-Party Reanalysis** | • `PRECTOTCORR`: Precipitation Corrected (mm/day)<br>• `T2M`: Temperature at 2 m (°C)<br>• `RH2M`: Relative Humidity at 2 m (%)<br>• `WS2M`: Wind Speed at 2 m (m/s) |
| **Hydrological Feature Store** | `data/processed/himachal_features.csv` | **Derived Feature Engineering Pipeline**<br>Computed deterministically via `src/features/feature_engineering.py` | Same as Raw Weather Baseline | **2015-01-01 to 2024-12-31** | **Derived Data** | • `RAIN_3DAY`, `RAIN_7DAY`, `RAIN_14DAY`: Rolling rainfall sums (mm)<br>• `RAIN_INTENSITY`: Single-day to 3-day ratio<br>• `TEMP_HUMIDITY`: Heat-moisture product (°C · %)<br>• `API_7DAY`: Antecedent Precipitation Index ($\sum_{i=1}^7 P_{t-i} \cdot 0.85^i$) |
| **Ground-Truth Disaster Target** | Column `DOCUMENTED_FLOOD_EVENT` in `data/processed/himachal_flood_labeled.csv` | **Government Disaster Chronicles:**<br>1. Himachal Pradesh State Disaster Management Authority (HPSDMA)<br>2. National Disaster Management Authority (NDMA)<br>3. South Asia Network on Dams, Rivers and People (SANDRP)<br>4. IMD Climate Hazards Reports | Statewide documented cloudburst and flash flood events across Sutlej, Beas, Ravi, and Giri catchments | **2015-01-01 to 2024-12-31**<br>*(52 verified disaster dates)* | **Verified Official Government Archive** | • Binary indicator (1 = Documented Disaster Event, 0 = Non-disaster day) |
| **IMD Rule-Based Baseline Alert** | Column `RULE_BASELINE_ALERT` in `data/processed/himachal_flood_labeled.csv` | **India Meteorological Department (IMD) Guidelines**<br>IMD Heavy Rainfall Standard (≥ 64.5 mm/day) and Hydrological Antecedent Saturation Criteria | Same as Raw Weather Baseline | **2015-01-01 to 2024-12-31**<br>*(36 rule alert dates)* | **Derived Heuristic Benchmark** | • Binary indicator (1 = Rule Alert Triggered, 0 = Normal) |
| **Real-Time Regional Telemetry** | Fetched via `src/data/live_weather.py` | **Open-Meteo REST API (Numerical Weather Prediction)**<br>Blends ECMWF IFS (9 km), DWD ICON (7 km), and NOAA GFS with regional station grids<br>Reference: `https://open-meteo.com` | 8 Mountain Catchments across Sutlej, Beas, Ravi, and Yamuna River Basins | Real-time current weather + past 7-day observed rainfall sums | **Real-Time Numerical Weather Prediction (NWP)** | • Current Precipitation, Temperature, Humidity, Wind Speed<br>• Antecedent 3-Day & 7-Day Precipitation sums |
| **Official River Monitoring Grid** | GIS Marker Network in `dashboard/app.py` | **Central Water Commission (CWC) River Flood Monitoring Network**<br>Ministry of Jal Shakti, Government of India<br>Reference: `https://ffs.cwc.gov.in` | Official river gauge coordinates across Himachal Pradesh | Permanent monitoring infrastructure | **Official Government Station Metadata** | • CWC Station Name, River, Basin, Latitude, Longitude, Station Type |

---

## 2. Official Central Water Commission (CWC) Station Network

The real-time and GIS components monitor the following 8 official river stations in Himachal Pradesh:

| Station Name | River | River Basin | Latitude | Longitude | Elevation (m) | CWC Role & Operational Significance |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Rampur-1 / Bayal** | Satluj | Sutlej Basin | 31.4392°N | 77.6255°E | 1,000 m | CWC Level Monitoring Station downstream of Nathpa Jhakri Dam; monitored during the July 2024 Samej disaster. |
| **Manali** | Beas | Upper Beas Basin | 32.2396°N | 77.1887°E | 1,950 m | CWC Hydrological Observation Site; critical gauge during July 2023 Beas river surge. |
| **Mandi / Thalout** | Beas | Beas Basin | 31.6708°N | 77.1264°E | 860 m | CWC Level Station upstream of Pandoh Dam; narrow gorge floodgate management node. |
| **Dharamshala** | Gaj / Chari | Kangra / Gaj Basin | 32.2190°N | 76.3234°E | 1,457 m | IMD / State Hydrometric Station in the highest rainfall belt of Himachal Pradesh (Dhauladhar range). |
| **Chamba** | Ravi | Ravi River Basin | 32.5534°N | 76.1258°E | 1,006 m | CWC Hydrological Station monitoring Ravi river canyon discharge towards Chamera reservoirs. |
| **Khab / Pooh** | Satluj & Spiti | Upper Sutlej Basin | 31.8167°N | 78.6500°E | 2,600 m | CWC Trans-Himalayan Gauge Station at the confluence of Satluj and Spiti rivers near the Indo-Tibetan border. |
| **Suni / Tatapani** | Satluj | Sutlej Basin | 31.2464°N | 77.1206°E | 655 m | CWC Sutlej Level Monitoring Station upstream of Bhakra Dam backwaters. |
| **Yashwant Nagar** | Giri | Giri / Yamuna Basin | 30.7333°N | 77.2000°E | 1,100 m | CWC Hydrological Station on Giri River (Yamuna drainage system) along the southern Himalayan foothills. |

---

## 3. Verified Historical Disaster Event Chronicle (52 Events: 2015–2024)

Every single date below is backed by published government SITREPs, HPSDMA annual reports, or IMD extreme weather summaries:

- **2015 (4 events):** `2015-03-02` (March snowmelt surge), `2015-07-11` to `2015-07-13` (Manikaran cloudburst & Parbati flood).
- **2016 (2 events):** `2016-08-01`, `2016-08-06` (Mandi & Kangra seasonal flash floods).
- **2017 (3 events):** `2017-07-28`, `2017-08-13` (Kotropi landslide disaster, 46 dead), `2017-08-21` (Chamba flood).
- **2018 (5 events):** `2018-07-03`, `2018-08-13`, `2018-09-23` to `2018-09-25` (Historic late-September deluge; Beas floodgate discharge).
- **2019 (4 events):** `2019-07-15`, `2019-08-17` to `2019-08-19` (Shimla/Rohru cloudbursts; 22 deaths).
- **2020 (2 events):** `2020-08-10`, `2020-08-20` (Mandi & Sirmaur monsoon torrents).
- **2021 (4 events):** `2021-07-12`, `2021-07-13` (Boh Valley mudslide & Dharamshala surge), `2021-07-27`, `2021-07-28` (Tozing Nullah Lahaul flood).
- **2022 (7 events):** `2022-07-06` (Manikaran cloudburst), `2022-07-13`, `2022-07-30`, `2022-08-11`, `2022-08-19`, `2022-08-20` (Chakki bridge collapse), `2022-09-24`.
- **2023 (15 events):** `2023-06-25`, `2023-07-08` to `2023-07-11` (Historic Himalayan deluge; Beas washed away highways & temples), `2023-07-14`, `2023-07-20`, `2023-07-22`, `2023-07-25`, `2023-08-13` to `2023-08-15` (Summer Hill Shiv Temple collapse & Shimla landslides), `2023-08-22` to `2023-08-24`.
- **2024 (6 events):** `2024-07-05`, `2024-07-25`, `2024-07-31` to `2024-08-02` (Samej Khad Rampur cloudburst & hydel plant burst; 36 casualties), `2024-08-11`.

---

## 4. Purged / Non-Historical Elements

1. **Year 2025 Daily Weather Data:**  
   The raw dataset originally extended into 2025. Because observational reanalysis for late 2025 does not exist, all 2025 records have been removed from the machine learning training and testing partitions.
2. **8 Inferred 2025 Disaster Dates:**  
   The dates `2025-06-29, 2025-06-30, 2025-07-01, 2025-08-25, 2025-08-31, 2025-09-01, 2025-09-02, 2025-09-03` were algorithmically selected based on high rainfall in the 2025 raw file. They have been purged from the ground-truth target `DOCUMENTED_FLOOD_EVENT`.
3. **Misleading "Satellite Data" Label:**  
   All previous dashboard captions referring to Open-Meteo as "real-time satellite data" have been corrected to "Numerical Weather Prediction (NWP) Model Reanalysis and Station Telemetry".
4. **Single-Point Baseline vs. Multi-Catchment Scope:**  
   The historical ML baseline is trained on NASA POWER/MERRA-2 meteorological data from the Shimla-area grid point. Terrain and catchment variables are planned for the SIH extension. The 8-catchment Open-Meteo telemetry feature serves as multi-location real-time evaluation, not multi-catchment training.
5. **Shimla Elevation Qualification:**  
   2,200 m is an approximate representative elevation of the Shimla Ridge, not a model feature, model elevation, catchment average, training elevation, DEM-derived elevation, or calibration elevation. The NASA POWER MERRA-2 0.5° × 0.625° grid average elevation for the coordinate cell is 897.65 m.

