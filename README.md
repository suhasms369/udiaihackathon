# 🇮🇳 Aadhaar Operational Intelligence Dashboard

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Gemini AI](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google%20bard&logoColor=white)

> **A decision-support system transforming aggregated UIDAI data into actionable operational insights.**
## TL;DR

- Built an interactive dashboard to analyze aggregated Aadhaar enrolment and update data.
- Identifies regional, demographic, and temporal operational patterns.
- Provides state-level and district-level geospatial insights.
- Includes optional AI-assisted pattern summarization (Gemini).
- Designed for reproducibility, clarity, and governance-safe analytics.


## Overview

This repository contains an **Operational Intelligence Dashboard** built using aggregated Aadhaar enrolment and update datasets provided by UIDAI.  
The dashboard uncovers **regional patterns, demographic trends, temporal dynamics, and operational anomalies** to support data-driven decision-making in governance systems.

The solution prioritizes:
- Reproducibility  
- Visual clarity  
- Technical rigor  
- Responsible AI usage  
- Strict data privacy  

---

## Problem Statement

**Unlocking Societal Trends in Aadhaar Enrolment and Updates**

The goal is to analyze Aadhaar enrolment and update operations to:
- Identify meaningful regional and demographic patterns  
- Detect temporal surges and operational volatility  
- Highlight under-served or anomalous regions  
- Translate aggregated data into actionable insights  

---

## Approach

The dashboard combines:
- Deterministic statistical analysis  
- Geospatial visualization (state & district levels)  
- Temporal and demographic analytics  
- Optional AI-assisted pattern summarization  

All conclusions are derived from **aggregated operational data only**.

---

## Datasets Used

- UIDAI Aadhaar enrolment and update datasets (CSV format)
- Aggregated by:
  - State
  - District
  - Date
  - Age group
  - Operation type (Enrolment / Biometric Update / Demographic Update)

**No personal, biometric, or individual-level data is used.**

---

## Data Preprocessing & Methodology

### Dataset Consolidation
- Multiple CSV files with identical schemas were merged per operation type.
- This ensured completeness while preserving UIDAI’s aggregation structure.

### Data Cleaning
- Column names standardized
- Invalid or incomplete records excluded
- Dates parsed consistently

### Name Normalization
- State and district names normalized to handle naming variations
- Explicit mapping applied to align dataset values with GeoJSON boundaries

### Feature Engineering
- `child_ops`: operations for ages 0–17  
- `adult_ops`: operations for ages 18+  
- `total_ops`: combined operational volume  

### Aggregation Strategy
- All analysis uses grouped aggregates
- No row-level or individual inference is performed

---

## Visualisation & Analysis

### Geospatial Views
- **All India View**: State-level operational intensity  
- **State Drill-down**: District-level operational distribution  
- Adaptive zooming for clarity across states and UTs  

### Analytical Views
- Temporal trends (daily and monthly)
- Top vs bottom performing districts
- Operation type composition
- Child vs adult demographic correlation
- Weekly operational patterns
- District-level volatility analysis

Dark, high-contrast themes ensure readability in PDFs and presentations.

---

## AI-Assisted Insights (Optional)

The dashboard includes an **optional AI-assisted analysis module** using Google Gemini.

### Purpose
- Summarize observed patterns
- Assist exploratory analysis
- Highlight areas for further investigation

### Safeguards
- AI receives only aggregated summaries
- No raw data exposure
- AI outputs clearly marked as assistive
- Deterministic analysis remains the primary source of truth

> AI-generated insights should be validated by analysts and are not treated as authoritative conclusions.

---

## Technology Stack

- Python  
- Streamlit  
- Pandas  
- Plotly (including Mapbox-based choropleths)  
- GeoJSON (state and district boundaries)  
- Google Gemini API (optional)  

---

## Running the Application (Streamlit Cloud)

### Repository Structure

```
./
├── app.py/
├──*.csv
├── file2_normalized.geojson
├── india (2).geojson
└── README.md

```


### Deployment
- Designed for direct deployment on **Streamlit Cloud**
- Entry point: `app.py`
- AI API keys (if used) are managed via **Streamlit Secrets**

---

## Reproducibility

- Dashboard runs fully without AI
- AI usage is optional and non-deterministic
- All visualizations rely on deterministic aggregation logic
- Results are reproducible with the same datasets

---

## Impact & Applicability

This dashboard can support:
- Operational planning
- Resource allocation
- Monitoring enrolment/update campaigns
- Identifying under-served regions
- Detecting unusual surges or volatility

The approach is extensible to other large-scale governance datasets.

---

## Disclaimer

This project is developed solely for analytical and educational purposes using aggregated datasets provided for the hackathon.  
It does **not** represent official UIDAI systems, policies, or conclusions.

## License
Distributed under the MIT License. See LICENSE for more information.
