# ProcWatch👁️‍🗨️
*A tool designed for collecting and analyzing process information with machine learning.*

Welcome to ProcWatch! This project is a work in progress.

## Features
- Live Process Monitoring: Collects system metrics (CPU, memory usage, etc.) for forensic analysis
- Anomaly Simulation: Safely triggers controlled CPU and memory spikes, I/O stress, process anomalies, etc.
- Dataset Preparation: Labels and organzies events and extracts statistical features for ML training
- Machine Learning Evaluation: Compares a heuristic baseline test and a Random Forest model for anomaly detection accuracy
- Session-Based Logging: Tracks data based on sessions for distinction and organization

## Requirements  
Python 3.10+  
### Install dependencies:
```bash
pip install pandas psutil scikit-learn joblib stress-ng  
```

## Quick install
```bash
git clone https://github.com/rcs8888/ProcWatch.git
cd ProcWatch
```

## Usage
### 1. Begin recording process information (stop with CTRL + C):
```python
python3 process_collector.py --session session_name --interval 1.0
```

### 2. Simulate anomalies (if desired):    
```python
python3 spawn_children_anomaly.py
```
Choose from the menu options to simulate:  
1. Rapid process creation  
2. CPU stress  
3. Memory stress  
4. Mixed behavior  
5. Idle baseline  
6. Exit  

Note: Timestamped logs are located in:  
```bash
logs/<session_name>/process_stream.csv
```

  
### 3. Prepare the dataset:   
After collecting the desired amount of data (at least 5 minutes is recommended for ML training), run:
```python
python3 prepare_dataset.py
```
Process metrics and logs are automatically merged and labeled. A clean, readable CSV is outputted:  
```bash
logs/<session_name>/anomaly_events.csv  
logs/<session_name>/labeled_dataset.csv
```

### 4. Train and evaluate ML model:  
Train a Random Forest classifier and compare against the heuristic detector.  
```python
python3 train_eval.py
```
This also prints precision, recall, F1, and a confusion matrix for the heuristic and ML scores.  

### Work-in-progress steps:
I aim to do (at least) the following:  
- Add more anomaly types, possibly evasive malware indicators such as process injection/hollowing
- Add a web-based dashboard for visualization, including graphs

## Author
Copyright (c) 2025 Rachel Catherine Soubier. All rights reserved.  
This code is not public. Contact rcs2002@uncw.edu for collaboration requests.  
  
The latest update to this project was: 11/10/2025  
💫💫💫  
If you are interested in reading the associated research paper, the link will be provided below when it is finished.  
**Post-Exploitation Malware Analysis: Leveraging Memory Forensics and Machine Learning for Real-Time Threat Intelligence**

   
