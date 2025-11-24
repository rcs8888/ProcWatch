# ProcWatch👁️‍🗨️
*A tool designed for collecting and analyzing process information with machine learning.*

This project is a work in progress.

## Features
- Process Monitoring: Collects system metrics (CPU, memory usage, etc.) for forensic analysis
- Anomaly Simulation: Safely triggers controlled CPU and memory spikes, I/O stress, process anomalies, etc.
- Dataset Preparation: Labels and organzies events and extracts statistical features for ML training
- Machine Learning Evaluation: Compares a heuristic baseline test with XGBoost and LightGBM for detecting anomalies
- Session-Based Logging: Tracks data based on sessions for distinction and organization
- (Beta) Live Detection Alerts: Outputs anomalous PID, suspicious metrics, and potential causes
- (Beta) Web Dashboard: Visualize 

## Requirements  
Python 3.10+  
### Install dependencies:
```bash
pip install pandas psutil scikit-learn joblib stress-ng flask
```

## Quick install
```bash
git clone https://github.com/rcs8888/ProcWatch.git
cd ProcWatch
```

## Research-Based Usage:
### 1. Begin recording process information (stop with CTRL + C):
```python
python3 process_collector.py --session session_name
```

### 2. Simulate anomalies (if desired):    
```python
python3 spawn_process_2.py --session session_name
```
Choose from the menu options to simulate:  
1. Normal background (benign, 30s) 
2. CPU anomaly
3. CPU anomaly (gradual escalation)  
4. Memory anomaly
5. I/O anomaly
6. Network anomaly
7. File descriptor leak
8. Fork bomb light
9. Mixed anomaly 
10. Quit 

Note: Timestamped logs are located in:  
```bash
logs/<session_name>/process_stream.csv
```

  
### 3. Prepare the dataset:   
After collecting the desired amount of data (at least 5 minutes is recommended for ML training), run:
```python
python3 prepare_dataset.py --session session_name
```  
Process metrics and logs are automatically labeled. A clean, readable CSV is outputted:  
```bash
logs/<session_name>/anomaly_events.csv  
logs/<session_name>/labeled_dataset.csv
```
Be sure to combine all sessions when finished:  
```bash
python3 combine_sessions.py --sessions session1 session2 --output all_sessions
```  

### 4. Train and evaluate ML model:  
Train XGBoost & LightGBM models and compare against the heuristic detector. Tuning hyperparameters before beginning is optional but recommended:  
``` bash
python3 hyperparameter_tuning.py --session all --trials 50
```
This will optimize performance of the XGBoost and LightGBM models. Next, train the model with the tuned parameters:  
``` bash
python3 train_with_tuned_params.py --session all --model xgboost
```
Alternatively:  
```bash
python3 train_with_tuned_params.py --session all --params-file models/tuned_params_xgboost_all.txt
```
If not using tuned parameters, run:  
```bash
python3 train_anomaly_model_advanced.py --session all --use-smote
```
This will output a confusion matrix, as well as relevant success metrics. Using SMOTE is optional.   

### 5. Visualize Results:  
This will produce graphs/tables with relevant statistics, including comparison of ML models with heuristic baseline:  
```bash
python3 generate_visualizations.py --session all --model-path models/best_model_all.pkl  
```
Visualizations include: Confusion matrix, top 20 most important features (CSV and bar chart), ROC and precision/recall curves, performance table (CSV and table), and metrics comparison for XGBoost, LightGBM, and Heuristic models.  

## Detection-Based Usage:  
To utilize this tool for live process detection, the pre-trained model is accessible for use; however, it may be more desirable to train from scratch, as detailed above.  
Pre-trained model (basic usage):  
``` bash
python3 live_detector.py --model models/tuned_model_all2.pkl
```
This will use a default interval of 2 seconds and a default anomaly probability threshold of 0.7. These options can be customized through --interval and --threshold tags.  
### Example output:  
``` bash
================================================================================
!!! ANOMALY DETECTED - 2025-11-24 15:30:45
================================================================================

 [+] Process Information:
   PID:        12345
   Parent PID: 1
   Name:       suspicious_miner
   Executable: /tmp/cryptominer

 [+] Detection Metrics:
   Anomaly Score:  92% (threshold: 70%)
   Confidence:     HIGH

 [+] Suspicious Metrics:
   • cpu=95.30
   • cpu_spike=4.80
   • proc_age=125.00
   • cpu_burst=1.00
   • delta_cpu=23.50

[+] Likely Attack Type: Cryptocurrency Mining
   Confidence: 80%
   Sustained high CPU usage detected. May indicate cryptocurrency mining malware.

[+] Indicators:
   • CPU usage: 95.3%
   • Prolonged compute-intensive activity
   • Process age: 125s

[+] Recommended Actions:
   1. Check process name against known miners
   2. Monitor network connections to mining pools
   3. Examine process for packed/obfuscated code

================================================================================
```
The web dashboard is currently not working. 


## Author
Copyright (c) 2025 Rachel Catherine Soubier. All rights reserved.  
This code is not public. Contact rcs2002@uncw.edu for collaboration requests.  
  
The latest update to this project was: 11/24/2025  
💫💫💫  
If you are interested in reading the associated research paper, the link will be provided below when it is finished.  
**Post-Exploitation Malware Analysis: Leveraging Memory Forensics and Machine Learning for Real-Time Threat Intelligence**

   
