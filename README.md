# ProcWatch👁️‍🗨️
*A tool designed for collecting and analyzing process information with machine learning.*

Welcome to ProcWatch! This project is a work in progress. No promises that anything works as intended!

## Features
- Real-time process monitoring feature
- Machine learning based anomaly scoring
- Heuristic detection layer for suspicious behavior
- JSON logging
- Anomaly generation scripts for basic testing

## Quick start
```bash
git clone https://github.com/rcs8888/ProcWatch.git
cd ProcWatch
```
## Usage
1. Begin recording process information:
```python
python3 process_collector.py
```
2. If using the tool for simulated anomaly detection/research, scripts are provided. Run in a seperate terminal.
  
 For child process spawning/CPU stress:    
```python
python3 spawn_children_anomaly.py --duration 30 --workers 6 --mode cpu
```  
 For memory stress:  
```python
python3 spawn_children_anomaly.py --duration 20 --workers 8 --mode mem --mem-mb 150  
```
  
 All options can be customized at runtime.
   
3. Stop collector (CTRL + C)  
4. Save labeled process stream to a CSV file:  
```python
python3 prepare_dataset.py
```
Process information will automatically be merged and labeled.

## Author
This project was designed and implemented by: Rachel Soubier  
💫💫💫  
If you are interested in reading the associated research paper, the link will be provided below when it is finished.  
**Post-Exploitation Malware Analysis: Leveraging Memory Forensics and Machine Learning for Real-Time Threat Intelligence**

   
