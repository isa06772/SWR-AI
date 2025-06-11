import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from typing import Union, List, Dict
import rippl_AI
import os
from dataLoader import DataLoader

class SWRDetectionAgent:
    def __init__(self):
        self.available_architectures = {
            'CNN1D': {'model': 5, 'predictions': [0]},
            'CNN2D': {'model': 5, 'predictions': [0]},
            'LSTM': {'model': 5, 'predictions': [0]},
            'SVM': {'model': 3, 'predictions': [0]},
            'XGBoost': {'model': 3, 'predictions': [0]}
        }
        self.metrics = {}
    
    #def load_data(self, data_path):
        """Load and preprocess EEG data"""
        # 1. Create an instance first
        #loader = DataLoader(rhd_file="demo.rhd")  # You need to provide the required rhd_file parameter

        # 2. Now call the method on the instance
        #lfp_data, orig_rate, new_rate = loader.load_raw_data(new_rate=1000)

        #print(f"Resampled from {orig_rate}Hz to {new_rate}Hz")
    
    def detect_swrs(self, lfp: np.ndarray, sf: float, architectures: Union[str, List[str]] = 'all',
                    model_numbers: Union[str, List[str]] = 'all', channels: List[int] = [0]) -> Dict[str, np.ndarray]:
        
        if len(lfp.shape) == 1:
            lfp = lfp[:, np.newaxis]  # Convert to 2D if single channel
            
        if architectures == 'all':
            architectures = list(self.available_architectures.keys())
            
        if model_numbers == 'all':
            model_numbers = {arch: list(range(1, self.available_architectures[arch]['models']+1))
                            for arch in architectures}
        elif isinstance(model_numbers, int):
            model_numbers = {arch: [model_numbers] for arch in architectures}
        
        results = {}
        for arch in architectures:
            if arch not in self.available_architectures:
                print(f"Warning: invalid model")
                continue

            arch_results = []
            for model_num in model_numbers:
                try:
                    channel_probs = []
                    for ch in channels:
                        swr_prob=rippl_AI.predict(lfp[:, ch],sf,arch=arch,model_number=model_num,channels=[0])
                        channel_probs.append(swr_prob)

                        arch_results.append(np.mean(channel_probs, axis=0))
                except Exception as e:
                    print(f"Error running {arch} model {model_num}: {str(e)}")

            if arch_results:
                results.append(np.mean(arch_results, axis=0))

        return results


