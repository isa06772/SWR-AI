import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from typing import Union, List, Dict
import rippl_AI

class SWRDetectionAgent:
    def __init__(self):
        self.available_architectures = {
            'CNN1D': {'model': 5, 'predictions': [0], 'type': 'tensorflow', 'path': 'optimized_models/CNN1D_1_Ch8_W60_Ts16_OGmodel12/'},
            'CNN2D': {'model': 5, 'predictions': [0], 'type': 'tensorflow', 'path': 'optimized_models/CNN2D_1_Ch8_W60_Ts40_OgModel/'},
            'LSTM': {'model': 5, 'predictions': [0], 'type': 'tensorflow', 'path': 'optimized_models/LSTM_2_Ch8_W60_Ts16_Bi0_L4_U25_E05_TB256/'},
            'SVM': {'model': 3, 'predictions': [0], 'type': 'pickle', 'path': 'Models_output/SVM.pickle'},
            'XGBoost': {'model': 3, 'predictions': [0], 'type': 'pickle', 'path': 'Models_output/XGBOOST.pickle'}
        }
        self.metrics = {}
    
    def _load_model(self, arch):
        config = self.available_architectures[arch]
        try:
            if config['type'] == 'tensorflow':
                import tensorflow as tf
                return tf.keras.models.load_model(config['path'])
            elif config['type'] == 'pickle':
                import joblib
                return joblib.load(config['path'])
            else:
                raise ValueError(f"Unsupported model type: {config['type']}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {arch} model: {str(e)}")
    
    def detect_swrs(self, lfp: np.ndarray, sf: float, architectures: Union[str, List[str]] = 'all',
                    model_numbers: Union[str, List[str]] = 'all', channels: List[int] = [0]) -> Dict[str, np.ndarray]:
        
        if len(lfp.shape) == 1:
            lfp = lfp[:, np.newaxis]  
            
        if architectures == 'all':
            architectures = list(self.available_architectures.keys())
            
        if model_numbers == 'all':
            model_numbers = {arch: list(range(1, self.available_architectures[arch]['model']+1))
                            for arch in architectures}
        elif isinstance(model_numbers, int):
            model_numbers = {arch: [model_numbers] for arch in architectures}
        
        results = {}
        for arch in architectures:
            if arch not in self.available_architectures:
                print(f"Warning: {arch} is not a valid model")
                continue           
            
            try:
                model = self._load_model(arch)
                channel_probs = []
                
                if arch=='CNN2D':
                    channels=[0,3,7]
                elif arch=='XGBOOST':
                    channels=[0]
                else:
                    channels=[0,1,2,3,4,5,6,7]

                for ch in channels:
                    swr_prob = rippl_AI.predict(
                        LFP=lfp,
                        sf=sf,
                        arch=arch,
                        model_number=self.available_architectures[arch]['model'],
                        channels=channels
                    )

                    # Postprocess: Remove extra dimensions
                    if isinstance(swr_prob, np.ndarray) and swr_prob.ndim > 1:
                        swr_prob = np.squeeze(swr_prob)  

                    channel_probs.append(swr_prob)

                    results[arch] = np.mean(np.stack(channel_probs), axis=0)
            
            except Exception as e:
                print(f"Error running {arch} model: {str(e)}")

        return results


