from SWR_AI import SWRDetectionAgent
from dataLoader import DataLoader
import numpy as np

loader = DataLoader(rhd_file="demo.rhd")
lfp_data, orig_rate, new_rate = loader.load_raw_data(new_rate=1250)

print("Input data shape:", lfp_data.shape)  # Should be (n_samples, timesteps, 8)

agent = SWRDetectionAgent()
results = agent.detect_swrs(lfp_data, new_rate)

#np.save("results.npy", results)