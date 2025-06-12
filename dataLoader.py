import spikeinterface.extractors as se
from spikeinterface import preprocessing as si
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self, rhd_file="demo.rhd", sampling_rate=1250):
        self.rhd_file = Path(rhd_file)
        self.sampling_rate = sampling_rate
        self.locations = np.array([[0, 630], [0, 600], [0, 570], [0, 540], [0, 510], [0, 480], [0, 660], [0, 690], 
                              [0, 570], [0, 540], [0, 510], [0, 480], [0, 570], [0, 540], [0, 510], [0, 480], 
                              [0, 570], [0, 540], [0, 510], [0, 480], [0, 570], [0, 540], [0, 510], [0, 480], 
                              [0, 570], [0, 540], [0, 510], [0, 480], [0, 570], [0, 540], [0, 510], [0, 480]])
    
    def load_raw_data(self, new_rate=1000):
        recording = se.read_intan(self.rhd_file, stream_id = "0")
        original_rate = recording.get_sampling_frequency()
        
        recording.set_channel_locations(self.locations.astype(np.float32))
        recording = si.bandpass_filter(recording, freq_min=50, freq_max=300)
        #recording = si.notch_filter(recording, 50)
        

        recording_resampled = si.resample(recording, 1250)
        new_rate_actual = recording_resampled.get_sampling_frequency()

        lfp_data = recording_resampled.get_traces().T

        lfp_data = lfp_data.reshape(32, -1, 8)

        # seconds_of_data = 2_250_016 / 1250  # ≈1800 seconds (30 minutes)
        # 60-second windows:
        window_size = 60 * 1250  # 75,000 timesteps
        lfp_data = lfp_data[:, :window_size, :]  # Trim to (32, 75000, 8)


        output_path = "data/lfp_data.npy"
        np.save(output_path, lfp_data)
        print(f"Data saved to {output_path}")
        print(f"Array shape: {lfp_data.shape}, dtype: {lfp_data.dtype}")
        print(f"Original rate: {original_rate}Hz, New rate: {new_rate_actual}Hz")

        return lfp_data, original_rate, new_rate_actual







