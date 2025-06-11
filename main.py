from SWR_AI import SWRDetectionAgent
from dataLoader import DataLoader

loader = DataLoader(rhd_file="demo.rhd")
lfp_data, orig_rate, new_rate = loader.load_raw_data(new_rate=1000)

agent = SWRDetectionAgent()
results = agent.detect_swrs(lfp_data, new_rate)
