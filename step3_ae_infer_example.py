# Step 3 (AE): load scaler + features -> transform -> AE -> error -> compare with threshold
import pickle, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# files
scaler_path = r"/mnt/data/exp52_step2_W300_S30_extracted/scaler_W300_S30.pkl"
feature_cols_path = r"/mnt/data/exp52_step2_W300_S30_extracted/feature_columns.txt"
model_path = r"/mnt/data/exp52_step3_AE_W300_S30/ae_model_torch_state_dict.pt"
meta_path = r"/mnt/data/exp52_step3_AE_W300_S30/step3_ae_meta.json"

with open(feature_cols_path,'r') as f:
    feature_cols = [line.strip() for line in f if line.strip()]
with open(scaler_path,'rb') as f:
    scaler = pickle.load(f)
with open(meta_path,'r') as f:
    meta = json.load(f)
thr = meta["thr_ae_p99"]

class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d,16), nn.ReLU(), nn.Linear(16,8), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,d))
    def forward(self, x):
        return self.decoder(self.encoder(x))

# load model
ckpt = torch.load(model_path, map_location="cpu")
d_in = ckpt["arch"]["d_in"]
model = AE(d_in)
model.load_state_dict(ckpt["state_dict"])
model.eval()

def window_df_to_errors(df):
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
    Xn = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(Xn)
        xh = model(x)
        err = ((x-xh)**2).mean(dim=1).numpy()
    return err

# Example:
# df = pd.read_csv("some_windows.csv")
# err = window_df_to_errors(df)
# pred = (err > thr).astype(int)
# print(err[:5], pred[:5])
