# read-me
Another Repo
and mention tacos
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def aralik_mask(df, aralik):
    lo, hi = aralik
    return ((df["DONEM"] >= lo) & (df["DONEM"] <= hi)).values

# ---- OOT'yi genişlet: 3 gözlem yerine 12 ----
CONFIG["TRAIN_TEST_RANGE"] = (201910, 202407)   # tüm geçmişi kullan
CONFIG["OOT_RANGE"]        = (202408, 202507)   # 12 gözlem
CONFIG["N_SPLITS"]         = 5

m_oot      = aralik_mask(train_df, CONFIG["OOT_RANGE"])
m_pool_raw = aralik_mask(train_df, CONFIG["TRAIN_TEST_RANGE"])
m_pool     = m_pool_raw & (~m_oot)

# ---- KRONOLOJİK sıra (shuffle YOK) ----
pool_idx = np.where(m_pool)[0]
pool_idx = pool_idx[np.argsort(train_df["DONEM"].values[pool_idx])]

if len(pool_idx) < 20:
    raise ValueError(f"Pool'da {len(pool_idx)} gözlem var, CV için az.")

tscv = TimeSeriesSplit(n_splits=CONFIG["N_SPLITS"])
CV_FOLDS = [(pool_idx[tr], pool_idx[te]) for tr, te in tscv.split(pool_idx)]

# Son kat = "holdout test", tüm pool = final fit havuzu
m_train_full = np.zeros(len(train_df), dtype=bool)
m_train_full[pool_idx] = True

print("=== SPLIT ÖZETİ ===")
print(f"Pool: {len(pool_idx)} gözlem | OOT: {m_oot.sum()} gözlem")
for i, (tr, te) in enumerate(CV_FOLDS, 1):
    d = train_df["DONEM"].values
    print(f"  Kat {i}: train {d[tr].min()}-{d[tr].max()} ({len(tr)}) "
          f"-> test {d[te].min()}-{d[te].max()} ({len(te)})")
print(f"OOT: {train_df['DONEM'].values[m_oot].min()}-"
      f"{train_df['DONEM'].values[m_oot].max()}")
