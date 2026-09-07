# =============================================================================
# ÖNCÜLLERDEN 12M90 TAHMİN MODELİ — TAM KONSOLİDE VERSİYON (v3)
# =============================================================================
# Kohort mantığı:
#   DONEM = kartın açıldığı ay (vintage)
#   2M20  = kohortun 2 ay içinde 20+ gecikme görme oranı (limit ağırlıklı)
#   2M30  = kohortun 2 ay içinde 30+ gecikme görme oranı
#   12M90 = kohortun 12 ay içinde 90+ gecikme (batık) oranı -> kohort+12'de bilinir
#   Amaç  : erken öncüllerle 12M90'ı ~10 ay önceden tahmin etmek (202602 kohortu)
#
# Bölümler:
#   1  Importlar
#   2  Ham veri
#   3  Veri hazırlık
#   4  Konfigürasyon
#   5  Split maskeleri + walk-forward CV katları
#   6  Feature üretimi + model factory + metrikler
#   7  Deneme fonksiyonu + tüm kombinasyon taraması
#   8  Top-10 model
#   9  En iyi adayın raporu + CHAMPION dondurma
#   10 Doğrusallık (linearity) kontrolü
#   11 Train-CV-OOT grafikler + model haritası
#   12 OLS diagnostik: DW + Ljung-Box + HAC + VIF + korelasyon
#   12B Dış kitle için sabit production formülü (raw uzay)
#   13 Modelin kitleye fiti (df_kitle0 / df_kitle1 / df_genel)
#   14 Görseller (kitle bazlı)
#   15 TOP-3 model formül ve parametrik tahmin bloğu
#   16 Sağlık kontrol listesi
# =============================================================================


# =============================================================================
# 1) IMPORTLAR
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from itertools import combinations
from scipy import stats

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =============================================================================
# 2) BDS ÖNCÜLLER — HAM VERİ
# =============================================================================
# Tab ayraçlı, virgüllü ondalık. DONEM + 10 metrik kolonu.
raw_data = """DONEM	T Takip 90 + L12M	T Takip 90 + L6M	T Takip 90 + L5M	T Takip 60 + L5M	T Takip 90 + L4M	T Takip 60 + L4M	T Takip 30 + L4M	T Takip 30 + L3M	T Takip 30 + L2M	T Takip 20 + L2M
201910	0,023045	0,008181	0,005876	0,011534	0,001803	0,007762	0,021960	0,015000	0,003000	0,004000
"""  # <-- TAM VERİYİ BURAYA YAPIŞTIR

rename_map = {
    "T Takip 90 + L12M": "12M90",
    "T Takip 90 + L6M":  "6M90",
    "T Takip 90 + L5M":  "5M90",
    "T Takip 60 + L5M":  "5M60",
    "T Takip 90 + L4M":  "4M90",
    "T Takip 60 + L4M":  "4M60",
    "T Takip 30 + L4M":  "4M30",
    "T Takip 30 + L3M":  "3M30",
    "T Takip 30 + L2M":  "2M30",
    "T Takip 20 + L2M":  "2M20",
}


# =============================================================================
# 3) VERİ HAZIRLIK
# =============================================================================
lines  = [l for l in raw_data.strip().splitlines() if l.strip()]
header = lines[0].split("\t")
value_cols = header[1:]

rows = []
for line in lines[1:]:
    parts = line.split("\t")
    row = {"DONEM": int(parts[0])}
    for col, val in zip(value_cols, parts[1:]):
        row[rename_map[col]] = float(val.replace(",", ".")) * 100   # oran -> yüzde
    rows.append(row)

train_df = pd.DataFrame(rows).sort_values("DONEM").reset_index(drop=True)
train_df = train_df[["DONEM"] + list(rename_map.values())]

print(f"Veri: {len(train_df)} kohort | "
      f"{train_df['DONEM'].min()} - {train_df['DONEM'].max()}")


# =============================================================================
# 4) KONFİGÜRASYON
# =============================================================================
CONFIG = {
    "TARGET":   "12M90",
    "FEATURES": ["5M60", "4M60", "4M30", "3M30", "2M30", "2M20"],

    "POOL_RANGE": (202201, 202504),   # model geliştirme havuzu
    "OOT_RANGE":  (202504, 202507),   # son 3 ay, kör test
    "N_SPLITS":   5,                  # walk-forward kat sayısı

    "APPLY_DONEM": "202602",
    "MIN_FEATURES": 1,
    "ALPHAS": np.logspace(-3, 4, 80),
    "MODEL_OPTIONS": ["ridge", "lasso", "ols"],
    "LASSO_MAX_ITER": 200000,

    "SEASON_OPTIONS":  [False],
    "ROLLING_OPTIONS": [False],
    "ROLLING_PENCERE": 3,

    # Geçerlilik eşikleri (oran-hata odaklı; R2 kullanılmıyor)
    "ESIK_CV_WMAPE":    18.0,
    "ESIK_OOT_WMAPE":   22.0,
    "ESIK_DRIFT_WMAPE":  8.0,
    "ESIK_OVERFIT_WMAPE": 8.0,
    "ESIK_CV_MAPE_STD":  6.0,
}

FEATURES = CONFIG["FEATURES"]
y = train_df[CONFIG["TARGET"]].astype(float).values


# =============================================================================
# 5) SPLIT MASKELERİ + WALK-FORWARD CV (shuffle YOK)
# =============================================================================
def aralik_mask(df, aralik):
    lo, hi = aralik
    return ((df["DONEM"] >= lo) & (df["DONEM"] <= hi)).values


m_oot      = aralik_mask(train_df, CONFIG["OOT_RANGE"])
m_pool_raw = aralik_mask(train_df, CONFIG["POOL_RANGE"])
m_pool     = m_pool_raw & (~m_oot)          # çakışma engellenir

# KRONOLOJİK sıra — random shuffle YOK
pool_idx = np.where(m_pool)[0]
pool_idx = pool_idx[np.argsort(train_df["DONEM"].values[pool_idx])]

if len(pool_idx) < 20:
    raise ValueError(f"Pool'da sadece {len(pool_idx)} gözlem var, CV için yetersiz.")

tscv = TimeSeriesSplit(n_splits=CONFIG["N_SPLITS"])
CV_FOLDS = [(pool_idx[tr], pool_idx[te]) for tr, te in tscv.split(pool_idx)]

m_train_full = np.zeros(len(train_df), dtype=bool)
m_train_full[pool_idx] = True

if (m_train_full & m_oot).any():
    raise ValueError("Pool ile OOT çakışıyor, range kontrol et.")

print("=== SPLIT ÖZETİ (WALK-FORWARD) ===")
print(f"Pool: {len(pool_idx)} gözlem | OOT: {m_oot.sum()} gözlem")
_d = train_df["DONEM"].values
for i, (tr, te) in enumerate(CV_FOLDS, 1):
    print(f"  Kat {i}: train {_d[tr].min()}-{_d[tr].max()} ({len(tr):>2}) "
          f"-> test {_d[te].min()}-{_d[te].max()} ({len(te):>2})")
print(f"  OOT  : {_d[m_oot].min()}-{_d[m_oot].max()} ({m_oot.sum()})")


# =============================================================================
# 6) FEATURE ÜRETİMİ + MODEL FACTORY + METRİKLER
# =============================================================================
def mevsim_ekle(donem_serisi):
    ay = donem_serisi.astype(str).str[-2:].astype(int)
    return pd.DataFrame({
        "sin_ay": np.sin(2 * np.pi * ay / 12),
        "cos_ay": np.cos(2 * np.pi * ay / 12),
    }, index=donem_serisi.index)


def make_X(metric_df, feats, donem_series=None, mevsim=False, rolling=False):
    X = metric_df[feats].reset_index(drop=True).copy()

    if rolling:
        p = CONFIG["ROLLING_PENCERE"]
        for f in feats:
            X[f"{f}_ma{p}"] = X[f].rolling(p, min_periods=1).mean()

    if mevsim:
        if donem_series is None:
            raise ValueError("mevsim=True için donem_series gerekli.")
        X = pd.concat([X, mevsim_ekle(donem_series.reset_index(drop=True))], axis=1)

    return X


def pipe_factory(model_name):
    m = str(model_name).lower()
    if m == "ridge":
        model = RidgeCV(alphas=CONFIG["ALPHAS"])
    elif m == "lasso":
        model = LassoCV(alphas=CONFIG["ALPHAS"], cv=5,
                        max_iter=CONFIG["LASSO_MAX_ITER"])
    elif m in ["ols", "linearregression"]:
        model = LinearRegression()
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    return Pipeline([("scale", StandardScaler()), ("model", model)])


def metrikler(y_true, y_pred, n_feat):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    if n == 0:
        return {k: np.nan for k in ["R2", "AdjR2", "RMSE", "MAE", "MAPE", "WMAPE"]}

    r2   = r2_score(y_true, y_pred) if n > 1 else np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    eps   = 1e-12
    mape  = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100
    wmape = (np.sum(np.abs(y_true - y_pred))
             / np.maximum(np.sum(np.abs(y_true)), eps)) * 100

    adj = (1 - (1 - r2) * (n - 1) / (n - n_feat - 1)
           if (not np.isnan(r2) and n - n_feat - 1 > 0) else np.nan)

    return {"R2": r2, "AdjR2": adj, "RMSE": rmse,
            "MAE": mae, "MAPE": mape, "WMAPE": wmape}


def trend_metrikleri(y_true, y_pred):
    """n<6 ise devre dışı: n=3'te TrendCorr her zaman ±1 çıkıyor (bilgisiz)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 6:
        return {"TrendCorr": np.nan, "DirectionAcc": np.nan}

    dy = np.diff(y_true)
    dp = np.diff(y_pred)

    if np.std(dy) < 1e-12 or np.std(dp) < 1e-12:
        trend_corr = np.nan
    else:
        trend_corr = np.corrcoef(dy, dp)[0, 1]

    direction_acc = np.mean(np.sign(dy) == np.sign(dp))
    return {"TrendCorr": trend_corr, "DirectionAcc": direction_acc}


def nz(x, default):
    return default if pd.isna(x) else x


# =============================================================================
# 7) DENEME FONKSİYONU + TÜM KOMBİNASYON TARAMASI
# =============================================================================
def deneme_yap(feats, mevsim, rolling, model_name):
    feats = list(feats)
    X = make_X(train_df, feats, train_df["DONEM"], mevsim, rolling)
    n_feat = X.shape[1]

    # ---- 1) WALK-FORWARD CV (asıl performans ölçümü) ----
    cv_mae, cv_mape, cv_wmape, cv_r2 = [], [], [], []
    for tr_idx, te_idx in CV_FOLDS:
        mdl = pipe_factory(model_name).fit(X.iloc[tr_idx], y[tr_idx])
        d = metrikler(y[te_idx], mdl.predict(X.iloc[te_idx]), n_feat)
        cv_mae.append(d["MAE"]);    cv_mape.append(d["MAPE"])
        cv_wmape.append(d["WMAPE"]); cv_r2.append(d["R2"])

    CV_MAE       = np.mean(cv_mae)
    CV_MAPE      = np.mean(cv_mape)
    CV_WMAPE     = np.mean(cv_wmape)
    CV_R2        = np.mean(cv_r2)
    CV_MAPE_STD  = np.std(cv_mape)
    CV_WMAPE_STD = np.std(cv_wmape)

    # ---- 2) Tüm pool ile fit -> OOT değerlendirmesi ----
    model = pipe_factory(model_name).fit(X[m_train_full], y[m_train_full])
    mtr = metrikler(y[m_train_full], model.predict(X[m_train_full]), n_feat)
    moo = metrikler(y[m_oot],        model.predict(X[m_oot]),        n_feat)

    overfit_wmape = abs(mtr["WMAPE"] - CV_WMAPE)   # ezber göstergesi
    drift_wmape   = abs(CV_WMAPE - moo["WMAPE"])   # rejim kayması
    overfit_mape  = abs(mtr["MAPE"] - CV_MAPE)
    drift_mape    = abs(CV_MAPE - moo["MAPE"])

    tr_oot     = trend_metrikleri(y[m_oot], model.predict(X[m_oot]))
    trend_corr = tr_oot["TrendCorr"]
    dir_acc    = tr_oot["DirectionAcc"]

    # ---- 3) SEÇİM SKORU (küçük = iyi), WMAPE odaklı ----
    trend_pen = 0.0 if pd.isna(trend_corr) else ((1 - np.clip(trend_corr, -1, 1)) / 2) * 5
    dir_pen   = 0.0 if pd.isna(dir_acc)    else (1 - dir_acc) * 5

    secim_skoru = (
        0.45 * CV_WMAPE +
        0.25 * moo["WMAPE"] +
        0.10 * (CV_MAPE_STD * 2) +
        0.10 * drift_wmape +
        0.10 * overfit_wmape +
        trend_pen + dir_pen
    )

    sk_model  = model.named_steps["model"]
    alpha_val = getattr(sk_model, "alpha_", np.nan)

    return {
        "model": model_name,
        "alpha": round(float(alpha_val), 6) if not pd.isna(alpha_val) else np.nan,
        "feats": ",".join(feats),
        "n_feat": len(feats),
        "mevsim": mevsim,
        "rolling": rolling,

        "CV_R2":        round(CV_R2, 3),
        "CV_MAE":       round(CV_MAE, 4),
        "CV_MAPE":      round(CV_MAPE, 2),
        "CV_MAPE_STD":  round(CV_MAPE_STD, 2),
        "CV_WMAPE":     round(CV_WMAPE, 2),
        "CV_WMAPE_STD": round(CV_WMAPE_STD, 2),

        "Train_R2":    round(mtr["R2"], 3),
        "Train_MAE":   round(mtr["MAE"], 4),
        "Train_MAPE":  round(mtr["MAPE"], 2),
        "Train_WMAPE": round(mtr["WMAPE"], 2),

        "OOT_R2":    round(moo["R2"], 3),
        "OOT_RMSE":  round(moo["RMSE"], 4),
        "OOT_MAE":   round(moo["MAE"], 4),
        "OOT_MAPE":  round(moo["MAPE"], 2),
        "OOT_WMAPE": round(moo["WMAPE"], 2),

        "Overfit_MAPE":  round(overfit_mape, 3),
        "Overfit_WMAPE": round(overfit_wmape, 3),
        "Drift_MAPE":    round(drift_mape, 3),
        "Drift_WMAPE":   round(drift_wmape, 3),

        "TrendCorr_OOT":    np.nan if pd.isna(trend_corr) else round(trend_corr, 3),
        "DirectionAcc_OOT": np.nan if pd.isna(dir_acc)    else round(dir_acc, 3),

        "SecimSkoru": round(secim_skoru, 4),
    }


kayit = []
for model_name in CONFIG["MODEL_OPTIONS"]:
    for r in range(CONFIG["MIN_FEATURES"], len(FEATURES) + 1):
        for feats in combinations(FEATURES, r):
            for mevsim in CONFIG["SEASON_OPTIONS"]:
                for rolling in CONFIG["ROLLING_OPTIONS"]:
                    kayit.append(deneme_yap(feats, mevsim, rolling, model_name))

tum_denemeler = pd.DataFrame(kayit)

# Geçerlilik filtresi — oran-hata odaklı, R2 KULLANILMAZ
tum_denemeler["Gecerli"] = (
    (tum_denemeler["CV_WMAPE"]      <= CONFIG["ESIK_CV_WMAPE"]) &
    (tum_denemeler["OOT_WMAPE"]     <= CONFIG["ESIK_OOT_WMAPE"]) &
    (tum_denemeler["Drift_WMAPE"]   <= CONFIG["ESIK_DRIFT_WMAPE"]) &
    (tum_denemeler["Overfit_WMAPE"] <= CONFIG["ESIK_OVERFIT_WMAPE"]) &
    (tum_denemeler["CV_MAPE_STD"]   <= CONFIG["ESIK_CV_MAPE_STD"])
)

tum_denemeler = tum_denemeler.sort_values(
    ["Gecerli", "SecimSkoru", "n_feat"],
    ascending=[False, True, True]
).reset_index(drop=True)
tum_denemeler["Sira"] = np.arange(1, len(tum_denemeler) + 1)

print(f"\nToplam {len(tum_denemeler)} deneme | "
      f"Gecerli: {tum_denemeler['Gecerli'].sum()} "
      f"({tum_denemeler['Gecerli'].mean()*100:.0f}%)")


# =============================================================================
# 8) TOP-10 MODEL
# =============================================================================
goster = [
    "Sira", "Gecerli", "model", "alpha", "feats", "n_feat",
    "CV_R2", "CV_MAPE", "CV_MAPE_STD", "CV_WMAPE", "CV_MAE",
    "Train_R2", "Train_MAPE", "Train_WMAPE",
    "OOT_R2", "OOT_MAPE", "OOT_WMAPE", "OOT_MAE",
    "Overfit_WMAPE", "Drift_WMAPE",
    "TrendCorr_OOT", "DirectionAcc_OOT", "SecimSkoru",
]
goster = [c for c in goster if c in tum_denemeler.columns]
top10 = tum_denemeler[goster].head(10).copy()

print("\n=== EN İYİ 10 MODEL ===")
print(top10.to_string(index=False))


# =============================================================================
# 9) EN İYİ ADAYIN RAPORU + CHAMPION DONDURMA
# =============================================================================
en_iyi = tum_denemeler.iloc[0].copy()

SECILEN_MODEL_KEY = str(en_iyi["model"]).lower()
SECILEN_FEATS     = [f.strip() for f in str(en_iyi["feats"]).split(",") if f.strip()]
SECILEN_MEVSIM    = bool(en_iyi["mevsim"])
SECILEN_ROLLING   = bool(en_iyi["rolling"])

X_model    = make_X(train_df, SECILEN_FEATS, train_df["DONEM"],
                    SECILEN_MEVSIM, SECILEN_ROLLING)
FEAT_MODEL = list(X_model.columns)

# --- 9a) DÜRÜST performans raporu: SADECE pool ile fit (OOT görülmemiş kalsın)
ana_model = pipe_factory(SECILEN_MODEL_KEY).fit(X_model[m_train_full], y[m_train_full])

perf_rows = []
for i, (tr_idx, te_idx) in enumerate(CV_FOLDS, 1):
    mdl = pipe_factory(SECILEN_MODEL_KEY).fit(X_model.iloc[tr_idx], y[tr_idx])
    d = metrikler(y[te_idx], mdl.predict(X_model.iloc[te_idx]), len(FEAT_MODEL))
    perf_rows.append({"Split": f"CV-Kat{i}", "n": len(te_idx),
                      "R2": round(d["R2"], 3), "RMSE": round(d["RMSE"], 4),
                      "MAE": round(d["MAE"], 4), "MAPE": round(d["MAPE"], 2),
                      "WMAPE": round(d["WMAPE"], 2)})

for ad, m in [("Train(pool)", m_train_full), ("OOT", m_oot)]:
    d = metrikler(y[m], ana_model.predict(X_model[m]), len(FEAT_MODEL))
    perf_rows.append({"Split": ad, "n": int(m.sum()),
                      "R2": round(d["R2"], 3), "RMSE": round(d["RMSE"], 4),
                      "MAE": round(d["MAE"], 4), "MAPE": round(d["MAPE"], 2),
                      "WMAPE": round(d["WMAPE"], 2)})

perf_df = pd.DataFrame(perf_rows)
print("\n=== ANA MODEL PERFORMANS (pool ile fit, leakage'sız) ===")
print(perf_df.to_string(index=False))

coef = pd.Series(np.ravel(ana_model.named_steps["model"].coef_), index=FEAT_MODEL)
print(f"\nIntercept (std): {float(ana_model.named_steps['model'].intercept_):.4f}")
print(pd.DataFrame({"Feature": FEAT_MODEL,
                    "Katsayi_std": coef.round(4).values}).to_string(index=False))

# --- 9b) PRODUCTION CHAMPION: model seçimi bitti -> pool + OOT ile refit
m_final = m_train_full | m_oot
champion_model = pipe_factory(SECILEN_MODEL_KEY).fit(X_model[m_final], y[m_final])

sk_model = champion_model.named_steps["model"]
scaler   = champion_model.named_steps["scale"]

coef_std      = np.ravel(sk_model.coef_).astype(float)
intercept_std = float(sk_model.intercept_)
scaler_mean   = scaler.mean_.astype(float)
scaler_std    = scaler.scale_.astype(float)

# Ham (orijinal) uzay eşdeğeri: y = b0_raw + Σ b_raw * x
coef_raw      = coef_std / scaler_std
intercept_raw = intercept_std - np.sum(coef_raw * scaler_mean)

CHAMPION = {
    "model_key":      SECILEN_MODEL_KEY,
    "features_base":  SECILEN_FEATS,
    "features_model": FEAT_MODEL,
    "mevsim":         SECILEN_MEVSIM,
    "rolling":        SECILEN_ROLLING,
    "pipeline":       champion_model,
    "coef_std":       coef_std,
    "intercept_std":  intercept_std,
    "scaler_mean":    scaler_mean,
    "scaler_std":     scaler_std,
    "coef_raw":       coef_raw,
    "intercept_raw":  intercept_raw,
    "alpha_refit":    float(getattr(sk_model, "alpha_", np.nan)),   # GERÇEK alpha
    "alpha_scan":     float(en_iyi["alpha"]) if not pd.isna(en_iyi["alpha"]) else np.nan,
    "n_fit":          int(m_final.sum()),
}

print("\n=== CHAMPION DONDURULDU ===")
print("Model        :", CHAMPION["model_key"])
print("Features     :", CHAMPION["features_model"])
print("Fit veri     :", CHAMPION["n_fit"], "gözlem (pool + OOT)")
print("Alpha (scan) :", CHAMPION["alpha_scan"], " <- tarama sırasında pool ile seçilen")
print("Alpha (refit):", CHAMPION["alpha_refit"], " <- production fit'te seçilen")
print("Intercept    :", round(intercept_std, 6))
print("Coef (std)   :", [round(v, 6) for v in coef_std])


# =============================================================================
# 10) DOĞRUSALLIK (LINEARITY) KONTROLÜ — FULL vs MASKED
# =============================================================================
def linearity_raporu(df, feats, target, mask=None, etiket="FULL"):
    d = df if mask is None else df[mask]
    rows = []
    for f in feats:
        pr, pp = stats.pearsonr(d[f], d[target])
        sr, sp = stats.spearmanr(d[f], d[target])
        rows.append({"feature": f,
                     "Pearson": round(pr, 3), "p_Pearson": round(pp, 4),
                     "Spearman": round(sr, 3), "p_Spearman": round(sp, 4),
                     "n": len(d)})
    out = pd.DataFrame(rows).sort_values("Pearson", ascending=False)
    print(f"\n=== LINEARITY [{etiket}] ===")
    print(out.to_string(index=False))
    return out


lin_full   = linearity_raporu(train_df, FEATURES, CONFIG["TARGET"],
                              None, "FULL (tüm dönem)")
lin_masked = linearity_raporu(train_df, FEATURES, CONFIG["TARGET"],
                              m_pool, "MASKED (pool)")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, f in zip(axes.ravel(), FEATURES):
    ax.scatter(train_df.loc[m_pool, f], train_df.loc[m_pool, CONFIG["TARGET"]],
               s=30, alpha=0.7, label="Pool")
    ax.scatter(train_df.loc[m_oot, f], train_df.loc[m_oot, CONFIG["TARGET"]],
               s=50, color="green", edgecolor="k", label="OOT")
    z = np.polyfit(train_df.loc[m_pool, f], train_df.loc[m_pool, CONFIG["TARGET"]], 1)
    xs = np.linspace(train_df[f].min(), train_df[f].max(), 50)
    ax.plot(xs, np.polyval(z, xs), "r--", lw=1.2)
    ax.set_xlabel(f); ax.set_ylabel(CONFIG["TARGET"]); ax.grid(alpha=0.25)
axes.ravel()[0].legend()
plt.suptitle("Öncül vs 12M90 — doğrusallık kontrolü")
plt.tight_layout(); plt.show()


# =============================================================================
# 11) TRAIN-CV-OOT GRAFİKLER + MODEL HARİTASI
# =============================================================================
def plot_trend_odakli_ozet(df, y, pred, m_pool_, m_oot_, target_col):
    y    = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    donem = df["DONEM"].values
    xpos  = np.arange(len(donem))

    y_trend = pd.Series(y).rolling(3, min_periods=1).mean().values
    p_trend = pd.Series(pred).rolling(3, min_periods=1).mean().values

    # --- 1) Zaman serisi ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(xpos, y,       "o-", color="#1f77b4", alpha=0.35, label=f"Gerçek {target_col}")
    ax.plot(xpos, pred,    "^-", color="red",     alpha=0.35, label="Tahmin")
    ax.plot(xpos, y_trend, "-",  color="#1f77b4", lw=2.2, label="Gerçek trend(3A)")
    ax.plot(xpos, p_trend, "-",  color="red",     lw=2.2, label="Tahmin trend(3A)")

    for p in np.where(m_pool_)[0]:
        ax.axvspan(p - 0.5, p + 0.5, color="#1f77b4", alpha=0.05)
    for p in np.where(m_oot_)[0]:
        ax.axvspan(p - 0.5, p + 0.5, color="green", alpha=0.10)

    ax.set_title(f"Gerçek vs Tahmin + Trend ({target_col})")
    ax.set_xlabel("Dönem"); ax.set_ylabel(f"{target_col} (%)")
    ax.set_xticks(xpos); ax.set_xticklabels(donem, rotation=90)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(alpha=0.25); ax.legend()
    plt.tight_layout(); plt.show()

    # --- 2) Tahmin vs Gerçek scatter ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y[m_pool_], pred[m_pool_], color="#1f77b4", s=50, label="Pool")
    ax.scatter(y[m_oot_],  pred[m_oot_],  color="green", s=80,
               edgecolor="k", label="OOT")
    lim = [min(y.min(), pred.min()) - 0.05, max(y.max(), pred.max()) + 0.05]
    ax.plot(lim, lim, "k--", lw=1, label="İdeal y=x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title("Tahmin vs Gerçek")
    ax.set_xlabel(f"Gerçek {target_col} (%)"); ax.set_ylabel("Tahmin (%)")
    ax.grid(alpha=0.25); ax.legend()
    plt.tight_layout(); plt.show()

    # --- 3) Artıklar ---
    artik = y - pred
    renk  = np.where(m_oot_, "green", np.where(m_pool_, "#4c78a8", "#cccccc"))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(xpos, artik, color=renk)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("Artıklar (Gerçek - Tahmin)")
    ax.set_xlabel("Dönem"); ax.set_ylabel("Hata")
    ax.set_xticks(xpos); ax.set_xticklabels(donem, rotation=90)
    ax.grid(alpha=0.25)
    plt.tight_layout(); plt.show()


pred_all = np.asarray(ana_model.predict(X_model), dtype=float)
plot_trend_odakli_ozet(train_df, y, pred_all, m_train_full, m_oot, CONFIG["TARGET"])

# --- Model haritası: CV_MAPE vs OOT_MAPE ---
top30 = tum_denemeler.head(30).copy()
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(top30["CV_MAPE"], top30["OOT_MAPE"],
                c=top30["SecimSkoru"], cmap="viridis", s=70)
for _, r in top30.iterrows():
    ax.text(r["CV_MAPE"], r["OOT_MAPE"], str(int(r["Sira"])), fontsize=8)
cbar = plt.colorbar(sc, ax=ax); cbar.set_label("SecimSkoru (düşük iyi)")
ax.set_xlabel("CV_MAPE (düşük iyi)"); ax.set_ylabel("OOT_MAPE (düşük iyi)")
ax.set_title("Model Haritası — köşegen üstü = OOT'de bozulan modeller")
lim = [0, max(top30["CV_MAPE"].max(), top30["OOT_MAPE"].max()) * 1.1]
ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
ax.grid(alpha=0.25)
plt.tight_layout(); plt.show()


# =============================================================================
# 12) OLS DİAGNOSTİK (train set üzerinde) + VIF + DW + KORELASYON
# =============================================================================
Xc    = sm.add_constant(X_model[m_train_full].values)
ols   = sm.OLS(y[m_train_full], Xc).fit()
resid = ols.resid

print("\n=== OLS DİAGNOSTİK (pool üzerinde) ===")
print(ols.summary())

# --- Durbin-Watson ---
dw = durbin_watson(resid)
print(f"\nDurbin-Watson: {dw:.3f}")
print("  <1.5 : pozitif otokorelasyon VAR")
print("  ~2.0 : temiz")
print("  >2.5 : negatif otokorelasyon")

# --- Ljung-Box ---
maxlag = min(12, len(resid) // 4)
lb = acorr_ljungbox(resid, lags=list(range(1, maxlag + 1)), return_df=True)
print("\nLjung-Box (p<0.05 -> otokorelasyon VAR):")
print(lb.round(4).to_string())

# --- HAC (Newey-West) düzeltilmiş standart hatalar ---
ols_hac = sm.OLS(y[m_train_full], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
karsilastirma = pd.DataFrame({
    "degisken":  ["const"] + FEAT_MODEL,
    "katsayi":   ols.params.round(4),
    "se_normal": ols.bse.round(4),
    "se_HAC":    ols_hac.bse.round(4),
    "t_normal":  ols.tvalues.round(2),
    "t_HAC":     ols_hac.tvalues.round(2),
    "p_HAC":     ols_hac.pvalues.round(4),
})
print("\n--- Normal vs HAC standart hata ---")
print("(t_HAC << t_normal ise otokorelasyon anlamlılığı şişirmiş demektir)")
print(karsilastirma.to_string(index=False))

# --- Artık otokorelasyon grafiği ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(resid, "o-", lw=1)
axes[0].axhline(0, color="gray", lw=0.8)
axes[0].set_title(f"Artıklar (DW={dw:.2f})")
axes[0].grid(alpha=0.25)

sm.graphics.tsa.plot_acf(resid, lags=min(20, len(resid) // 2 - 1), ax=axes[1])
axes[1].set_title("Artık ACF")
plt.tight_layout(); plt.show()

# --- VIF ---
# DİKKAT: statsmodels VIF'i SABİT TERİMLİ matris ister. Sabit yoksa VIF şişer.
if len(FEAT_MODEL) >= 2:
    Xv_const = sm.add_constant(X_model[m_train_full].values)     # DOĞRU kullanım
    vif = pd.DataFrame({
        "feature": FEAT_MODEL,
        "VIF": [variance_inflation_factor(Xv_const, i + 1)
                for i in range(len(FEAT_MODEL))],
    })
    print("\n--- VIF (>5 dikkat, >10 ciddi çoklu bağlantı) ---")
    print(vif.round(2).to_string(index=False))

    print("\n--- Feature korelasyon matrisi ---")
    print(X_model[m_train_full].corr().round(3).to_string())
else:
    print("\nVIF: tek feature -> hesaplanamaz (çoklu bağlantı riski yok)")


# =============================================================================
# 12B) DIŞ KİTLE İÇİN SABİT PRODUCTION FORMÜLÜ (RAW UZAY)
# =============================================================================
# CHAMPION zaten dondurulmuş: coef_raw ve intercept_raw scaler'ı içeriyor.
feats_raw     = list(CHAMPION["features_base"])
coef_raw_v    = np.asarray(CHAMPION["coef_raw"], dtype=float)
intercept_raw_v = float(CHAMPION["intercept_raw"])

terms = " + ".join([f"({coef_raw_v[i]:.6f} * {feats_raw[i]})"
                    for i in range(len(feats_raw))])
formula_raw = f"12M90_tahmin = {intercept_raw_v:.6f} + {terms}"

print("=== DIŞ KİTLE PRODUCTION FORMÜLÜ (mean/std GEREKMEZ) ===")
print(f"Model     : {CHAMPION['model_key']}")
print(f"Features  : {feats_raw}")
print(f"Intercept : {intercept_raw_v:.6f}")
for i, f in enumerate(feats_raw):
    print(f"  Katsayi[{f}] = {coef_raw_v[i]:.6f}")
print("\nFORMÜL:")
print(formula_raw)


def predict_12m90_raw(**kwargs):
    """Dış kitle için: predict_12m90_raw(**{'2M30': 0.34, '2M20': 0.80})"""
    x = np.array([float(kwargs[f]) for f in feats_raw], dtype=float)
    return intercept_raw_v + np.dot(coef_raw_v, x)


ornek = {"2M30": 0.342621, "2M20": 0.804575}
print("\n=== ÖRNEK TAHMİN ===")
print(f"Girdi : {ornek}")
print(f"Tahmin: {predict_12m90_raw(**ornek):.4f}  (12M90 %)")

# Pipeline ile tutarlılık kontrolü
X_chk = np.array([[ornek[f] for f in feats_raw]], dtype=float)
pipe_pred = float(CHAMPION["pipeline"].predict(
    pd.DataFrame(X_chk, columns=CHAMPION["features_model"]))[0])
print(f"Pipeline: {pipe_pred:.4f}  -> raw formul ile fark: "
      f"{abs(pipe_pred - predict_12m90_raw(**ornek)):.10f}")


# =============================================================================
# 13) MODELİN KİTLEYE FİTİ
# =============================================================================
# Gereken dış segment dataframe'leri: df_kitle0, df_kitle1, df_genel
# Her biri: KITLE kolonu + SECILEN_FEATS kolonları + TARGET kolonu
def normalize_model_name(model_name: str) -> str:
    m = str(model_name).strip().lower()
    alias = {"ols": "ols", "linearregression": "ols",
             "ridge": "ridge", "ridgecv": "ridge",
             "lasso": "lasso", "lassocv": "lasso"}
    if m not in alias:
        raise ValueError(f"Desteklenmeyen model adi: {model_name}")
    return alias[m]


def ensure_1d(a):
    return np.asarray(a, dtype=float).reshape(-1)


TARGET_C = CONFIG["TARGET"]

if "SECILEN_MODEL_KEY" in globals():
    model_key_raw = SECILEN_MODEL_KEY
elif "en_iyi" in globals() and "model" in en_iyi.index:
    model_key_raw = en_iyi["model"]
else:
    raise ValueError("Secilen model bilgisi bulunamadi.")
SECILEN_MODEL_KEY = normalize_model_name(model_key_raw)

# Refit YOK — dondurulmuş champion kullanılır
final_model = CHAMPION["pipeline"]
if not hasattr(final_model, "named_steps"):
    raise ValueError("final_model pipeline degil. pipe_factory ciktisini kontrol et.")

scaler_f = final_model.named_steps.get("scale", None)
sk_model = final_model.named_steps.get("model", None)
if sk_model is None:
    raise ValueError("Pipeline icinde 'model' adimi bulunamadi.")

model_sinif = sk_model.__class__.__name__
alpha_txt   = getattr(sk_model, "alpha_", None)

gereken_segler = ["df_kitle0", "df_kitle1", "df_genel"]
eksik_segler   = [x for x in gereken_segler if x not in globals()]
if eksik_segler:
    raise ValueError(f"Segment dataframe'leri eksik: {eksik_segler}")

seg = pd.concat([df_kitle0, df_kitle1, df_genel], ignore_index=True).copy()

eksik = [f for f in SECILEN_FEATS if f not in seg.columns]
if eksik:
    raise ValueError(f"Segment verisinde eksik kolonlar: {eksik}")
if TARGET_C not in seg.columns:
    raise ValueError(f"Segment verisinde target kolonu yok: {TARGET_C}")

apply_donem = (int(CONFIG["APPLY_DONEM"])
               if str(CONFIG["APPLY_DONEM"]).isdigit() else CONFIG["APPLY_DONEM"])

X_seg_full = make_X(seg, SECILEN_FEATS,
                    pd.Series([apply_donem] * len(seg)),
                    SECILEN_MEVSIM, SECILEN_ROLLING)

X_seg      = X_seg_full[FEAT_MODEL].astype(float).copy()
y_true_seg = seg[TARGET_C].astype(float).values
y_pred_seg = ensure_1d(final_model.predict(X_seg))

seg_tahmin = pd.DataFrame({
    "KITLE": seg["KITLE"].astype(str).values,
    f"Gercek_{TARGET_C}_%": np.round(y_true_seg, 3),
    f"Tahmin_{TARGET_C}_%": np.round(y_pred_seg, 3),
})
seg_tahmin["Fark_%"] = np.round(
    seg_tahmin[f"Tahmin_{TARGET_C}_%"] - seg_tahmin[f"Gercek_{TARGET_C}_%"], 3)

# --- Katsayılar ve katkılar ---
if not hasattr(sk_model, "coef_"):
    raise ValueError("Secilen model linear coef_ vermiyor.")

coef_vals  = np.ravel(sk_model.coef_).astype(float)
feat_names = list(X_seg.columns)
if len(coef_vals) != len(feat_names):
    raise ValueError("Katsayi sayisi ile feature sayisi uyusmuyor.")

coef_series   = pd.Series(coef_vals, index=feat_names,
                          name="Katsayi_std").sort_values(ascending=False)
intercept_val = float(getattr(sk_model, "intercept_", np.nan))

X_used = scaler_f.transform(X_seg) if scaler_f is not None else X_seg.values

contrib    = X_used * coef_vals
contrib_df = pd.DataFrame(contrib, columns=[f"Katki_{c}" for c in feat_names])
contrib_df.insert(0, "Base_Intercept", intercept_val)

recon    = intercept_val + contrib.sum(axis=1)
max_diff = np.max(np.abs(recon - y_pred_seg))

terms_s     = " + ".join([f"({coef_vals[i]:.4f} * z_{feat_names[i]})"
                          for i in range(len(feat_names))])
formula_txt = f"y_hat = {intercept_val:.4f} + {terms_s}"

print("=== MODEL BILGISI ===")
print(f"Model key: {SECILEN_MODEL_KEY}")
print(f"Model class: {model_sinif}")
print(f"Alpha: {alpha_txt if alpha_txt is not None else 'N/A'}")
print(f"Feature set ({len(feat_names)}): {feat_names}")
print(f"Intercept: {intercept_val:.6f}")
print(f"Reconstruction max abs diff: {max_diff:.12f}")

print("\n=== KATSAYILAR (std uzay) ===")
print(coef_series.to_string())
print("\n=== FORMUL (std uzay) ===")
print(formula_txt)

input_df  = seg[["KITLE"] + feat_names].copy()
report_df = pd.concat([
    input_df.reset_index(drop=True),
    contrib_df.reset_index(drop=True),
    seg_tahmin[[f"Gercek_{TARGET_C}_%", f"Tahmin_{TARGET_C}_%", "Fark_%"]]
        .reset_index(drop=True),
], axis=1)

print(f"\n=== {CONFIG['APPLY_DONEM']} KITLE RAPORU (Input + Katki + Tahmin) ===")
print(report_df.to_string(index=False))


# =============================================================================
# 14) GÖRSELLER (kitle bazlı)
# =============================================================================
# 14.1 Gerçek vs Tahmin bar chart
fig, ax = plt.subplots(figsize=(10, 5))
xp = np.arange(len(seg_tahmin))
w  = 0.38

bars_true = ax.bar(xp - w / 2, seg_tahmin[f"Gercek_{TARGET_C}_%"],
                   width=w, label="Gercek", color="#66c2a5")
bars_pred = ax.bar(xp + w / 2, seg_tahmin[f"Tahmin_{TARGET_C}_%"],
                   width=w, label="Tahmin", color="#fc8d62")

for bars in [bars_true, bars_pred]:
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.2f}%",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(xp)
ax.set_xticklabels([f"KITLE {k}" for k in seg_tahmin["KITLE"]])
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.set_ylabel(f"{TARGET_C} (%)")
ax.set_title(f"{CONFIG['APPLY_DONEM']} - Kitle Bazli Gercek vs Tahmin")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True); ax.legend()
plt.tight_layout(); plt.show()

# 14.2 Input heatmap (kitle x feature)
inp = seg[feat_names].astype(float).values
fig, ax = plt.subplots(figsize=(1.2 * len(feat_names) + 3, 0.8 * len(seg) + 3))
im = ax.imshow(inp, aspect="auto", cmap="YlGnBu")

ax.set_xticks(np.arange(len(feat_names)))
ax.set_xticklabels(feat_names, rotation=45, ha="right")
ax.set_yticks(np.arange(len(seg)))
ax.set_yticklabels([f"KITLE {k}" for k in seg["KITLE"].astype(str)])

for i in range(inp.shape[0]):
    for j in range(inp.shape[1]):
        ax.text(j, i, f"{inp[i, j]:.3f}", ha="center", va="center",
                fontsize=8, color="black")

ax.set_title("Kitle Inputlari (ham degerler)")
fig.colorbar(im, ax=ax, shrink=0.85)
plt.tight_layout(); plt.show()

# 14.3 Katkı dağılımı (intercept + feature katkıları, stacked)
fig, ax = plt.subplots(figsize=(11, 5))
xp = np.arange(len(seg))

base = np.full(len(seg), intercept_val)
ax.bar(xp, base, label="Intercept", color="#bdbdbd")

pos_bottom = base.copy()
neg_bottom = base.copy()

for j, f in enumerate(feat_names):
    vals = contrib[:, j]
    pos  = np.where(vals > 0, vals, 0.0)
    neg  = np.where(vals < 0, vals, 0.0)

    ax.bar(xp, pos, bottom=pos_bottom, label=f"Katki_{f}")
    ax.bar(xp, neg, bottom=neg_bottom)

    pos_bottom += pos
    neg_bottom += neg

ax.plot(xp, y_pred_seg, "ko--", label="Toplam Tahmin")
ax.set_xticks(xp)
ax.set_xticklabels([f"KITLE {k}" for k in seg["KITLE"].astype(str)])
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.set_ylabel(f"{TARGET_C} (%)")
ax.set_title("Tahminin hesaplanmasi: Intercept + Feature Katkilari")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(loc="best", fontsize=8, ncol=2)
plt.tight_layout(); plt.show()


# =============================================================================
# 15) TOP-3 MODEL FORMÜL VE PARAMETRİK TAHMİN BLOĞU
# =============================================================================
gerekli = ["train_df", "y", "tum_denemeler", "m_train_full", "m_oot",
           "make_X", "pipe_factory"]
for g in gerekli:
    if g not in globals():
        raise ValueError(f"Gerekli obje eksik: {g}")

FIT_MODE = "pool_oot"   # "pool" veya "pool_oot"; production'da pool_oot


def get_fit_mask():
    if FIT_MODE == "pool":
        return m_train_full
    elif FIT_MODE == "pool_oot":
        return (m_train_full | m_oot)
    else:
        raise ValueError("FIT_MODE sadece 'pool' veya 'pool_oot' olabilir.")


fit_mask = get_fit_mask()

top3         = tum_denemeler.head(3).copy()
TOP3_MODELS  = {}
summary_rows = []
param_rows   = []

for i, row in top3.iterrows():
    rank      = int(row["Sira"]) if "Sira" in top3.columns else int(i + 1)
    model_key = str(row["model"]).lower()
    feats     = [f.strip() for f in str(row["feats"]).split(",") if f.strip()]
    mevsim    = bool(row["mevsim"])  if "mevsim"  in row.index else False
    rolling   = bool(row["rolling"]) if "rolling" in row.index else False
    alpha_selected = row["alpha"] if "alpha" in row.index else np.nan

    X = make_X(train_df, feats, train_df["DONEM"], mevsim=mevsim, rolling=rolling)
    feat_model = list(X.columns)

    pipe = pipe_factory(model_key).fit(X[fit_mask], y[fit_mask])

    sk_m   = pipe.named_steps["model"]
    scl    = pipe.named_steps["scale"]

    coef_std_i      = np.ravel(sk_m.coef_).astype(float)
    intercept_std_i = float(sk_m.intercept_)
    scaler_mean_i   = scl.mean_.astype(float)
    scaler_std_i    = scl.scale_.astype(float)

    # Ham uzaya dönüşüm: y = b0_raw + Σ b_raw * x
    coef_raw_i      = coef_std_i / scaler_std_i
    intercept_raw_i = intercept_std_i - np.sum(coef_raw_i * scaler_mean_i)

    terms_i = " + ".join([f"({coef_raw_i[j]:.6f} * {feat_model[j]})"
                          for j in range(len(feat_model))])
    formula_raw_i = (f"12M90_tahmin = {intercept_raw_i:.6f}"
                     + (f" + {terms_i}" if terms_i else ""))

    TOP3_MODELS[rank] = {
        "rank": rank,
        "model_key": model_key,
        "alpha_scan":  alpha_selected,                          # taramadaki alpha
        "alpha_refit": float(getattr(sk_m, "alpha_", np.nan)),  # fit'teki GERÇEK alpha
        "features_base": feats,
        "features_model": feat_model,
        "mevsim": mevsim,
        "rolling": rolling,
        "pipeline": pipe,
        "coef_std": coef_std_i,
        "intercept_std": intercept_std_i,
        "scaler_mean": scaler_mean_i,
        "scaler_std": scaler_std_i,
        "coef_raw": coef_raw_i,
        "intercept_raw": intercept_raw_i,
        "formula_raw": formula_raw_i,
        "fit_n": int(fit_mask.sum()),
    }

    summary_rows.append({
        "Model_Rank": rank,
        "Model": model_key,
        "Alpha_Scan":  np.nan if pd.isna(alpha_selected) else float(alpha_selected),
        "Alpha_Refit": float(getattr(sk_m, "alpha_", np.nan)),
        "Kolonlar": ",".join(feat_model),
        "Input_Alanlari": ", ".join(feat_model),
        "Intercept_Raw": round(intercept_raw_i, 6),
        "Formul_Raw": formula_raw_i,
        "Fit_N": int(fit_mask.sum()),
    })

    param_rows.append({"Model_Rank": rank, "Parametre": "Intercept",
                       "Deger": float(intercept_raw_i)})
    for j, f in enumerate(feat_model):
        param_rows.append({"Model_Rank": rank, "Parametre": f,
                           "Deger": float(coef_raw_i[j])})

summary_df = pd.DataFrame(summary_rows).sort_values("Model_Rank")
params_df  = pd.DataFrame(param_rows).sort_values(
    ["Model_Rank", "Parametre"]).reset_index(drop=True)

print("\n=== TOP-3 MODEL ÖZET TABLO (KOLONLAR + INPUT + INTERCEPT + FORMÜL) ===")
print(summary_df.to_string(index=False))

print("\n=== TOP-3 PARAMETRE TABLOSU (RAW UZAY) ===")
print(params_df.to_string(index=False))

print("\n=== TOP-3 FORMÜLLER ===")
for rnk in sorted(TOP3_MODELS.keys()):
    m = TOP3_MODELS[rnk]
    print(f"\nModel {rnk} | {m['model_key']} | alpha_refit={m['alpha_refit']}")
    print(m["formula_raw"])


def predict_12m90_top3(model_rank, input_dict):
    """
    model_rank: 1/2/3 (top3 sıralamasındaki rank)
    input_dict: örn {"2M30": 0.34, "2M20": 0.80}
    """
    if model_rank not in TOP3_MODELS:
        raise ValueError(f"Model rank bulunamadi: {model_rank}")

    m     = TOP3_MODELS[model_rank]
    feats = m["features_model"]

    eksik = [f for f in feats if f not in input_dict]
    if eksik:
        raise ValueError(f"Eksik input alanlari: {eksik}")

    x     = np.array([float(input_dict[f]) for f in feats], dtype=float)
    y_hat = float(m["intercept_raw"] + np.dot(m["coef_raw"], x))

    x_df   = pd.DataFrame([input_dict])[feats]
    y_pipe = float(m["pipeline"].predict(x_df)[0])

    return {
        "Model_Rank": model_rank,
        "Model": m["model_key"],
        "Inputs": {f: float(input_dict[f]) for f in feats},
        "Prediction_12M90": y_hat,
        "Formula": m["formula_raw"],
        "Pipeline_Check": y_pipe,
        "Abs_Diff_Formula_vs_Pipeline": abs(y_hat - y_pipe),
    }


# ---- Manuel input tablosu (ilk 3 model için) ----
all_feats = sorted(list(set(sum([v["features_model"] for v in TOP3_MODELS.values()], []))))

manual_inputs = pd.DataFrame([
    {"Case": "Kitle_A", **{f: np.nan for f in all_feats}},
    {"Case": "Kitle_B", **{f: np.nan for f in all_feats}},
    {"Case": "Kitle_C", **{f: np.nan for f in all_feats}},
])
print("\n=== MANUEL INPUT ŞABLONU (DEĞERLERİ DOLDUR) ===")
print(manual_inputs.to_string(index=False))

ornek_inputs = [
    {"Case": "Kitle_0", "2M30": 0.342621, "2M20": 0.804575},
    {"Case": "Kitle_1", "2M30": 0.000000, "2M20": 0.000000},
    {"Case": "GENEL",   "2M30": 0.342621, "2M20": 0.804574},
]
ornek_df = pd.DataFrame(ornek_inputs)

pred_rows = []
for _, r in ornek_df.iterrows():
    in_dict = r.drop(labels=["Case"]).dropna().to_dict()

    for model_rank in sorted(TOP3_MODELS.keys()):
        feats_need = TOP3_MODELS[model_rank]["features_model"]
        if all([f in in_dict for f in feats_need]):
            out = predict_12m90_top3(model_rank, in_dict)
            pred_rows.append({
                "Case": r["Case"],
                "Model_Rank": out["Model_Rank"],
                "Model": out["Model"],
                "Features": ",".join(feats_need),
                "Inputs": str(out["Inputs"]),
                "Intercept": round(TOP3_MODELS[model_rank]["intercept_raw"], 6),
                "Prediction_12M90": round(out["Prediction_12M90"], 6),
                "Formula_vs_Pipeline_Diff": out["Abs_Diff_Formula_vs_Pipeline"],
            })

pred_df = pd.DataFrame(pred_rows).sort_values(
    ["Case", "Model_Rank"]).reset_index(drop=True)

print("\n=== TOP-3 MODEL PARAMETRİK TAHMİN TABLOSU ===")
print(pred_df.to_string(index=False))


# =============================================================================
# 16) SAĞLIK KONTROL LİSTESİ
# =============================================================================
print("\n" + "=" * 70)
print("SAĞLIK KONTROLLERİ")
print("=" * 70)

kontroller = []

cv_wmape  = float(en_iyi["CV_WMAPE"])
oot_wmape = float(en_iyi["OOT_WMAPE"])
cv_mape   = float(en_iyi["CV_MAPE"])
cv_std    = float(en_iyi["CV_MAPE_STD"])
drift_w   = float(en_iyi["Drift_WMAPE"])
ovf_w     = float(en_iyi["Overfit_WMAPE"])
tr_r2     = float(en_iyi["Train_R2"])
cv_r2     = float(en_iyi["CV_R2"])

kontroller.append(("CV_WMAPE <= 18", cv_wmape <= 18, f"{cv_wmape:.2f}"))
kontroller.append(("OOT_WMAPE <= 22", oot_wmape <= 22, f"{oot_wmape:.2f}"))
kontroller.append(("Drift_WMAPE <= 8", drift_w <= 8, f"{drift_w:.2f}"))
kontroller.append(("Overfit_WMAPE <= 8", ovf_w <= 8, f"{ovf_w:.2f}"))
kontroller.append(("CV kararli (std < MAPE/2)", cv_std < cv_mape / 2,
                   f"std={cv_std:.2f} vs MAPE={cv_mape:.2f}"))
kontroller.append(("CV_R2 <= Train_R2 (sizinti yok)", cv_r2 <= tr_r2 + 0.02,
                   f"CV={cv_r2:.3f} vs Train={tr_r2:.3f}"))
kontroller.append(("Durbin-Watson > 1.5", dw > 1.5, f"{dw:.3f}"))
kontroller.append(("Katsayilar pozitif (oncul mantigi)",
                   bool(np.all(CHAMPION["coef_raw"] > 0)),
                   str([round(v, 4) for v in CHAMPION["coef_raw"]])))
kontroller.append(("Alpha tutarli (scan == refit)",
                   bool(np.isclose(nz(CHAMPION["alpha_scan"], -1),
                                   nz(CHAMPION["alpha_refit"], -2))),
                   f"scan={CHAMPION['alpha_scan']} refit={CHAMPION['alpha_refit']}"))
kontroller.append(("Raw formul == pipeline",
                   abs(pipe_pred - predict_12m90_raw(**ornek)) < 1e-8,
                   f"{abs(pipe_pred - predict_12m90_raw(**ornek)):.2e}"))

for ad, gecti, deger in kontroller:
    print(f"  [{'OK ' if gecti else 'UYARI'}] {ad:<38} -> {deger}")

basarisiz = [k[0] for k in kontroller if not k[1]]
if basarisiz:
    print(f"\n{len(basarisiz)} kontrol gecmedi — modeli kullanmadan once incele:")
    for b in basarisiz:
        print(f"   - {b}")
else:
    print("\nTum kontroller gecti.")
