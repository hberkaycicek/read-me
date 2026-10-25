# ============================================================
# BLOK 12 — RAPORLAMA PAKETİ
# 12.1 tam metrik tablosu (tüm basamaklar × üç bölge)
# 12.2 şampiyon modelin formülü ve katsayıları
# 12.3 gerçekleşen vs tahmin serisi (TRAIN/ARA/OOT) + grafik
# 12.4 gelecek tahminleri (202510–202603)
# ============================================================
import numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

HEDEF = CONFIG.get("HEDEF_KOLON") or [c for c in d.columns if "12M90" in str(c).upper()][0]

# --- bölge sınırları (BLOK 4'teki değerlerle aynı olmalı) -----------------
P = lambda s: pd.Period(s, freq="M")
SINIR = {"TRAIN": (P("2021-01"), P("2024-03")),
         "ARA":   (P("2024-04"), P("2024-09")),
         "OOT":   (P("2024-10"), P("2025-09"))}

def bolge_idx(ad):
    a, b = SINIR[ad]
    m = (d["AY"] >= a) & (d["AY"] <= b) & d[HEDEF].notna()
    return d.index[m]

# --- metrik seti ---------------------------------------------------------
def metrikler(y, yhat, w=None):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    w = np.ones(len(y)) if w is None else np.asarray(w, float)[ok]
    e = yhat - y
    out = {
        "N":        len(y),
        "MAE":      np.mean(np.abs(e)),
        "MSE":      np.mean(e**2),
        "RMSE":     np.sqrt(np.mean(e**2)),
        "MAPE_%":   100*np.mean(np.abs(e)/y),
        "WMAPE_%":  100*np.sum(w*np.abs(e))/np.sum(w*y),
        "SAPMA":    np.mean(e),                      # + = fazla tahmin
        "BIAS_ORANI": abs(np.mean(e))/np.mean(np.abs(e)),
        "PEARSON":  stats.pearsonr(y, yhat)[0] if len(y) > 2 else np.nan,
        "SPEARMAN": stats.spearmanr(y, yhat)[0] if len(y) > 2 else np.nan,
        "ARTI_HATA": int((e > 0).sum()),
    }
    out["R2"] = 1 - np.sum(e**2)/np.sum((y-y.mean())**2) if len(y) > 2 else np.nan
    return out

# --- 12.1  BASAMAK × BÖLGE TABLOSU --------------------------------------
# BLOK 11'de seçilen nihai kurulumları buraya yaz (basamak adı -> wf_pencereli argümanları)
FINAL = {
  # ÖRNEK — BLOK 11 çıktısındaki gerçek değerlerle değiştir:
  # "OLGUN_6M": dict(kolonlar=[...], aile="OLGUN", bilgi_ay=6,
  #                  max_gecmis=30, oran_paydasi="KK_TAKIP_30_2_TUTARSAL",
  #                  duzelt_pencere=6, ek_kolonlar=["MM_SKOR","AY_SIN","AY_COS"]),
}

satir = []
for ad, kw in FINAL.items():
    for bolge in ["TRAIN", "ARA", "OOT"]:
        idx = bolge_idx(bolge)
        res = wf_pencereli(hedef_idx=idx, **kw)
        r = res if isinstance(res, pd.DataFrame) else pd.DataFrame(res)
        gy  = d.loc[r.index, HEDEF].values
        w   = d.loc[r.index, "BASVURU_ADEDI"].values
        yh  = r.iloc[:, 0].values if "TAHMIN" not in r else r["TAHMIN"].values
        satir.append({"BASAMAK": ad, "BOLGE": bolge, **metrikler(gy, yh, w)})

TABLO = pd.DataFrame(satir)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(TABLO.round(4).to_string(index=False))
TABLO.to_excel("12_metrik_tablosu.xlsx", index=False)

# --- 12.2  ŞAMPİYONUN KATSAYILARI --------------------------------------
SAMPIYON = "OLGUN_6M"        # FINAL içindeki şampiyon anahtarı
kw = FINAL[SAMPIYON]
kols = list(kw["kolonlar"]) + list(kw.get("ek_kolonlar") or [])
son_idx = bolge_idx("OOT")                 # en güncel eğitim penceresi
tr = d.loc[d["AY"] <= SINIR["OOT"][1]].dropna(subset=[HEDEF]).sort_values("AY")
if kw.get("max_gecmis"): tr = tr.iloc[-kw["max_gecmis"]:]

pipe = pipeline_kur(kw["aile"]) if "pipeline_kur" in dir() else None   # BLOK 5'teki kurucu
X, y = tr[kols], tr[HEDEF]
payda = kw.get("oran_paydasi")
if payda: y = y / tr[payda]
_fit_agirlikli(pipe, X, y, _agirlik(tr.index), kw["aile"])

def katsayilar(pipe, kols):
    m = pipe
    for attr in ("regressor_", "named_steps"):
        if hasattr(m, "regressor_"): m = m.regressor_
    sc = m.named_steps.get("s"); reg = m.named_steps.get("m")
    b_std = np.ravel(reg.coef_)
    b_raw = b_std / sc.scale_ if sc is not None else b_std
    sabit = float(reg.intercept_) - (np.sum(b_raw * sc.mean_) if sc is not None else 0)
    return pd.DataFrame({"DEGISKEN": kols, "KATSAYI_STD": b_std,
                         "KATSAYI_HAM": b_raw}), sabit

KATS, SABIT = katsayilar(pipe, kols)
KATS["ETKI_%"] = 100*np.abs(KATS.KATSAYI_STD)/np.abs(KATS.KATSAYI_STD).sum()
print(KATS.round(6).to_string(index=False)); print("SABIT =", round(SABIT, 6))
print("\nFORMÜL:\n  tahmin =", round(SABIT, 5), "".join(
    f"\n           {'+' if b>=0 else '-'} {abs(b):.5f} × {k}"
    for k, b in zip(KATS.DEGISKEN, KATS.KATSAYI_HAM)))
if payda: print(f"  (sonuç {payda} ile çarpılır — oransal hedef)")
if kw.get("duzelt_pencere"): print(f"  (son {kw['duzelt_pencere']} kohortun ortalama sapması düşülür)")
KATS.to_excel("12_katsayilar.xlsx", index=False)

# --- 12.3  GERÇEKLEŞEN vs TAHMİN SERİSİ --------------------------------
parcalar = []
for bolge in ["TRAIN", "ARA", "OOT"]:
    idx = bolge_idx(bolge)
    res = wf_pencereli(hedef_idx=idx, **kw)
    r = res if isinstance(res, pd.DataFrame) else pd.DataFrame(res)
    yh = r["TAHMIN"].values if "TAHMIN" in r else r.iloc[:, 0].values
    parcalar.append(pd.DataFrame({
        "AY": d.loc[r.index, "AY"].astype(str).values, "BOLGE": bolge,
        "GERCEK": d.loc[r.index, HEDEF].values, "TAHMIN": yh}))
SERI = pd.concat(parcalar).sort_values("AY")
SERI["HATA"] = SERI.TAHMIN - SERI.GERCEK
SERI["HATA_%"] = 100*SERI.HATA/SERI.GERCEK
print(SERI.round(4).to_string(index=False))
SERI.to_excel("12_seri.xlsx", index=False)

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(SERI.AY, SERI.GERCEK, "o-", lw=2.5, color="#16253D", label="Gerçekleşen")
ax.plot(SERI.AY, SERI.TAHMIN, "s--", lw=2, color="#C25E00", label="Tahmin")
for b, c in [("ARA", "#FDF1E3"), ("OOT", "#E7EFEC")]:
    s = SERI[SERI.BOLGE == b]
    if len(s): ax.axvspan(s.AY.iloc[0], s.AY.iloc[-1], color=c, zorder=0)
ax.set_title("12M90 — gerçekleşen vs tahmin (TRAIN / ARA / OOT)")
ax.legend(); plt.xticks(rotation=90, fontsize=7); plt.tight_layout(); plt.show()

# --- 12.4  GELECEK TAHMİNLERİ (202510–202603) --------------------------
gelecek = d[d[HEDEF].isna()].sort_values("AY")
Xg = gelecek[kols]
yg = pipe.predict(Xg)
if payda: yg = yg * gelecek[payda].values
if kw.get("duzelt_pencere"):
    son = SERI[SERI.BOLGE == "OOT"].tail(kw["duzelt_pencere"])
    yg = yg - son.HATA.mean()
GELECEK = pd.DataFrame({"AY": gelecek["AY"].astype(str).values,
                        "TAHMIN_%": yg,
                        "OLGUNLUK_AY": [(P("2026-10") - a).n for a in gelecek["AY"]]})
print(GELECEK.round(4).to_string(index=False))
GELECEK.to_excel("12_gelecek.xlsx", index=False)
