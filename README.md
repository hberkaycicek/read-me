# ============================================================
# BLOK 12 — RAPORLAMA PAKETİ (v5)
#   wf_pencereli'yi yeniden koşmaz; OOT_SONUC / DOGRU / SECILEN üzerinden çalışır
#   12.0  temel + ay normalizasyonu
#   12.1  tam metrik tablosu (OOT + referanslar + nihai senaryolar)
#   12.2  her basamağın formülü ve katsayıları
#   12.3  gerçekleşen vs tahmin serisi + grafikler
#   12.4  gelecek tahminleri (202510–202603)
#   12.5  v2 senaryo tanımları
# ============================================================
import numpy as np, pandas as pd, inspect, re
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LinearRegression, RidgeCV, LassoCV,
                                  ElasticNetCV, HuberRegressor)

pd.set_option("display.width", 320)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

# --- ay normalizasyonu: her şey 'YYYYMM' --------------------------------
def AYS(x):
    return re.sub(r"\D", "", str(x))[:6]

P         = lambda s: pd.Period(AYS(s), freq="M")
BUGUN     = P(CONFIG.get("BILGI_AYI", "2026-10"))
HEDEF_HAM = "KK_TAKIP_90_12_TUTARSAL"

_mask = (d["OLGUN"].astype(bool) if "OLGUN" in d.columns
         else d["HEDEF_KULLANILABILIR"].astype(bool))
d["HEDEF_OLGUN"] = d[HEDEF_HAM].where(_mask)
HEDEF = "HEDEF_OLGUN"

d["_AYS"] = d["AY"].map(AYS)
AY2IDX    = dict(zip(d["_AYS"], d.index))

TR0, TR1 = AYS(CONFIG["TRAIN_ARALIK"][0]), AYS(CONFIG["TRAIN_ARALIK"][1])
AR0, AR1 = AYS(CONFIG["ARA_ARALIK"][0]),   AYS(CONFIG["ARA_ARALIK"][1])
OT0, OT1 = AYS(CONFIG["OOT_ARALIK"][0]),   AYS(CONFIG["OOT_ARALIK"][1])

print("=" * 120)
print("12.0  TEMEL")
print("=" * 120)
print(f"hedef            : {HEDEF_HAM}   ({CONFIG['HEDEF_TUR']}, {CONFIG['HEDEF_SON']})")
print(f"bilgi ayı        : {BUGUN}   |  olgunlaşma {CONFIG['HEDEF_OLGUNLASMA_AY']} ay "
      f"+ rapor gecikmesi {CONFIG['RAPOR_GECIKME_AY']} ay")
print(f"olgun kohort     : {d[HEDEF].notna().sum()}  "
      f"({d.loc[d[HEDEF].notna(),'_AYS'].min()} – {d.loc[d[HEDEF].notna(),'_AYS'].max()})")
print(f"tahmin edilecek  : {d[HEDEF].isna().sum()}  "
      f"({d.loc[d[HEDEF].isna(),'_AYS'].min()} – {d.loc[d[HEDEF].isna(),'_AYS'].max()})")
print(f"TRAIN {TR0}–{TR1}   ARA {AR0}–{AR1}   OOT {OT0}–{OT1}")

# ============================================================
# METRİKLER   (WMAPE ağırlıksız = BLOK 11 ile aynı tanım)
# ============================================================
def metrikler(y, yhat, w=None):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    e = yhat - y
    out = {
        "N":           len(y),
        "WMAPE_%":     100*np.sum(np.abs(e))/np.sum(y),
        "MAPE_%":      100*np.mean(np.abs(e)/y),
        "MAE":         np.mean(np.abs(e)),
        "RMSE":        np.sqrt(np.mean(e**2)),
        "MSE":         np.mean(e**2),
        "SAPMA":       np.mean(e),
        "BIAS_ORANI":  abs(np.mean(e))/np.mean(np.abs(e)),
        "POZ":         int((e > 0).sum()),
        "PEARSON":     stats.pearsonr(y, yhat)[0],
        "SPEARMAN":    stats.spearmanr(y, yhat)[0],
        "R2_donem":    1 - np.sum(e**2)/np.sum((y-y.mean())**2),
        "EN_KOTU":     np.max(np.abs(e)),
    }
    if w is not None:
        w = np.asarray(w, float)
        out["WMAPE_HACIM_%"] = 100*np.sum(w*np.abs(e))/np.sum(w*y)
    return out

# ============================================================
# 12.1  TAM METRİK TABLOSU
# ============================================================
_s = []
for ad, v in OOT_SONUC.items():
    don = [AYS(t) for t in v["_don"]]
    ger, tah = np.array(v["_ger"], float), np.array(v["_tah"], float)
    w = np.array([d.loc[AY2IDX[t], "BASVURU_ADEDI"] for t in don], float)
    m = metrikler(ger, tah, w)
    m.update({"MODEL": ad,
              "TIP": "referans" if ad.startswith("REF") else "merdiven",
              "EGITIM_ILK": v.get("EGITIM_N_ILK"), "EGITIM_SON": v.get("EGITIM_N_SON")})
    _s.append(m)
TABLO = pd.DataFrame(_s)

_ref = TABLO.loc[TABLO.MODEL.str.startswith("REF_Dummy"),  "WMAPE_%"].iloc[0]
_onc = TABLO.loc[TABLO.MODEL.str.startswith("REF_Onceki"), "WMAPE_%"].iloc[0]
TABLO["SKILL_vs_REF"]    = 1 - TABLO["WMAPE_%"]/_ref
TABLO["SKILL_vs_ONCEKI"] = 1 - TABLO["WMAPE_%"]/_onc

_kol = ["MODEL","TIP","N","WMAPE_%","WMAPE_HACIM_%","MAPE_%","MAE","RMSE","MSE",
        "SAPMA","BIAS_ORANI","POZ","PEARSON","SPEARMAN","R2_donem","EN_KOTU",
        "SKILL_vs_REF","SKILL_vs_ONCEKI","EGITIM_ILK","EGITIM_SON"]
print("\n" + "=" * 120)
print(f"12.1  KİLİTLİ TEST (OOT {OT0}–{OT1}) — v1 TABAN KURGU, TÜM METRİKLER")
print("=" * 120)
print(TABLO[_kol].to_string(index=False))
print("\n  WMAPE_%      : ağırlıksız  Σ|hata| / Σgerçek   (BLOK 11 ile aynı tanım)")
print("  WMAPE_HACIM_%: başvuru adediyle ağırlıklı — büyük kohortlara daha çok ağırlık")
print("  R2_donem     : OOT ortalamasına göre; 12 dönemde negatif çıkması olağan")

print("\n--- NİHAİ SENARYOLAR (BLOK 11 seçimi) ---")
_dg = DOGRU.copy()
_dg["SKILL_vs_REF"]    = 1 - _dg["WMAPE"]/_ref
_dg["SKILL_vs_ONCEKI"] = 1 - _dg["WMAPE"]/_onc
print(_dg.to_string(index=False))
print(f"\n  referans (Dummy) WMAPE = {_ref:.4f}   |   önceki çalışma WMAPE = {_onc:.4f}")

print("\n--- v1 → nihai senaryo kazancı ---")
_kars = (TABLO[TABLO.TIP == "merdiven"][["MODEL","WMAPE_%"]]
         .rename(columns={"MODEL":"MERDIVEN","WMAPE_%":"v1_WMAPE"})
         .merge(DOGRU[["MERDIVEN","senaryo","WMAPE","skill","PEARSON"]], on="MERDIVEN"))
_kars["KAZANC_%"] = 100*(_kars.v1_WMAPE - _kars.WMAPE)/_kars.v1_WMAPE
print(_kars.to_string(index=False))

# ============================================================
# 12.2  HER BASAMAĞIN FORMÜLÜ VE KATSAYILARI
# ============================================================
def pipe_kur(aile):
    reg = {"Linear":     LinearRegression(),
           "Ridge":      RidgeCV(alphas=CONFIG["ALPHALAR"]),
           "Lasso":      LassoCV(alphas=CONFIG["ALPHALAR"], max_iter=50000),
           "ElasticNet": ElasticNetCV(alphas=CONFIG["ALPHALAR"], max_iter=50000),
           "Huber":      HuberRegressor(max_iter=2000)}[aile]
    return Pipeline([("s", StandardScaler()), ("m", reg)])

def katsayilar(pipe, kols):
    sc, reg = pipe.named_steps["s"], pipe.named_steps["m"]
    b_std = np.ravel(reg.coef_)
    b_ham = b_std / sc.scale_
    sabit = float(np.ravel(reg.intercept_)[0]) - float(np.sum(b_ham * sc.mean_))
    K = pd.DataFrame({"DEGISKEN": kols, "KATSAYI_STD": b_std, "KATSAYI_HAM": b_ham})
    K["ETKI_%"] = 100*np.abs(K.KATSAYI_STD)/np.abs(K.KATSAYI_STD).sum()
    K["YON"]    = np.where(K.KATSAYI_STD >= 0, "artirir", "azaltir")
    return K.sort_values("ETKI_%", ascending=False).reset_index(drop=True), sabit

tr_tum = d[d[HEDEF].notna()].sort_values("AY")

MODELLER = {}
print("\n" + "=" * 120)
print("12.2  MODEL FORMÜLLERİ")
print("=" * 120)
for ad, cfg in SECILEN.items():
    kols  = list(cfg["kolonlar"]) + list(cfg.get("ek") or [])
    eksik = [c for c in kols if c not in d.columns]
    if eksik:
        print(f"\n--- {ad} --- atlandı, kolon yok: {eksik}")
        continue

    trm = tr_tum.dropna(subset=kols)                    # <-- NaN'li satırları at
    atilan = len(tr_tum) - len(trm)
    w = trm["BASVURU_ADEDI"].values.astype(float)
    w = w/w.mean() if CONFIG.get("HACIM_AGIRLIK") else np.ones(len(trm))

    p = pipe_kur(cfg["aile"])
    try:
        p.fit(trm[kols], trm[HEDEF], m__sample_weight=w)
        agirlikli = True
    except (TypeError, ValueError):
        p.fit(trm[kols], trm[HEDEF])
        agirlikli = False

    K, SABIT = katsayilar(p, kols)
    MODELLER[ad] = {"pipe": p, "kols": kols, "cfg": cfg, "sabit": SABIT,
                    "kats": K, "egitim": trm}

    print(f"\n--- {ad}   (bilgi_ay={cfg['bilgi_ay']}, model={cfg['aile']}, "
          f"hacim ağırlığı={'evet' if agirlikli else 'hayır'}) ---")
    print(f"  eğitim: {trm._AYS.iloc[0]} – {trm._AYS.iloc[-1]}, {len(trm)} kohort"
          + (f"  ({atilan} kohort lag/MA boşluğu nedeniyle atıldı)" if atilan else ""))
    print(K.to_string(index=False))
    print(f"  SABIT = {SABIT:.8f}"
          + (f"   alpha = {p.named_steps['m'].alpha_:.5f}"
             if hasattr(p.named_steps['m'], "alpha_") else ""))
    print(f"  FORMÜL:  tahmin = {SABIT:+.8f}")
    for _k, _b in zip(K.DEGISKEN, K.KATSAYI_HAM):
        print(f"                    {'+' if _b >= 0 else '-'} {abs(_b):.8f} × {_k}")
    _ic = p.predict(trm[kols]); _y = trm[HEDEF].values
    print(f"  eğitim içi: WMAPE %{100*np.sum(np.abs(_ic-_y))/np.sum(_y):.2f}"
          f"   R² {1-np.sum((_ic-_y)**2)/np.sum((_y-_y.mean())**2):.4f}")

# ============================================================
# 12.3  GERÇEKLEŞEN vs TAHMİN
# ============================================================
SERI = pd.DataFrame({"AY":     [AYS(t) for t in OOT_SONUC["CHAMPION"]["_don"]],
                     "GERCEK": OOT_SONUC["CHAMPION"]["_ger"]})
for ad in OOT_SONUC:
    SERI[ad] = OOT_SONUC[ad]["_tah"]
SERI["CHAMPION_HATA"]   = SERI["CHAMPION"] - SERI["GERCEK"]
SERI["CHAMPION_HATA_%"] = 100*SERI["CHAMPION_HATA"]/SERI["GERCEK"]

print("\n" + "=" * 120)
print("12.3  OOT — GERÇEKLEŞEN vs TÜM MODELLERİN TAHMİNİ")
print("=" * 120)
print(SERI.to_string(index=False))

tam = (d[d[HEDEF].notna()].sort_values("AY")[["_AYS", HEDEF]]
       .rename(columns={"_AYS": "AY", HEDEF: "GERCEK"})
       .merge(SERI[["AY", "CHAMPION", "REF_Dummy"]], on="AY", how="left"))
tam["BOLGE"] = np.select([tam.AY <= TR1, tam.AY <= AR1], ["TRAIN", "ARA"], default="OOT")
print("\n--- bölge özeti ---")
print(tam.groupby("BOLGE").agg(N=("GERCEK","size"), GERCEK_ORT=("GERCEK","mean"),
                               MIN=("GERCEK","min"), MAKS=("GERCEK","max")
      ).reindex(["TRAIN","ARA","OOT"]).to_string())

fig, ax = plt.subplots(figsize=(17, 5.5))
ax.axvspan(AR0, AR1, color="#FDF1E3", zorder=0, label="ARA (hakem)")
ax.axvspan(OT0, OT1, color="#E7EFEC", zorder=0, label="OOT (kilitli)")
ax.plot(tam.AY, tam.GERCEK,    "o-",  lw=2.4, ms=4, color="#16253D", label="Gerçekleşen")
ax.plot(tam.AY, tam.CHAMPION,  "s--", lw=2.2, ms=5, color="#C25E00", label="Şampiyon (v1)")
ax.plot(tam.AY, tam.REF_Dummy, ":",   lw=2,         color="#B4BCC6", label="Referans")
ax.set_title("12M90 — TRAIN / ARA / OOT gerçekleşen ve kilitli test tahminleri", fontsize=13)
ax.set_ylabel("12M90 oranı"); ax.legend(ncol=5, fontsize=9)
plt.xticks(rotation=90, fontsize=7); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(17, 4.5))
for ad, c in [("CHAMPION","#C25E00"), ("CHALLENGER","#2E6B5E"),
              ("ULTRA_ERKEN","#8595AF"), ("REF_Dummy","#B4BCC6")]:
    if ad in SERI:
        ax.plot(SERI.AY, SERI[ad], "--", lw=2, color=c, label=ad)
ax.plot(SERI.AY, SERI.GERCEK, "o-", lw=3, ms=6, color="#16253D", label="Gerçekleşen")
ax.set_title("OOT yakın plan — basamakların karşılaştırması", fontsize=13)
ax.legend(ncol=5, fontsize=9); plt.xticks(rotation=45, fontsize=8)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(17, 3))
ax.bar(SERI.AY, SERI.CHAMPION_HATA,
       color=np.where(SERI.CHAMPION_HATA > 0, "#C25E00", "#2E6B5E"))
ax.axhline(0, color="#16253D", lw=1)
ax.set_title("Şampiyon (v1) tahmin hatası  (+ fazla / − eksik)", fontsize=12)
plt.xticks(rotation=45, fontsize=8); plt.tight_layout(); plt.show()

# ============================================================
# 12.4  GELECEK TAHMİNLERİ
# ============================================================
gel   = d[d[HEDEF].isna()].sort_values("AY").copy()
son12 = tr_tum.tail(12)[HEDEF].mean()
_rows, _atlanan = [], []
for ad, M in MODELLER.items():
    ba = M["cfg"]["bilgi_ay"]
    for _, r in gel.iterrows():
        yas = (BUGUN - r["AY"]).n
        bos = [c for c in M["kols"] if pd.isna(r[c])]
        if yas < ba or bos:
            _atlanan.append({"AY": r["_AYS"], "BASAMAK": ad, "YAS_AY": yas,
                             "SEBEP": "olgun değil" if yas < ba else f"boş: {bos}"})
            continue
        _rows.append({"AY": r["_AYS"], "YAS_AY": yas, "BASAMAK": ad, "BILGI_AY": ba,
                      "TAHMIN": float(M["pipe"].predict(r[M["kols"]].to_frame().T)[0])})
GELECEK = pd.DataFrame(_rows)

print("\n" + "=" * 120)
print("12.4  GELECEK TAHMİNLERİ   (v1 taban kurgu — v2c düzeltmesi henüz uygulanmadı)")
print("=" * 120)
if GELECEK.empty:
    print("  Tahmin üretilemedi.")
else:
    GELECEK["SON12_ORT"] = son12
    GELECEK["FARK_%"]    = 100*(GELECEK.TAHMIN - son12)/son12
    print(GELECEK.to_string(index=False))
    print("\n--- basamak × ay matrisi ---")
    print(GELECEK.pivot_table(index="AY", columns="BASAMAK", values="TAHMIN").to_string())
    print(f"\n  son 12 olgun kohort ortalaması : {son12:.4f}")
    _ch = GELECEK[GELECEK.BASAMAK == "CHAMPION"]
    if len(_ch):
        print(f"  şampiyon tahmin ortalaması     : {_ch.TAHMIN.mean():.4f}"
              f"   ({'YUKARI' if _ch.TAHMIN.mean() > son12 else 'AŞAĞI'})")

    fig, ax = plt.subplots(figsize=(17, 5))
    ax.plot(tam.AY, tam.GERCEK, "o-", lw=2.4, ms=4, color="#16253D", label="Gerçekleşen")
    if len(_ch):
        ax.plot(_ch.AY, _ch.TAHMIN, "D--", lw=2.6, ms=9, color="#C25E00",
                label="Gelecek (şampiyon)")
    ax.axvline(tam.AY.iloc[-1], color="#8A93A0", ls=":", lw=1.5)
    ax.set_title("12M90 — gerçekleşen seri ve olgunlaşmamış kohort tahminleri", fontsize=13)
    ax.set_ylabel("12M90 oranı"); ax.legend()
    plt.xticks(rotation=90, fontsize=7); plt.tight_layout(); plt.show()

if _atlanan:
    print("\n--- tahmin üretilemeyen (basamak, ay) çiftleri ---")
    print(pd.DataFrame(_atlanan).to_string(index=False))

# ============================================================
# 12.5  v2 SENARYO TANIMLARI
# ============================================================
print("\n" + "=" * 120)
print("12.5  SENARYO TANIMLARI")
print("=" * 120)
_bulundu = False
for n, v in list(globals().items()):
    if callable(v) and re.search(r"senaryo|v2|oransal|duzelt|kosullu|wf_", n, re.I):
        _bulundu = True
        try:
            print(f"\n--- {n} ---"); print(inspect.getsource(v))
        except Exception:
            print(f"\n--- {n} --- kaynak okunamadı")
if not _bulundu:
    print("  senaryo fonksiyonu bulunamadı — BLOK 10/11 hücresini tekrar çalıştırman gerekebilir.")
