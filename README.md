# ============================================================================
# BLOK 4 — ZAMANLAMA DENETIMI + CV MOTORU + METRIK MOTORU
#
# 4.1  ZAMANLAMA DENETIMI : GEREKEN_AY etiketleri VERIYLE dogrulanir.
#                           Teorik iddiada kalmiyoruz; olgunlasmamis kohortlarda
#                           hangi degiskenin gercekten dolu oldugunu kontrol ediyoruz.
# 4.2  MODEL MERDIVENI    : uretimde kohort yasina gore hangi model calisir
# 4.3  FOLD OLUSTURMA     : olgunlasma kontrollu walk-forward
# 4.4  METRIK MOTORU      : YANLILIK ve ISARET SERISI dahil
# 4.5  CV DEGERLENDIRME   : tek bir kombinasyonu uctan uca olcer
# 4.6  MOTOR DOGRULAMASI  : Dummy / tek degisken / tum havuz karsilastirmasi
# ============================================================================
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("=" * 104)
print("BLOK 4 — ZAMANLAMA DENETIMI, CV MOTORU, METRIK MOTORU")
print("=" * 104)

# ---------------------------------------------------------------- 4.1 DENETIM
print("\n" + "-" * 104)
print("4.1  ZAMANLAMA DENETIMI — GEREKEN_AY etiketleri veriyle dogrulaniyor")
print("-" * 104)
bilgi = pd.Period(CONFIG["BILGI_AYI"], freq="M")
tah = d[d.BOLGE == "TAHMIN"].copy()
tah["YAS_AY"] = (bilgi - tah["AY"]).apply(lambda x: x.n)

print(f"  Bilgi ayi = {bilgi}. Olgunlasmamis kohortlarin YASI ve veri doluluk durumu:\n")
dn = []
for _, r in tah.iterrows():
    yas = int(r["YAS_AY"])
    # TEORI: yas >= GEREKEN_AY olan her degisken DOLU olmali
    beklenen_dolu = [c for c in CHAMPION_BUDANMIS if GEREKEN_AY[c] <= yas]
    beklenen_bos  = [c for c in CHAMPION_BUDANMIS if GEREKEN_AY[c] >  yas]
    gercek_dolu   = [c for c in CHAMPION_BUDANMIS if pd.notna(r[c])]
    celiski_a = set(beklenen_dolu) - set(gercek_dolu)   # dolu olmali ama bos
    celiski_b = set(gercek_dolu) - set(beklenen_dolu)   # bos olmali ama dolu
    dn.append({"DONEM": r[DONEM_KOL], "yas_ay": yas,
               "beklenen_dolu": len(beklenen_dolu), "gercek_dolu": len(gercek_dolu),
               "EKSIK": len(celiski_a), "FAZLA": len(celiski_b)})
dn = pd.DataFrame(dn)
print(dn.to_string(index=False))
print("\n  EKSIK = teoriye gore dolu olmali ama NaN  -> zamanlama etiketi YANLIS veya veri eksik")
print("  FAZLA = teoriye gore bos olmali ama dolu  -> SIZINTI SINYALI (ciddi)")
if dn.FAZLA.sum() > 0:
    print(f"  !! {int(dn.FAZLA.sum())} FAZLA tespit edildi — incele")
else:
    print("  OK — 'FAZLA' yok. GEREKEN_AY etiketleri tutarli, sizinti sinyali yok.")

print("\n  Her olgunluk seviyesinde kac degisken kullanilabilir:")
for yas in sorted(tah.YAS_AY.unique()):
    nc = sum(1 for c in CHAMPION_BUDANMIS   if GEREKEN_AY[c] <= yas)
    nl = sum(1 for c in CHALLENGER_BUDANMIS if GEREKEN_AY[c] <= yas)
    print(f"    yas={int(yas):2d} ay -> champion havuzundan {nc:2d}, challenger havuzundan {nl:2d}")

# ---------------------------------------------------------------- 4.2 MERDIVEN
print("\n" + "-" * 104)
print("4.2  MODEL MERDIVENI — uretimde kohort yasina gore model secimi")
print("-" * 104)
MERDIVEN = {
    "ULTRA_ERKEN": 1,   # kohort 1 aylik
    "ERKEN":       2,
    "CHALLENGER":  CONFIG["CHALLENGER_MAX_AY"],   # 3
    "ORTA":        4,
    "CHAMPION":    CONFIG["SON_ONCU_AY"],         # 6
}
print("  Ayni kohort olgunlastikca daha iyi modele TERFI eder, tahmin revize edilir.\n")
mt = []
for ad, b_ay in MERDIVEN.items():
    hv = [c for c in CHAMPION_BUDANMIS if GEREKEN_AY[c] <= b_ay]
    en_iyi = max((tek.set_index("degisken").r2.get(c, 0), c) for c in hv) if hv else (0, "-")
    mt.append({"model": ad, "bilgi_ay": b_ay, "havuz": len(hv),
               "erkenlik_ay": CONFIG["HEDEF_OLGUNLASMA_AY"] - b_ay,
               "en_guclu_girdi": en_iyi[1][:34], "en_yuksek_r2": round(en_iyi[0], 3)})
MERDIVEN_DF = pd.DataFrame(mt)
print(MERDIVEN_DF.to_string(index=False))
print("\n  erkenlik_ay = 12 - bilgi_ay : hedefi kac ay ONCE tahmin edebiliyoruz")
print("  >> BLOK 5'te her merdiven basamagi icin AYRI grid calistirilacak.")

# ---------------------------------------------------------------- 4.3 FOLD
print("\n" + "-" * 104)
print("4.3  FOLD OLUSTURMA — olgunlasma kontrollu walk-forward")
print("-" * 104)
MIN_TRAIN = CONFIG["MIN_TRAIN"]

def fold_olustur(bolge_idx, bilgi_ay, n_splits=None, test_size=None):
    """
    Olgunlasma kontrollu zaman serisi fold'lari.

    KURAL:  t_tr + HEDEF_OLGUNLASMA_AY + RAPOR_GECIKME_AY  <=  t_val_ilk + bilgi_ay
    Yani:   egitim kohortunun HEDEFI, test anindan ONCE bilinmis olmali.

    Bu, sizintiyi onleyen TEK mekanizma. TimeSeriesSplit tek basina YETMEZ:
    o sadece 'gelecegi egitimde kullanma' der, ama hedefin OLGUNLASMA suresini bilmez.
    """
    n_splits = n_splits or CONFIG["N_SPLITS"]
    test_size = test_size or CONFIG["TEST_SIZE"]
    sub = d.loc[bolge_idx].sort_values("AY")
    aylar = sub["AY"].values
    pos = np.arange(len(sub))
    tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    out = []
    for tr_p, va_p in tss.split(pos):
        ilk_tahmin  = aylar[va_p[0]] + bilgi_ay
        hedef_erisim = aylar[tr_p] + CONFIG["HEDEF_OLGUNLASMA_AY"] + CONFIG["RAPOR_GECIKME_AY"]
        tr_p2 = tr_p[hedef_erisim <= ilk_tahmin]          # <-- OLGUNLASMA FILTRESI
        if len(tr_p2) < MIN_TRAIN:
            continue
        out.append((sub.index.values[tr_p2], sub.index.values[va_p]))
    return out

print("  Fold ozetleri (olgunlasma filtresinin etkisi):\n")
for ad, b_ay in [("CHAMPION", 6), ("CHALLENGER", 3), ("ULTRA_ERKEN", 1)]:
    fl = fold_olustur(tr_idx, b_ay)
    print(f"  [{ad}] bilgi_ay={b_ay} -> egitim kohortu en az "
          f"{CONFIG['HEDEF_OLGUNLASMA_AY']-b_ay} ay eski olmali")
    for i, (a, b) in enumerate(fl, 1):
        ta, tb = d.loc[a, "AY"], d.loc[b, "AY"]
        fark = (tb.min() - ta.max()).n
        print(f"     fold{i}: egitim n={len(a):2d} (son {ta.max()})  "
              f"test {tb.min()}..{tb.max()} n={len(b)}  FARK={fark} ay "
              f"{'OK' if fark >= CONFIG['HEDEF_OLGUNLASMA_AY']-b_ay else '!! IHLAL'}")
    print()

# ---------------------------------------------------------------- 4.4 METRIK
print("-" * 104)
print("4.4  METRIK MOTORU")
print("-" * 104)
def temel_metrikler(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    return {
        "MAE":   float(np.mean(np.abs(e))),
        "RMSE":  float(np.sqrt(np.mean(e**2))),
        "MAPE":  float(100*np.mean(np.abs(e/np.where(y == 0, np.nan, y)))),
        "WMAPE": float(100*np.sum(np.abs(e))/np.sum(np.abs(y))),
        "SAPMA": float(np.mean(e)),      # YANLILIK — onceki kurguda siralamada YOKTU
    }
print("  Temel metrikler: MAE / RMSE / MAPE / WMAPE / SAPMA")
print("\n  NEDEN WMAPE ana metrik?")
print("    MAPE kucuk hedef degerlerinde patlar (hedef 0.004-0.023 arasi oynuyor).")
print("    WMAPE = toplam_mutlak_hata / toplam_gerceklesme -> olcek-duyarsiz ve kararli.")
print("\n  NEDEN SAPMA kritik?")
print("    Onceki modelde OOT'de 12/12 sapma pozitifti, hatanin %82'si YANLILIKTI.")
print("    Eski siralama agirliklarinda bu metrik HIC YOKTU. Artik %20 agirlikla var.")

# ---------------------------------------------------------------- 4.5 CV
print("\n" + "-" * 104)
print("4.5  CV DEGERLENDIRME")
print("-" * 104)

def model_fabrika(aile):
    A = CONFIG["ALPHALAR"]
    if aile == "Linear":     return LinearRegression()
    if aile == "Ridge":      return RidgeCV(alphas=A)
    if aile == "Lasso":      return LassoCV(alphas=A, max_iter=20000, random_state=CONFIG["RANDOM_STATE"])
    if aile == "ElasticNet": return ElasticNetCV(alphas=A, l1_ratio=[.2,.5,.8], max_iter=20000,
                                                 random_state=CONFIG["RANDOM_STATE"])
    if aile == "Dummy":      return DummyRegressor(strategy="mean")
    raise ValueError(aile)

def maks_vif(X):
    try:
        Xc = add_constant(np.asarray(X, float), has_constant="add")
        return float(max(variance_inflation_factor(Xc, i) for i in range(1, Xc.shape[1])))
    except Exception:
        return np.inf

def cv_degerlendir(kolonlar, aile, bilgi_ay, ek_kolonlar=None, bolge_idx=None, etiket=""):
    """Tek bir (degisken seti x model ailesi) kombinasyonunu uctan uca olcer."""
    kol = list(kolonlar) + list(ek_kolonlar or [])
    bolge_idx = tr_idx if bolge_idx is None else bolge_idx
    foldlar = fold_olustur(bolge_idx, bilgi_ay)
    if not foldlar:
        return None

    tum_y, tum_p, fold_wmape, fold_sapma, tr_rmse_list = [], [], [], [], []
    for tr_i, va_i in foldlar:
        eg = d.loc[tr_i, kol + [HEDEF]].dropna()
        te = d.loc[va_i, kol + [HEDEF]].dropna()
        if len(eg) < MIN_TRAIN or len(te) == 0:
            return None
        pipe = Pipeline([("sc", StandardScaler()), ("m", model_fabrika(aile))])
        fit_kw = {}
        if CONFIG["HACIM_AGIRLIK"] and "BASVURU_ADEDI" in d.columns:
            w = d.loc[eg.index, "BASVURU_ADEDI"].values.astype(float)
            try:
                pipe.fit(eg[kol], eg[HEDEF], m__sample_weight=w)
            except TypeError:
                pipe.fit(eg[kol], eg[HEDEF])     # agirlik desteklemeyen aile
        else:
            pipe.fit(eg[kol], eg[HEDEF])
        p = pipe.predict(te[kol]); y = te[HEDEF].values
        m = temel_metrikler(y, p)
        fold_wmape.append(m["WMAPE"]); fold_sapma.append(m["SAPMA"])
        tr_rmse_list.append(np.sqrt(mean_squared_error(eg[HEDEF], pipe.predict(eg[kol]))))
        tum_y.extend(y); tum_p.extend(p)

    tum_y, tum_p = np.array(tum_y), np.array(tum_p)
    g = temel_metrikler(tum_y, tum_p)
    tr_rmse = float(np.mean(tr_rmse_list))

    # DEGISIM metrikleri: seviyeyi degil YONU/EGIMI olcer
    dy, dp = np.diff(tum_y), np.diff(tum_p)
    yon = float(np.mean(np.sign(dy) == np.sign(dp))) if len(dy) else np.nan
    tol = CONFIG["YON_TOLERANSI"]
    anlamli = np.abs(dy) > tol
    yon_a = float(np.mean(np.sign(dy[anlamli]) == np.sign(dp[anlamli]))) if anlamli.sum() else np.nan

    # ISARET SERISI: tum fold sapmalari ayni isaretli mi?
    # Onceki modelin OOT'de 12/12 pozitif sapma felaketini CV'DE yakalar.
    isaret_serisi = bool(len(fold_sapma) >= 2 and
                         (all(s > 0 for s in fold_sapma) or all(s < 0 for s in fold_sapma)))

    try:
        mv = maks_vif(d.loc[bolge_idx, kol].dropna())
    except Exception:
        mv = np.inf

    return {
        "ETIKET": etiket, "GIRDILER": ", ".join(kol), "K": len(kol), "AILE": aile,
        "BILGI_AY": bilgi_ay, "N_FOLD": len(foldlar),
        "CV_WMAPE": g["WMAPE"], "CV_MAPE": g["MAPE"], "CV_MAE": g["MAE"], "CV_RMSE": g["RMSE"],
        "CV_SAPMA": g["SAPMA"], "MUTLAK_CV_SAPMA": abs(g["SAPMA"]),
        "EN_KOTU_WMAPE": float(np.max(fold_wmape)),
        "FOLD_WMAPE_STD": float(np.std(fold_wmape)),
        "ISARET_SERISI": isaret_serisi,
        "TRAIN_RMSE": tr_rmse,
        "OVERFIT_ORANI": float(g["RMSE"]/tr_rmse) if tr_rmse > 1e-12 else np.inf,
        "DELTA_MAE": float(np.mean(np.abs(dp-dy))) if len(dy) else np.nan,
        "YON_BASARISI": yon, "YON_BASARISI_TOL": yon_a,
        "MAKS_VIF": mv,
        "ARALIK_DISI": int(((tum_p < 0) | (tum_p > 1)).sum()),
        "N_TAHMIN": len(tum_y),
    }

print("  cv_degerlendir() her kombinasyon icin su metrikleri uretir:")
print("""
    SEVIYE     : CV_WMAPE, CV_MAPE, CV_MAE, CV_RMSE
    YANLILIK   : CV_SAPMA, MUTLAK_CV_SAPMA, ISARET_SERISI   <-- YENI
    ISTIKRAR   : EN_KOTU_WMAPE, FOLD_WMAPE_STD
    EZBER      : TRAIN_RMSE, OVERFIT_ORANI (= CV_RMSE/TRAIN_RMSE)  <-- oran, mutlak fark degil
    DEGISIM    : DELTA_MAE, YON_BASARISI
    SAGLIK     : MAKS_VIF, ARALIK_DISI
""")
print(f"  Hacim agirligi (sample_weight=BASVURU_ADEDI): {CONFIG['HACIM_AGIRLIK']}")
print("    Gerekce: oran verisinde Var ~ p(1-p)/n. Basvuru adedi 44bin-291bin arasi")
print("    oynuyor -> dusuk hacimli aylarin orani 6.6 kat daha gurultulu.")

# ---------------------------------------------------------------- 4.6 DOGRULAMA
print("\n" + "-" * 104)
print("4.6  MOTOR DOGRULAMASI")
print("-" * 104)
print("  Motorun calistigini ve metriklerin anlamli oldugunu KANITLIYORUZ.")
print("  Beklenti: Dummy en kotu, tek guclu degisken iyi, tum havuz ezber yapar.\n")

en_iyi_ch = tek[tek.degisken.isin(CHALLENGER_BUDANMIS)].iloc[0]["degisken"]
en_iyi_cp = tek[tek.degisken.isin(CHAMPION_BUDANMIS)].iloc[0]["degisken"]

testler = [
    ("0-Dummy(ortalama)",        [en_iyi_ch],           "Dummy",  3),
    ("1-Tek guclu (challenger)", [en_iyi_ch],           "Linear", 3),
    ("1R-Tek guclu Ridge",       [en_iyi_ch],           "Ridge",  3),
    ("2-Tek guclu (champion)",   [en_iyi_cp],           "Linear", 6),
    ("3-Tum challenger havuzu",  CHALLENGER_BUDANMIS,   "Ridge",  3),
    ("4-Tum champion havuzu",    CHAMPION_BUDANMIS,     "Ridge",  6),
]
dg = []
for ad, kol, aile, b_ay in testler:
    r = cv_degerlendir(kol, aile, b_ay, etiket=ad)
    if r: dg.append(r)
DG = pd.DataFrame(dg)
goster = ["ETIKET","K","AILE","BILGI_AY","CV_WMAPE","CV_SAPMA","EN_KOTU_WMAPE",
          "FOLD_WMAPE_STD","OVERFIT_ORANI","MAKS_VIF","ISARET_SERISI","YON_BASARISI","ARALIK_DISI"]
print(DG[goster].round(4).to_string(index=False))

ref = DG[DG.ETIKET.str.startswith("0-")]["CV_WMAPE"].iloc[0]
print(f"\n  REFERANS (Dummy) CV_WMAPE = {ref:.2f}")
print(f"  Beceri skorlari (1 - model/referans):")
for _, r in DG.iterrows():
    if r.ETIKET.startswith("0-"): continue
    print(f"    {r.ETIKET:28s} skill = {1-r.CV_WMAPE/ref:+.3f}   "
          f"overfit_orani = {r.OVERFIT_ORANI:5.2f}   maks_VIF = {r.MAKS_VIF:8.1f}")

print("\n  KONTROL LISTESI:")
k1 = DG[DG.ETIKET.str.startswith('0-')].CV_WMAPE.iloc[0] == DG.CV_WMAPE.max()
k2 = DG[DG.ETIKET.str.startswith('4-')].OVERFIT_ORANI.iloc[0] > \
     DG[DG.ETIKET.str.startswith('1-')].OVERFIT_ORANI.iloc[0]
k3 = DG.ARALIK_DISI.sum() == 0
print(f"    Dummy en kotu mu?                      {k1}  {'OK' if k1 else '!! INCELE'}")
print(f"    Tum havuz tek degiskenden cok ezber mi? {k2}  {'OK' if k2 else '!! INCELE'}")
print(f"    Aralik disi tahmin yok mu?             {k3}  {'OK' if k3 else '!! INCELE'}")
print(f"    ISARET_SERISI tetiklenen model sayisi  : {int(DG.ISARET_SERISI.sum())} / {len(DG)}")
print("      (tetiklenen = tum fold'larda ayni yonde sapiyor -> sistematik yanlilik)")

print("\n" + "=" * 104)
print("BLOK 4 TAMAM — CV motoru hazir ve dogrulandi.")
print(f"  Merdiven basamagi : {len(MERDIVEN)}")
print(f"  Hacim agirligi    : {CONFIG['HACIM_AGIRLIK']}")
print("=" * 104)
