# ============================================================================
# BLOK 5 — MERDIVEN BAZLI GRID ARAMASI
#
# BLOK 4 BULGULARINA GORE IKI DUZELTME:
#   1) ESIK KALIBRASYONU: OVERFIT_ORANI_MAX=2.0 cok sikiydi — Dummy bile 2.94.
#      Esikler artik DUMMY REFERANSINA gore otomatik hesaplaniyor, sihirli sayi yok.
#   2) ISARET_SERISI tek basina diskalifiye edemez (3 fold'da sans eseri %25).
#      Yeni kural: isaret serisi VE sapma buyuk -> ele.
#
# AYRICA:
#   - Siralama min-max yerine YUZDELIK RANK (havuza bagimli degil)
#   - ORANSAL model ailesi eklendi (BLOK 3.2: egim CV %55.8 vs carpan CV %17.6)
#   - Grid'de sadece hizli aileler (Linear/Ridge/Oransal); Lasso/ElasticNet BLOK 6'da
#   - Her MERDIVEN basamagi icin AYRI grid
# ============================================================================
import time
from itertools import combinations

print("=" * 106)
print("BLOK 5 — MERDIVEN BAZLI GRID ARAMASI")
print("=" * 106)

# ---------------------------------------------------------------- 5.1 ORANSAL MODEL
class OransalModel(BaseEstimator, RegressorMixin):
    """
    y = k1*x1 + k2*x2 + ...   (SABIT TERIM YOK, ham olcekte)

    NEDEN: BLOK 3.2 kayan pencere sonucu
             carpan k CV = %17.6
             egim     CV = %55.8   -> 3.2 KAT daha oynak
           Sabit terim olmadan model olcek degisimine dayanikli olur.
           Onceki modelde sabit terim 0.0447 idi (hedef ortalamasinin 2.9 kati),
           KKB terimi onu kesiyordu -> kirilgan parametrizasyon.
    """
    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, float); y = np.asarray(y, float)
        if sample_weight is not None:
            w = np.sqrt(np.asarray(sample_weight, float))
            self.coef_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        else:
            self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        return self
    def predict(self, X):
        return np.asarray(X, float) @ self.coef_

_eski_fabrika = model_fabrika
def model_fabrika(aile):
    if aile == "Oransal": return OransalModel()
    return _eski_fabrika(aile)

# Oransal model STANDARDIZE EDILMEMELI (sabit terimsizligin anlami kalmaz)
def pipeline_kur(aile):
    if aile == "Oransal":
        return Pipeline([("m", OransalModel())])
    return Pipeline([("sc", StandardScaler()), ("m", model_fabrika(aile))])

# cv_degerlendir'i oransal modeli destekleyecek sekilde guncelle
def cv_degerlendir(kolonlar, aile, bilgi_ay, ek_kolonlar=None, bolge_idx=None, etiket=""):
    kol = list(kolonlar) + list(ek_kolonlar or [])
    bolge_idx = tr_idx if bolge_idx is None else bolge_idx
    foldlar = fold_olustur(bolge_idx, bilgi_ay)
    if not foldlar: return None

    tum_y, tum_p, fold_wmape, fold_sapma, tr_rmse_list = [], [], [], [], []
    for tr_i, va_i in foldlar:
        eg = d.loc[tr_i, kol + [HEDEF]].dropna()
        te = d.loc[va_i, kol + [HEDEF]].dropna()
        if len(eg) < MIN_TRAIN or len(te) == 0: return None
        pipe = pipeline_kur(aile)
        if CONFIG["HACIM_AGIRLIK"] and "BASVURU_ADEDI" in d.columns:
            w = d.loc[eg.index, "BASVURU_ADEDI"].values.astype(float)
            try:    pipe.fit(eg[kol], eg[HEDEF], m__sample_weight=w)
            except TypeError: pipe.fit(eg[kol], eg[HEDEF])
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
    dy, dp = np.diff(tum_y), np.diff(tum_p)
    yon = float(np.mean(np.sign(dy) == np.sign(dp))) if len(dy) else np.nan
    isaret_serisi = bool(len(fold_sapma) >= 2 and
                         (all(s > 0 for s in fold_sapma) or all(s < 0 for s in fold_sapma)))
    try:    mv = maks_vif(d.loc[bolge_idx, kol].dropna())
    except Exception: mv = np.inf
    return {
        "ETIKET": etiket, "GIRDILER": ", ".join(kol), "K": len(kol), "AILE": aile,
        "BILGI_AY": bilgi_ay, "N_FOLD": len(foldlar),
        "CV_WMAPE": g["WMAPE"], "CV_MAPE": g["MAPE"], "CV_MAE": g["MAE"], "CV_RMSE": g["RMSE"],
        "CV_SAPMA": g["SAPMA"], "MUTLAK_CV_SAPMA": abs(g["SAPMA"]),
        "EN_KOTU_WMAPE": float(np.max(fold_wmape)), "FOLD_WMAPE_STD": float(np.std(fold_wmape)),
        "ISARET_SERISI": isaret_serisi, "TRAIN_RMSE": tr_rmse,
        "OVERFIT_ORANI": float(g["RMSE"]/tr_rmse) if tr_rmse > 1e-12 else np.inf,
        "DELTA_MAE": float(np.mean(np.abs(dp-dy))) if len(dy) else np.nan,
        "YON_BASARISI": yon, "MAKS_VIF": mv,
        "ARALIK_DISI": int(((tum_p < 0) | (tum_p > 1)).sum()), "N_TAHMIN": len(tum_y),
    }

# ---------------------------------------------------------------- 5.2 ESIK KALIBRASYONU
print("\n" + "-" * 106)
print("5.2  ESIK KALIBRASYONU — Dummy referansina gore, sihirli sayi yok")
print("-" * 106)
REF = {}
for ad, b_ay in MERDIVEN.items():
    hv = [c for c in CHAMPION_BUDANMIS if GEREKEN_AY[c] <= b_ay]
    r = cv_degerlendir([hv[0]], "Dummy", b_ay, etiket=f"REF_{ad}")
    if r: REF[ad] = r

ref_df = pd.DataFrame(REF).T[["CV_WMAPE","MUTLAK_CV_SAPMA","OVERFIT_ORANI","EN_KOTU_WMAPE"]]
print("  Her merdiven basamaginda DUMMY (ortalama) referansi:")
print(ref_df.round(4).to_string())

ESIK = {}
for ad in MERDIVEN:
    r = REF[ad]
    ESIK[ad] = {
        # Dummy'nin hatasinin en fazla %45'i  (Dummy 52.3 -> esik ~23.5)
        "CV_WMAPE_MAX":      0.45 * r["CV_WMAPE"],
        "EN_KOTU_WMAPE_MAX": 0.60 * r["EN_KOTU_WMAPE"],
        # Dummy'nin yanliliginin en fazla %30'u
        "MUTLAK_SAPMA_MAX":  max(0.30 * r["MUTLAK_CV_SAPMA"], 1e-4),
        # Dummy'nin ezber oraninin uzerine cikma (yapisal taban bu)
        "OVERFIT_ORANI_MAX": 1.10 * r["OVERFIT_ORANI"],
        "MAKS_VIF_MAX":      10.0,
    }
print("\n  KALIBRE EDILMIS ESIKLER:")
print(pd.DataFrame(ESIK).T.round(4).to_string())
print("\n  Eski sabit esikler: CV_WMAPE<=20, EN_KOTU<=28, OVERFIT<=2.0, SAPMA<=0.0015")
print("  >> OVERFIT 2.0 idi ama DUMMY BILE 2.94 veriyordu -> hicbir model gecemezdi.")
print("  >> Artik her basamagin kendi veri yapisina gore kalibre.")

# ---------------------------------------------------------------- 5.3 GECERLILIK
def gecerlilik_uygula(df, ad):
    e = ESIK[ad]
    df = df.copy()
    df["G_WMAPE"]   = df.CV_WMAPE        <= e["CV_WMAPE_MAX"]
    df["G_ENKOTU"]  = df.EN_KOTU_WMAPE   <= e["EN_KOTU_WMAPE_MAX"]
    df["G_SAPMA"]   = df.MUTLAK_CV_SAPMA <= e["MUTLAK_SAPMA_MAX"]
    df["G_OVERFIT"] = df.OVERFIT_ORANI   <= e["OVERFIT_ORANI_MAX"]
    df["G_VIF"]     = df.MAKS_VIF        <= e["MAKS_VIF_MAX"]
    df["G_ARALIK"]  = df.ARALIK_DISI     == 0
    # ISARET SERISI: tek basina degil, SAPMA ILE BIRLIKTE
    # 3 fold'da tum isaretlerin ayni olmasi sans eseri %25 -> tek basina anlamsiz.
    # Asil tehlike: ayni yonde VE buyuk sapma (onceki modelin OOT'de 12/12 +0.283pp'si)
    df["G_ISARET"]  = ~(df.ISARET_SERISI & (df.MUTLAK_CV_SAPMA > 0.5*e["MUTLAK_SAPMA_MAX"]))
    df["GECERLI"]   = (df.G_WMAPE & df.G_ENKOTU & df.G_SAPMA &
                       df.G_OVERFIT & df.G_VIF & df.G_ARALIK & df.G_ISARET)
    return df

# ---------------------------------------------------------------- 5.4 SIRALAMA
def rank_ekle(df):
    """
    Siralama: MIN-MAX degil YUZDELIK RANK.
    Neden: min-max havuza bagimli — tek bir felaket model girerse iyiler 0'a sikisir.
           Yuzdelik rank dagilim sekline dayaniksiz.
    """
    df = df.copy()
    for m, w in CONFIG["AGIRLIK"].items():
        if m not in df.columns:
            df[f"R_{m}"] = 0.0; continue
        df[f"R_{m}"] = df[m].rank(pct=True, na_option="bottom")
    df["RANK_SKORU"] = sum(CONFIG["AGIRLIK"][m] * df[f"R_{m}"] for m in CONFIG["AGIRLIK"])
    return df.sort_values(["GECERLI", "RANK_SKORU"], ascending=[False, True])

# ---------------------------------------------------------------- 5.5 GRID
GRID_AILELERI = ["Linear", "Ridge", "Oransal"]
print("\n" + "-" * 106)
print("5.5  GRID ARAMASI")
print("-" * 106)
print(f"  Grid aileleri : {GRID_AILELERI}")
print(f"    Lasso/ElasticNet GRID'DE YOK — ic CV'leri yavas ve degisken secimini")
print(f"    zaten grid yapiyor. BLOK 6'da (Optuna) genis aile seti denenecek.")
print(f"  Oransal       : sabit terimsiz, standardize edilmez (ham olcek)")

MEVSIM_SET = [[]] + ([["SIN_AY","COS_AY"]] if CONFIG["MEVSIM_DENE"] else [])

def grid_calistir(ad, bilgi_ay, havuz):
    aday = [c for c in havuz if GEREKEN_AY[c] <= bilgi_ay]
    kmbs = [a for k in range(1, CONFIG["MAX_K"]+1)
              for a in combinations(aday, k) if kombinasyon_gecerli(a)]
    top = len(kmbs) * len(MEVSIM_SET) * len(GRID_AILELERI)
    print(f"\n  [{ad}] bilgi_ay={bilgi_ay}  havuz={len(aday)}  "
          f"kombinasyon={len(kmbs):,}  varyant={top:,}")
    t0 = time.time(); out = []; i = 0
    for alt in kmbs:
        for ek in MEVSIM_SET:
            for aile in GRID_AILELERI:
                i += 1
                r = cv_degerlendir(alt, aile, bilgi_ay, ek_kolonlar=ek, etiket=ad)
                if r:
                    r["MEVSIM"] = int(len(ek) > 0); r["MERDIVEN"] = ad
                    out.append(r)
        if i % max(1, top//8) < len(MEVSIM_SET)*len(GRID_AILELERI):
            print(f"      ... %{100*i/top:5.1f}  ({time.time()-t0:5.1f}s)", end="\r")
    df = pd.DataFrame(out)
    print(f"      tamamlandi: {len(df):,} gecerli hesap, {time.time()-t0:.1f}s" + " "*20)
    return rank_ekle(gecerlilik_uygula(df, ad))

SONUC = {}
for ad, b_ay in MERDIVEN.items():
    SONUC[ad] = grid_calistir(ad, b_ay, CHAMPION_BUDANMIS)

# ---------------------------------------------------------------- 5.6 SONUCLAR
print("\n" + "=" * 106)
print("5.6  SONUCLAR")
print("=" * 106)
GOSTER = ["GIRDILER","K","AILE","MEVSIM","CV_WMAPE","CV_SAPMA","EN_KOTU_WMAPE",
          "FOLD_WMAPE_STD","OVERFIT_ORANI","MAKS_VIF","ISARET_SERISI","YON_BASARISI",
          "GECERLI","RANK_SKORU"]

ozet = []
for ad, b_ay in MERDIVEN.items():
    df = SONUC[ad]
    g = df[df.GECERLI]
    print(f"\n{'='*106}")
    print(f"[{ad}]  bilgi_ay={b_ay}  erkenlik={CONFIG['HEDEF_OLGUNLASMA_AY']-b_ay} ay  "
          f"|  {len(df):,} varyant, {len(g):,} GECERLI (%{100*len(g)/max(len(df),1):.1f})")
    print("=" * 106)
    if len(g) == 0:
        print("  !! HIC GECERLI MODEL YOK. Elenme sebepleri:")
        for kr in ["G_WMAPE","G_ENKOTU","G_SAPMA","G_OVERFIT","G_VIF","G_ARALIK","G_ISARET"]:
            print(f"     {kr:12s} gecen: {int(df[kr].sum()):5,} / {len(df):,}")
        continue
    t = g.head(8)[GOSTER].copy()
    t["GIRDILER"] = t.GIRDILER.str.replace("KK_TAKIP_","").str.replace(f"_{T}","")
    print(t.round(4).to_string(index=False))
    b = g.iloc[0]
    ozet.append({"MERDIVEN": ad, "bilgi_ay": b_ay,
                 "erkenlik_ay": CONFIG["HEDEF_OLGUNLASMA_AY"]-b_ay,
                 "GIRDILER": b.GIRDILER.replace("KK_TAKIP_","").replace(f"_{T}",""),
                 "K": b.K, "AILE": b.AILE, "MEVSIM": b.MEVSIM,
                 "CV_WMAPE": b.CV_WMAPE, "CV_SAPMA": b.CV_SAPMA,
                 "OVERFIT_ORANI": b.OVERFIT_ORANI, "MAKS_VIF": b.MAKS_VIF,
                 "gecerli_n": len(g), "skill": 1 - b.CV_WMAPE/REF[ad]["CV_WMAPE"]})

print("\n" + "=" * 106)
print("MERDIVEN OZETI — her basamagin sampiyonu")
print("=" * 106)
OZET = pd.DataFrame(ozet)
print(OZET.round(4).to_string(index=False))

print("\n  ERKENLIK / DOGRULUK DENGESI:")
for _, r in OZET.iterrows():
    print(f"    {r.MERDIVEN:12s} {int(r.erkenlik_ay):2d} ay erken  "
          f"CV_WMAPE={r.CV_WMAPE:5.2f}  skill={r.skill:+.3f}  "
          f"sapma={r.CV_SAPMA:+.5f}")
if len(OZET) >= 2:
    en_erken = OZET.iloc[OZET.erkenlik_ay.idxmax()] if False else OZET.sort_values("erkenlik_ay").iloc[-1]
    en_iyi   = OZET.sort_values("CV_WMAPE").iloc[0]
    print(f"\n  En ERKEN  : {en_erken.MERDIVEN} ({int(en_erken.erkenlik_ay)} ay) WMAPE={en_erken.CV_WMAPE:.2f}")
    print(f"  En DOGRU  : {en_iyi.MERDIVEN} ({int(en_iyi.erkenlik_ay)} ay) WMAPE={en_iyi.CV_WMAPE:.2f}")
    print(f"  BEDEL     : {en_erken.CV_WMAPE - en_iyi.CV_WMAPE:+.2f} WMAPE puani karsiliginda "
          f"{int(en_erken.erkenlik_ay - en_iyi.erkenlik_ay)} ay erkenlik")

print("\n" + "-" * 106)
print("AILE VE MEVSIM DAGILIMI (gecerli modeller icinde)")
print("-" * 106)
for ad in MERDIVEN:
    g = SONUC[ad][SONUC[ad].GECERLI]
    if len(g) == 0: continue
    a = g.AILE.value_counts().to_dict(); m = int(g.MEVSIM.sum())
    print(f"  {ad:12s} aile={a}  mevsimli={m}/{len(g)} (%{100*m/len(g):.0f})")
print("\n  >> Oransal model ust siralarda cikiyorsa: sabit terimsiz yapi dogrulanmis demektir.")
print("  >> Mevsimli model orani dusukse: BLOK 1'deki lag-12=0.116 bulgusu teyit edilmis olur.")

print("\n" + "=" * 106)
print("BLOK 5 TAMAM")
print("=" * 106)
