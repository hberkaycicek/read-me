# ============================================================================
# BLOK 6 — HEDEFLI DENEYLER VE MODEL AILESI GENISLETME
#
# Optuna YERINE kontrollu deneyler yapiyoruz. Gerekce: onceki calismada 60 trial
# Optuna hicbir sey eklemedi (birinci degismedi). Grid zaten kapsamli.
# Bunun yerine CEVAPLANMASI GEREKEN SORULARA odaklaniyoruz:
#
#   6.1  ETKIN_BILGI_AY  : her model GERCEKTEN kac ay olgunluk istiyor? (seffaflik)
#   6.2  MM_SKOR ABLASYON: 5/5 basamakta secildi — gercek bilgi mi, intercept vekili mi?
#   6.3  LOGIT DONUSUM   : hedef bir ORAN; logit heteroskedastisiteyi giderir,
#                          tahmini (0,1)'de tutar. Hic denenmedi.
#   6.4  ZAMAN AGIRLIGI  : Chow 202311'de kirilma buldu. Son kohortlara daha cok
#                          agirlik vermek rejim kaymasini telafi eder mi?
#   6.5  GENIS AILELER   : Lasso / ElasticNet / Huber (grid'de yoktu)
#   6.6  BIRLESIK SIRALAMA
# ============================================================================
from sklearn.linear_model import HuberRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer

print("=" * 106)
print("BLOK 6 — HEDEFLI DENEYLER")
print("=" * 106)

# ---------------------------------------------------------------- 6.1 ETKIN AY
print("\n" + "-" * 106)
print("6.1  ETKIN_BILGI_AY — her model gercekten kac ay olgunluk istiyor?")
print("-" * 106)
def etkin_ay(girdiler_str):
    kols = [c.strip() for c in girdiler_str.split(",")]
    return max([GEREKEN_AY.get(c, 0) for c in kols] or [0])

for ad in MERDIVEN:
    g = SONUC[ad][SONUC[ad].GECERLI]
    if len(g) == 0: continue
    b = g.iloc[0]
    ea = etkin_ay(b.GIRDILER)
    print(f"  {ad:12s} basamak_bilgi_ay={MERDIVEN[ad]}  ETKIN_BILGI_AY={ea}  "
          f"{'<-- 4. ay verisi KULLANILMIYOR' if ea < MERDIVEN[ad] else ''}")
print("""
  YORUM: ETKIN < basamak ise, model o basamagin verisini kullanmiyor demektir.
         Farki yaratan sey BILGI degil EGITIM VERISI:
           bilgi_ay=4 -> egitim kohortu 8 ay eski olabilir
           bilgi_ay=3 -> egitim kohortu 9 ay eski olmali
         Yani '1 ay fazla beklemenin bedeli' mesru bir takas, ama RAPORDA NET OLMALI.
""")

# ---------------------------------------------------------------- 6.2 ABLASYON
print("-" * 106)
print("6.2  MM_SKOR ABLASYON TESTI")
print("-" * 106)
print("  MM_SKOR 5/5 basamakta secildi. Tek basina r2=0.357 (zayif), CV=%13.6.")
print("  SORU: gercek bilgi mi, yoksa neredeyse-sabit bir seri olarak")
print("        modele ek serbestlik veren 'intercept vekili' mi?\n")
print("  TEST: sampiyondan MM_SKOR'u cikar, yerine (a) hicbir sey (b) KKB_MERKEZ koy.\n")

abl = []
for ad in MERDIVEN:
    g = SONUC[ad][SONUC[ad].GECERLI]
    if len(g) == 0: continue
    b = g.iloc[0]
    kols = [c.strip() for c in b.GIRDILER.split(",")]
    ek   = [c for c in kols if c in ("SIN_AY","COS_AY")]
    ana  = [c for c in kols if c not in ("SIN_AY","COS_AY")]
    senaryolar = [("TAM (sampiyon)", ana)]
    if "MM_SKOR" in ana:
        senaryolar.append(("MM_SKOR cikarildi", [c for c in ana if c != "MM_SKOR"]))
        if "KKB_MERKEZ" in CHAMPION_BUDANMIS:
            senaryolar.append(("MM_SKOR -> KKB_MERKEZ",
                               [c if c != "MM_SKOR" else "KKB_MERKEZ" for c in ana]))
    for sad, kk in senaryolar:
        if not kk: continue
        r = cv_degerlendir(kk, b.AILE, MERDIVEN[ad], ek_kolonlar=ek)
        if r:
            abl.append({"MERDIVEN": ad, "senaryo": sad, "K": r["K"],
                        "CV_WMAPE": r["CV_WMAPE"], "CV_SAPMA": r["CV_SAPMA"],
                        "OVERFIT": r["OVERFIT_ORANI"], "EN_KOTU": r["EN_KOTU_WMAPE"]})
ABL = pd.DataFrame(abl)
print(ABL.round(4).to_string(index=False))

print("\n  KATKI HESABI (TAM'a gore WMAPE degisimi):")
for ad in ABL.MERDIVEN.unique():
    s = ABL[ABL.MERDIVEN == ad].set_index("senaryo")
    if "TAM (sampiyon)" not in s.index: continue
    taban = s.loc["TAM (sampiyon)", "CV_WMAPE"]
    for sen in s.index:
        if sen == "TAM (sampiyon)": continue
        fark = s.loc[sen, "CV_WMAPE"] - taban
        print(f"    {ad:12s} {sen:24s} {fark:+6.2f} WMAPE  "
              f"{'-> MM_SKOR GERCEK KATKI' if fark > 0.5 else '-> katki ZAYIF/BELIRSIZ'}")
print("\n  >> MM_SKOR cikinca WMAPE belirgin KOTULESIYORSA gercek bilgi tasiyor.")
print("  >> Degismiyorsa veya KKB_MERKEZ ayni isi goruyorsa, intercept vekili demektir.")

# ---------------------------------------------------------------- 6.3 LOGIT
print("\n" + "-" * 106)
print("6.3  LOGIT HEDEF DONUSUMU")
print("-" * 106)
print("  Hedef bir ORAN (0.004-0.023). Dogrusal model teorik olarak NEGATIF tahmin")
print("  uretebilir. logit(p) = ln(p/(1-p)) donusumu:")
print("    (a) tahmini daima (0,1) araliginda tutar")
print("    (b) oran verisindeki dogal heteroskedastisiteyi azaltir")
print("    (c) dusuk seviyelerde carpimsal davranir -> rejim kaymasina dayanikli\n")

def _lg(p):  p = np.clip(np.asarray(p, float), 1e-9, 1-1e-9); return np.log(p/(1-p))
def _ilg(z): return 1.0/(1.0+np.exp(-np.asarray(z, float)))
LOGIT_TR = FunctionTransformer(_lg, inverse_func=_ilg, check_inverse=False, validate=False)

def pipeline_kur(aile):
    """Guncellendi: 'LOGIT_' onekli aileler hedefi logit uzayinda modeller."""
    if aile.startswith("LOGIT_"):
        taban = aile.replace("LOGIT_", "")
        ic = Pipeline([("sc", StandardScaler()), ("m", model_fabrika(taban))])
        return TransformedTargetRegressor(regressor=ic, transformer=LOGIT_TR)
    if aile == "Oransal":
        return Pipeline([("m", OransalModel())])
    return Pipeline([("sc", StandardScaler()), ("m", model_fabrika(aile))])

def model_fabrika(aile):
    A = CONFIG["ALPHALAR"]
    if aile == "Oransal":    return OransalModel()
    if aile == "Linear":     return LinearRegression()
    if aile == "Ridge":      return RidgeCV(alphas=A)
    if aile == "Lasso":      return LassoCV(alphas=A, max_iter=20000,
                                            random_state=CONFIG["RANDOM_STATE"])
    if aile == "ElasticNet": return ElasticNetCV(alphas=A, l1_ratio=[.2,.5,.8], max_iter=20000,
                                                 random_state=CONFIG["RANDOM_STATE"])
    if aile == "Huber":      return HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=2000)
    if aile == "Dummy":      return DummyRegressor(strategy="mean")
    raise ValueError(aile)

# ---------------------------------------------------------------- 6.4 ZAMAN AGIRLIGI
print("-" * 106)
print("6.4  ZAMAN AGIRLIKLI EGITIM")
print("-" * 106)
print("  BLOK 3.2 Chow testi: en guclu kirilma 202311 (F=28.75), TRAIN'in SONUNDA.")
print("  Yani TRAIN'in son ceyregi farkli rejim. Eski kohortlara esit agirlik vermek")
print("  modeli ESKI rejime cekiyor olabilir.\n")
print("  TEST: ustel zaman agirligi w_t = yarilanma^((son-t)/12)")
print("        yarilanma=1.0 -> agirlik yok (mevcut)")
print("        yarilanma=0.5 -> 12 ay onceki kohort yari agirlikta\n")

CONFIG["ZAMAN_YARILANMA"] = 1.0   # deneyde degistirilecek

def _agirlik(idx_list):
    """Hacim agirligi x zaman agirligi."""
    w = np.ones(len(idx_list), float)
    if CONFIG["HACIM_AGIRLIK"] and "BASVURU_ADEDI" in d.columns:
        w *= d.loc[idx_list, "BASVURU_ADEDI"].values.astype(float)
    y = CONFIG.get("ZAMAN_YARILANMA", 1.0)
    if y < 1.0:
        aylar = d.loc[idx_list, "AY"]
        yas = np.array([(aylar.max() - a).n for a in aylar], float)
        w *= y ** (yas/12.0)
    return w

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
        w = _agirlik(eg.index)
        try:
            if aile.startswith("LOGIT_"): pipe.fit(eg[kol], eg[HEDEF], regressor__m__sample_weight=w)
            else:                         pipe.fit(eg[kol], eg[HEDEF], m__sample_weight=w)
        except (TypeError, ValueError):
            pipe.fit(eg[kol], eg[HEDEF])
        p = np.asarray(pipe.predict(te[kol]), float); y = te[HEDEF].values
        m = temel_metrikler(y, p)
        fold_wmape.append(m["WMAPE"]); fold_sapma.append(m["SAPMA"])
        tr_rmse_list.append(np.sqrt(mean_squared_error(eg[HEDEF], pipe.predict(eg[kol]))))
        tum_y.extend(y); tum_p.extend(p)
    tum_y, tum_p = np.array(tum_y), np.array(tum_p)
    g = temel_metrikler(tum_y, tum_p); tr_rmse = float(np.mean(tr_rmse_list))
    dy, dp = np.diff(tum_y), np.diff(tum_p)
    yon = float(np.mean(np.sign(dy) == np.sign(dp))) if len(dy) else np.nan
    isaret = bool(len(fold_sapma) >= 2 and
                  (all(s > 0 for s in fold_sapma) or all(s < 0 for s in fold_sapma)))
    try:    mv = maks_vif(d.loc[bolge_idx, kol].dropna())
    except Exception: mv = np.inf
    return {"ETIKET": etiket, "GIRDILER": ", ".join(kol), "K": len(kol), "AILE": aile,
            "BILGI_AY": bilgi_ay, "N_FOLD": len(foldlar),
            "CV_WMAPE": g["WMAPE"], "CV_MAPE": g["MAPE"], "CV_MAE": g["MAE"], "CV_RMSE": g["RMSE"],
            "CV_SAPMA": g["SAPMA"], "MUTLAK_CV_SAPMA": abs(g["SAPMA"]),
            "EN_KOTU_WMAPE": float(np.max(fold_wmape)), "FOLD_WMAPE_STD": float(np.std(fold_wmape)),
            "ISARET_SERISI": isaret, "TRAIN_RMSE": tr_rmse,
            "OVERFIT_ORANI": float(g["RMSE"]/tr_rmse) if tr_rmse > 1e-12 else np.inf,
            "DELTA_MAE": float(np.mean(np.abs(dp-dy))) if len(dy) else np.nan,
            "YON_BASARISI": yon, "MAKS_VIF": mv,
            "ARALIK_DISI": int(((tum_p < 0) | (tum_p > 1)).sum()), "N_TAHMIN": len(tum_y)}

# ---------------------------------------------------------------- 6.5 DENEYLER
print("-" * 106)
print("6.5  DENEY MATRISI — her basamagin sampiyon degisken seti uzerinde")
print("-" * 106)
YENI_AILELER = ["Lasso", "ElasticNet", "Huber",
                "LOGIT_Linear", "LOGIT_Ridge", "LOGIT_Huber"]
YARILANMALAR = [1.0, 0.7, 0.5]

print(f"  Yeni aileler : {YENI_AILELER}")
print(f"  Zaman yarilanmasi: {YARILANMALAR}")
print(f"  Her basamagin SAMPIYON degisken seti sabit tutulur -> adil karsilastirma.\n")

dny = []
t0 = time.time()
for ad in MERDIVEN:
    g = SONUC[ad][SONUC[ad].GECERLI]
    if len(g) == 0: continue
    b = g.iloc[0]
    kols = [c.strip() for c in b.GIRDILER.split(",")]
    ek   = [c for c in kols if c in ("SIN_AY","COS_AY")]
    ana  = [c for c in kols if c not in ("SIN_AY","COS_AY")]
    for aile in [b.AILE] + YENI_AILELER:
        for yar in YARILANMALAR:
            CONFIG["ZAMAN_YARILANMA"] = yar
            r = cv_degerlendir(ana, aile, MERDIVEN[ad], ek_kolonlar=ek, etiket=ad)
            if r:
                r["MERDIVEN"] = ad; r["YARILANMA"] = yar
                r["MEVSIM"] = int(len(ek) > 0)
                r["TABAN_MI"] = (aile == b.AILE and yar == 1.0)
                dny.append(r)
CONFIG["ZAMAN_YARILANMA"] = 1.0
DNY = pd.DataFrame(dny)
print(f"  {len(DNY)} deney, {time.time()-t0:.1f}s\n")

G = ["MERDIVEN","AILE","YARILANMA","CV_WMAPE","CV_SAPMA","EN_KOTU_WMAPE",
     "OVERFIT_ORANI","ISARET_SERISI","ARALIK_DISI","TABAN_MI"]
for ad in MERDIVEN:
    s = DNY[DNY.MERDIVEN == ad]
    if len(s) == 0: continue
    taban = s[s.TABAN_MI].CV_WMAPE.iloc[0] if s.TABAN_MI.any() else np.nan
    print(f"\n  [{ad}]  taban (grid sampiyonu) CV_WMAPE = {taban:.4f}")
    t = s.nsmallest(6, "CV_WMAPE")[G].copy()
    t["KAZANC"] = (taban - t.CV_WMAPE).round(3)
    print(t.round(4).to_string(index=False))

# ---------------------------------------------------------------- 6.6 BIRLESIK
print("\n" + "=" * 106)
print("6.6  BIRLESIK SIRALAMA — grid + deneyler")
print("=" * 106)
NIHAI = {}
for ad in MERDIVEN:
    gr = SONUC[ad].copy(); gr["YARILANMA"] = 1.0; gr["KAYNAK"] = "grid"
    dn = DNY[DNY.MERDIVEN == ad].copy()
    if len(dn): dn["KAYNAK"] = "deney"
    hep = pd.concat([gr, dn], ignore_index=True)
    hep = rank_ekle(gecerlilik_uygula(hep, ad))
    NIHAI[ad] = hep
    g = hep[hep.GECERLI]
    print(f"\n[{ad}] toplam {len(hep):,} aday, {len(g):,} gecerli")
    if len(g) == 0: continue
    t = g.head(5)[["KAYNAK","GIRDILER","K","AILE","YARILANMA","MEVSIM","CV_WMAPE",
                   "CV_SAPMA","OVERFIT_ORANI","MAKS_VIF","RANK_SKORU"]].copy()
    t["GIRDILER"] = t.GIRDILER.str.replace("KK_TAKIP_","").str.replace(f"_{T}","")
    print(t.round(4).to_string(index=False))

print("\n" + "=" * 106)
print("NIHAI MERDIVEN — BLOK 7 hakemligine gidecek adaylar")
print("=" * 106)
fin = []
for ad, b_ay in MERDIVEN.items():
    g = NIHAI[ad][NIHAI[ad].GECERLI]
    if len(g) == 0: continue
    b = g.iloc[0]
    fin.append({"MERDIVEN": ad, "bilgi_ay": b_ay,
                "erkenlik": CONFIG["HEDEF_OLGUNLASMA_AY"]-b_ay,
                "etkin_ay": etkin_ay(b.GIRDILER),
                "GIRDILER": b.GIRDILER.replace("KK_TAKIP_","").replace(f"_{T}",""),
                "AILE": b.AILE, "YARILANMA": b.YARILANMA, "KAYNAK": b.KAYNAK,
                "CV_WMAPE": b.CV_WMAPE, "CV_SAPMA": b.CV_SAPMA,
                "OVERFIT": b.OVERFIT_ORANI,
                "skill": 1 - b.CV_WMAPE/REF[ad]["CV_WMAPE"]})
FIN = pd.DataFrame(fin)
print(FIN.round(4).to_string(index=False))

print("\n  KARAR NOTLARI:")
print(f"    Logit kazanan basamak  : {int((FIN.AILE.str.startswith('LOGIT_')).sum())} / {len(FIN)}")
print(f"    Zaman agirligi kazanan : {int((FIN.YARILANMA < 1.0).sum())} / {len(FIN)}")
print(f"    Deneyden gelen sampiyon: {int((FIN.KAYNAK=='deney').sum())} / {len(FIN)}")
print(f"    Mevsimli sampiyon      : {int((FIN.GIRDILER.str.contains('SIN_AY')).sum())} / {len(FIN)}")
print("\n  >> Logit kazaniyorsa: oran yapisi dogru modellenmis demektir.")
print("  >> Zaman agirligi kazaniyorsa: Chow'un buldugu rejim kirilmasi telafi edilmis.")
print("=" * 106)
