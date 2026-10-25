# ============================================================================
# BLOK 7 — ARA BOLGESI HAKEMLIGI
#
# NEDEN BU BLOK VAR:
#   Onceki calismada CV sampiyonu CHALLENGER'i (CV 9.63) gosteriyordu,
#   OOT ise tam tersini soyledi (OOT 21.36, sonuncu). CV->OOT bozulmasi %61-122.
#   Sebep: 9.260 varyant arasindan CV'ye gore secim -> SECIM YANLILIGI.
#
#   TRAIN'de en iyi gorunen model, CV fold'larinin gurultusune de uymustur.
#   Bunu kiracak tek sey: SECIMDE HIC KULLANILMAMIS bagimsiz bir bolge.
#
# BU BLOKTA:
#   7.1  ARA walk-forward motoru (uretim senaryosunun birebir simulasyonu)
#   7.2  Her basamagin ilk N adayi ARA'da sinanir
#   7.3  CV -> ARA siralama tutarliligi (Spearman) = secim yanliliginin OLCUSU
#   7.4  Birlesik secim: CV + ARA
#   7.5  Nihai adaylar -> OOT'ye SADECE bunlar gider
#
# ONEMLI UYARI: ARA'da yalnizca 6 kohort var. WMAPE gurultulu olacak.
#   Bu yuzden ARA tek basina karar vermez; CV ile BIRLIKTE degerlendirilir.
# ============================================================================
print("=" * 108)
print("BLOK 7 — ARA BOLGESI HAKEMLIGI")
print("=" * 108)

# ---------------------------------------------------------------- 7.1
print("\n" + "-" * 108)
print("7.1  ARA WALK-FORWARD MOTORU")
print("-" * 108)
print("""
  Her hedef kohort icin:
    tahmin_ay = kohort_ay + bilgi_ay
    egitim    = HEDEF_ERISIM < tahmin_ay olan TUM olgun kohortlar
    -> yani 'o tarihte elimde ne vardiysa' ile egit, sonra tahmin et.

  Bu, BLOK 4'teki CV'den FARKLI ve daha gercekci:
    CV      : sabit fold yapisi, 3 blok
    walk-fwd: her kohort icin model YENIDEN egitilir, egitim seti BUYUR
  Uretimde olan budur.
""")

def walkforward(kolonlar, aile, bilgi_ay, hedef_idx, yarilanma=1.0, ek_kolonlar=None):
    kol = list(kolonlar) + list(ek_kolonlar or [])
    olgun = d[d.HEDEF_KULLANILABILIR]
    tah, ger, eg_n, don = [], [], [], []
    for i in hedef_idx:
        c = d.loc[i]
        tahmin_ay = c["AY"] + bilgi_ay
        eg = olgun[olgun["HEDEF_ERISIM"] < tahmin_ay][kol + [HEDEF]].dropna()
        if len(eg) < MIN_TRAIN:
            continue
        x = d.loc[[i], kol]
        if x.isna().any(axis=None):
            continue
        pipe = pipeline_kur(aile)
        _eski = CONFIG.get("ZAMAN_YARILANMA", 1.0)
        CONFIG["ZAMAN_YARILANMA"] = yarilanma
        w = _agirlik(eg.index)
        _fit_agirlikli(pipe, eg[kol], eg[HEDEF], w, aile)
        CONFIG["ZAMAN_YARILANMA"] = _eski
        tah.append(float(np.clip(pipe.predict(x)[0], 0, 1)))
        ger.append(float(c[HEDEF])); eg_n.append(len(eg)); don.append(c[DONEM_KOL])
    if len(tah) == 0:
        return None
    y, p = np.array(ger), np.array(tah)
    m = temel_metrikler(y, p)
    e = p - y
    return {"N": len(y), "WMAPE": m["WMAPE"], "MAPE": m["MAPE"], "MAE": m["MAE"],
            "RMSE": m["RMSE"], "SAPMA": m["SAPMA"], "MUTLAK_SAPMA": abs(m["SAPMA"]),
            "BIAS_ORANI": abs(m["SAPMA"])/m["MAE"] if m["MAE"] > 0 else np.nan,
            "POZ_SAPMA": int((e > 0).sum()),
            "EN_KOTU_MUTLAK": float(np.max(np.abs(e))),
            "EGITIM_N_ILK": eg_n[0], "EGITIM_N_SON": eg_n[-1],
            "_don": don, "_ger": ger, "_tah": tah}

# Motor dogrulamasi: ARA'da egitim seti gercekten buyuyor mu?
_test = walkforward([CHALLENGER_BUDANMIS[0]], "Linear", 3, ara_idx)
if _test:
    print(f"  Dogrulama [ARA, bilgi_ay=3]: {_test['N']} kohort tahmin edildi, "
          f"egitim seti {_test['EGITIM_N_ILK']} -> {_test['EGITIM_N_SON']} satir")
    print(f"  Donemler: {_test['_don']}")
print("  >> Egitim seti BUYUYORSA walk-forward dogru calisiyor.")

# ---------------------------------------------------------------- 7.2
print("\n" + "-" * 108)
print("7.2  ADAYLARI ARA'DA SINA")
print("-" * 108)
ARA_N = CONFIG.get("ARA_ADAY_N", 20)
print(f"  Her basamagin ilk {ARA_N} GECERLI adayi ARA bolgesinde sinaniyor.")
print(f"  ARA = {d.loc[ara_idx, DONEM_KOL].min()} .. {d.loc[ara_idx, DONEM_KOL].max()}  "
      f"({len(ara_idx)} kohort)\n")

ARA_SONUC = {}
for ad, b_ay in MERDIVEN.items():
    kaynak = NIHAI[ad] if "NIHAI" in dir() else SONUC[ad]
    g = kaynak[kaynak.GECERLI].head(ARA_N)
    if len(g) == 0:
        print(f"  [{ad}] gecerli aday yok, atlaniyor"); continue
    rows = []
    for _, b in g.iterrows():
        kols = [c.strip() for c in b.GIRDILER.split(",")]
        ek   = [c for c in kols if c in ("SIN_AY","COS_AY")]
        ana  = [c for c in kols if c not in ("SIN_AY","COS_AY")]
        yar  = float(b.YARILANMA) if "YARILANMA" in b.index and pd.notna(b.YARILANMA) else 1.0
        r = walkforward(ana, b.AILE, b_ay, ara_idx, yarilanma=yar, ek_kolonlar=ek)
        if r is None: continue
        rows.append({
            "GIRDILER": b.GIRDILER, "K": int(b.K), "AILE": b.AILE, "YARILANMA": yar,
            "MEVSIM": int(b.MEVSIM) if "MEVSIM" in b.index else 0,
            "CV_WMAPE": b.CV_WMAPE, "CV_SAPMA": b.CV_SAPMA,
            "CV_RANK": b.RANK_SKORU,
            "ARA_WMAPE": r["WMAPE"], "ARA_SAPMA": r["SAPMA"],
            "ARA_MUTLAK_SAPMA": r["MUTLAK_SAPMA"], "ARA_BIAS_ORANI": r["BIAS_ORANI"],
            "ARA_POZ": f"{r['POZ_SAPMA']}/{r['N']}",
            "ARA_EN_KOTU": r["EN_KOTU_MUTLAK"], "ARA_N": r["N"],
        })
    if not rows: continue
    D_ = pd.DataFrame(rows)
    D_["BOZULMA"]   = D_.ARA_WMAPE - D_.CV_WMAPE
    D_["BOZULMA_%"] = 100*(D_.ARA_WMAPE/D_.CV_WMAPE - 1)
    ARA_SONUC[ad] = D_
    print(f"\n  [{ad}]  {len(D_)} aday ARA'da sinandi")
    t = D_.sort_values("ARA_WMAPE").head(6).copy()
    t["GIRDILER"] = t.GIRDILER.str.replace("KK_TAKIP_","").str.replace(f"_{T}","")
    print(t[["GIRDILER","AILE","YARILANMA","CV_WMAPE","ARA_WMAPE","BOZULMA_%",
             "ARA_SAPMA","ARA_POZ","ARA_EN_KOTU"]].round(4).to_string(index=False))

# ---------------------------------------------------------------- 7.3
print("\n" + "=" * 108)
print("7.3  SECIM YANLILIGININ OLCUSU — CV siralamasi ARA'da tutuyor mu?")
print("=" * 108)
print("""
  Spearman korelasyonu: CV siralamasi ile ARA siralamasi ne kadar ortusuyor?
    rho ~ +1 : CV guvenilir, secim yanliligi kucuk
    rho ~  0 : CV siralamasi ARA'da ANLAMSIZ -> secim sansa dayali
    rho <  0 : CV'de iyi olan ARA'da KOTU -> klasik overfit imzasi
""")
yan = []
for ad, D_ in ARA_SONUC.items():
    if len(D_) < 5: continue
    rho, p = stats.spearmanr(D_.CV_WMAPE, D_.ARA_WMAPE)
    rho_r, _ = stats.spearmanr(D_.CV_RANK, D_.ARA_WMAPE)
    cv_en_iyi  = D_.sort_values("CV_WMAPE").iloc[0]
    ara_en_iyi = D_.sort_values("ARA_WMAPE").iloc[0]
    yan.append({
        "MERDIVEN": ad, "n_aday": len(D_),
        "Spearman_WMAPE": rho, "p": p, "Spearman_RANK": rho_r,
        "CV_birincisinin_ARA_sirasi": int(D_.ARA_WMAPE.rank().loc[cv_en_iyi.name]),
        "ort_BOZULMA_%": D_["BOZULMA_%"].mean(),
        "CV1_ARA_WMAPE": cv_en_iyi.ARA_WMAPE,
        "ARA_EN_IYI_WMAPE": ara_en_iyi.ARA_WMAPE,
        "KAYIP": cv_en_iyi.ARA_WMAPE - ara_en_iyi.ARA_WMAPE,
    })
YAN = pd.DataFrame(yan)
print(YAN.round(4).to_string(index=False))
print("""
  OKUMA:
    CV_birincisinin_ARA_sirasi : 1 ise CV mukemmel secmis; buyukse yanlilik var
    KAYIP : CV birincisini secerek ARA'da kaybettigimiz WMAPE puani
    ort_BOZULMA_% : ARA'da hatanin CV'ye gore ne kadar arttigi
""")

# ---------------------------------------------------------------- 7.4
print("-" * 108)
print("7.4  BIRLESIK SECIM — CV + ARA")
print("-" * 108)
print("""
  Ne saf CV'ye ne saf ARA'ya guveniyoruz:
    CV : 39 kohort, 3 fold, ama SECIMDE KULLANILDI -> yanli
    ARA:  6 kohort, bagimsiz, ama COK KUCUK -> gurultulu

  Birlesik skor = 0.40*CV_rank + 0.45*ARA_rank + 0.15*ARA_yanlilik_rank
  Ayrica ELEME: ARA'da tum sapmalar ayni yonlu VE buyukse diskalifiye.
""")
NIHAI_ADAY = {}
for ad, D_ in ARA_SONUC.items():
    X = D_.copy()
    X["R_CV"]    = X.CV_WMAPE.rank(pct=True)
    X["R_ARA"]   = X.ARA_WMAPE.rank(pct=True)
    X["R_YANLI"] = X.ARA_MUTLAK_SAPMA.rank(pct=True)
    X["BIRLESIK"] = 0.40*X.R_CV + 0.45*X.R_ARA + 0.15*X.R_YANLI
    # Eleme: ARA'da tek yonlu VE buyuk sapma (onceki modelin 12/12 felaketi)
    n_ara = X.ARA_N.iloc[0]
    tek_yon = X.ARA_POZ.isin([f"{n_ara}/{n_ara}", f"0/{n_ara}"])
    buyuk   = X.ARA_MUTLAK_SAPMA > d.loc[ara_idx, HEDEF].mean()*0.12
    X["ARA_ELENDI"] = tek_yon & buyuk
    X = X.sort_values(["ARA_ELENDI", "BIRLESIK"], ascending=[True, True])
    NIHAI_ADAY[ad] = X
    print(f"\n  [{ad}]  elenen: {int(X.ARA_ELENDI.sum())}/{len(X)}")
    t = X.head(5).copy()
    t["GIRDILER"] = t.GIRDILER.str.replace("KK_TAKIP_","").str.replace(f"_{T}","")
    print(t[["GIRDILER","AILE","YARILANMA","CV_WMAPE","ARA_WMAPE","ARA_SAPMA",
             "ARA_POZ","BIRLESIK","ARA_ELENDI"]].round(4).to_string(index=False))

# ---------------------------------------------------------------- 7.5
print("\n" + "=" * 108)
print("7.5  NIHAI MERDIVEN — OOT'ye gidecek modeller")
print("=" * 108)
SECILEN = {}
fin = []
for ad, b_ay in MERDIVEN.items():
    if ad not in NIHAI_ADAY: continue
    X = NIHAI_ADAY[ad]
    X = X[~X.ARA_ELENDI] if (~X.ARA_ELENDI).any() else X
    b = X.iloc[0]
    kols = [c.strip() for c in b.GIRDILER.split(",")]
    SECILEN[ad] = {"kolonlar": [c for c in kols if c not in ("SIN_AY","COS_AY")],
                   "ek": [c for c in kols if c in ("SIN_AY","COS_AY")],
                   "aile": b.AILE, "bilgi_ay": b_ay, "yarilanma": float(b.YARILANMA)}
    fin.append({"MERDIVEN": ad, "bilgi_ay": b_ay,
                "erkenlik": CONFIG["HEDEF_OLGUNLASMA_AY"]-b_ay,
                "etkin_ay": max(GEREKEN_AY.get(c,0) for c in kols),
                "GIRDILER": b.GIRDILER.replace("KK_TAKIP_","").replace(f"_{T}",""),
                "AILE": b.AILE, "YARILANMA": b.YARILANMA,
                "CV_WMAPE": b.CV_WMAPE, "ARA_WMAPE": b.ARA_WMAPE,
                "ARA_SAPMA": b.ARA_SAPMA, "ARA_POZ": b.ARA_POZ,
                "BOZULMA_%": b["BOZULMA_%"]})
FINAL = pd.DataFrame(fin)
print(FINAL.round(4).to_string(index=False))

print("\n  ERKENLIK / DOGRULUK (ARA bolgesinde, bagimsiz olcum):")
for _, r in FINAL.iterrows():
    print(f"    {r.MERDIVEN:12s} {int(r.erkenlik):2d} ay erken  "
          f"CV={r.CV_WMAPE:5.2f} -> ARA={r.ARA_WMAPE:5.2f}  "
          f"({r['BOZULMA_%']:+6.1f}%)  sapma={r.ARA_SAPMA:+.5f}")

print("\n  SECIM DEGISTI MI? (TRAIN sampiyonu vs ARA hakemliginden sonra)")
for ad in MERDIVEN:
    if ad not in NIHAI_ADAY: continue
    kaynak = NIHAI[ad] if "NIHAI" in dir() else SONUC[ad]
    cv1 = kaynak[kaynak.GECERLI].iloc[0]
    ara1 = NIHAI_ADAY[ad][~NIHAI_ADAY[ad].ARA_ELENDI].iloc[0] \
           if (~NIHAI_ADAY[ad].ARA_ELENDI).any() else NIHAI_ADAY[ad].iloc[0]
    degisti = cv1.GIRDILER != ara1.GIRDILER or cv1.AILE != ara1.AILE
    print(f"    {ad:12s} {'DEGISTI' if degisti else 'ayni'}")
    if degisti:
        print(f"        TRAIN : {cv1.GIRDILER[:62]} ({cv1.AILE})")
        print(f"        ARA   : {ara1.GIRDILER[:62]} ({ara1.AILE})")

print("\n" + "=" * 108)
print("BLOK 7 TAMAM — OOT hala DOKUNULMADI.")
print(f"  Secilen model sayisi: {len(SECILEN)}")
print(f"  Ortalama CV->ARA bozulmasi: %{FINAL['BOZULMA_%'].mean():.1f}")
print(f"  (Onceki calismada CV->OOT bozulmasi %61-122 idi)")
print("=" * 108)
