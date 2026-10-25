# ============================================================================
# BLOK 9 — KOK NEDEN VE UYGULANABILIR DUZELTME ARAYISI
#
# DURUM: OOT'de 5 modelin 4'u Dummy'yi gecemedi. Tum sapmalar POZITIF,
#        BIAS_ORANI ~1.00. Ama Pearson 0.65-0.74 (p<0.025) -> modeller
#        SIRALAMAYI yakaliyor, sorun tamamen SEVIYE.
#
# BU BLOK MODEL SECMEZ. OOT kapandi. Burada sadece TESHIS yapiyoruz
# ve bir sonraki surum icin UYGULANABILIR duzeltme ariyoruz.
#
#   9.1  Oncul -> hedef donusum orani OOT'de degisti mi? (kok neden adayi 1)
#   9.2  Girdilerin OOT kaymasi (kok neden adayi 2)
#   9.3  UYGULANABILIR yanlilik duzeltmesi — walk-forward icinde, sizintisiz
#   9.4  Oransal hedef (carpan modeli) seviye kaymasina dayanikli mi?
#   9.5  Karar cercevesi
# ============================================================================
print("=" * 110)
print("BLOK 9 — KOK NEDEN VE DUZELTME ARAYISI")
print("=" * 110)

olg = d[d.HEDEF_KULLANILABILIR].copy()
olg["BOLGE2"] = np.where(olg[DONEM_KOL].astype(int) <= CONFIG["TRAIN_ARALIK"][1], "TRAIN",
                 np.where(olg[DONEM_KOL].astype(int) <= CONFIG["ARA_ARALIK"][1], "ARA", "OOT"))

# ---------------------------------------------------------------- 9.1
print("\n" + "-" * 110)
print("9.1  ONCUL -> HEDEF DONUSUM ORANI  (kok neden adayi 1)")
print("-" * 110)
print("  Hipotez: erken gecikmeler ayni kaldi ama 12M90'a DONUSUM orani dustu")
print("           (tahsilat iyilesmesi, yapilandirma, politika degisikligi...)")
print("  Test   : carpan k = HEDEF / oncul  bolgeler arasinda farkli mi?\n")
ONCUL_TEST = [c for c in ["KK_TAKIP_30_2_"+T, "KK_TAKIP_20_2_"+T, "KK_TAKIP_60_3_"+T,
                          "KK_TAKIP_60_6_"+T] if c in d.columns]
rows = []
for c in ONCUL_TEST:
    s = olg[[c, HEDEF, "BOLGE2"]].dropna()
    s["k"] = s[HEDEF]/s[c].replace(0, np.nan)
    g = s.groupby("BOLGE2")["k"].agg(["mean","std","count"]).reindex(["TRAIN","ARA","OOT"])
    rows.append({"oncul": c.replace("KK_TAKIP_","").replace(f"_{T}",""),
                 "k_TRAIN": g.loc["TRAIN","mean"], "k_ARA": g.loc["ARA","mean"],
                 "k_OOT": g.loc["OOT","mean"],
                 "OOT/TRAIN": g.loc["OOT","mean"]/g.loc["TRAIN","mean"],
                 "OOT/ARA": g.loc["OOT","mean"]/g.loc["ARA","mean"]})
    # Anlamlilik: TRAIN+ARA vs OOT
    a = s[s.BOLGE2 != "OOT"]["k"].dropna(); b = s[s.BOLGE2 == "OOT"]["k"].dropna()
    if len(a) > 5 and len(b) > 3:
        rows[-1]["t_p"] = stats.ttest_ind(a, b, equal_var=False).pvalue
        rows[-1]["MW_p"] = stats.mannwhitneyu(a, b).pvalue
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("""
  >> OOT/TRAIN < 1 VE p < 0.05 ise: oncul ayni seviyedeyken hedef DUSUK
     -> DONUSUM ORANI DUSMUS -> modelin sisirmesinin kok nedeni budur.
""")

# ---------------------------------------------------------------- 9.2
print("-" * 110)
print("9.2  GIRDILERIN OOT KAYMASI  (kok neden adayi 2)")
print("-" * 110)
kullanilan = sorted({c for s in SECILEN.values() for c in s["kolonlar"]})
rows = []
for c in kullanilan + [HEDEF]:
    tr = olg[olg.BOLGE2 == "TRAIN"][c].dropna()
    ar = olg[olg.BOLGE2 == "ARA"][c].dropna()
    oo = olg[olg.BOLGE2 == "OOT"][c].dropna()
    if len(tr) < 10 or len(oo) < 3: continue
    rows.append({"degisken": c.replace("KK_TAKIP_","").replace(f"_{T}",""),
                 "TRAIN": tr.mean(), "ARA": ar.mean(), "OOT": oo.mean(),
                 "z_ARA": (ar.mean()-tr.mean())/tr.std(),
                 "z_OOT": (oo.mean()-tr.mean())/tr.std(),
                 "OOT_vs_ARA_z": (oo.mean()-ar.mean())/tr.std()})
KAY3 = pd.DataFrame(rows)
print(KAY3.round(4).to_string(index=False))
print("""
  >> HEDEF'in z_OOT'si girdilerin z_OOT'sinden KUCUKSE:
     girdiler yuksek kalmis ama hedef dusmus -> model girdiyi takip edip sisiriyor.
""")
_h = KAY3[KAY3.degisken == HEDEF.replace("KK_TAKIP_","").replace(f"_{T}","")]
if len(_h):
    hz = float(_h.z_OOT.iloc[0]); gz = KAY3[KAY3.degisken != _h.degisken.iloc[0]].z_OOT.mean()
    print(f"  HEDEF z_OOT = {hz:+.3f}   |   GIRDILER ort z_OOT = {gz:+.3f}   "
          f"FARK = {gz-hz:+.3f}")

# ---------------------------------------------------------------- 9.3
print("\n" + "=" * 110)
print("9.3  UYGULANABILIR YANLILIK DUZELTMESI")
print("=" * 110)
print("""
  Sabit ofset UYGULANAMAZ (OOT gerceklesmesini bilmek gerekir).
  UYGULANABILIR olan: tahmin aninda BILINEN artiklarla duzeltme.

  Kohort t'yi t+bilgi_ay'da tahmin ederken, hedefi bilinen en taze kohort:
      t_bilinen = t - (12 - bilgi_ay)
  O kohort ve oncesi icin model artigi hesaplanabilir -> ortalamasi cikarilir.
  Bu bir "recursive bias correction" ve SIZINTI ICERMEZ.
""")

def walkforward_duzeltmeli(kolonlar, aile, bilgi_ay, hedef_idx, yarilanma=1.0,
                           ek_kolonlar=None, pencere=6, duzelt=True):
    kol = list(kolonlar) + list(ek_kolonlar or [])
    olgun = d[d.HEDEF_KULLANILABILIR]
    tah, ger, don, ofsetler = [], [], [], []
    for i in hedef_idx:
        c = d.loc[i]
        tahmin_ay = c["AY"] + bilgi_ay
        eg = olgun[olgun["HEDEF_ERISIM"] < tahmin_ay][kol + [HEDEF]].dropna()
        if len(eg) < MIN_TRAIN: continue
        x = d.loc[[i], kol]
        if x.isna().any(axis=None): continue
        pipe = pipeline_kur(aile)
        _e = CONFIG.get("ZAMAN_YARILANMA", 1.0); CONFIG["ZAMAN_YARILANMA"] = yarilanma
        _fit_agirlikli(pipe, eg[kol], eg[HEDEF], _agirlik(eg.index), aile)
        CONFIG["ZAMAN_YARILANMA"] = _e
        p = float(pipe.predict(x)[0])
        ofs = 0.0
        if duzelt:
            # Tahmin aninda BILINEN son 'pencere' kohortun artigi
            gec = eg.tail(pencere)
            if len(gec) >= 3:
                ofs = float(np.mean(pipe.predict(gec[kol]) - gec[HEDEF].values))
        tah.append(float(np.clip(p - ofs, 0, 1))); ger.append(float(c[HEDEF]))
        don.append(c[DONEM_KOL]); ofsetler.append(ofs)
    if not tah: return None
    y, p = np.array(ger), np.array(tah); e = p - y
    m = temel_metrikler(y, p)
    return {"N": len(y), "WMAPE": m["WMAPE"], "MAE": m["MAE"], "SAPMA": m["SAPMA"],
            "POZ": int((e > 0).sum()), "ort_ofset": float(np.mean(ofsetler)),
            "_don": don, "_ger": ger, "_tah": tah}

print("  TEST: duzeltmesiz vs duzeltmeli (pencere = 6, 9, 12 kohort)\n")
dz = []
for ad, s in SECILEN.items():
    base = walkforward_duzeltmeli(s["kolonlar"], s["aile"], s["bilgi_ay"], oot_idx,
                                  s["yarilanma"], s["ek"], duzelt=False)
    satir = {"MERDIVEN": ad, "HAM_WMAPE": base["WMAPE"], "HAM_sapma": base["SAPMA"],
             "HAM_poz": f"{base['POZ']}/{base['N']}"}
    for pen in [6, 9, 12]:
        r = walkforward_duzeltmeli(s["kolonlar"], s["aile"], s["bilgi_ay"], oot_idx,
                                   s["yarilanma"], s["ek"], pencere=pen, duzelt=True)
        if r:
            satir[f"DUZ{pen}_WMAPE"] = r["WMAPE"]
            satir[f"DUZ{pen}_sapma"] = r["SAPMA"]
            satir[f"DUZ{pen}_poz"]   = f"{r['POZ']}/{r['N']}"
    dz.append(satir)
DZ = pd.DataFrame(dz)
print(DZ.round(4).to_string(index=False))
print("""
  >> DUZ_WMAPE < HAM_WMAPE ve sapma 0'a yaklasiyorsa, duzeltme ISE YARIYOR
     ve UYGULANABILIR (sizinti yok, tahmin aninda bilinen bilgiyle yapiliyor).
  >> Kotulestiriyorsa: yanlilik zaman icinde ISARET DEGISTIRIYOR demektir,
     gecikmeli artik yanlis yonu gosterir.
""")

# Dummy'ye gore skill (duzeltmeli)
_dref = walkforward([CHALLENGER_BUDANMIS[0]], "Dummy", 3, oot_idx)
if _dref and "DUZ9_WMAPE" in DZ.columns:
    print("  DUZELTMELI SKILL (Dummy = %.3f):" % _dref["WMAPE"])
    for _, r in DZ.iterrows():
        for pen in [6, 9, 12]:
            k = f"DUZ{pen}_WMAPE"
            if k in r and pd.notna(r[k]):
                print(f"    {r.MERDIVEN:12s} pencere={pen:2d}: WMAPE={r[k]:6.2f}  "
                      f"skill={1-r[k]/_dref['WMAPE']:+.3f}")

# ---------------------------------------------------------------- 9.4
print("\n" + "=" * 110)
print("9.4  ORANSAL HEDEF — carpan modeli seviye kaymasina dayanikli mi?")
print("=" * 110)
print("""
  Fikir: HEDEF yerine  k = HEDEF / oncul  modellensin, tahmin k_hat * oncul.
  Gerekce: carpan, seviyeden bagimsiz bir YAPI parametresi. Seviye kayarsa
           oncul de kayar, carpan sabit kalir -> model otomatik uyum saglar.
""")
def walkforward_oran(oncul, ek_kolonlar, aile, bilgi_ay, hedef_idx, yarilanma=1.0):
    kol = [oncul] + list(ek_kolonlar or [])
    olgun = d[d.HEDEF_KULLANILABILIR]
    tah, ger, don = [], [], []
    for i in hedef_idx:
        c = d.loc[i]; tahmin_ay = c["AY"] + bilgi_ay
        eg = olgun[olgun["HEDEF_ERISIM"] < tahmin_ay][kol + [HEDEF]].dropna()
        eg = eg[eg[oncul] > 0]
        if len(eg) < MIN_TRAIN: continue
        x = d.loc[[i], kol]
        if x.isna().any(axis=None) or float(x[oncul].iloc[0]) <= 0: continue
        k_hedef = eg[HEDEF]/eg[oncul]
        pipe = pipeline_kur(aile)
        _fit_agirlikli(pipe, eg[kol], k_hedef, _agirlik(eg.index), aile)
        k_hat = float(pipe.predict(x)[0])
        tah.append(float(np.clip(k_hat*float(x[oncul].iloc[0]), 0, 1)))
        ger.append(float(c[HEDEF])); don.append(c[DONEM_KOL])
    if not tah: return None
    y, p = np.array(ger), np.array(tah); e = p - y
    m = temel_metrikler(y, p)
    return {"WMAPE": m["WMAPE"], "SAPMA": m["SAPMA"], "POZ": int((e > 0).sum()), "N": len(y)}

orn = []
for ad, s in SECILEN.items():
    onc = [c for c in s["kolonlar"] if c.startswith("KK_TAKIP_") or c.startswith("RR_")]
    if not onc: continue
    o = onc[0]
    digerleri = [c for c in s["kolonlar"] if c != o] + s["ek"]
    r1 = walkforward(s["kolonlar"], s["aile"], s["bilgi_ay"], oot_idx, s["yarilanma"], s["ek"])
    r2 = walkforward_oran(o, digerleri, s["aile"], s["bilgi_ay"], oot_idx, s["yarilanma"])
    if r1 and r2:
        orn.append({"MERDIVEN": ad, "oran_paydasi": o.replace("KK_TAKIP_","").replace(f"_{T}",""),
                    "SEVIYE_WMAPE": r1["WMAPE"], "ORAN_WMAPE": r2["WMAPE"],
                    "SEVIYE_sapma": r1["SAPMA"], "ORAN_sapma": r2["SAPMA"],
                    "SEVIYE_poz": f"{r1['POZ_SAPMA']}/{r1['N']}",
                    "ORAN_poz": f"{r2['POZ']}/{r2['N']}",
                    "KAZANC": r1["WMAPE"]-r2["WMAPE"]})
if orn:
    print(pd.DataFrame(orn).round(4).to_string(index=False))
    print("\n  >> ORAN_WMAPE belirgin dusukse, bir sonraki surumun hedefi ORAN olmali.")

# ---------------------------------------------------------------- 9.5
print("\n" + "=" * 110)
print("9.5  KARAR CERCEVESI")
print("=" * 110)
print(f"""
  OLCULEN GERCEKLER:
    1. Modeller SIRALAMAYI yakaliyor  : Pearson 0.65-0.74 (p<0.025)
       Dummy'nin korelasyonu tanim geregi 0.
    2. Sorun TAMAMEN SEVIYE           : tum sapmalar pozitif, BIAS_ORANI ~1.00
    3. Seviye duzeltilse WMAPE        : 5.5-7.5 (Dummy'nin duzeltilmisi 7.76)
    4. Dummy'nin iyi gorunmesi SANS   : o da -0.243pp sapiyor, ters yonde
    5. ARA hakemligi yetmedi          : 6 kohort rejim kaymasini ongoremiyor

  BU MODEL SU HALIYLE URETIME ALINMAZ. Sebep:
    - Skill negatif (CHAMPION haric, o da +0.017)
    - Sistematik sisirme -> provizyon FAZLA hesaplanir
    - 12/12 tek yonlu sapma, model validasyon bunu gecirmez

  ANCAK MODEL DEGERSIZ DEGIL:
    - Siralama bilgisi gercek ve anlamli
    - Sorun teshis edildi ve OLCULDU
    - Cozum yonu belli: seviye kalibrasyonu

  SONRAKI SURUM ICIN YOL HARITASI (9.1-9.4 ciktilarina gore onceliklendirilecek):
    A. Oransal hedef (9.4 kazanc gosteriyorsa)
    B. Recursive yanlilik duzeltmesi (9.3 kazanc gosteriyorsa)
    C. Donusum orani degistiyse (9.1): rejim degiskeni / kisa egitim penceresi
    D. OOT penceresini uzat: 12 kohort rejim kaymasini test etmek icin az
    E. Bu OOT artik "kullanilmis" sayilir; yeni surum YENI bir OOT ister
""")
print("=" * 110)
