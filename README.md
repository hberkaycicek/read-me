# ============================================================================
# BLOK 10 — v2 TASARIMI
#
# BLOK 9 BULGULARI:
#   1. Donusum orani k = HEDEF/oncul COKTU:  60_3 %33, 30_2 %12, 60_6 %12
#      AMA 20_2'ninki DEGISMEDI (OOT/TRAIN=0.999, p=0.858)
#   2. Dusus TRAIN->ARA arasinda; ARA->OOT'de k stabil (OOT/ARA ~ 1.00)
#   3. Recursive duzeltme sadece buyuk bilgi_ay'da calisiyor (CHAMPION +0.122)
#   4. Oransal hedef, k'si degisen onculde kazandiriyor (+1.44, +1.80)
#
# v2'NIN UC DEGISIKLIGI:
#   A. K-STABILITE KRITERI  : feature secimine yeni filtre
#   B. KISA EGITIM PENCERESI: eski rejimi unut (k dusus TRAIN'de kaldi)
#   C. KOSULLU DUZELTME     : sadece bilgi_ay >= 4 oldugunda recursive duzeltme
#
# !!! DURUSTLUK NOTU: Bu OOT (202410-202509) artik HARCANDI. Asagidaki
#     sonuclar KESIF niteliginde, bagimsiz dogrulama DEGIL.
#     v2'nin gercek testi 202510+ kohortlari olgunlastiginda yapilmalidir.
# ============================================================================
print("=" * 110)
print("BLOK 10 — v2 TASARIMI")
print("=" * 110)
print("""
  !!! UYARI: OOT (202410-202509) v1'de kullanildi ve HARCANDI.
      Buradaki sonuclar KESIF amaclidir, bagimsiz dogrulama degildir.
      v2'nin gercek sinavi: 202510+ kohortlari olgunlastiginda.
""")

# ---------------------------------------------------------------- 10.A
print("-" * 110)
print("10.A  K-STABILITE SKORU — her oncul icin donusum orani ne kadar sabit?")
print("-" * 110)
print("""
  Mevcut secim kriteri: r2 + VIF + kayma (z, disari)
  EKSIK OLAN: oncul ile hedef arasindaki ORANIN zaman icinde sabitligi.

  k_t = HEDEF_t / oncul_t
  Olculer:
    k_CV      : k'nin degisim katsayisi (dusuk = sabit)
    k_TREND_p : k'de dogrusal trend var mi (p>0.05 istiyoruz)
    k_SON_ORAN: son 12 kohortun k ortalamasi / ilk 12'nin  (1'e yakin istiyoruz)
""")
olg = d[d.HEDEF_KULLANILABILIR].copy().sort_values("AY").reset_index(drop=True)
kst = []
for c in CHAMPION_BUDANMIS:
    s = olg[[c, HEDEF]].dropna()
    s = s[s[c] > 0]
    if len(s) < 25: continue
    k = (s[HEDEF]/s[c]).values
    t = np.arange(len(k))
    sl, ic, r_, p_, se = stats.linregress(t, k)
    n2 = max(8, len(k)//3)
    kst.append({"oncul": c.replace("KK_TAKIP_","").replace(f"_{T}",""),
                "tam_ad": c, "n": len(k),
                "k_ort": k.mean(), "k_CV": 100*k.std()/k.mean(),
                "k_trend_p": p_, "k_trend_R2": r_**2,
                "k_son/ilk": k[-n2:].mean()/k[:n2].mean(),
                "r2_hedef": float(tek.set_index("degisken").r2.get(c, np.nan))})
KST = pd.DataFrame(kst)
# K-STABILITE SKORU: dusuk CV + trendsiz + son/ilk 1'e yakin
KST["K_SKOR"] = (KST.k_CV.rank(pct=True)*0.40
                 + (1-KST.k_trend_p).rank(pct=True)*0.30
                 + (KST["k_son/ilk"]-1).abs().rank(pct=True)*0.30)
KST = KST.sort_values("K_SKOR")
print(KST.drop(columns=["tam_ad"]).round(4).to_string(index=False))
print("""
  K_SKOR dusuk = donusum orani STABIL = uretimde guvenilir
  >> r2_hedef yuksek ama K_SKOR kotu olan onculler v1'in tuzagidir:
     gecmisi iyi aciklar, gelecegi sisirir.
""")

# ---------------------------------------------------------------- 10.B
print("-" * 110)
print("10.B  v2 HAVUZU — r2 VE k-stabilite birlikte")
print("-" * 110)
KST["r2_rank"] = KST.r2_hedef.rank(pct=True, ascending=False)
KST["BIRLESIK_SKOR"] = 0.45*KST.r2_rank + 0.55*KST.K_SKOR
V2 = KST.sort_values("BIRLESIK_SKOR")
print(V2[["oncul","r2_hedef","k_CV","k_trend_p","k_son/ilk","K_SKOR","BIRLESIK_SKOR"]]
        .head(14).round(4).to_string(index=False))
V2_HAVUZ = V2.head(10).tam_ad.tolist() + \
           [c for c in ["MM_SKOR","KKB_MERKEZ"] if c in d.columns]
print(f"\n  v2 havuzu ({len(V2_HAVUZ)}): "
      f"{[c.replace('KK_TAKIP_','').replace(f'_{T}','') for c in V2_HAVUZ]}")
_atilan = set(CHAMPION_BUDANMIS) - set(V2_HAVUZ)
print(f"  v1'de olup v2'de ELENEN: "
      f"{[c.replace('KK_TAKIP_','').replace(f'_{T}','') for c in sorted(_atilan)]}")

# ---------------------------------------------------------------- 10.C
print("\n" + "-" * 110)
print("10.C  KISA EGITIM PENCERESI — eski rejimi unut")
print("-" * 110)
print("""
  BLOK 9.1: k dususu TRAIN icinde oldu, ARA->OOT'de k STABIL.
  Yani son ~24 kohort ayni rejimde. Tum gecmisi egitime katmak
  modeli ESKI (yuksek k) rejime cekiyor.

  TEST: egitim setini son N kohortla sinirla.
""")
def wf_pencereli(kolonlar, aile, bilgi_ay, hedef_idx, max_gecmis=None,
                 ek_kolonlar=None, yarilanma=1.0, oran_paydasi=None,
                 duzelt_pencere=None):
    kol = list(kolonlar) + list(ek_kolonlar or [])
    olgun = d[d.HEDEF_KULLANILABILIR]
    tah, ger, don = [], [], []
    for i in hedef_idx:
        c = d.loc[i]; tahmin_ay = c["AY"] + bilgi_ay
        eg = olgun[olgun["HEDEF_ERISIM"] < tahmin_ay]
        if max_gecmis: eg = eg.tail(max_gecmis)
        kk = kol + ([oran_paydasi] if oran_paydasi and oran_paydasi not in kol else [])
        eg = eg[kk + [HEDEF]].dropna()
        if oran_paydasi: eg = eg[eg[oran_paydasi] > 0]
        if len(eg) < MIN_TRAIN: continue
        x = d.loc[[i], kk]
        if x.isna().any(axis=None): continue
        if oran_paydasi and float(x[oran_paydasi].iloc[0]) <= 0: continue
        pipe = pipeline_kur(aile)
        _e = CONFIG.get("ZAMAN_YARILANMA",1.0); CONFIG["ZAMAN_YARILANMA"] = yarilanma
        y_fit = (eg[HEDEF]/eg[oran_paydasi]) if oran_paydasi else eg[HEDEF]
        _fit_agirlikli(pipe, eg[kol], y_fit, _agirlik(eg.index), aile)
        CONFIG["ZAMAN_YARILANMA"] = _e
        p = float(pipe.predict(x[kol])[0])
        if oran_paydasi: p *= float(x[oran_paydasi].iloc[0])
        if duzelt_pencere:
            gec = eg.tail(duzelt_pencere)
            if len(gec) >= 3:
                pg = pipe.predict(gec[kol])
                if oran_paydasi: pg = pg*gec[oran_paydasi].values
                p -= float(np.mean(pg - gec[HEDEF].values))
        tah.append(float(np.clip(p,0,1))); ger.append(float(c[HEDEF])); don.append(c[DONEM_KOL])
    if not tah: return None
    y,p = np.array(ger), np.array(tah); e = p-y
    m = temel_metrikler(y,p)
    return {"WMAPE":m["WMAPE"],"MAE":m["MAE"],"SAPMA":m["SAPMA"],
            "POZ":int((e>0).sum()),"N":len(y),
            "PEARSON":float(stats.pearsonr(y,p)[0]) if len(y)>3 else np.nan,
            "_don":don,"_ger":ger,"_tah":tah}

_dref = walkforward([CHALLENGER_BUDANMIS[0]], "Dummy", 3, oot_idx)
DUMMY_W = _dref["WMAPE"]
print(f"  Referans Dummy WMAPE = {DUMMY_W:.3f}\n")
pen = []
for ad, s in SECILEN.items():
    for mg in [None, 36, 30, 24, 18]:
        r = wf_pencereli(s["kolonlar"], s["aile"], s["bilgi_ay"], oot_idx,
                         max_gecmis=mg, ek_kolonlar=s["ek"], yarilanma=s["yarilanma"])
        if r: pen.append({"MERDIVEN":ad,"egitim_penceresi":mg or "TUMU",
                          "WMAPE":r["WMAPE"],"SAPMA":r["SAPMA"],
                          "POZ":f"{r['POZ']}/{r['N']}","skill":1-r["WMAPE"]/DUMMY_W})
PEN = pd.DataFrame(pen)
for ad in PEN.MERDIVEN.unique():
    print(f"\n  [{ad}]")
    print(PEN[PEN.MERDIVEN==ad].round(4).to_string(index=False))

# ---------------------------------------------------------------- 10.D
print("\n" + "=" * 110)
print("10.D  v2 BIRLESIK: k-stabil payda + kisa pencere + kosullu duzeltme")
print("=" * 110)
# En k-stabil oncul (paydaya aday)
EN_STABIL = V2.iloc[0].tam_ad
print(f"  En k-stabil oncul: {EN_STABIL.replace('KK_TAKIP_','').replace(f'_{T}','')}"
      f"  (k_CV={V2.iloc[0].k_CV:.1f}%, trend_p={V2.iloc[0].k_trend_p:.3f})\n")

v2 = []
for ad, s in SECILEN.items():
    b_ay = s["bilgi_ay"]
    payda_aday = [c for c in V2_HAVUZ if GEREKEN_AY.get(c,99) <= b_ay
                  and (c.startswith("KK_TAKIP_") or c.startswith("RR_"))]
    payda = payda_aday[0] if payda_aday else None
    duz = 6 if b_ay >= 4 else None          # 9.3: sadece buyuk bilgi_ay'da calisiyor
    senaryolar = [
        ("v1 (taban)",            dict(max_gecmis=None, oran_paydasi=None,  duzelt_pencere=None)),
        ("v2a kisa pencere",      dict(max_gecmis=30,   oran_paydasi=None,  duzelt_pencere=None)),
        ("v2b + oransal",         dict(max_gecmis=30,   oran_paydasi=payda, duzelt_pencere=None)),
        ("v2c + kosullu duzelt",  dict(max_gecmis=30,   oran_paydasi=payda, duzelt_pencere=duz)),
    ]
    for sad, kw in senaryolar:
        if kw.get("oran_paydasi") is None and "oransal" in sad: continue
        r = wf_pencereli(s["kolonlar"], s["aile"], b_ay, oot_idx,
                         ek_kolonlar=s["ek"], yarilanma=s["yarilanma"], **kw)
        if r: v2.append({"MERDIVEN":ad,"senaryo":sad,"WMAPE":r["WMAPE"],
                         "SAPMA":r["SAPMA"],"POZ":f"{r['POZ']}/{r['N']}",
                         "PEARSON":r["PEARSON"],"skill":1-r["WMAPE"]/DUMMY_W})
V2S = pd.DataFrame(v2)
for ad in V2S.MERDIVEN.unique():
    print(f"\n  [{ad}]")
    print(V2S[V2S.MERDIVEN==ad].round(4).to_string(index=False))

print("\n" + "=" * 110)
print("v2 OZET — her basamagin en iyi senaryosu")
print("=" * 110)
best = V2S.loc[V2S.groupby("MERDIVEN").WMAPE.idxmin()]
best["erkenlik"] = best.MERDIVEN.map(
    {k: CONFIG["HEDEF_OLGUNLASMA_AY"]-v["bilgi_ay"] for k,v in SECILEN.items()})
print(best.sort_values("erkenlik", ascending=False).round(4).to_string(index=False))
print(f"""
  Dummy WMAPE = {DUMMY_W:.3f}   (skill > 0 ise model Dummy'den iyi)

  KARAR:
    skill > 0.15 olan basamak varsa  -> v2 yonu DOGRU, yeni OOT ile dogrula
    hicbiri gecmiyorsa               -> sorun feature'da degil REJIMDE;
                                        makro degisken / disal sinyal gerekiyor
""")
print("=" * 110)
