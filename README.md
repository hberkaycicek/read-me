# ============================================================================
# BLOK 6.7 — MM_SKOR GECIKME VE STABILITE TESTI
#
# BULGU (6.2-EK S2): MM_SKOR(t-5) hedefle r=+0.859, MM_SKOR(t) ise r=+0.598.
#   -> MM_SKOR ONCU gosterge. Gecikmeli hali modelde KULLANILMIYOR.
#   -> MM_SKOR(t-k) kohort t icin GEREKEN_AY=0 (gecmis kohortun skoru, bugun belli)
#   -> Sizinti riski SIFIR, erkenlik bedeli SIFIR.
#
# AMA UYARI (6.2-EK S2): z-fark std=0.897, ilk/son yari kaymasi 0.64 std
#   -> Iliski tam stabil degil. Gecikmeli korelasyon TREND ARTEFAKTI olabilir.
#
# BU BLOK:
#   T1  Trend arindirilmis korelasyon — k=+5 bulgusu gercek mi?
#   T2  Kayan pencerede korelasyon stabilitesi
#   T3  MM_SKOR gecikmelerini URET ve sampiyon setlerde TEST et
#   T4  Karar: havuza eklensin mi?
# ============================================================================
print("=" * 106)
print("BLOK 6.7 — MM_SKOR GECIKME VE STABILITE TESTI")
print("=" * 106)
MM = "MM_SKOR"

# ---------------------------------------------------------------- T1
print("\n" + "-" * 106)
print("T1  TREND ARINDIRILMIS KORELASYON — k=+5 bulgusu gercek mi?")
print("-" * 106)
print("  Iki seri de trendliyse gecikmeli korelasyon YAPAY sisik cikar.")
print("  Test: her iki seriden dogrusal trendi cikar, korelasyonu TEKRAR olc.\n")

def _detrend(s):
    s = pd.Series(s).dropna()
    t = np.arange(len(s))
    return pd.Series(s.values - np.polyval(np.polyfit(t, s.values, 1), t), index=s.index)

def _fark(s):
    return pd.Series(s).diff()

_base = d.loc[tr_idx]
rows = []
for k in range(0, 8):
    x = d[MM].shift(k)
    m_ = pd.concat([x.rename("x"), d[HEDEF].rename("y")], axis=1).loc[tr_idx].dropna()
    if len(m_) < 18: continue
    r_ham = m_.corr().iloc[0, 1]
    r_det = _detrend(m_.x).corr(_detrend(m_.y))
    md = pd.concat([_fark(m_.x), _fark(m_.y)], axis=1).dropna()
    r_dif = md.corr().iloc[0, 1] if len(md) > 10 else np.nan
    rows.append({"k": k, "n": len(m_), "r_HAM": r_ham,
                 "r_TRENDSIZ": r_det, "r_FARK": r_dif})
T1 = pd.DataFrame(rows)
print(T1.round(4).to_string(index=False))
print("""
  OKUMA:
    r_HAM      : ham korelasyon (6.2-EK'teki deger)
    r_TRENDSIZ : dogrusal trend cikarildiktan sonra
    r_FARK     : birinci farklar uzerinde (en sert test, seviyeyi tamamen siler)

  >> r_HAM yuksek ama r_TRENDSIZ/r_FARK DUSUKSE -> TREND ARTEFAKTI, kullanma.
  >> Ucu de yuksekse -> GERCEK onculuk, havuza al.
""")
_en_iyi_k = int(T1.loc[T1.r_TRENDSIZ.idxmax(), "k"]) if T1.r_TRENDSIZ.notna().any() else 0
print(f"  Trendsiz korelasyonun EN YUKSEK oldugu gecikme: k={_en_iyi_k}")

# ---------------------------------------------------------------- T2
print("\n" + "-" * 106)
print("T2  KAYAN PENCEREDE KORELASYON STABILITESI")
print("-" * 106)
print("  6.2-EK: ilk yari ort fark -0.329, son yari +0.313 -> 0.64 std kayma.")
print("  Iliski zaman icinde kayiyorsa, uretimde bozulur.\n")
W = 18
olg2 = d[d.HEDEF_KULLANILABILIR].reset_index(drop=True)
kay2 = []
for i in range(len(olg2) - W + 1):
    w = olg2.iloc[i:i+W]
    sat = {"pencere": f"{w[DONEM_KOL].iloc[0]}-{w[DONEM_KOL].iloc[-1]}"}
    for k in [0, 3, 5]:
        xx = olg2[MM].shift(k).iloc[i:i+W]
        mm_ = pd.concat([xx.rename("x"), w[HEDEF].rename("y")], axis=1).dropna()
        sat[f"r_k{k}"] = mm_.corr().iloc[0, 1] if len(mm_) > 10 else np.nan
    kay2.append(sat)
KAY2 = pd.DataFrame(kay2)
print(KAY2.round(3).to_string(index=False))
print("\n  KORELASYON OYNAKLIGI (dusuk = stabil):")
for k in [0, 3, 5]:
    s = KAY2[f"r_k{k}"].dropna()
    print(f"    k={k}: ort={s.mean():+.3f}  std={s.std():.3f}  "
          f"min={s.min():+.3f}  max={s.max():+.3f}  aralik={s.max()-s.min():.3f}")
print("  >> std kucuk VE isaret hic degismiyor -> guvenilir. Aksi halde riskli.")

# ---------------------------------------------------------------- T3
print("\n" + "-" * 106)
print("T3  MM_SKOR GECIKMELERINI URET VE SAMPIYON SETLERDE TEST ET")
print("-" * 106)
# Gecikmeli MM_SKOR: kohort t icin MM_SKOR(t-k), basvuru aninda ZATEN bilinir
for k in [1, 2, 3, 4, 5, 6]:
    ad = f"{MM}_L{k}"
    d[ad] = d[MM].shift(k)
    GEREKEN_AY[ad] = 0           # gecmis kohortun skoru -> bugun belli
    AILE.setdefault("G_KKB", []).append(ad)
d[f"{MM}_MA3"]   = d[MM].rolling(3, min_periods=3).mean(); GEREKEN_AY[f"{MM}_MA3"] = 0
d[f"{MM}_MA3L3"] = d[MM].shift(3).rolling(3, min_periods=3).mean(); GEREKEN_AY[f"{MM}_MA3L3"] = 0
MM_ADAYLAR = [MM] + [f"{MM}_L{k}" for k in [1,2,3,4,5,6]] + [f"{MM}_MA3", f"{MM}_MA3L3"]
print(f"  Uretildi: {MM_ADAYLAR}")
print(f"  Hepsinin GEREKEN_AY = 0  (gecmis kohortun skoru, kohort acildiginda belli)\n")

sonuc67 = []
for ad in MERDIVEN:
    g = NIHAI[ad][NIHAI[ad].GECERLI] if "NIHAI" in dir() else SONUC[ad][SONUC[ad].GECERLI]
    if len(g) == 0: continue
    b = g.iloc[0]
    kols = [c.strip() for c in b.GIRDILER.split(",")]
    ek   = [c for c in kols if c in ("SIN_AY","COS_AY")]
    ana  = [c for c in kols if c not in ("SIN_AY","COS_AY")]
    if MM not in ana: continue
    yar = float(b.YARILANMA) if "YARILANMA" in b.index else 1.0
    for mm_var in MM_ADAYLAR:
        kol = [c if c != MM else mm_var for c in ana]
        CONFIG["ZAMAN_YARILANMA"] = yar
        r = cv_degerlendir(kol, b.AILE, MERDIVEN[ad], ek_kolonlar=ek)
        CONFIG["ZAMAN_YARILANMA"] = 1.0
        if r:
            sonuc67.append({"MERDIVEN": ad, "MM_varyant": mm_var,
                            "CV_WMAPE": r["CV_WMAPE"], "CV_SAPMA": r["CV_SAPMA"],
                            "EN_KOTU": r["EN_KOTU_WMAPE"], "OVERFIT": r["OVERFIT_ORANI"],
                            "MAKS_VIF": r["MAKS_VIF"], "ISARET": r["ISARET_SERISI"]})
S67 = pd.DataFrame(sonuc67)
for ad in S67.MERDIVEN.unique():
    s = S67[S67.MERDIVEN == ad].sort_values("CV_WMAPE")
    taban = s[s.MM_varyant == MM].CV_WMAPE.iloc[0]
    s = s.copy(); s["KAZANC"] = (taban - s.CV_WMAPE).round(3)
    print(f"\n  [{ad}]  taban (MM_SKOR es zamanli) = {taban:.4f}")
    print(s.round(4).to_string(index=False))

# ---------------------------------------------------------------- T4
print("\n" + "=" * 106)
print("T4  KARAR")
print("=" * 106)
kaz = []
for ad in S67.MERDIVEN.unique():
    s = S67[S67.MERDIVEN == ad]
    taban = s[s.MM_varyant == MM].CV_WMAPE.iloc[0]
    en = s.sort_values("CV_WMAPE").iloc[0]
    kaz.append({"MERDIVEN": ad, "taban": taban, "en_iyi_varyant": en.MM_varyant,
                "en_iyi_WMAPE": en.CV_WMAPE, "KAZANC": taban - en.CV_WMAPE,
                "sapma": en.CV_SAPMA, "overfit": en.OVERFIT})
KAZ = pd.DataFrame(kaz)
print(KAZ.round(4).to_string(index=False))

_ort_kazanc = KAZ.KAZANC.mean()
_kazanan_n  = int((KAZ.en_iyi_varyant != MM).sum())
print(f"\n  Gecikmeli varyant kazanan basamak : {_kazanan_n} / {len(KAZ)}")
print(f"  Ortalama kazanc                   : {_ort_kazanc:+.3f} WMAPE")
print(f"""
  KARAR KURALI:
    EKLE   eger (a) T1'de r_TRENDSIZ ve r_FARK da yuksek  VE
                 (b) T2'de korelasyon isaret degistirmiyor, std dusuk  VE
                 (c) T3'te >=3 basamakta kazanc > 0.3 WMAPE
    EKLEME eger T1'de sadece r_HAM yuksekse -> TREND ARTEFAKTI

  >> EKLENIRSE: BLOK 3.8 havuzlarina MM_SKOR gecikmeleri dahil edilip
     BLOK 5 grid'i YENIDEN calistirilmali (havuz degisti).
  >> EKLENMEZSE: mevcut merdiven aynen BLOK 7 hakemligine gider.
""")

# Havuza otomatik ekleme (kriterler saglanirsa)
_T1ok = bool(T1.loc[T1.k == _en_iyi_k, "r_FARK"].abs().iloc[0] > 0.30) if len(T1) else False
_T2ok = bool(all(KAY2[f"r_k{_en_iyi_k}"].dropna() > 0)) if f"r_k{_en_iyi_k}" in KAY2 else False
_T3ok = bool((KAZ.KAZANC > 0.3).sum() >= 3)
print(f"  T1 (trend arindirilmis guclu) : {_T1ok}")
print(f"  T2 (isaret stabil)            : {_T2ok}")
print(f"  T3 (>=3 basamakta kazanc)     : {_T3ok}")
MM_GECIKME_EKLE = _T1ok and _T2ok and _T3ok
print(f"\n  >>> SONUC: MM_SKOR gecikmeleri havuza "
      f"{'EKLENECEK -> BLOK 5 yeniden calistirilmali' if MM_GECIKME_EKLE else 'EKLENMEYECEK -> BLOK 7 ile devam'}")
print("=" * 106)
