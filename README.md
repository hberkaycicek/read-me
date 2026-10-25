# ============================================================================
# BLOK 6.2-EK — MM_SKOR DERINLEMESINE TESHIS
#
# BULGU: MM_SKOR cikinca WMAPE 3-5 KAT kotulesiyor VE sapma -0.004/-0.006'ya
#        firliyor. Bu bir 'feature katkisi' degil, SEVIYE CIPASI davranisi.
#
# CEVAPLANACAK SORULAR:
#   S1  MM_SKOR tek basina ne yapiyor?
#   S2  Hedefle ES ZAMANLI mi hareket ediyor? (seviye cipasi tezi)
#   S3  Katsayi isareti IS MANTIGIYLA tutarli mi?
#   S4  Alternatif seviye cipalari onun yerini tutar mi?
#   S5  Fold bazinda sapma yapisi nasil degisiyor?
#   S6  MODEL RISKI: MM_SKOR kayarsa tahmin ne kadar oynar?
# ============================================================================
print("=" * 106)
print("BLOK 6.2-EK — MM_SKOR DERINLEMESINE TESHIS")
print("=" * 106)

MM = "MM_SKOR"
_s = d.loc[tr_idx, MM].dropna()
print(f"\n  MM_SKOR: aralik=[{_s.min():.4f}, {_s.max():.4f}]  ort={_s.mean():.4f}  "
      f"std={_s.std():.4f}  CV=%{100*_s.std()/_s.mean():.1f}")
print(f"  HEDEF  : aralik=[{d.loc[tr_idx,HEDEF].min():.4f}, {d.loc[tr_idx,HEDEF].max():.4f}]  "
      f"ort={d.loc[tr_idx,HEDEF].mean():.4f}")
print(f"  MM_SKOR / HEDEF orani: {_s.mean()/d.loc[tr_idx,HEDEF].mean():.2f} kat")
print(f"  >> MM_SKOR bir OLASILIK gibi gorunuyor (muhtemelen ortalama PD).")

# ---------------------------------------------------------------- S1
print("\n" + "-" * 106)
print("S1  MM_SKOR TEK BASINA")
print("-" * 106)
solo = []
for ad, b_ay in MERDIVEN.items():
    for kk, et in [([MM], "MM_SKOR tek"),
                   ([tek.iloc[0]["degisken"]], "en guclu oncul tek")]:
        if max(GEREKEN_AY.get(c, 0) for c in kk) > b_ay: continue
        r = cv_degerlendir(kk, "Linear", b_ay)
        if r: solo.append({"MERDIVEN": ad, "model": et, "CV_WMAPE": r["CV_WMAPE"],
                           "CV_SAPMA": r["CV_SAPMA"], "OVERFIT": r["OVERFIT_ORANI"],
                           "YON": r["YON_BASARISI"]})
print(pd.DataFrame(solo).round(4).to_string(index=False))
print("\n  >> MM_SKOR tek basina ZAYIFSA ama kombinasyonda KRITIKSE,")
print("     tamamlayici bilgi tasiyor demektir (seviye), oncul ise SIRALAMA tasiyor.")

# ---------------------------------------------------------------- S2
print("\n" + "-" * 106)
print("S2  ES ZAMANLI HAREKET — seviye cipasi tezi")
print("-" * 106)
_tr = d.loc[tr_idx, [DONEM_KOL, MM, HEDEF]].dropna()
print("  Gecikmeli korelasyonlar (MM_SKOR(t-k) ~ HEDEF(t)):")
for k in range(-3, 7):
    x = d[MM].shift(k)
    m_ = pd.concat([x.rename("x"), d[HEDEF]], axis=1).loc[tr_idx].dropna()
    if len(m_) > 15:
        r_ = m_.corr().iloc[0, 1]
        isaret = " <-- ES ZAMANLI" if k == 0 else ""
        print(f"    k={k:+2d} : r={r_:+.4f}  (n={len(m_)}){isaret}")
print("\n  Seviye takibi: iki serinin NORMALIZE edilmis hali ne kadar ortusuyor?")
_z1 = (_tr[MM] - _tr[MM].mean())/_tr[MM].std()
_z2 = (_tr[HEDEF] - _tr[HEDEF].mean())/_tr[HEDEF].std()
print(f"    z-skor farkinin std'si: {(_z1-_z2).std():.4f}  (0'a yakin = birlikte hareket)")
print(f"    ilk yari ort fark: {(_z1-_z2).iloc[:len(_z1)//2].mean():+.4f}   "
      f"son yari: {(_z1-_z2).iloc[len(_z1)//2:].mean():+.4f}")
print("    >> Iki yari arasindaki fark kucukse, iliski ZAMAN ICINDE STABIL.")

# ---------------------------------------------------------------- S3
print("\n" + "-" * 106)
print("S3  KATSAYI ISARETI VE BUYUKLUGU")
print("-" * 106)
for ad in MERDIVEN:
    g = SONUC[ad][SONUC[ad].GECERLI]
    if len(g) == 0: continue
    b = g.iloc[0]; kk = [c.strip() for c in b.GIRDILER.split(",")]
    sub = d.loc[tr_idx, kk + [HEDEF]].dropna()
    pp = pipeline_kur("Linear"); pp.fit(sub[kk], sub[HEDEF])
    sc, reg = pp[0], pp[-1]
    ham = reg.coef_ / sc.scale_
    b0 = float(reg.intercept_ - np.dot(ham, sc.mean_))
    print(f"\n  [{ad}]  sabit={b0:+.6f}")
    for c, kat in zip(kk, ham):
        etki = kat * d.loc[tr_idx, c].std()     # 1 std oynamanin etkisi
        print(f"    {c:32s} katsayi={kat:+12.6f}   1-std etkisi={etki:+.5f}  "
              f"({100*etki/d.loc[tr_idx,HEDEF].mean():+.1f}% hedefin)")
print("\n  >> MM_SKOR bir PD ise katsayisi POZITIF olmali (PD artar -> 12M90 artar).")
print("  >> 1-std etkisi, degiskenin GERCEK agirligini gosterir (katsayi x oynaklik).")

# ---------------------------------------------------------------- S4
print("\n" + "-" * 106)
print("S4  ALTERNATIF SEVIYE CIPALARI — MM_SKOR'un yerini tutan var mi?")
print("-" * 106)
ADAY_CIPA = [MM, "KKB_MERKEZ", "KALIBRE_PD_AGIRLIKLI", "HEDEF_L9_12_ORT", "HEDEF_L9"]
ADAY_CIPA = [c for c in ADAY_CIPA if c in d.columns]
cip = []
for ad in ["CHALLENGER", "CHAMPION"]:
    g = SONUC[ad][SONUC[ad].GECERLI]
    if len(g) == 0: continue
    b = g.iloc[0]; kk = [c.strip() for c in b.GIRDILER.split(",")]
    cekirdek = [c for c in kk if c != MM]
    for cp in [None] + ADAY_CIPA:
        kol = cekirdek + ([cp] if cp else [])
        if not kol: continue
        if cp and GEREKEN_AY.get(cp, 0) > MERDIVEN[ad]: continue
        r = cv_degerlendir(kol, b.AILE, MERDIVEN[ad])
        if r: cip.append({"MERDIVEN": ad, "cipa": cp or "(yok)", "K": r["K"],
                          "CV_WMAPE": r["CV_WMAPE"], "CV_SAPMA": r["CV_SAPMA"],
                          "OVERFIT": r["OVERFIT_ORANI"], "ISARET": r["ISARET_SERISI"]})
CIP = pd.DataFrame(cip)
print(CIP.round(4).to_string(index=False))
print("\n  >> Hicbir alternatif MM_SKOR'a yaklasamiyorsa, model ona BAGIMLI demektir.")
print("     Bu bir bulgu, ama ayni zamanda MODEL RISKI — S6'da olculecek.")

# ---------------------------------------------------------------- S5
print("\n" + "-" * 106)
print("S5  FOLD BAZINDA SAPMA YAPISI")
print("-" * 106)
for ad in ["CHALLENGER", "ORTA"]:
    g = SONUC[ad][SONUC[ad].GECERLI]
    if len(g) == 0: continue
    b = g.iloc[0]; kk = [c.strip() for c in b.GIRDILER.split(",")]
    r1 = cv_degerlendir(kk, b.AILE, MERDIVEN[ad])
    r2 = cv_degerlendir([c for c in kk if c != MM], b.AILE, MERDIVEN[ad])
    print(f"\n  [{ad}]")
    print(f"    MM_SKOR ILE  : fold sapmalari = {r1['FOLD_SAPMALAR']}  isaret_serisi={r1['ISARET_SERISI']}")
    if r2:
        print(f"    MM_SKOR'SUZ  : fold sapmalari = {r2['FOLD_SAPMALAR']}  isaret_serisi={r2['ISARET_SERISI']}")
print("\n  >> MM_SKOR'suz TUM fold'lar ayni yonde sapiyorsa, model seviyeyi")
print("     yakalayamiyor demektir. Onceki modelin OOT'deki 12/12 felaketinin aynisi.")

# ---------------------------------------------------------------- S6
print("\n" + "-" * 106)
print("S6  MODEL RISKI — MM_SKOR kayarsa tahmin ne kadar oynar?")
print("-" * 106)
print("  SENARYO: uretimde skor karti rekalibre edilirse MM_SKOR seviyesi kayar.")
print("  Duyarlilik: MM_SKOR'a +/- x std sok uygulanirsa tahmin ne olur?\n")
for ad in ["CHALLENGER", "CHAMPION"]:
    g = SONUC[ad][SONUC[ad].GECERLI]
    if len(g) == 0: continue
    b = g.iloc[0]; kk = [c.strip() for c in b.GIRDILER.split(",")]
    if MM not in kk: continue
    sub = d.loc[tr_idx, kk + [HEDEF]].dropna()
    pp = pipeline_kur(b.AILE); pp.fit(sub[kk], sub[HEDEF])
    taban = pp.predict(sub[kk]).mean()
    sd = sub[MM].std()
    print(f"  [{ad}]  taban ortalama tahmin = {taban:.5f}  (gerceklesen {sub[HEDEF].mean():.5f})")
    for sok in [-2, -1, -0.5, 0.5, 1, 2]:
        s2 = sub.copy(); s2[MM] = s2[MM] + sok*sd
        yeni = pp.predict(s2[kk]).mean()
        print(f"    MM_SKOR {sok:+.1f} std -> tahmin {yeni:.5f}  "
              f"({100*(yeni/taban-1):+6.1f}%)")
print("\n  >> +/-1 std sokta tahmin %20'den fazla oynuyorsa, MM_SKOR icin")
print("     AYLIK PSI IZLEMESI zorunlu hale gelir. Izleme esigi BLOK 10'a yazilacak.")

print("\n" + "=" * 106)
print("MM_SKOR KARAR CERCEVESI")
print("=" * 106)
print("""
  MM_SKOR'u MODELDE TUT eger:
    - Katsayi isareti is mantigiyla tutarli (PD artar -> 12M90 artar)
    - Es zamanli korelasyon gecikmelilerden guclu (seviye cipasi tezi)
    - z_OOT dusuk ve disari=0 (zaten dogrulandi: 0.600 / 0.000)
    - Alternatif cipa yok

  ANCAK zorunlu onlemler:
    1. MM_SKOR icin AYLIK PSI izlemesi (esik: |z| > 1.0 -> alarm)
    2. Skor karti versiyon degisikligi -> MODEL YENIDEN KALIBRE EDILMELI
    3. Model dokumanina 'tek degisken bagimliligi' riski yazilmali
    4. MM_SKOR'suz yedek model (degraded mode) hazir tutulmali
""")
print("=" * 106)
