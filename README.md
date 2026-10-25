# ============================================================================
# BLOK 7.6 — ADAPTIF AGIRLIK VE NIHAI SECIM
#
# SORUN: 7.4'te sabit agirlik (0.40 CV / 0.45 ARA) kullandik.
#        Ama 7.3 gosterdi ki CV'nin guvenilirligi BASAMAGA GORE DEGISIYOR:
#          ULTRA_ERKEN : Spearman -0.785 (p<0.0001)  -> CV TERS calisiyor
#          ORTA        : Spearman +0.603 (p=0.005)   -> CV guvenilir
#        Sabit agirlik, CV'nin ters calistigi basamakta YANLIS secim yapiyor.
#
# COZUM: CV agirligi, o basamaktaki CV-ARA uyumuna gore UYARLANIR.
#        Bu bir hiperparametre ayari degil; olculen bir gercegin modele yansitilmasi.
#
# AYRICA:
#   - Yanlilik agirligi 0.15 -> 0.20 (bu projenin ana riski)
#   - ARA_POZ dengesi kriteri eklendi (0/6 veya 6/6 -> tek yonlu sapma)
#   - "Sadece ARA ile secseydik ne olurdu?" karsilastirmasi
# ============================================================================
print("=" * 108)
print("BLOK 7.6 — ADAPTIF AGIRLIK VE NIHAI SECIM")
print("=" * 108)

YANB = YAN.set_index("MERDIVEN")

def adaptif_agirlik(rho_rank, rho_wmape):
    """
    CV agirligi, CV siralamasinin ARA'da ne kadar tuttuguna gore belirlenir.
    Iki Spearman'in KUCUGU esas alinir (muhafazakar).
    """
    rho = min(rho_rank, rho_wmape)
    if   rho >= 0.50: return 0.45, 0.35, 0.20, "CV GUVENILIR"
    elif rho >= 0.20: return 0.35, 0.45, 0.20, "CV kismen guvenilir"
    elif rho >= 0.00: return 0.25, 0.55, 0.20, "CV zayif"
    else:             return 0.10, 0.70, 0.20, "CV TERS CALISIYOR"

print("\n  BASAMAK BAZLI AGIRLIKLAR (7.3'teki Spearman'a gore):\n")
AG = {}
for ad in ARA_SONUC:
    if ad not in YANB.index: continue
    rw = float(YANB.loc[ad, "Spearman_WMAPE"]); rr = float(YANB.loc[ad, "Spearman_RANK"])
    w_cv, w_ara, w_yan, etiket = adaptif_agirlik(rr, rw)
    AG[ad] = (w_cv, w_ara, w_yan)
    print(f"    {ad:12s} rho_WMAPE={rw:+.3f}  rho_RANK={rr:+.3f}  ->  "
          f"CV={w_cv:.2f} ARA={w_ara:.2f} YANLI={w_yan:.2f}   [{etiket}]")

print("\n" + "-" * 108)
print("NIHAI SECIM")
print("-" * 108)
NIHAI_ADAY2, fin2, kiyas = {}, [], []
for ad, b_ay in MERDIVEN.items():
    if ad not in ARA_SONUC: continue
    X = ARA_SONUC[ad].copy()
    w_cv, w_ara, w_yan = AG.get(ad, (0.35, 0.45, 0.20))
    X["R_CV"]    = X.CV_WMAPE.rank(pct=True)
    X["R_ARA"]   = X.ARA_WMAPE.rank(pct=True)
    X["R_YANLI"] = X.ARA_MUTLAK_SAPMA.rank(pct=True)
    X["BIRLESIK2"] = w_cv*X.R_CV + w_ara*X.R_ARA + w_yan*X.R_YANLI

    # ELEME 1: tek yonlu VE buyuk sapma
    n_ara = int(X.ARA_N.iloc[0])
    tek_yon = X.ARA_POZ.isin([f"{n_ara}/{n_ara}", f"0/{n_ara}"])
    buyuk   = X.ARA_MUTLAK_SAPMA > d.loc[ara_idx, HEDEF].mean()*0.12
    X["ELE_YANLI"] = tek_yon & buyuk
    # ELEME 2: ARA'da en kotu tekil hata cok buyukse
    X["ELE_UC"] = X.ARA_EN_KOTU > d.loc[ara_idx, HEDEF].mean()*0.45
    X["ELENDI"] = X.ELE_YANLI | X.ELE_UC

    X = X.sort_values(["ELENDI", "BIRLESIK2"], ascending=[True, True])
    NIHAI_ADAY2[ad] = X
    G = X[~X.ELENDI] if (~X.ELENDI).any() else X
    b = G.iloc[0]

    # Kiyas: eski sabit agirlik / sadece ARA / sadece CV ne secerdi?
    eski = NIHAI_ADAY[ad][~NIHAI_ADAY[ad].ARA_ELENDI].iloc[0] \
           if ad in NIHAI_ADAY and (~NIHAI_ADAY[ad].ARA_ELENDI).any() else None
    sade_ara = X.sort_values("ARA_WMAPE").iloc[0]
    sade_cv  = X.sort_values("CV_WMAPE").iloc[0]
    kiyas.append({"MERDIVEN": ad,
                  "SADECE_CV_ARA_WMAPE":  sade_cv.ARA_WMAPE,
                  "SABIT_AGIRLIK_ARA":    eski.ARA_WMAPE if eski is not None else np.nan,
                  "ADAPTIF_ARA":          b.ARA_WMAPE,
                  "SADECE_ARA_ARA_WMAPE": sade_ara.ARA_WMAPE})

    kols = [c.strip() for c in b.GIRDILER.split(",")]
    SECILEN[ad] = {"kolonlar": [c for c in kols if c not in ("SIN_AY","COS_AY")],
                   "ek": [c for c in kols if c in ("SIN_AY","COS_AY")],
                   "aile": b.AILE, "bilgi_ay": b_ay, "yarilanma": float(b.YARILANMA)}
    fin2.append({"MERDIVEN": ad, "bilgi_ay": b_ay,
                 "erkenlik": CONFIG["HEDEF_OLGUNLASMA_AY"]-b_ay,
                 "etkin_ay": max(GEREKEN_AY.get(c,0) for c in kols),
                 "GIRDILER": b.GIRDILER.replace("KK_TAKIP_","").replace(f"_{T}",""),
                 "AILE": b.AILE, "YARILANMA": b.YARILANMA,
                 "CV_WMAPE": b.CV_WMAPE, "ARA_WMAPE": b.ARA_WMAPE,
                 "ARA_SAPMA": b.ARA_SAPMA, "ARA_POZ": b.ARA_POZ,
                 "ARA_EN_KOTU": b.ARA_EN_KOTU, "elenen": int(X.ELENDI.sum())})

    print(f"\n  [{ad}]  agirlik CV={w_cv:.2f}/ARA={w_ara:.2f}  elenen={int(X.ELENDI.sum())}/{len(X)}")
    t = G.head(4).copy()
    t["GIRDILER"] = t.GIRDILER.str.replace("KK_TAKIP_","").str.replace(f"_{T}","")
    print(t[["GIRDILER","AILE","CV_WMAPE","ARA_WMAPE","ARA_SAPMA","ARA_POZ",
             "ARA_EN_KOTU","BIRLESIK2"]].round(4).to_string(index=False))

# ---------------------------------------------------------------- KIYAS
print("\n" + "=" * 108)
print("SECIM STRATEJILERININ KARSILASTIRMASI (ARA_WMAPE uzerinden)")
print("=" * 108)
K = pd.DataFrame(kiyas)
print(K.round(4).to_string(index=False))
print(f"""
  ORTALAMA ARA_WMAPE:
    Sadece CV ile secim      : {K.SADECE_CV_ARA_WMAPE.mean():.4f}   <- onceki calismanin yontemi
    Sabit agirlik (7.4)      : {K.SABIT_AGIRLIK_ARA.mean():.4f}
    ADAPTIF agirlik (7.6)    : {K.ADAPTIF_ARA.mean():.4f}
    Sadece ARA ile secim     : {K.SADECE_ARA_ARA_WMAPE.mean():.4f}   <- ust sinir (ARA'ya overfit)

  NOT: 'Sadece ARA' en dusuk cikar ama o da ARA'ya overfit olur — OOT'de bozulur.
       ADAPTIF, CV'nin guvenilir oldugu yerde CV'yi, olmadigi yerde ARA'yi dinler.
""")

# ---------------------------------------------------------------- NIHAI
print("=" * 108)
print("NIHAI MERDIVEN — OOT'ye gidecek 5 model")
print("=" * 108)
FINAL2 = pd.DataFrame(fin2)
print(FINAL2.round(5).to_string(index=False))

print("\n  ERKENLIK / DOGRULUK:")
for _, r in FINAL2.iterrows():
    print(f"    {r.MERDIVEN:12s} {int(r.erkenlik):2d} ay erken  "
          f"CV={r.CV_WMAPE:5.2f}  ARA={r.ARA_WMAPE:5.2f}  "
          f"sapma={r.ARA_SAPMA:+.5f}  poz={r.ARA_POZ}")

print("\n  7.4 -> 7.6 DEGISIKLIKLERI:")
for ad in MERDIVEN:
    if ad not in NIHAI_ADAY or ad not in NIHAI_ADAY2: continue
    e = NIHAI_ADAY[ad][~NIHAI_ADAY[ad].ARA_ELENDI].iloc[0] \
        if (~NIHAI_ADAY[ad].ARA_ELENDI).any() else NIHAI_ADAY[ad].iloc[0]
    y = NIHAI_ADAY2[ad][~NIHAI_ADAY2[ad].ELENDI].iloc[0] \
        if (~NIHAI_ADAY2[ad].ELENDI).any() else NIHAI_ADAY2[ad].iloc[0]
    if e.GIRDILER != y.GIRDILER or e.AILE != y.AILE:
        print(f"    {ad:12s} DEGISTI  ARA_WMAPE {e.ARA_WMAPE:.3f} -> {y.ARA_WMAPE:.3f}  "
              f"| sapma {e.ARA_SAPMA:+.5f} -> {y.ARA_SAPMA:+.5f}")
        print(f"        once : {e.GIRDILER[:58]} ({e.AILE})")
        print(f"        simdi: {y.GIRDILER[:58]} ({y.AILE})")
    else:
        print(f"    {ad:12s} ayni")

print("\n" + "=" * 108)
print("BLOK 7.6 TAMAM — OOT hala DOKUNULMADI. Secim KILITLENDI.")
print(f"  Ortalama ARA_WMAPE : {FINAL2.ARA_WMAPE.mean():.4f}")
print(f"  Ortalama |sapma|   : {FINAL2.ARA_SAPMA.abs().mean():.5f}  "
      f"(hedef ort {d.loc[ara_idx,HEDEF].mean():.5f}'in "
      f"%{100*FINAL2.ARA_SAPMA.abs().mean()/d.loc[ara_idx,HEDEF].mean():.1f}'i)")
print("=" * 108)
