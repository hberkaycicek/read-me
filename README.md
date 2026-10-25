# ============================================================================
# BLOK 11 — SAMPIYON DOGRULAMA VE NIHAI KARAR
#
# DURUM: CHAMPION v2c skill +0.265 ile esigi gecti. AMA:
#   - 12 gozlem uzerinde olculdu
#   - Bu OOT v1'de harcandi
#   - ULTRA_ERKEN v2b Pearson -0.089 (WMAPE yaniltici)
#
# BU BLOK:
#   11.1  WMAPE + PEARSON birlikte -> dogru secim (WMAPE tek basina yetmez)
#   11.2  CHAMPION v2c donem bazinda detay
#   11.3  BOOTSTRAP: skill +0.265'in guven araligi
#   11.4  DIEBOLD-MARIANO: Dummy'den farki istatistiksel olarak anlamli mi?
#   11.5  Nihai karar ve uretim cercevesi
# ============================================================================
print("=" * 110)
print("BLOK 11 — SAMPIYON DOGRULAMA")
print("=" * 110)

# ---------------------------------------------------------------- 11.1
print("\n" + "-" * 110)
print("11.1  DOGRU SECIM: WMAPE + PEARSON birlikte")
print("-" * 110)
print("""
  10.D'deki ozet tablo WMAPE.idxmin() ile secim yapti -> YANLIS.
  ULTRA_ERKEN v2b: WMAPE 13.33 (iyi) ama PEARSON -0.089 (model bilgi tasimiyor).
  Dusuk WMAPE, "duz cizgi tahmin et" stratejisiyle de elde edilebilir.

  DOGRU KRITER: PEARSON > 0.30 SARTI + sonra WMAPE minimum
""")
V2S2 = V2S.copy()
V2S2["GECERLI"] = V2S2.PEARSON > 0.30
dogru = []
for ad in V2S2.MERDIVEN.unique():
    s = V2S2[(V2S2.MERDIVEN == ad) & V2S2.GECERLI]
    if len(s) == 0:
        s = V2S2[V2S2.MERDIVEN == ad]
        print(f"  !! [{ad}] hicbir senaryo PEARSON>0.30 saglamiyor")
    b = s.loc[s.WMAPE.idxmin()]
    dogru.append(b)
DOGRU = pd.DataFrame(dogru)
DOGRU["erkenlik"] = DOGRU.MERDIVEN.map(
    {k: CONFIG["HEDEF_OLGUNLASMA_AY"]-v["bilgi_ay"] for k, v in SECILEN.items()})
print("\n  PEARSON>0.30 sartiyla DOGRU secim:")
print(DOGRU[["MERDIVEN","erkenlik","senaryo","WMAPE","SAPMA","POZ","PEARSON","skill"]]
        .sort_values("erkenlik", ascending=False).round(4).to_string(index=False))
print(f"\n  Karsilastirma — 10.D'nin (hatali) secimi:")
print(best[["MERDIVEN","senaryo","WMAPE","PEARSON","skill"]].round(4).to_string(index=False))

# ---------------------------------------------------------------- 11.2
print("\n" + "-" * 110)
print("11.2  CHAMPION v2c — DONEM BAZINDA")
print("-" * 110)
s = SECILEN["CHAMPION"]
payda = [c for c in V2_HAVUZ if GEREKEN_AY.get(c, 99) <= s["bilgi_ay"]
         and (c.startswith("KK_TAKIP_") or c.startswith("RR_"))][0]
print(f"  Oransal payda : {payda.replace('KK_TAKIP_','').replace(f'_{T}','')}")
print(f"  Egitim penceresi: son 30 kohort  |  Duzeltme penceresi: 6 kohort")
print(f"  Girdiler: {[c.replace('KK_TAKIP_','').replace(f'_{T}','') for c in s['kolonlar']+s['ek']]}\n")

r_v2c = wf_pencereli(s["kolonlar"], s["aile"], s["bilgi_ay"], oot_idx,
                     max_gecmis=30, ek_kolonlar=s["ek"], yarilanma=s["yarilanma"],
                     oran_paydasi=payda, duzelt_pencere=6)
r_v1  = walkforward(s["kolonlar"], s["aile"], s["bilgi_ay"], oot_idx,
                    s["yarilanma"], s["ek"])
r_dum = walkforward([CHALLENGER_BUDANMIS[0]], "Dummy", 3, oot_idx)

DET = pd.DataFrame({"DONEM": r_v2c["_don"],
                    "GERCEK_%": [100*x for x in r_v2c["_ger"]],
                    "v2c_%":    [100*x for x in r_v2c["_tah"]],
                    "v1_%":     [100*x for x in r_v1["_tah"]],
                    "Dummy_%":  [100*x for x in r_dum["_tah"]]})
for k in ["v2c","v1","Dummy"]:
    DET[f"{k}_sapma"] = DET[f"{k}_%"] - DET["GERCEK_%"]
print(DET.round(3).to_string(index=False))
print(f"\n  v2c   : WMAPE={r_v2c['WMAPE']:.3f}  sapma={r_v2c['SAPMA']:+.5f}  "
      f"poz={r_v2c['POZ']}/{r_v2c['N']}  Pearson={r_v2c['PEARSON']:.4f}")
print(f"  v1    : WMAPE={r_v1['WMAPE']:.3f}  sapma={r_v1['SAPMA']:+.5f}  "
      f"poz={r_v1['POZ_SAPMA']}/{r_v1['N']}")
print(f"  Dummy : WMAPE={r_dum['WMAPE']:.3f}  sapma={r_dum['SAPMA']:+.5f}  "
      f"poz={r_dum['POZ_SAPMA']}/{r_dum['N']}")

# ---------------------------------------------------------------- 11.3
print("\n" + "-" * 110)
print("11.3  BOOTSTRAP — skill +0.265'in guven araligi (12 gozlem!)")
print("-" * 110)
rng = np.random.default_rng(42)
y  = np.array(r_v2c["_ger"]); p2 = np.array(r_v2c["_tah"])
p1 = np.array(r_v1["_tah"]);  pd_= np.array(r_dum["_tah"])
B = 5000; sk, dm_fark = np.empty(B), np.empty(B)
for i in range(B):
    ix = rng.integers(0, len(y), len(y))
    w2 = 100*np.sum(np.abs(p2[ix]-y[ix]))/np.sum(np.abs(y[ix]))
    wd = 100*np.sum(np.abs(pd_[ix]-y[ix]))/np.sum(np.abs(y[ix]))
    sk[i] = 1 - w2/wd
    dm_fark[i] = wd - w2
print(f"  skill ortalama        : {sk.mean():+.4f}")
print(f"  %90 guven araligi     : [{np.percentile(sk,5):+.4f}, {np.percentile(sk,95):+.4f}]")
print(f"  %95 guven araligi     : [{np.percentile(sk,2.5):+.4f}, {np.percentile(sk,97.5):+.4f}]")
print(f"  P(skill > 0)          : {100*(sk>0).mean():.1f}%")
print(f"  P(skill > 0.15)       : {100*(sk>0.15).mean():.1f}%")
print(f"\n  >> %90 alt sinir > 0 ise, Dummy'den ustunluk 12 gozlemde bile saglam.")
print(f"  >> Alt sinir negatifse, +0.265 SANS olabilir — yeni OOT sart.")

# ---------------------------------------------------------------- 11.4
print("\n" + "-" * 110)
print("11.4  DIEBOLD-MARIANO — v2c vs Dummy, v2c vs v1")
print("-" * 110)
def dm_testi(y, pa, pb, ad_a, ad_b):
    """H0: iki tahminin dogrulugu esit. Kayip farki uzerinde t-testi."""
    la = np.abs(pa-y); lb = np.abs(pb-y)
    dfark = la - lb
    n = len(dfark); ort = dfark.mean()
    # HAC varyans (Newey-West, lag=1) — kucuk orneklem icin
    g0 = np.mean((dfark-ort)**2)
    g1 = np.mean((dfark[1:]-ort)*(dfark[:-1]-ort)) if n > 2 else 0.0
    var = (g0 + 2*g1)/n
    dm = ort/np.sqrt(var) if var > 0 else np.nan
    p = 2*(1-stats.t.cdf(abs(dm), n-1)) if np.isfinite(dm) else np.nan
    print(f"  {ad_a} vs {ad_b}:  ort_kayip_farki={ort:+.6f}  DM={dm:+.3f}  p={p:.4f}  "
          f"{'-> ' + (ad_b if ort>0 else ad_a) + ' DAHA IYI' if p<0.10 else '-> fark anlamsiz'}")
    return p
dm_testi(y, pd_, p2, "Dummy", "v2c")
dm_testi(y, p1,  p2, "v1   ", "v2c")
print("\n  (DM testi 12 gozlemde dusuk gucludur; p<0.10 esigi kullanildi)")

# ---------------------------------------------------------------- 11.5
print("\n" + "=" * 110)
print("11.5  NIHAI KARAR")
print("=" * 110)
alt90 = np.percentile(sk, 5)
print(f"""
  OLCULEN:
    CHAMPION v2c  : WMAPE {r_v2c['WMAPE']:.2f}  skill {1-r_v2c['WMAPE']/r_dum['WMAPE']:+.3f}
                    Pearson {r_v2c['PEARSON']:.3f}  sapma {r_v2c['SAPMA']:+.5f}
                    skill %90 GA = [{np.percentile(sk,5):+.3f}, {np.percentile(sk,95):+.3f}]
    Erken basamaklar (1-3 ay): hicbiri Dummy'yi gecemedi

  YAPISAL BULGU:
    Oransal model HER basamakta Pearson'u yukseltiyor (siralama iyilesiyor)
    ama seviyeyi bozuyor. Seviye ancak TAZE ARTIK bilgisiyle geri kazanilabiliyor,
    o da sadece bilgi_ay >= 4'te mumkun.
    -> ERKENLIK/GUVENILIRLIK takasinin kritik esigi: 6 AY

  URETIM ONERISI:
    [EVET] CHAMPION v2c (6 ay) -> izlemeli pilot, su sartlarla:
             - aylik PSI: MM_SKOR, 60_6, 30_2_MA3
             - aylik sapma izleme: ardisik 3 ay ayni yonlu sapma -> ALARM
             - 202510+ kohortlari olgunlastikca GERCEK OOT ile dogrula
             - k-stabilite yeniden olc: k_son/ilk 0.85-1.15 disina cikarsa REKALIBRE
    [HAYIR] 1-3 ay basamaklari -> uretime alinmaz, arastirma devam

  v3 ICIN YON:
    1. k-stabilite filtresini ORTAK ALANLARA da uygula (MM_SKOR K_SKOR=0.94!)
    2. Erken basamaklar icin seviye cipasi ARA: makro (faiz, issizlik, TUFE),
       portfoy kompozisyonu, kampanya/limit politikasi degiskenleri
    3. OOT penceresini uzat (12 -> 18+) ve YENI bir OOT ayir
    4. ADETSEL hedefi ayni akista kosup tutarsalla karsilastir
""")
print("=" * 110)
