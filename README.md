# ============================================================================
# BLOK 8 — OOT ACILISI
#
# BU BLOK BIR KEZ CALISIR. Sonrasinda model secimi YAPILMAZ.
# OOT = 202410..202509 (12 kohort). Secim surecinde HIC kullanilmadi:
#   - grid/Optuna : sadece TRAIN (39 kohort)
#   - hakemlik    : sadece ARA  ( 6 kohort)
#   - OOT         : su ana kadar sifir temas
#
# ONEMLI: Burada cikan sonuc NE OLURSA OLSUN model degistirilmeyecek.
#         Aksi halde OOT de bir secim setine donusur ve bagimsizligini kaybeder.
#
# ICERIK:
#   8.1  OOT walk-forward (5 merdiven modeli + 2 referans)
#   8.2  HATA AYRISTIRMASI — hatanin yuzde kaci YANLILIK? (onceki model: %82)
#   8.3  ISARET TESTI — sapmalar tek yonlu mu? (onceki model: 12/12, p=0.0005)
#   8.4  ARA -> OOT TUTARLILIGI — hakemlik gercekten ise yaradi mi?
#   8.5  ONCEKI CALISMAYLA DOGRUDAN KARSILASTIRMA (ayni OOT penceresi!)
#   8.6  Donem bazinda detay
#   8.7  Gorsellestirme
# ============================================================================
print("=" * 110)
print("BLOK 8 — OOT ACILISI".center(110))
print("=" * 110)
print(f"\n  OOT: {d.loc[oot_idx, DONEM_KOL].min()} .. {d.loc[oot_idx, DONEM_KOL].max()}  "
      f"({len(oot_idx)} kohort)")
print(f"  Hedef ortalamasi (OOT): {d.loc[oot_idx, HEDEF].mean():.5f}")
print(f"  Hedef ortalamasi (ARA): {d.loc[ara_idx, HEDEF].mean():.5f}")
print(f"  Hedef ortalamasi (TRAIN): {d.loc[tr_idx, HEDEF].mean():.5f}")
print("  >> Seviye farki varsa, modelin rejim kaymasini ne kadar tasidigi burada gorulur.\n")

# ---------------------------------------------------------------- 8.1
print("-" * 110)
print("8.1  OOT WALK-FORWARD")
print("-" * 110)

# REFERANS 1: Dummy (ortalama)
# REFERANS 2: ONCEKI CALISMANIN MODELI — ayni OOT penceresi oldugu icin
#             dogrudan karsilastirilabilir (20_2 + 60_3 + KKB, RidgeCV, bilgi_ay=3)
ONCEKI = [oncul_kolon("20_2"), oncul_kolon("60_3"), "KKB_SKOR_O0120"]
ONCEKI_VAR = all(c in d.columns for c in ONCEKI)

OOT_SONUC, oot_rows = {}, []
for ad, s in SECILEN.items():
    r = walkforward(s["kolonlar"], s["aile"], s["bilgi_ay"], oot_idx,
                    yarilanma=s["yarilanma"], ek_kolonlar=s["ek"])
    if r is None: continue
    OOT_SONUC[ad] = r
    oot_rows.append({"MODEL": ad, "tip": "merdiven", "bilgi_ay": s["bilgi_ay"],
                     "erkenlik": CONFIG["HEDEF_OLGUNLASMA_AY"]-s["bilgi_ay"],
                     "N": r["N"], "WMAPE": r["WMAPE"], "MAPE": r["MAPE"],
                     "MAE": r["MAE"], "RMSE": r["RMSE"], "SAPMA": r["SAPMA"],
                     "BIAS_ORANI": r["BIAS_ORANI"], "POZ": r["POZ_SAPMA"],
                     "EN_KOTU": r["EN_KOTU_MUTLAK"]})

# Referanslar
_dref = walkforward([CHALLENGER_BUDANMIS[0]], "Dummy", 3, oot_idx)
if _dref:
    OOT_SONUC["REF_Dummy"] = _dref
    oot_rows.append({"MODEL": "REF_Dummy(ortalama)", "tip": "referans", "bilgi_ay": 3,
                     "erkenlik": 9, "N": _dref["N"], "WMAPE": _dref["WMAPE"],
                     "MAPE": _dref["MAPE"], "MAE": _dref["MAE"], "RMSE": _dref["RMSE"],
                     "SAPMA": _dref["SAPMA"], "BIAS_ORANI": _dref["BIAS_ORANI"],
                     "POZ": _dref["POZ_SAPMA"], "EN_KOTU": _dref["EN_KOTU_MUTLAK"]})
if ONCEKI_VAR:
    _oref = walkforward(ONCEKI, "Ridge", 3, oot_idx)
    if _oref:
        OOT_SONUC["REF_Onceki"] = _oref
        oot_rows.append({"MODEL": "REF_Onceki(20_2+60_3+KKB)", "tip": "referans",
                         "bilgi_ay": 3, "erkenlik": 9, "N": _oref["N"],
                         "WMAPE": _oref["WMAPE"], "MAPE": _oref["MAPE"],
                         "MAE": _oref["MAE"], "RMSE": _oref["RMSE"],
                         "SAPMA": _oref["SAPMA"], "BIAS_ORANI": _oref["BIAS_ORANI"],
                         "POZ": _oref["POZ_SAPMA"], "EN_KOTU": _oref["EN_KOTU_MUTLAK"]})

OOT = pd.DataFrame(oot_rows)
OOT["SKILL"] = 1 - OOT.WMAPE/OOT[OOT.MODEL.str.startswith("REF_Dummy")].WMAPE.iloc[0] \
               if (OOT.MODEL.str.startswith("REF_Dummy")).any() else np.nan
print(OOT.round(5).to_string(index=False))
print("""
  BIAS_ORANI = |ortalama sapma| / MAE
    1.00 -> hatanin TAMAMI sistematik (onceki modelde tam olarak 1.000 cikmisti)
    0.00 -> hata tamamen rastgele, yanlilik yok
""")

# ---------------------------------------------------------------- 8.2
print("-" * 110)
print("8.2  HATA AYRISTIRMASI — MSE = yanlilik^2 + varyans")
print("-" * 110)
ayr = []
for ad, r in OOT_SONUC.items():
    e = np.array(r["_tah"]) - np.array(r["_ger"])
    mse, bias2, var = float(np.mean(e**2)), float(np.mean(e))**2, float(np.var(e))
    ayr.append({"MODEL": ad, "MSE": mse, "yanlilik^2": bias2, "varyans": var,
                "YANLILIK_%": 100*bias2/mse if mse > 0 else np.nan,
                "DAGILIM_%": 100*var/mse if mse > 0 else np.nan})
AYR = pd.DataFrame(ayr).sort_values("YANLILIK_%")
print(AYR.round(6).to_string(index=False))
print("""
  ONCEKI CALISMA: hatanin %81.7'si YANLILIK idi (MSE=0.0977, bias^2=0.0798)
  -> Yeni modellerde bu oran DUSUKSE, yapisal sorun cozulmus demektir.
""")

# ---------------------------------------------------------------- 8.3
print("-" * 110)
print("8.3  ISARET TESTI — sapmalar tek yonlu mu?")
print("-" * 110)
print("  Onceki calismada OOT'de 12 sapmanin 12'si POZITIF idi (binom p=0.00049).")
print("  Bu, hatanin rastgele degil SISTEMATIK oldugunun kaniti.\n")
isr = []
for ad, r in OOT_SONUC.items():
    e = np.array(r["_tah"]) - np.array(r["_ger"])
    poz = int((e > 0).sum()); n = len(e)
    pb = stats.binomtest(poz, n, 0.5).pvalue
    try:    pw = stats.wilcoxon(e).pvalue
    except Exception: pw = np.nan
    isr.append({"MODEL": ad, "poz/n": f"{poz}/{n}", "binom_p": pb, "wilcoxon_p": pw,
                "ort_sapma": float(np.mean(e)),
                "SISTEMATIK": "EVET" if pb < 0.05 else "hayir"})
ISR = pd.DataFrame(isr)
print(ISR.round(5).to_string(index=False))

# ---------------------------------------------------------------- 8.4
print("\n" + "-" * 110)
print("8.4  ARA -> OOT TUTARLILIGI — hakemlik ise yaradi mi?")
print("-" * 110)
tut = []
for ad in SECILEN:
    if ad not in OOT_SONUC: continue
    a = FINAL2[FINAL2.MERDIVEN == ad].iloc[0]
    o = OOT_SONUC[ad]
    tut.append({"MERDIVEN": ad, "CV_WMAPE": a.CV_WMAPE, "ARA_WMAPE": a.ARA_WMAPE,
                "OOT_WMAPE": o["WMAPE"],
                "CV->ARA_%": 100*(a.ARA_WMAPE/a.CV_WMAPE - 1),
                "ARA->OOT_%": 100*(o["WMAPE"]/a.ARA_WMAPE - 1),
                "CV->OOT_%": 100*(o["WMAPE"]/a.CV_WMAPE - 1),
                "ARA_sapma": a.ARA_SAPMA, "OOT_sapma": o["SAPMA"]})
TUT = pd.DataFrame(tut)
print(TUT.round(4).to_string(index=False))
if len(TUT) >= 3:
    rho_ao, p_ao = stats.spearmanr(TUT.ARA_WMAPE, TUT.OOT_WMAPE)
    rho_co, p_co = stats.spearmanr(TUT.CV_WMAPE,  TUT.OOT_WMAPE)
    print(f"\n  Spearman(ARA, OOT) = {rho_ao:+.3f} (p={p_ao:.3f})")
    print(f"  Spearman(CV , OOT) = {rho_co:+.3f} (p={p_co:.3f})")
    print(f"  >> ARA korelasyonu CV'den YUKSEKSE, hakemlik katmani degerliydi.")
print(f"\n  Ortalama bozulma: CV->ARA %{TUT['CV->ARA_%'].mean():.1f}  |  "
      f"ARA->OOT %{TUT['ARA->OOT_%'].mean():.1f}  |  CV->OOT %{TUT['CV->OOT_%'].mean():.1f}")
print(f"  (Onceki calismada CV->OOT bozulmasi %61-122 idi)")

# ---------------------------------------------------------------- 8.5
print("\n" + "=" * 110)
print("8.5  ONCEKI CALISMAYLA DOGRUDAN KARSILASTIRMA")
print("=" * 110)
print("  Ayni OOT penceresi (202410-202509), ayni hedef, ayni walk-forward kurgusu.\n")
if "REF_Onceki" in OOT_SONUC:
    o = OOT_SONUC["REF_Onceki"]
    print(f"  ONCEKI MODEL (20_2 + 60_3 + KKB_SKOR, Ridge, 3 ay):")
    print(f"     WMAPE={o['WMAPE']:.3f}  sapma={o['SAPMA']:+.5f}  "
          f"poz={o['POZ_SAPMA']}/{o['N']}  bias_orani={o['BIAS_ORANI']:.3f}")
    print(f"     (orijinal notebook'ta rapor edilen: WMAPE 17.10, sapma +0.0028, poz 12/12)\n")
    kar = []
    for ad in SECILEN:
        if ad not in OOT_SONUC: continue
        n = OOT_SONUC[ad]
        kar.append({"MERDIVEN": ad,
                    "erkenlik": CONFIG["HEDEF_OLGUNLASMA_AY"]-SECILEN[ad]["bilgi_ay"],
                    "YENI_WMAPE": n["WMAPE"], "ONCEKI_WMAPE": o["WMAPE"],
                    "IYILESME_%": 100*(1 - n["WMAPE"]/o["WMAPE"]),
                    "YENI_sapma": n["SAPMA"], "ONCEKI_sapma": o["SAPMA"],
                    "YENI_poz": f"{n['POZ_SAPMA']}/{n['N']}",
                    "ONCEKI_poz": f"{o['POZ_SAPMA']}/{o['N']}"})
    print(pd.DataFrame(kar).round(4).to_string(index=False))

# ---------------------------------------------------------------- 8.6
print("\n" + "-" * 110)
print("8.6  DONEM BAZINDA DETAY")
print("-" * 110)
det = pd.DataFrame({"DONEM": OOT_SONUC[list(SECILEN)[0]]["_don"]})
det["GERCEK_%"] = [100*x for x in OOT_SONUC[list(SECILEN)[0]]["_ger"]]
for ad in list(SECILEN) + (["REF_Onceki"] if "REF_Onceki" in OOT_SONUC else []):
    if ad not in OOT_SONUC: continue
    det[f"{ad[:9]}_%"] = [100*x for x in OOT_SONUC[ad]["_tah"]]
    det[f"{ad[:9]}_sp"] = det[f"{ad[:9]}_%"] - det["GERCEK_%"]
print(det.round(3).to_string(index=False))

# ---------------------------------------------------------------- 8.7
print("\n" + "-" * 110)
print("8.7  GORSELLESTIRME")
print("-" * 110)
fig, ax = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle("BLOK 8 — OOT Sonuclari", fontsize=14, fontweight="bold")
don = OOT_SONUC[list(SECILEN)[0]]["_don"]
ger = np.array(OOT_SONUC[list(SECILEN)[0]]["_ger"])*100
renkler = plt.cm.viridis(np.linspace(0, .85, len(SECILEN)))

ax[0,0].plot(don, ger, "o-", color="black", lw=2.5, ms=6, label="GERCEK", zorder=5)
for (ad, c) in zip(SECILEN, renkler):
    if ad in OOT_SONUC:
        ax[0,0].plot(don, np.array(OOT_SONUC[ad]["_tah"])*100, "s--", color=c, ms=4,
                     lw=1.3, label=f"{ad} ({CONFIG['HEDEF_OLGUNLASMA_AY']-SECILEN[ad]['bilgi_ay']}ay)")
if "REF_Onceki" in OOT_SONUC:
    ax[0,0].plot(don, np.array(OOT_SONUC["REF_Onceki"]["_tah"])*100, "^:",
                 color="red", ms=5, lw=1.5, label="ONCEKI MODEL")
ax[0,0].set_title("OOT: Gercek vs Tahmin"); ax[0,0].set_ylabel("12M90 (%)")
ax[0,0].legend(fontsize=7); ax[0,0].tick_params(axis="x", rotation=90, labelsize=7)

for (ad, c) in zip(SECILEN, renkler):
    if ad in OOT_SONUC:
        e = (np.array(OOT_SONUC[ad]["_tah"]) - np.array(OOT_SONUC[ad]["_ger"]))*100
        ax[0,1].plot(don, e, "o-", color=c, ms=4, lw=1.2, label=ad)
if "REF_Onceki" in OOT_SONUC:
    e = (np.array(OOT_SONUC["REF_Onceki"]["_tah"]) - np.array(OOT_SONUC["REF_Onceki"]["_ger"]))*100
    ax[0,1].plot(don, e, "^:", color="red", ms=5, lw=1.5, label="ONCEKI")
ax[0,1].axhline(0, color="black", lw=1)
ax[0,1].set_title("Sapmalar (tahmin - gercek)\n0 cizgisinin IKI YANINDA olmali")
ax[0,1].set_ylabel("puan"); ax[0,1].legend(fontsize=7)
ax[0,1].tick_params(axis="x", rotation=90, labelsize=7)

M = OOT[OOT.tip == "merdiven"].sort_values("erkenlik")
ax[0,2].plot(M.erkenlik, M.WMAPE, "o-", color=RENK["ger"], ms=8, lw=2)
for _, r in M.iterrows():
    ax[0,2].annotate(r.MODEL[:9], (r.erkenlik, r.WMAPE),
                     textcoords="offset points", xytext=(0,8), fontsize=7, ha="center")
if "REF_Onceki" in OOT_SONUC:
    ax[0,2].axhline(OOT_SONUC["REF_Onceki"]["WMAPE"], color="red", ls="--",
                    label=f"onceki model ({OOT_SONUC['REF_Onceki']['WMAPE']:.1f})")
ax[0,2].invert_xaxis(); ax[0,2].set_xlabel("Erkenlik (ay)"); ax[0,2].set_ylabel("OOT WMAPE")
ax[0,2].set_title("Erkenlik / Dogruluk egrisi"); ax[0,2].legend(fontsize=8)

A = AYR[~AYR.MODEL.str.startswith("REF_Dummy")].sort_values("YANLILIK_%")
ax[1,0].barh(A.MODEL, A["YANLILIK_%"], color=RENK["tah"], label="yanlilik")
ax[1,0].barh(A.MODEL, A["DAGILIM_%"], left=A["YANLILIK_%"], color=RENK["tr"], label="dagilim")
ax[1,0].axvline(81.7, color="black", ls="--", lw=1.5, label="onceki model (%81.7)")
ax[1,0].set_xlabel("MSE payi (%)"); ax[1,0].set_title("Hata ayristirmasi")
ax[1,0].legend(fontsize=7); ax[1,0].tick_params(axis="y", labelsize=7)

x = np.arange(len(TUT)); w = 0.26
ax[1,1].bar(x-w, TUT.CV_WMAPE,  w, label="CV",  color=RENK["tr"])
ax[1,1].bar(x,   TUT.ARA_WMAPE, w, label="ARA", color=RENK["ara"])
ax[1,1].bar(x+w, TUT.OOT_WMAPE, w, label="OOT", color=RENK["oot"])
ax[1,1].set_xticks(x); ax[1,1].set_xticklabels(TUT.MERDIVEN, rotation=30, fontsize=7, ha="right")
ax[1,1].set_ylabel("WMAPE"); ax[1,1].set_title("CV / ARA / OOT tutarliligi"); ax[1,1].legend(fontsize=8)

for (ad, c) in zip(SECILEN, renkler):
    if ad in OOT_SONUC:
        ax[1,2].scatter(np.array(OOT_SONUC[ad]["_ger"])*100,
                        np.array(OOT_SONUC[ad]["_tah"])*100, s=45, color=c, label=ad)
lim = [ger.min()*0.85, ger.max()*1.15]
ax[1,2].plot(lim, lim, "k--", lw=1.2)
ax[1,2].set_xlim(lim); ax[1,2].set_ylim(lim)
ax[1,2].set_xlabel("Gercek (%)"); ax[1,2].set_ylabel("Tahmin (%)")
ax[1,2].set_title("Tahmin vs Gercek\n(cizginin USTU = sisirme)"); ax[1,2].legend(fontsize=7)
plt.tight_layout(); plt.show()

print("\n" + "=" * 110)
print("BLOK 8 TAMAM — OOT ACILDI VE KAPANDI. Bundan sonra model secimi YAPILMAZ.")
print("=" * 110)
