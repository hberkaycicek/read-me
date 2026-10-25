# ============================================================================
# BLOK 2 — OZELLIK MUHENDISLIGI VE BOLME
#
# BLOK 1 BULGUSUNA GORE REVIZE EDILDI:
#   lag-12 hedef otokorelasyonu 0.116 cikti -> HEDEF lag'i zayif.
#   Yeni strateji: ONCULLERIN gecikmeleri ve yumusatmalari.
#   Gerekce: 20_2(t-1) kohort t icin t+1'de bilinir -> hem TAZE hem ILISKILI.
#
# MERKEZI KAVRAM: "GEREKEN_AY"
#   Her degisken icin "kohort t acildiktan kac ay sonra bilinir" etiketi.
#   Champion/Challenger havuzu bu etikete gore filtrelenir.
#   Bu, sizintinin TEK kaynaginin kapatilmasi demektir.
#
# URETILEN AILELER:
#   A) ONCUL LAG       : X(t-1), X(t-2)        -> seviye hafizasi, cok taze
#   B) ONCUL YUMUSATMA : 3 aylik hareketli ort -> gurultu azaltma
#   C) ONCUL MOMENTUM  : X(t) / ort(X(t-1..3)) -> hiz, seviyeden bagimsiz
#   D) ROLL-RATE       : 60_3/30_3 vb.         -> bozulmanin sertlesme hizi
#   E) HEDEF LAG       : y(t-9..t-12)          -> zayif ama test edilsin
#   F) OLCEK TEMIZLEME : limit/adet, goreceli  -> enflasyon trendi kirilir
#   G) KKB MERKEZLEME + momentum
#   H) TREND / MEVSIM
# ============================================================================

T = CONFIG["HEDEF_TUR"]
def K(g, a): return f"KK_TAKIP_{g}_{a}_{T}"

# GEREKEN_AY kaydi: {kolon_adi: kohort acildiktan kac ay sonra bilinir}
GEREKEN_AY = {}

def kaydet(ad, seri, gereken_ay, aile):
    """Yeni degisken uret + ne zaman bilindigini kaydet."""
    d[ad] = seri
    GEREKEN_AY[ad] = gereken_ay
    AILE.setdefault(aile, []).append(ad)
    return ad

AILE = {}

print("=" * 100)
print("BLOK 2 — OZELLIK MUHENDISLIGI")
print("=" * 100)

# ---------------------------------------------------------------- 2.0 TEMEL
# Ham onculler: X(t) kohort t'nin {ay}. ayinda olculur -> t+{ay}'da bilinir
for kod in ONCU_KODLARI:
    c = oncul_kolon(kod)
    if c in d.columns:
        GEREKEN_AY[c] = oncul_ay(kod)
        AILE.setdefault("0_HAM_ONCUL", []).append(c)

# Ortak alanlar: basvuru aninda bilinir -> 0 ay
for c in ORTAK_ALANLAR:
    if c in d.columns:
        GEREKEN_AY[c] = 0
        AILE.setdefault("0_ORTAK", []).append(c)

# ---------------------------------------------------------------- 2.1 (A) ONCUL LAG
# X(t-L) kohort t icin  (t-L)+ay_X  aninda bilinir  ->  gereken = max(0, ay_X - L)
print("\n" + "-" * 100)
print("2.1 (A)  ONCUL GECIKMELERI — seviye hafizasi")
print("-" * 100)
print("  Mantik: X(t-1) kohort t icin X(t)'den 1 ay ONCE bilinir.")
print("          Hedefin lag'i zayif (r=0.116) ama oncullerin lag'i taze ve iliskili.\n")
for kod in ONCU_KODLARI:
    c = oncul_kolon(kod)
    if c not in d.columns: continue
    for L in [1, 2]:
        ad = f"{c}_L{L}"
        kaydet(ad, d[c].shift(L), max(0, oncul_ay(kod) - L), "A_ONCUL_LAG")
print(f"  Uretildi: {len(AILE.get('A_ONCUL_LAG', []))} degisken")
print(f"  Ornek   : {K(20,2)}_L1  ->  gereken_ay = {GEREKEN_AY.get(K(20,2)+'_L1')}")

# ---------------------------------------------------------------- 2.2 (B) YUMUSATMA
print("\n" + "-" * 100)
print("2.2 (B)  ONCUL YUMUSATMA — 3 aylik hareketli ortalama")
print("-" * 100)
print("  Tek ayin oranı gurultulu (BASVURU_ADEDI 44bin-291bin arasi oynuyor).")
print("  3 kohortluk ortalama seviyeyi daha istikrarli olcer.\n")
for kod in ONCU_KODLARI:
    c = oncul_kolon(kod)
    if c not in d.columns: continue
    # t, t-1, t-2'nin ortalamasi -> en gec bilinen X(t) -> gereken = ay_X
    kaydet(f"{c}_MA3", d[c].rolling(3, min_periods=3).mean(),
           oncul_ay(kod), "B_YUMUSATMA")
    # t-1, t-2, t-3 (gecmise kaydirilmis) -> gereken = max(0, ay_X - 1)
    kaydet(f"{c}_MA3L1", d[c].shift(1).rolling(3, min_periods=3).mean(),
           max(0, oncul_ay(kod) - 1), "B_YUMUSATMA")
print(f"  Uretildi: {len(AILE.get('B_YUMUSATMA', []))} degisken")

# ---------------------------------------------------------------- 2.3 (C) MOMENTUM
print("\n" + "-" * 100)
print("2.3 (C)  ONCUL MOMENTUM — seviyeden bagimsiz hiz")
print("-" * 100)
print("  X(t) / ort(X(t-1..t-3)) : 'bu kohort gecmis 3 kohorta gore nerede?'")
print("  Olcekten bagimsiz, enflasyon/trend tasimaz.\n")
for kod in ONCU_KODLARI:
    c = oncul_kolon(kod)
    if c not in d.columns: continue
    gecmis = d[c].shift(1).rolling(3, min_periods=3).mean()
    kaydet(f"{c}_MOM", d[c] / gecmis.replace(0, np.nan),
           oncul_ay(kod), "C_MOMENTUM")
print(f"  Uretildi: {len(AILE.get('C_MOMENTUM', []))} degisken")

# ---------------------------------------------------------------- 2.4 (D) ROLL-RATE
print("\n" + "-" * 100)
print("2.4 (D)  ROLL-RATE — bozulmanin sertlesme hizi")
print("-" * 100)
print("  60_3/30_3 : '30+ olanlarin ne kadari 60+'a ilerledi?'")
print("  VIF dusurur (20_2 ile 60_3 korelasyonu 0.88 idi) ve IS ANLAMI tasir.\n")
RR = [("RR_30_60_M3", 60, 3, 30, 3), ("RR_30_60_M4", 60, 4, 30, 4),
      ("RR_30_60_M5", 60, 5, 30, 5), ("RR_30_60_M6", 60, 6, 30, 6),
      ("RR_20_30_M2", 30, 2, 20, 2)]
for ad, gu, au, ga, aa in RR:
    cu, ca = K(gu, au), K(ga, aa)
    if cu in d.columns and ca in d.columns:
        kaydet(ad, d[cu] / d[ca].replace(0, np.nan), max(au, aa), "D_ROLLRATE")
# Aylar arasi gecis: 3. aydaki 60+ / 2. aydaki 30+
if K(60,3) in d.columns and K(30,2) in d.columns:
    kaydet("RR_M2_30_M3_60", d[K(60,3)] / d[K(30,2)].replace(0, np.nan), 3, "D_ROLLRATE")
print(f"  Uretildi: {len(AILE.get('D_ROLLRATE', []))} degisken")

# ---------------------------------------------------------------- 2.5 (E) HEDEF LAG
print("\n" + "-" * 100)
print("2.5 (E)  HEDEF GECIKMELERI — zayif ama test edilsin")
print("-" * 100)
print(f"  BLOK 1 bulgusu: lag-12 otokorelasyon = +0.116 (zayif).")
print(f"  Yine de grid karar versin; lag ortalamasi tek lag'den istikrarli olabilir.")
print(f"  Sizinti sarti: y(t-L) kohort t icin (t-L)+12'de bilinir -> gereken = max(0, 12-L)\n")
if CONFIG["LAG_KULLAN"]:
    for L in [9, 10, 11, 12]:
        kaydet(f"HEDEF_L{L}", d[HEDEF].shift(L),
               max(0, CONFIG["HEDEF_OLGUNLASMA_AY"] + CONFIG["RAPOR_GECIKME_AY"] - L),
               "E_HEDEF_LAG")
    # 4 lag'in ortalamasi: en gec bilinen y(t-9) -> gereken = 3
    kaydet("HEDEF_L9_12_ORT",
           pd.concat([d[f"HEDEF_L{L}"] for L in [9,10,11,12]], axis=1).mean(axis=1),
           3, "E_HEDEF_LAG")
    for L in [9, 10, 11, 12]:
        r = d[[f"HEDEF_L{L}", HEDEF]].dropna().corr().iloc[0,1]
        print(f"    HEDEF_L{L:<2d}  gereken_ay={GEREKEN_AY[f'HEDEF_L{L}']}  r(hedef)={r:+.3f}")
    r = d[["HEDEF_L9_12_ORT", HEDEF]].dropna().corr().iloc[0,1]
    print(f"    HEDEF_L9_12_ORT gereken_ay=3  r(hedef)={r:+.3f}   <-- ortalama daha iyi mi?")

# ---------------------------------------------------------------- 2.6 (F) OLCEK
print("\n" + "-" * 100)
print("2.6 (F)  OLCEK TEMIZLEME — enflasyon trendini kir")
print("-" * 100)
print(f"  TOPLAM_ONAYLANAN_LIMIT: 3.41e8 -> 1.05e10 = 31 KAT artis, CV=%70.3")
print(f"  Ham haliyle modele girerse SAF ZAMAN TRENDI proxy'si olur.")
print(f"  Cozum: adet basina + kendi gecmisine goreceli hale getir.\n")
if {"TOPLAM_ONAYLANAN_LIMIT", "BASVURU_ADEDI"} <= set(d.columns):
    ort_lim = d["TOPLAM_ONAYLANAN_LIMIT"] / d["BASVURU_ADEDI"].replace(0, np.nan)
    kaydet("ORT_ONAY_LIMIT", ort_lim, 0, "F_OLCEK")
    # Goreceli: kendi 12 aylik gecmisine oran -> olcek ve enflasyon notr
    kaydet("LIMIT_GORECELI",
           ort_lim / ort_lim.shift(1).rolling(12, min_periods=6).mean().replace(0, np.nan),
           0, "F_OLCEK")
    kaydet("ADET_GORECELI",
           d["BASVURU_ADEDI"] / d["BASVURU_ADEDI"].shift(1)
             .rolling(12, min_periods=6).mean().replace(0, np.nan),
           0, "F_OLCEK")
    kaydet("LOG_ADET", np.log(d["BASVURU_ADEDI"].replace(0, np.nan)), 0, "F_OLCEK")
    for c in ["ORT_ONAY_LIMIT", "LIMIT_GORECELI", "ADET_GORECELI"]:
        print(f"    {c:18s} CV=%{100*d[c].std()/d[c].mean():5.1f}  "
              f"ilk12={d[c].head(12).mean():.4g}  son12={d[c].tail(12).mean():.4g}")
    print(f"\n  >> LIMIT_GORECELI'nin CV'si ham limitten cok daha dusukse, trend kirilmistir.")

# ---------------------------------------------------------------- 2.7 (G) KKB
print("\n" + "-" * 100)
print("2.7 (G)  KKB MERKEZLEME VE MOMENTUM")
print("-" * 100)
if "KKB_SKOR_O0120" in d.columns:
    tr_mask = d[DONEM_KOL].astype(int).between(*CONFIG["TRAIN_ARALIK"]) & d.HEDEF_KULLANILABILIR
    KKB_REF = float(d.loc[tr_mask, "KKB_SKOR_O0120"].mean())   # SADECE train ortalamasi!
    print(f"  Merkezleme referansi (TRAIN ortalamasi): {KKB_REF:.2f}")
    print(f"  !! Tum veri ortalamasi kullanilsaydi OOT bilgisi sizardi. Train'den aliniyor.")
    if CONFIG["KKB_MERKEZLE"]:
        kaydet("KKB_MERKEZ", d["KKB_SKOR_O0120"] - KKB_REF, 0, "G_KKB")
        print(f"\n  Eski formul: 12M90 = 0.0447 + ... - 2.92e-05*KKB")
        print(f"               sabit terim hedefin 2.9 kati, KKB terimi onu kesiyordu.")
        print(f"  Yeni       : sabit terim = 'ortalama KKB'de beklenen 12M90' -> yorumlanabilir.")
    # KKB momentum: portfoy kalitesi iyilesiyor mu kotulesiyor mu
    kaydet("KKB_MOM", d["KKB_SKOR_O0120"] /
           d["KKB_SKOR_O0120"].shift(1).rolling(6, min_periods=3).mean().replace(0, np.nan),
           0, "G_KKB")
    kaydet("KKB_DEGISIM", d["KKB_SKOR_O0120"].diff(), 0, "G_KKB")
if "KALIBRE_PD_AGIRLIKLI" in d.columns:
    kaydet("PD_MOM", d["KALIBRE_PD_AGIRLIKLI"] /
           d["KALIBRE_PD_AGIRLIKLI"].shift(1).rolling(6, min_periods=3).mean().replace(0, np.nan),
           0, "G_KKB")

# ---------------------------------------------------------------- 2.8 (H) TREND/MEVSIM
print("\n" + "-" * 100)
print("2.8 (H)  TREND VE MEVSIM")
print("-" * 100)
d["TREND_IDX"] = (d["AY"].astype("int64") - d["AY"].astype("int64").min()).astype(float)
_ay = d["AY"].dt.month
d["SIN_AY"]  = np.sin(2*np.pi*_ay/12);  d["COS_AY"]  = np.cos(2*np.pi*_ay/12)
d["SIN_AY2"] = np.sin(4*np.pi*_ay/12);  d["COS_AY2"] = np.cos(4*np.pi*_ay/12)
for c in ["TREND_IDX", "SIN_AY", "COS_AY", "SIN_AY2", "COS_AY2"]:
    GEREKEN_AY[c] = 0
print("  TREND_IDX uretildi — DIKKAT: gelecege dogrusal ekstrapole eder.")
print("  Onceki calismada hic secilmedi; sans eseri felaket onlendi.")
print(f"  lag-12 otokorelasyon 0.116 -> 12 AYLIK DONGU YOK. Mevsim beklentisi dusuk.")

# ---------------------------------------------------------------- 2.9 HAVUZLAR
print("\n" + "-" * 100)
print("2.9  HAVUZ OLUSTURMA (GEREKEN_AY filtresiyle)")
print("-" * 100)
TUM_ADAY = [c for c in GEREKEN_AY if c in d.columns and c != HEDEF]
# 90+ yasagi (guvenlik agi)
TUM_ADAY = [c for c in TUM_ADAY if f"_90_" not in c or c.startswith("HEDEF_L")]

CHAMPION_HAVUZ   = sorted([c for c in TUM_ADAY if GEREKEN_AY[c] <= CONFIG["SON_ONCU_AY"]])
CHALLENGER_HAVUZ = sorted([c for c in TUM_ADAY if GEREKEN_AY[c] <= CONFIG["CHALLENGER_MAX_AY"]])

ozet = pd.DataFrame([{"aile": a, "adet": len(v),
                      "champion": sum(GEREKEN_AY[c] <= CONFIG["SON_ONCU_AY"] for c in v),
                      "challenger": sum(GEREKEN_AY[c] <= CONFIG["CHALLENGER_MAX_AY"] for c in v)}
                     for a, v in sorted(AILE.items())])
print(ozet.to_string(index=False))
print(f"\n  CHAMPION_HAVUZ   (<= {CONFIG['SON_ONCU_AY']} ay) : {len(CHAMPION_HAVUZ)} degisken")
print(f"  CHALLENGER_HAVUZ (<= {CONFIG['CHALLENGER_MAX_AY']} ay) : {len(CHALLENGER_HAVUZ)} degisken")

print("\n  GEREKEN_AY dagilimi (challenger havuzu):")
_g = pd.Series({c: GEREKEN_AY[c] for c in CHALLENGER_HAVUZ}).value_counts().sort_index()
for k, v in _g.items():
    print(f"    {k} ay bekleme : {v:3d} degisken")

# ---------------------------------------------------------------- 2.10 NaN
print("\n" + "-" * 100)
print("2.10  NaN DENETIMI (rolling/shift ilk satirlarda NaN uretir)")
print("-" * 100)
nan_say = d[TUM_ADAY].isna().sum()
nanli = nan_say[nan_say > 0].sort_values(ascending=False)
print(f"  NaN ureten degisken: {len(nanli)} / {len(TUM_ADAY)}")
if len(nanli):
    print(f"  En cok NaN: {dict(nanli.head(5))}")
ilk_tam = d[TUM_ADAY].notna().all(axis=1).idxmax()
print(f"  Tum degiskenlerin dolu oldugu ILK satir: index={ilk_tam}  "
      f"donem={d.loc[ilk_tam, DONEM_KOL]}")
print(f"  -> Bu donemden onceki {ilk_tam} kohort, TAM havuzla modellemede kullanilamaz.")
print(f"  -> Ama grid her kombinasyonu AYRI dropna ile isler; kucuk setler daha cok satir kullanir.")

# ---------------------------------------------------------------- 2.11 BOLME
print("\n" + "-" * 100)
print("2.11  TRAIN / ARA / OOT BOLMESI")
print("-" * 100)
dn = d[DONEM_KOL].astype(int)
d["BOLGE"] = np.where(dn.between(*CONFIG["TRAIN_ARALIK"]), "TRAIN",
              np.where(dn.between(*CONFIG["ARA_ARALIK"]),   "ARA",
              np.where(dn.between(*CONFIG["OOT_ARALIK"]),   "OOT", "TAHMIN")))
d.loc[~d.HEDEF_KULLANILABILIR, "BOLGE"] = "TAHMIN"

tr_idx  = d.index[(d.BOLGE == "TRAIN")].to_numpy()
ara_idx = d.index[(d.BOLGE == "ARA")].to_numpy()
oot_idx = d.index[(d.BOLGE == "OOT")].to_numpy()
tah_idx = d.index[(d.BOLGE == "TAHMIN")].to_numpy()

b = d.groupby("BOLGE").agg(n=(DONEM_KOL, "count"), bas=(DONEM_KOL, "min"),
                           son=(DONEM_KOL, "max"), hedef_ort=(HEDEF, "mean"),
                           hedef_min=(HEDEF, "min"), hedef_max=(HEDEF, "max")
                          ).reindex(["TRAIN", "ARA", "OOT", "TAHMIN"])
print(b.round(5).to_string())
print(f"\n  TRAIN : grid + Optuna SADECE burada")
print(f"  ARA   : BLOK 7 hakemligi — ilk {CONFIG['ARA_ADAY_N']} aday burada elenecek")
print(f"  OOT   : KILITLI, BLOK 8'de tek seferde acilacak")

# ---------------------------------------------------------------- 2.12 PSI
print("\n" + "-" * 100)
print("2.12  PSI — dagilim kaymasi (TRAIN referans)")
print("-" * 100)
def psi(bek, goz, bins=6):
    bek, goz = pd.Series(bek).dropna(), pd.Series(goz).dropna()
    if len(bek) < 10 or len(goz) < 3: return np.nan
    kes = np.unique(np.percentile(bek, np.linspace(0, 100, bins+1)))
    if len(kes) < 3: return np.nan
    kes[0], kes[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(bek, kes)[0] / len(bek), 1e-6, None)
    a = np.clip(np.histogram(goz, kes)[0] / len(goz), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

ps = []
for c in [HEDEF] + CHAMPION_HAVUZ:
    ps.append({"degisken": c, "gereken_ay": GEREKEN_AY.get(c, 0),
               "PSI_ARA": psi(d.loc[tr_idx, c], d.loc[ara_idx, c]),
               "PSI_OOT": psi(d.loc[tr_idx, c], d.loc[oot_idx, c])})
ps = pd.DataFrame(ps)
ps["durum"] = np.where(ps.PSI_OOT < 0.10, "OK",
               np.where(ps.PSI_OOT < 0.25, "ORTA", "CIDDI"))
print("  (PSI<0.10 kayma yok | 0.10-0.25 orta | >0.25 CIDDI)\n")
print("  EN COK KAYAN 15:")
print(ps.sort_values("PSI_OOT", ascending=False).head(15).round(4).to_string(index=False))
print(f"\n  Durum dagilimi: {dict(ps.durum.value_counts())}")
print(f"  >> 'CIDDI' isaretli degiskenler OOT'de bozulur. BLOK 5'te ceza uygulanacak.")

CIDDI_KAYAN = ps[ps.durum == "CIDDI"].degisken.tolist()
if CIDDI_KAYAN:
    print(f"  >> CIDDI kayanlar: {CIDDI_KAYAN[:10]}")

# ---------------------------------------------------------------- 2.13 ON ELEME
print("\n" + "-" * 100)
print("2.13  TEK DEGISKENLI ON TARAMA (TRAIN'de)")
print("-" * 100)
from scipy.stats import pearsonr, spearmanr
tr_ok = d.loc[tr_idx]
rows = []
for c in CHAMPION_HAVUZ:
    sub = tr_ok[[c, HEDEF]].dropna()
    if len(sub) < 15: continue
    r, _   = pearsonr(sub[c], sub[HEDEF])
    rho, _ = spearmanr(sub[c], sub[HEDEF])
    rows.append({"degisken": c, "gereken_ay": GEREKEN_AY[c], "n": len(sub),
                 "Pearson": r, "r2": r**2, "Spearman": rho, "S-P": rho - r,
                 "PSI_OOT": ps.set_index("degisken").PSI_OOT.get(c, np.nan)})
tek = pd.DataFrame(rows).sort_values("r2", ascending=False)
print("  EN GUCLU 20 (TRAIN):")
print(tek.head(20).round(4).to_string(index=False))

print("\n  CHALLENGER havuzundan (<=3 ay) EN GUCLU 12:")
print(tek[tek.gereken_ay <= CONFIG["CHALLENGER_MAX_AY"]].head(12).round(4).to_string(index=False))

print("\n  >> |S-P| > 0.10 olanlar: egrilik/aykiri deger var, dikkat")
egri = tek[tek["S-P"].abs() > 0.10]
print(f"     {len(egri)} degisken: {egri.degisken.head(6).tolist()}")

print("=" * 100)
print(f"BLOK 2 TAMAM — {len(TUM_ADAY)} aday degisken uretildi.")
print(f"  CHAMPION {len(CHAMPION_HAVUZ)} | CHALLENGER {len(CHALLENGER_HAVUZ)}")
print("=" * 100)
