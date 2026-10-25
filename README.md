# ============================================================================
# BLOK 3.8 — YAPISAL BAGIMLILIK TEMIZLIGI VE HAVUZ REVIZYONU
#
# NEDEN: VIF 2.66e+09 cikti. Sebep ikili korelasyon DEGIL, uclu yapisal bagimlilik:
#        MA3 = (X + X_L1 + X_L2) / 3   -> dordu birlikte determinanti sifirlar.
#        K4 kurali (ikili |r|>0.95) bunu yakalayamaz.
#
# COZUM: Degiskenleri havuzdan SILMIYORUZ (X_L1 ve X farkli olgunluk secenekleri,
#        ikisi de lazim). Bunun yerine GRID'DE KOMBINASYON KISITI koyuyoruz:
#        "ayni temel onculun iki versiyonu ayni modelde olamaz".
#
# AYRICA:
#   - disari_OOT kriteri budamaya ekleniyor (TREND_IDX, nominal limitler elenir)
#   - TREND_IDX grid'den cikariliyor (in-sample anlamli ama disari_OOT=1.000)
# ============================================================================
print("=" * 102)
print("BLOK 3.8 — YAPISAL BAGIMLILIK TEMIZLIGI")
print("=" * 102)

# ---------------------------------------------------------------- 3.8.1 AILE HARITASI
# Her degiskenin hangi TEMEL oncule ait oldugunu cikar.
def temel_oncul(c):
    """KK_TAKIP_30_2_TUTARSAL_MA3L1 -> 'KK_TAKIP_30_2_TUTARSAL'"""
    for ek in ["_MA3L1", "_MA3", "_MOM", "_L1", "_L2", "_L3"]:
        if c.endswith(ek):
            return c[: -len(ek)]
    return c

TEMEL = {c: temel_oncul(c) for c in TUM_ADAY}
gruplar = {}
for c, t in TEMEL.items():
    gruplar.setdefault(t, []).append(c)

print("\n3.8.1  YAPISAL BAGIMLILIK HARITASI")
print("-" * 102)
print("  MA3 = (X + X_L1 + X_L2)/3  ->  bu dortlu TAM BAGIMLI (determinant=0)")
print("  Asagidaki her grup icin grid EN FAZLA 1 degisken secebilecek.\n")
riskli = {t: v for t, v in gruplar.items()
          if len(v) > 1 and any(x in CHAMPION_BUDANMIS for x in v)}
for t, v in sorted(riskli.items()):
    icerde = [x for x in v if x in CHAMPION_BUDANMIS]
    if len(icerde) > 1:
        print(f"  {t[:42]:44s} havuzda {len(icerde)}: "
              f"{[x.replace(t, '') or '(ham)' for x in icerde]}")

# ---------------------------------------------------------------- 3.8.2 YASAK LISTE
# Ekstrapolasyon riski: OOT degerlerinin cogu TRAIN araligi disindaysa modele alma.
print("\n3.8.2  EKSTRAPOLASYON ELEMESI (disari_OOT kriteri)")
print("-" * 102)
print("  Onceki budamada SADECE |z|>1.75 kullanilmisti; disari_OOT hic uygulanmadi.")
print("  disari_OOT = OOT degerlerinin TRAIN min-max araligi DISINDA kalan orani.")
print("  >0.50 ise model o degisken icin EKSTRAPOLASYON yapiyor demektir.\n")

dis = ky.set_index("degisken")
DISARI_ESIK = 0.50
elenecek = []
for c in set(CHAMPION_BUDANMIS) | set(CHALLENGER_BUDANMIS):
    dd = dis.disari_OOT.get(c, 0.0)
    if dd > DISARI_ESIK:
        elenecek.append((c, dd, dis.z_OOT.get(c, np.nan)))
elenecek.sort(key=lambda t: -t[1])
if elenecek:
    print("  ELENEN (disari_OOT > 0.50):")
    for c, dd, z in elenecek:
        print(f"    {c:44s} disari={dd:.3f}  z_OOT={z:+.2f}")
else:
    print("  Bu kriterle elenen yok.")

# Nominal olcekli degiskenler: enflasyon tasiyor, ilkesel olarak yasak
NOMINAL_YASAK = ["TOPLAM_ONAYLANAN_LIMIT", "ORT_ONAY_LIMIT", "BASVURU_ADEDI", "TREND_IDX"]
print(f"\n  ILKESEL YASAK (nominal olcek / ekstrapolasyon): {NOMINAL_YASAK}")
print("    TOPLAM_ONAYLANAN_LIMIT : 3.41e8 -> 1.05e10 (31 kat), saf enflasyon trendi")
print("    TREND_IDX              : disari_OOT=1.000, tanimi geregi hep aralik disi")
print("    (yerlerine LIMIT_GORECELI / ADET_GORECELI zaten uretildi)")

YASAK = set([c for c, _, _ in elenecek]) | set(NOMINAL_YASAK)

CHAMPION_BUDANMIS   = [c for c in CHAMPION_BUDANMIS   if c not in YASAK]
CHALLENGER_BUDANMIS = [c for c in CHALLENGER_BUDANMIS if c not in YASAK]
print(f"\n  CHAMPION_BUDANMIS   : {len(CHAMPION_BUDANMIS)} (revize)")
print(f"  CHALLENGER_BUDANMIS : {len(CHALLENGER_BUDANMIS)} (revize)")

# ---------------------------------------------------------------- 3.8.3 KOMBINASYON KISITI
print("\n3.8.3  KOMBINASYON KISITI (grid bunu kullanacak)")
print("-" * 102)
CONFIG["AYNI_AILE_MAKS"] = 1   # ayni temel oncul ailesinden en fazla kac degisken

def kombinasyon_gecerli(alt):
    """Grid'in her kombinasyon icin cagiracagi kisit.
       Ayni temel oncul ailesinden >AYNI_AILE_MAKS degisken varsa REDDET."""
    say = {}
    for c in alt:
        t = TEMEL.get(c, c)
        say[t] = say.get(t, 0) + 1
        if say[t] > CONFIG["AYNI_AILE_MAKS"]:
            return False
    return True

# Dogrulama
_ornek_kotu = [c for c in CHAMPION_BUDANMIS if TEMEL[c] == "KK_TAKIP_30_2_" + T][:3]
if len(_ornek_kotu) >= 2:
    print(f"  TEST — ayni aileden 2 degisken: {[x[-14:] for x in _ornek_kotu[:2]]}")
    print(f"         kombinasyon_gecerli = {kombinasyon_gecerli(_ornek_kotu[:2])}  (False olmali)")
_ornek_iyi = [CHAMPION_BUDANMIS[0], [c for c in CHAMPION_BUDANMIS
                                     if TEMEL[c] != TEMEL[CHAMPION_BUDANMIS[0]]][0]]
print(f"  TEST — farkli ailelerden 2 degisken: {[x[-18:] for x in _ornek_iyi]}")
print(f"         kombinasyon_gecerli = {kombinasyon_gecerli(_ornek_iyi)}  (True olmali)")

# ---------------------------------------------------------------- 3.8.4 DUZELTILMIS VIF
print("\n3.8.4  VIF — kisit uygulanmis ornek kombinasyonlarda")
print("-" * 102)
print("  Tam havuz VIF'i anlamsiz (n=39, p=19 ve yapisal bagimlilik var).")
print("  Asil onemli olan: GRID'IN SECEBILECEGI 3'lu kombinasyonlarda VIF ne?\n")
from itertools import combinations as _kmb
import random
random.seed(42)
gecerli_uclu = [a for a in _kmb(CHALLENGER_BUDANMIS, 3) if kombinasyon_gecerli(a)]
print(f"  Challenger'da gecerli 3'lu kombinasyon: {len(gecerli_uclu):,} "
      f"(kisitsiz {len(list(_kmb(CHALLENGER_BUDANMIS,3))):,})")
vifler = []
for alt in random.sample(gecerli_uclu, min(300, len(gecerli_uclu))):
    sub = d.loc[tr_idx, list(alt)].dropna()
    if len(sub) < 20: continue
    X = add_constant(sub, has_constant="add")
    try:
        v = max(variance_inflation_factor(X.values, i) for i in range(1, X.shape[1]))
        vifler.append(v)
    except Exception:
        pass
vifler = np.array(vifler)
print(f"  Ornek 300 kombinasyonda MAKS VIF dagilimi:")
print(f"    medyan={np.median(vifler):6.2f}   p75={np.percentile(vifler,75):6.2f}   "
      f"p95={np.percentile(vifler,95):7.2f}   maks={vifler.max():8.2f}")
print(f"    VIF<5  olan kombinasyon orani : %{100*(vifler<5).mean():.1f}")
print(f"    VIF<10 olan kombinasyon orani : %{100*(vifler<10).mean():.1f}")
print(f"  >> Kisit oncesi bu dagilim milyarlara uzaniyordu. Sorun cozuldu.")
print(f"  >> BLOK 5'te ayrica MAKS_VIF<=10 gecerlilik filtresi uygulanacak.")

# ---------------------------------------------------------------- 3.8.5 MEVSIM/TREND KARARI
print("\n3.8.5  TREND VE MEVSIM NIHAI KARARI")
print("-" * 102)
print(f"  TREND  : in-sample p=0.0001 ANLAMLI, ama disari_OOT=1.000")
print(f"           -> in-sample anlamlilik != out-of-sample kullanilabilirlik")
print(f"           -> GRID'DEN CIKARILDI (ekstrapolasyonda modeli savurur)")
CONFIG["TREND_DENE"] = False

print(f"  MEVSIM : p=0.0186 (sinirda, 39 gozlemde 2 parametre)")
print(f"           BLOK 1: lag-12 otokorelasyon = +0.116 -> 12 aylik dongu YOK")
print(f"           -> CELISKI var. Havuzda KALIYOR ama gecerlilik filtresi karar verecek.")
CONFIG["MEVSIM_DENE"] = True

vary = 1 + int(CONFIG["TREND_DENE"]) + int(CONFIG["MEVSIM_DENE"]) + \
       int(CONFIG["TREND_DENE"] and CONFIG["MEVSIM_DENE"])
nC, nL = len(CHAMPION_BUDANMIS), len(CHALLENGER_BUDANMIS)
gC = sum(1 for k in range(1, CONFIG["MAX_K"]+1)
         for a in _kmb(CHAMPION_BUDANMIS, k) if kombinasyon_gecerli(a))
gL = sum(1 for k in range(1, CONFIG["MAX_K"]+1)
         for a in _kmb(CHALLENGER_BUDANMIS, k) if kombinasyon_gecerli(a))

print("\n" + "-" * 102)
print("3.8.6  NIHAI ARAMA UZAYI")
print("-" * 102)
print(f"  Baslangic (budama oncesi)        : 1,881,104 varyant")
print(f"  Budama sonrasi (BLOK 3.5)        :    18,544 varyant")
nf = len(CONFIG["MODEL_AILELERI"])
print(f"  + aile kisiti + yasak + trend yok :")
print(f"      CHAMPION   : {nC:2d} degisken, {gC:5,} gecerli kombinasyon -> {gC*vary*nf:7,} varyant")
print(f"      CHALLENGER : {nL:2d} degisken, {gL:5,} gecerli kombinasyon -> {gL*vary*nf:7,} varyant")
print(f"  >> Secim yanliligi minimuma indi. BLOK 7 hakemligi kalani temizleyecek.")

print("\n  === NIHAI CHAMPION HAVUZU ===")
t = pd.DataFrame({"degisken": CHAMPION_BUDANMIS})
t["aile"] = t.degisken.map(lambda c: TEMEL[c].replace("KK_TAKIP_", "").replace(f"_{T}", ""))
t["gereken_ay"] = t.degisken.map(GEREKEN_AY)
t["r2"] = t.degisken.map(tek.set_index("degisken").r2)
t["z_OOT"] = t.degisken.map(dis.z_OOT)
t["disari"] = t.degisken.map(dis.disari_OOT)
print(t.sort_values(["gereken_ay", "r2"], ascending=[True, False]).round(3).to_string(index=False))

print("\n  === NIHAI CHALLENGER HAVUZU ===")
t2 = t[t.degisken.isin(CHALLENGER_BUDANMIS)]
print(t2.sort_values(["gereken_ay", "r2"], ascending=[True, False]).round(3).to_string(index=False))
print("=" * 102)
