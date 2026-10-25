# ============================================================================
# BLOK 1 — VERI YUKLEME VE KALITE DENETIMI
#
# AMAC: Modellemeye baslamadan once veriye guvenebildigimizi KANITLAMAK.
#       63 kohort x 51 kolonu gozle kontrol etmek imkansiz; otomatik tarama sart.
#
# MEVCUT veri_hazirla'ya gore eklenenler:
#   1) KOLON ENVANTERI    -> hangi kolon kullaniliyor, hangisi disarida, neden
#   2) TAKVIM SUREKLILIGI -> eksik ay var mi (kohort serisinde bosluk)
#   3) MONOTONLUK TESTI   -> kumulatif metrikler mantikli mi (12M90>=6M90>=5M90...)
#                            Bu, VERI HATASINI yakalayan en guclu tek kontrol.
#   4) AYKIRI DEGER       -> IQR taramasi, hangi donemde
#   5) OLGUNLASMA IMZASI  -> hedef/oncul orani son donemlerde 1'e yaklasiyor mu
#                            (bolum 9'da KISMI_GERCEK'te gordugumuz duzenin testi)
#   6) KKB ARALIK RAPORU  -> merkezleme ve kaldirac hesabi icin gerekli
#
# CIKTI: 5 tablo + olgunlasma ozeti. Hicbir modelleme yapilmaz.
# ============================================================================
import re
from math import comb

# ---------------------------------------------------------------- 1.1 YUKLEME
# df_raw zaten elinde (63 x 51). Yoksa asagidakilerden birini ac:
# df_raw = pd.read_excel("BDS_L12M90.xlsx")
# df_raw = pd.read_csv("veri.csv")
# df_raw = pd.read_clipboard()

assert "df_raw" in dir(), "df_raw bulunamadi. Once veriyi yukle."

d = df_raw.copy()

# Kolon adlarini normalize et (bosluk/buyuk-kucuk harf tutarsizligi)
d.columns = d.columns.astype(str).str.strip().str.upper()

# DONEM_1 string'e sabitle, kronolojik sirala
DONEM_KOL = "DONEM_1" if "DONEM_1" in d.columns else "DONEM"
d[DONEM_KOL] = d[DONEM_KOL].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
d = d.sort_values(DONEM_KOL).reset_index(drop=True)
d["AY"] = pd.PeriodIndex(pd.to_datetime(d[DONEM_KOL], format="%Y%m"), freq="M")

print("=" * 100)
print("BLOK 1 — VERI KALITE DENETIMI")
print("=" * 100)
print(f"Boyut: {d.shape[0]} kohort x {d.shape[1]} kolon")
print(f"Aralik: {d[DONEM_KOL].iloc[0]}  ->  {d[DONEM_KOL].iloc[-1]}")
print(f"Hedef : {HEDEF}")
assert HEDEF in d.columns, f"HEDEF kolonu yok: {HEDEF}"

# ---------------------------------------------------------------- 1.2 ENVANTER
# Her kolonu siniflandir: hangisi modele giriyor, hangisi disarida, NEDEN?
def kolon_sinifla(c):
    if c in (DONEM_KOL, "DONEM", "AY"):                return "KIMLIK"
    if c == HEDEF:                                      return "HEDEF"
    if c in ORTAK_ALANLAR:                              return "ORTAK (kullanilir)"
    m = re.match(r"^KK_TAKIP_(\d+)_(\d+)_(TUTARSAL|ADETSEL)$", c)
    if m:
        gec, ay, tur = int(m.group(1)), int(m.group(2)), m.group(3)
        if tur != CONFIG["HEDEF_TUR"]:                  return f"DIGER TUR ({tur})"
        if gec == 90:                                   return "90+ YASAK (hedefle ayni olay)"
        if f"{gec}_{ay}" in ONCU_KODLARI:
            return ("ONCU <=3ay (challenger+champion)" if ay <= CONFIG["CHALLENGER_MAX_AY"]
                    else "ONCU 4-6ay (sadece champion)")
        return "ONCU (havuz disi)"
    if re.match(r"^KK_TAKIP_\d+$", c):                  return "?? BELIRSIZ (incelenecek)"
    return "KULLANILMIYOR"

env = pd.DataFrame({"kolon": d.columns})
env["sinif"] = env.kolon.map(kolon_sinifla)
print("\n" + "-" * 100)
print("1.2  KOLON ENVANTERI")
print("-" * 100)
print(env.groupby("sinif").agg(adet=("kolon", "count"),
                               ornekler=("kolon", lambda s: ", ".join(list(s)[:4]) +
                                         ("..." if len(s) > 4 else ""))
                               ).sort_values("adet", ascending=False).to_string())

belirsiz = env[env.sinif.str.startswith("??")].kolon.tolist()
if belirsiz:
    print(f"\n  !! INCELENECEK ({len(belirsiz)} kolon): {belirsiz}")
    print("     Bunlar gecikme gunu belirtmiyor. Ne olctuklerini teyit et;")
    print("     kullanilabilir bilgi tasiyorlarsa havuza eklenebilir.")
    print("\n     Hedefle korelasyonlari (olgun satirlarda):")
    _ol = d[d[HEDEF].notna()]
    for c in belirsiz:
        if pd.api.types.is_numeric_dtype(d[c]):
            sub = _ol[[c, HEDEF]].dropna()
            if len(sub) > 10:
                print(f"       {c:22s} r={sub[c].corr(sub[HEDEF]):+.3f}  "
                      f"n={len(sub):3d}  aralik=[{sub[c].min():.4g}, {sub[c].max():.4g}]")

# ---------------------------------------------------------------- 1.3 TAKVIM
print("\n" + "-" * 100)
print("1.3  TAKVIM SUREKLILIGI")
print("-" * 100)
bekl = pd.period_range(d.AY.min(), d.AY.max(), freq="M")
eksik = sorted(set(bekl) - set(d.AY))
print(f"  Beklenen ay: {len(bekl)}  |  Mevcut: {len(d)}  |  EKSIK: {len(eksik)}")
if eksik:
    print(f"  !! Eksik donemler: {[str(x) for x in eksik]}")
    print("     Kesintili seri, 'zaman' CV'de ValueError uretir (fold_olustur kontrolu).")
else:
    print("  OK — seri kesintisiz, walk-forward CV guvenli.")

mukerrer = d[DONEM_KOL].duplicated().sum()
print(f"  Mukerrer donem: {mukerrer}  {'!! TEMIZLE' if mukerrer else 'OK'}")

# ---------------------------------------------------------------- 1.4 TEMEL KALITE
ONCU_KOLONLAR = [oncul_kolon(k) for k in ONCU_KODLARI if oncul_kolon(k) in d.columns]
IZLENEN = [HEDEF] + ONCU_KOLONLAR + [c for c in ORTAK_ALANLAR if c in d.columns]

print("\n" + "-" * 100)
print("1.4  TEMEL KALITE (NaN / sifir / negatif / aralik)")
print("-" * 100)
kal = pd.DataFrame({
    "NaN":      d[IZLENEN].isna().sum(),
    "sifir":    (d[IZLENEN] == 0).sum(),
    "negatif":  (d[IZLENEN] < 0).sum(),
    "min":      d[IZLENEN].min(),
    "ort":      d[IZLENEN].mean(),
    "max":      d[IZLENEN].max(),
    "CV_%":     100 * d[IZLENEN].std() / d[IZLENEN].mean(),
})
print(kal.round(4).to_string())

# KKB ozel raporu — merkezleme ve kaldirac icin
if "KKB_SKOR_O0120" in d.columns:
    k = d["KKB_SKOR_O0120"]
    print(f"\n  KKB_SKOR_O0120 detay:")
    print(f"    aralik = [{k.min():.1f}, {k.max():.1f}]   ortalama = {k.mean():.1f}   std = {k.std():.1f}")
    print(f"    p5/p50/p95 = {k.quantile(.05):.1f} / {k.quantile(.5):.1f} / {k.quantile(.95):.1f}")
    print(f"    ilk 12 ay ort = {k.head(12).mean():.1f}   son 12 ay ort = {k.tail(12).mean():.1f}"
          f"   -> KAYMA = {k.tail(12).mean()-k.head(12).mean():+.1f} puan")
    print(f"    >> Onceki modelde katsayi -2.92e-05 idi. Bu kayma tahmini "
          f"{abs(-2.92e-05*(k.tail(12).mean()-k.head(12).mean()))*100:.3f} pp oynatir.")

# ---------------------------------------------------------------- 1.5 MONOTONLUK
# En guclu veri-hatasi yakalayicisi: kumulatif metrikler mantikli siralanmali.
print("\n" + "-" * 100)
print("1.5  MANTIK / MONOTONLUK KONTROLU")
print("-" * 100)
T = CONFIG["HEDEF_TUR"]
def K(g, a): return f"KK_TAKIP_{g}_{a}_{T}"

kurallar = [
    # (aciklama, buyuk_kolon, kucuk_kolon)
    ("12M90 >= 6M90  (sure arttikca birikim artar)", HEDEF,     K(90,6)),
    ("6M90  >= 5M90",                                K(90,6),   K(90,5)),
    ("5M90  >= 4M90",                                K(90,5),   K(90,4)),
    ("6M60  >= 5M60",                                K(60,6),   K(60,5)),
    ("5M60  >= 4M60",                                K(60,5),   K(60,4)),
    ("4M60  >= 3M60",                                K(60,4),   K(60,3)),
    ("6M30  >= 5M30",                                K(30,6),   K(30,5)),
    ("4M30  >= 3M30",                                K(30,4),   K(30,3)),
    ("3M30  >= 2M30",                                K(30,3),   K(30,2)),
    ("4M30 >= 4M60  (dusuk esik daha cok hesap)",    K(30,4),   K(60,4)),
    ("4M60 >= 4M90",                                 K(60,4),   K(90,4)),
    ("5M60 >= 5M90",                                 K(60,5),   K(90,5)),
    ("2M20 >= 2M30  (20 gun esigi 30'dan genis)",    K(20,2),   K(30,2)),
]
ihlal_toplam = 0
for ad, a, b in kurallar:
    if a in d.columns and b in d.columns:
        sub = d[[DONEM_KOL, a, b]].dropna()
        ihl = sub[sub[a] < sub[b] - 1e-9]
        ihlal_toplam += len(ihl)
        durum = "OK" if len(ihl) == 0 else f"!! {len(ihl)} IHLAL"
        print(f"  {ad:46s} {durum}")
        if len(ihl):
            print(f"      donemler: {list(ihl[DONEM_KOL].values)[:8]}")
print(f"\n  TOPLAM IHLAL: {ihlal_toplam}")
if ihlal_toplam:
    print("  !! Ihlal varsa ya veri hatasi ya da olgunlasmamis satir. Once bunu coz.")
else:
    print("  OK — kumulatif mantik tutarli, veri guvenilir.")

# ---------------------------------------------------------------- 1.6 AYKIRI
print("\n" + "-" * 100)
print("1.6  AYKIRI DEGER TARAMASI (IQR)")
print("-" * 100)
bulundu = False
for c in IZLENEN:
    q1, q3 = d[c].quantile([.25, .75]); iqr = q3 - q1
    ay_ = d[(d[c] < q1 - 1.5*iqr) | (d[c] > q3 + 1.5*iqr)]
    if len(ay_):
        bulundu = True
        print(f"  {c:34s} {len(ay_):2d} aykiri -> " +
              ", ".join(f"{r[DONEM_KOL]}({r[c]:.4g})" for _, r in ay_.head(5).iterrows()))
if not bulundu:
    print("  Aykiri deger yok.")
print("\n  NOT: Aykirilar COVID/yapilandirma donemine denk geliyorsa veri hatasi degil,")
print("       REJIM etkisidir. BLOK 3'te Chow testiyle ayristiracagiz.")

# ---------------------------------------------------------------- 1.7 OLGUNLASMA
print("\n" + "-" * 100)
print("1.7  OLGUNLASMA HESABI")
print("-" * 100)
bilgi = pd.Period(CONFIG["BILGI_AYI"], freq="M")
d["HEDEF_ERISIM"] = d["AY"] + CONFIG["HEDEF_OLGUNLASMA_AY"] + CONFIG["RAPOR_GECIKME_AY"]
d["OLGUN"] = d["HEDEF_ERISIM"] < bilgi
d["OLGUNLUK_AY"] = np.clip(
    CONFIG["HEDEF_OLGUNLASMA_AY"] - (d["HEDEF_ERISIM"] - bilgi).apply(lambda x: x.n),
    0, CONFIG["HEDEF_OLGUNLASMA_AY"])

hedef_gecerli = (d[HEDEF].notna() & np.isfinite(d[HEDEF].astype(float))
                 & d[HEDEF].between(0, 1))
d["HEDEF_KULLANILABILIR"] = d["OLGUN"] & hedef_gecerli

print(f"  Bilgi ayi                  : {bilgi}")
print(f"  Toplam kohort              : {len(d)}")
print(f"  Olgun (HEDEF_ERISIM<bilgi) : {int(d.OLGUN.sum())}")
print(f"  Hedef gecerli (0-1, notna) : {int(hedef_gecerli.sum())}")
print(f"  KULLANILABILIR             : {int(d.HEDEF_KULLANILABILIR.sum())}")
print(f"  Olgunlasmamis (tahmin edilecek): {int((~d.HEDEF_KULLANILABILIR).sum())}"
      f"  -> {list(d.loc[~d.HEDEF_KULLANILABILIR, DONEM_KOL].values)}")

gecersiz = d[d.OLGUN & ~hedef_gecerli]
if len(gecersiz):
    print(f"\n  !! Olgun ama hedefi GECERSIZ {len(gecersiz)} satir: "
          f"{list(gecersiz[DONEM_KOL].values)}  -> incele")

# ---------------------------------------------------------------- 1.8 OLGUNLASMA IMZASI
# Bolum 9'da KISMI_GERCEK'in 1.68 -> 0.71'e dusmesi tam da bu duzendi.
# Burada ONCEDEN dogruluyoruz: olgunlasmamis satirlarda oran bozulmali.
print("\n" + "-" * 100)
print("1.8  OLGUNLASMA IMZASI  (hedef / 6M90 orani)")
print("-" * 100)
ref = K(90, 6)
if ref in d.columns:
    d["_oran"] = d[HEDEF] / d[ref].replace(0, np.nan)
    olg = d[d.HEDEF_KULLANILABILIR]
    print(f"  OLGUN kohortlarda oran : ort={olg._oran.mean():.3f}  std={olg._oran.std():.3f}  "
          f"CV={100*olg._oran.std()/olg._oran.mean():.1f}%")
    print(f"                           min={olg._oran.min():.3f}  max={olg._oran.max():.3f}")
    print("\n  SON 10 KOHORT (olgunlasma bozulmasi burada gorunur):")
    son = d.tail(10)[[DONEM_KOL, "OLGUNLUK_AY", HEDEF, ref, "_oran", "HEDEF_KULLANILABILIR"]]
    print(son.rename(columns={HEDEF: "hedef", ref: "6M90", "_oran": "oran",
                              "HEDEF_KULLANILABILIR": "kullanilir"}
                     ).round(4).to_string(index=False))
    print(f"\n  >> Oran OLGUN ortalamasindan ({olg._oran.mean():.2f}) 1'e dogru dusuyorsa,")
    print(f"     o satir olgunlasmamistir. Filtre dogru calisiyorsa hepsi kullanilir=False olmali.")
    # Capraz dogrulama
    _yanlis = d[(~d.HEDEF_KULLANILABILIR) & (d._oran > olg._oran.mean() * 0.9)]
    _kacak  = d[(d.HEDEF_KULLANILABILIR)  & (d._oran < olg._oran.mean() * 0.6)]
    print(f"  Capraz kontrol: olgunlasmamis ama orani normal = {len(_yanlis)}"
          f"  |  olgun ama orani dusuk = {len(_kacak)}")
    if len(_kacak):
        print(f"     !! SUPHELI (olgun sayildi ama orani dusuk): "
              f"{list(_kacak[DONEM_KOL].values)}")
    d.drop(columns=["_oran"], inplace=True)

# ---------------------------------------------------------------- 1.9 HEDEF SERISI
print("\n" + "-" * 100)
print("1.9  HEDEF SERISININ ZAMAN YAPISI")
print("-" * 100)
y = d.loc[d.HEDEF_KULLANILABILIR, HEDEF]
print(f"  n={len(y)}  ort={y.mean():.5f}  std={y.std():.5f}  CV={100*y.std()/y.mean():.1f}%")
print(f"  min={y.min():.5f} ({d.loc[y.idxmin(), DONEM_KOL]})  "
      f"max={y.max():.5f} ({d.loc[y.idxmax(), DONEM_KOL]})")
for lag in [1, 2, 3, 6, 12]:
    print(f"  lag-{lag:<2d} otokorelasyon = {y.autocorr(lag):+.3f}")
rho = y.autocorr(1)
print(f"\n  >> lag-1 rho={rho:.3f} -> ETKIN orneklem ~ {len(y)*(1-rho)/(1+rho):.1f} (nominal {len(y)})")
print(f"  >> lag-12 otokorelasyon yuksekse, BLOK 2'deki HEDEF_LAG12 guclu bir girdi olacak.")

fig, ax = plt.subplots(2, 1, figsize=(15, 7), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1]})
m = d.HEDEF_KULLANILABILIR
ax[0].plot(d.loc[m, "AY"].dt.to_timestamp(), d.loc[m, HEDEF]*100, "o-",
           color=RENK["ger"], lw=1.8, ms=4, label="Gerceklesen (olgun)")
ax[0].plot(d.loc[~m, "AY"].dt.to_timestamp(), d.loc[~m, HEDEF]*100, "s--",
           color=RENK["gri"], lw=1.2, ms=5, label="Kismi (olgunlasmamis)")
ax[0].set_ylabel(f"{HEDEF} (%)"); ax[0].legend(fontsize=8)
ax[0].set_title("Hedef serisi — olgun vs olgunlasmamis", fontweight="bold")
ax[1].bar(d["AY"].dt.to_timestamp(), d["OLGUNLUK_AY"], width=20,
          color=np.where(m, RENK["oot"], RENK["ara"]))
ax[1].axhline(CONFIG["HEDEF_OLGUNLASMA_AY"], color=RENK["tah"], ls="--", lw=1)
ax[1].set_ylabel("Olgunluk (ay)"); ax[1].set_xlabel("Kohort ayi")
plt.tight_layout(); plt.show()

print("=" * 100)
print("BLOK 1 TAMAM — veriye guvenilebilirse BLOK 2'ye gec.")
print("=" * 100)
