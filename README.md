# ============================================================================
# BLOK 3 — KESIF, REJIM TESTLERI VE HAVUZ BUDAMA
#
# BU BLOGUN ASIL TESLIMATI: HAVUZ BUDAMA
#   CHAMPION 89 degisken -> C(89,3)x4x4 = 1.881.104 varyant
#   Hem calismaz hem FELAKET duzeyde secim yanliligi uretir.
#   Onceki calismada 9.260 varyanttan secim CV->OOT'de %61 bozulma yaratmisti.
#   Hedef: havuzu ~25'e indirip varyanti ~37bin'e dusurmek.
#
# AYRICA:
#   3.1  Kayma analizi — PSI yerine KUCUK ORNEKLEME UYGUN metrikler
#   3.2  Rejim / yapisal kirilma (Chow + kayan pencere)
#   3.3  Trend ve mevsim: EK KATKI testi (onculden sonra hala anlamli mi?)
#   3.4  KKB confounding: iliskisi gercek mi, ortak trend mi?
#   3.5  Korelasyon kumeleme + BUDAMA
# ============================================================================
from scipy import stats
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("=" * 102)
print("BLOK 3 — KESIF, REJIM TESTLERI, HAVUZ BUDAMA")
print("=" * 102)

# ---------------------------------------------------------------- 3.1 KAYMA
# PSI'yi BIRAKTIK. n_ara=6, n_oot=12 -> bin basina <2 gozlem -> bos bin -> sahte PSI~11.
# Kucuk orneklemde calisan 3 metrik:
#   z_kayma : (ort_hedef - ort_train) / std_train     -> olcekten bagimsiz seviye kaymasi
#   disari  : hedefin TRAIN min-max araligi disinda kalan orani -> EKSTRAPOLASYON riski
#   KS p    : dagilim esitligi testi (kucuk n'de de gecerli)
print("\n" + "-" * 102)
print("3.1  DAGILIM KAYMASI (PSI yerine kucuk-orneklem metrikleri)")
print("-" * 102)
print("  NOT: Onceki PSI hesabi n=6/12'de BOS BIN urettigi icin sahte ~11 degerler")
print("       veriyordu. PSI bin basina ~10 gozlem ister; burada uygulanamaz.\n")

def kayma(tr_seri, hedef_seri):
    tr = pd.Series(tr_seri).dropna(); hd = pd.Series(hedef_seri).dropna()
    if len(tr) < 10 or len(hd) < 3: return dict(z=np.nan, disari=np.nan, ks_p=np.nan)
    sd = tr.std()
    return dict(
        z      = (hd.mean() - tr.mean()) / sd if sd > 0 else np.nan,
        disari = float(((hd < tr.min()) | (hd > tr.max())).mean()),
        ks_p   = float(stats.ks_2samp(tr, hd).pvalue))

ky = []
for c in [HEDEF] + CHAMPION_HAVUZ:
    a = kayma(d.loc[tr_idx, c], d.loc[ara_idx, c])
    o = kayma(d.loc[tr_idx, c], d.loc[oot_idx, c])
    ky.append({"degisken": c, "gereken_ay": GEREKEN_AY.get(c, 0),
               "z_ARA": a["z"], "z_OOT": o["z"],
               "disari_OOT": o["disari"], "KS_p_OOT": o["ks_p"]})
ky = pd.DataFrame(ky)
ky["durum"] = np.where(ky.z_OOT.abs() < 0.75, "OK",
               np.where(ky.z_OOT.abs() < 1.75, "ORTA", "CIDDI"))
print("  (|z|<0.75 OK | 0.75-1.75 ORTA | >1.75 CIDDI)")
print("  disari_OOT: OOT degerlerinin TRAIN araligi disinda kalan orani -> ekstrapolasyon\n")
print("  EN COK KAYAN 15:")
print(ky.reindex(ky.z_OOT.abs().sort_values(ascending=False).index)
        .head(15).round(3).to_string(index=False))
print(f"\n  Durum dagilimi: {dict(ky.durum.value_counts())}")
_h = ky[ky.degisken == HEDEF].iloc[0]
print(f"  HEDEFin kendi kaymasi: z_OOT={_h.z_OOT:+.3f}  disari={_h.disari_OOT:.2f}")
print(f"  >> Hedef z pozitifse OOT donemi TRAIN'den YUKSEK seviyede demektir.")

KAYAN = set(ky[ky.durum == "CIDDI"].degisken) - {HEDEF}
print(f"  >> CIDDI kayan {len(KAYAN)} degisken budamada cezalandirilacak.")

# ---------------------------------------------------------------- 3.2 REJIM
print("\n" + "-" * 102)
print("3.2  REJIM / YAPISAL KIRILMA")
print("-" * 102)
olg = d[d.HEDEF_KULLANILABILIR].copy().reset_index(drop=True)
REF = K(60, 3) if K(60, 3) in d.columns else K(30, 2)
olg["_k"] = olg[HEDEF] / olg[REF].replace(0, np.nan)

print(f"  Referans oncul: {REF}")
print(f"  Carpan k = hedef/{REF}: ort={olg._k.mean():.3f} std={olg._k.std():.3f} "
      f"CV=%{100*olg._k.std()/olg._k.mean():.1f}")

def chow(df, x, y, i):
    n, kk = len(df), 2
    X = add_constant(df[[x]].values); Y = df[y].values
    if i <= kk + 2 or n - i <= kk + 2: return np.nan, np.nan
    rp = OLS(Y, X).fit().ssr
    r1 = OLS(Y[:i], X[:i]).fit().ssr; r2 = OLS(Y[i:], X[i:]).fit().ssr
    F = ((rp - (r1 + r2)) / kk) / ((r1 + r2) / (n - 2*kk))
    return F, 1 - stats.f.cdf(F, kk, n - 2*kk)

ch = []
for i in range(8, len(olg) - 8):
    F, p = chow(olg, REF, HEDEF, i)
    ch.append({"idx": i, "DONEM": olg[DONEM_KOL].iloc[i], "F": F, "p": p})
ch = pd.DataFrame(ch)
print(f"\n  CHOW TESTI — en guclu 6 kirilma noktasi:")
print(ch.nlargest(6, "F").round(4).to_string(index=False))
kritik_F = stats.f.ppf(0.95, 2, len(olg) - 4)
print(f"  %95 kritik F = {kritik_F:.3f}")
print(f"  >> F > kritik olan donemler yapisal kirilma. TRAIN bolgesinin ICINDE")
print(f"     kirilma varsa, model iki farkli rejimi birlikte ogrenmeye calisiyor.")

W = 12
kay = []
for i in range(len(olg) - W + 1):
    w = olg.iloc[i:i+W]
    kay.append({"pencere": f"{w[DONEM_KOL].iloc[0]}-{w[DONEM_KOL].iloc[-1]}",
                "k_ort": w._k.mean(),
                "egim": OLS(w[HEDEF], add_constant(w[[REF]])).fit().params.iloc[1]})
kay = pd.DataFrame(kay)
print(f"\n  KAYAN PENCERE (12 ay):")
print(f"    carpan k : CV=%{100*kay.k_ort.std()/kay.k_ort.mean():.1f}  "
      f"aralik=[{kay.k_ort.min():.2f}, {kay.k_ort.max():.2f}]")
print(f"    egim     : CV=%{100*kay.egim.std()/kay.egim.mean():.1f}  "
      f"aralik=[{kay.egim.min():.2f}, {kay.egim.max():.2f}]")
print(f"  >> Carpanin CV'si egimin CV'sinden DUSUKSE, sabit terimsiz (oransal)")
print(f"     model daha saglamdir. Bunu BLOK 5'te ayri bir aday olarak test edecegiz.")

# ---------------------------------------------------------------- 3.3 TREND/MEVSIM
print("\n" + "-" * 102)
print("3.3  TREND VE MEVSIM — EK KATKI TESTI")
print("-" * 102)
print("  Dogru soru 'mevsim var mi' degil, 'ONCUL kontrol edildikten SONRA hala")
print("  anlamli mi'. Cunku mevsimsellik zaten oncule yansimis olabilir (cifte sayma).\n")
EN_IYI = tek.iloc[0]["degisken"]
tr_ok = d.loc[tr_idx].dropna(subset=[EN_IYI, HEDEF])
print(f"  Kontrol degiskeni: {EN_IYI} (TRAIN'de en guclu, r2={tek.iloc[0]['r2']:.3f})")

def ek_katki(taban, ek, ad):
    X0 = add_constant(tr_ok[taban]); X1 = add_constant(tr_ok[taban + ek])
    m0, m1 = OLS(tr_ok[HEDEF], X0).fit(), OLS(tr_ok[HEDEF], X1).fit()
    F = ((m0.ssr - m1.ssr) / len(ek)) / (m1.ssr / m1.df_resid)
    p = 1 - stats.f.cdf(F, len(ek), m1.df_resid)
    print(f"  {ad:34s} R2 {m0.rsquared:.4f} -> {m1.rsquared:.4f}  "
          f"F={F:6.3f}  p={p:.4f}  {'>>> EKLE' if p < 0.05 else 'ekleme'}")
    return p

p_tr = ek_katki([EN_IYI], ["TREND_IDX"],            "+ TREND")
p_ms = ek_katki([EN_IYI], ["SIN_AY", "COS_AY"],     "+ MEVSIM (1. harmonik)")
p_m2 = ek_katki([EN_IYI], ["SIN_AY","COS_AY","SIN_AY2","COS_AY2"], "+ MEVSIM (2 harmonik)")
print(f"\n  BLOK 1: lag-12 otokorelasyon = +0.116 -> 12 aylik dongu YOK.")
print(f"  Bu test de ayni sonucu verirse, mevsim varyantlari GRID'DEN CIKARILMALI")
print(f"  (arama uzayini 4 kat kuculturur, secim yanliligini azaltir).")

CONFIG["TREND_DENE"]  = bool(p_tr < 0.05)
CONFIG["MEVSIM_DENE"] = bool(p_ms < 0.05 or p_m2 < 0.05)
print(f"\n  -> CONFIG guncellendi: TREND_DENE={CONFIG['TREND_DENE']}  "
      f"MEVSIM_DENE={CONFIG['MEVSIM_DENE']}")

# ---------------------------------------------------------------- 3.4 KKB
print("\n" + "-" * 102)
print("3.4  KKB CONFOUNDING TESTI")
print("-" * 102)
print("  BLOK 1: KKB ilk12=1415.4 -> son12=1368.0  (-47.4 puan = 1.26 std)")
print("  Hedef de 2023'te zirve yapip dustu. Korelasyonun ne kadari GERCEK?\n")
if "KKB_SKOR_O0120" in d.columns:
    s = d.loc[tr_idx].dropna(subset=["KKB_SKOR_O0120", HEDEF, EN_IYI, "TREND_IDX"])
    r_ham, _ = stats.pearsonr(s["KKB_SKOR_O0120"], s[HEDEF])
    print(f"  Ham korelasyon  KKB ~ hedef        : {r_ham:+.4f}")

    def kismi(x, y, kontrol):
        rx = OLS(s[x], add_constant(s[kontrol])).fit().resid
        ry = OLS(s[y], add_constant(s[kontrol])).fit().resid
        return stats.pearsonr(rx, ry)

    for kon, ad in [(["TREND_IDX"], "TREND kontrol edilince"),
                    ([EN_IYI], f"{EN_IYI[:22]} kontrol edilince"),
                    ([EN_IYI, "TREND_IDX"], "IKISI birden kontrol")]:
        rr, pp = kismi("KKB_SKOR_O0120", HEDEF, kon)
        print(f"  Kismi korelasyon {ad:36s}: {rr:+.4f}  (p={pp:.4f})")
    print(f"\n  >> Kismi korelasyon ~0'a dusuyorsa KKB'nin katkisi BAGIMSIZ DEGIL;")
    print(f"     ya zaman trendini ya da oncul bilgisini tekrarliyor.")
    print(f"  >> Anlamli kalirsa KKB gercek ek bilgi tasiyor, havuzda kalmali.")

# ---------------------------------------------------------------- 3.5 BUDAMA
print("\n" + "-" * 102)
print("3.5  HAVUZ BUDAMA  (bu blogun asil teslimati)")
print("-" * 102)
nc = len(CHAMPION_HAVUZ)
from math import comb
eski = sum(comb(nc, k) for k in range(1, CONFIG["MAX_K"]+1)) * 4 * len(CONFIG["MODEL_AILELERI"])
print(f"  MEVCUT: CHAMPION {nc} degisken -> {eski:,} varyant")
print(f"  Onceki calismada 9.260 varyanttan secim CV->OOT'de %61 bozulma yaratmisti.")
print(f"  Hedef: havuzu ~25'e indir, varyanti ~37bin'e dusur.\n")

print("  BUDAMA KURALLARI:")
print("   K1  Minimum guc     : TRAIN r2 >= 0.35")
print("   K2  Yeterli gozlem  : n >= 30 (NaN uretenler elenir)")
print("   K3  Kayma cezasi    : |z_OOT| > 1.75 olanlar elenir (ekstrapolasyon riski)")
print("   K4  Tekrar temizligi: aralarinda |r| > 0.95 olanlardan SADECE EN GUCLUSU kalir")
print("   K5  Aile kotasi     : her aileden en fazla 4 degisken")
print("   K6  Olgunluk cesitliligi: her GEREKEN_AY seviyesinden en az 2 temsilci\n")

ad_bilgi = tek.set_index("degisken")
kayma_z  = ky.set_index("degisken").z_OOT.abs()

def buda(havuz, maks_ay, etiket, hedef_boyut=25):
    # K1 + K2 + K3
    aday = []
    for c in havuz:
        if c not in ad_bilgi.index: continue
        b = ad_bilgi.loc[c]
        if b["r2"] < 0.35 or b["n"] < 30: continue
        if kayma_z.get(c, 0) > 1.75: continue
        aday.append((c, float(b["r2"])))
    aday.sort(key=lambda t: -t[1])
    print(f"  [{etiket}] K1-K3 sonrasi: {len(aday)} / {len(havuz)}")

    # K4: tekrar temizligi (greedy, en gucluden basla)
    kor = d.loc[tr_idx, [c for c, _ in aday]].corr().abs()
    secili, aile_say = [], {}
    def hangi_aile(c):
        for a, v in AILE.items():
            if c in v: return a
        return "?"
    for c, r2 in aday:
        if any(kor.loc[c, s] > 0.95 for s in secili):   # K4
            continue
        a = hangi_aile(c)
        if aile_say.get(a, 0) >= 4:                     # K5
            continue
        secili.append(c); aile_say[a] = aile_say.get(a, 0) + 1
        if len(secili) >= hedef_boyut: break

    # K6: olgunluk cesitliligi — her seviyeden en az 2
    for ay in sorted({GEREKEN_AY[c] for c in havuz if GEREKEN_AY[c] <= maks_ay}):
        var = [c for c in secili if GEREKEN_AY[c] == ay]
        if len(var) >= 2: continue
        ek = [c for c, _ in aday if GEREKEN_AY[c] == ay and c not in secili]
        for c in ek[:2 - len(var)]:
            secili.append(c)
    print(f"  [{etiket}] BUDANMIS: {len(secili)} degisken  (aile dagilimi: {aile_say})")
    return secili

CHAMPION_BUDANMIS   = buda(CHAMPION_HAVUZ,   CONFIG["SON_ONCU_AY"],       "CHAMPION",   25)
CHALLENGER_BUDANMIS = buda(CHALLENGER_HAVUZ, CONFIG["CHALLENGER_MAX_AY"], "CHALLENGER", 20)

for ad, hv in [("CHAMPION", CHAMPION_BUDANMIS), ("CHALLENGER", CHALLENGER_BUDANMIS)]:
    t = pd.DataFrame({"degisken": hv})
    t["gereken_ay"] = t.degisken.map(GEREKEN_AY)
    t["r2"]         = t.degisken.map(ad_bilgi.r2)
    t["z_OOT"]      = t.degisken.map(ky.set_index("degisken").z_OOT)
    print(f"\n  === {ad} BUDANMIS HAVUZ ({len(hv)}) ===")
    print(t.sort_values(["gereken_ay", "r2"], ascending=[True, False]).round(3).to_string(index=False))

n2 = len(CHAMPION_BUDANMIS)
vary = 1 + int(CONFIG["TREND_DENE"]) + int(CONFIG["MEVSIM_DENE"]) + \
       int(CONFIG["TREND_DENE"] and CONFIG["MEVSIM_DENE"])
yeni = sum(comb(n2, k) for k in range(1, CONFIG["MAX_K"]+1)) * vary * len(CONFIG["MODEL_AILELERI"])
print(f"\n  VARYANT SAYISI: {eski:,}  ->  {yeni:,}   ({100*(1-yeni/eski):.1f}% azalma)")
print(f"  >> Secim yanliligi bu oranda azalir. BLOK 7 hakemligi kalani temizleyecek.")

# ---------------------------------------------------------------- 3.6 VIF
print("\n" + "-" * 102)
print("3.6  VIF — BUDANMIS HAVUZDA")
print("-" * 102)
for ad, hv in [("CHAMPION", CHAMPION_BUDANMIS), ("CHALLENGER", CHALLENGER_BUDANMIS)]:
    sub = d.loc[tr_idx, hv].dropna()
    if len(sub) <= len(hv) + 2:
        print(f"  [{ad}] n={len(sub)} <= p={len(hv)}: tam havuz VIF'i hesaplanamaz (beklenen).")
        print(f"         Grid zaten en fazla K={CONFIG['MAX_K']} degisken kullanacak.")
        continue
    X = add_constant(sub, has_constant="add")
    v = pd.DataFrame({"degisken": X.columns,
                      "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
    v = v[v.degisken != "const"].sort_values("VIF", ascending=False)
    print(f"\n  [{ad}] en yuksek 8 VIF:")
    print(v.head(8).round(2).to_string(index=False))

# ---------------------------------------------------------------- 3.7 GORSEL
print("\n" + "-" * 102)
print("3.7  GORSELLESTIRME")
print("-" * 102)
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle("BLOK 3 — Kesif Paneli", fontsize=13, fontweight="bold")

bg = {"TRAIN": RENK["tr"], "ARA": RENK["ara"], "OOT": RENK["oot"], "TAHMIN": RENK["gri"]}
for b_, g in d.groupby("BOLGE"):
    ax[0,0].scatter(g["AY"].dt.to_timestamp(), g[HEDEF]*100, s=30, color=bg[b_], label=b_)
ax[0,0].plot(d["AY"].dt.to_timestamp(), d[HEDEF]*100, "-", color=RENK["gri"], lw=.8, zorder=0)
ax[0,0].set_title("Hedef + bolgeler"); ax[0,0].legend(fontsize=7); ax[0,0].set_ylabel("%")
ax[0,0].tick_params(axis="x", rotation=45, labelsize=7)

ax[0,1].plot(ch.DONEM, ch.F, "-", color=RENK["tah"], lw=1.5)
ax[0,1].axhline(kritik_F, color=RENK["gri"], ls="--", label="%95 kritik")
ax[0,1].set_title("Chow F — kirilma noktasi"); ax[0,1].legend(fontsize=7)
ax[0,1].tick_params(axis="x", rotation=90, labelsize=6)

ax[0,2].plot(range(len(kay)), kay.k_ort, "o-", color=RENK["oot"], ms=3)
ax[0,2].set_title(f"Kayan carpan (12 ay)\nCV=%{100*kay.k_ort.std()/kay.k_ort.mean():.1f}")
ax[0,2].set_xticks(range(0, len(kay), 5))
ax[0,2].set_xticklabels(kay.pencere[::5], rotation=90, fontsize=6)

tt = tek[tek.degisken.isin(CHAMPION_BUDANMIS)].sort_values("r2").tail(18)
ax[1,0].barh(range(len(tt)), tt.r2,
             color=[RENK["oot"] if a <= CONFIG["CHALLENGER_MAX_AY"] else RENK["tr"]
                    for a in tt.gereken_ay])
ax[1,0].set_yticks(range(len(tt)))
ax[1,0].set_yticklabels([f"{c[:26]} ({a}ay)" for c, a in zip(tt.degisken, tt.gereken_ay)],
                        fontsize=6)
ax[1,0].set_xlim(0, 1); ax[1,0].set_title("Budanmis havuz r2\nyesil=challenger(<=3ay)")

mx = tek[tek.gereken_ay <= CONFIG["CHALLENGER_MAX_AY"]].iloc[0]["degisken"]
for b_, g in d.groupby("BOLGE"):
    if b_ == "TAHMIN": continue
    ax[1,1].scatter(g[mx], g[HEDEF]*100, s=35, color=bg[b_], label=b_)
_s = d.loc[tr_idx, [mx, HEDEF]].dropna()
z_ = np.polyfit(_s[mx], _s[HEDEF]*100, 1)
xs = np.linspace(d[mx].min(), d[mx].max(), 50)
ax[1,1].plot(xs, np.polyval(z_, xs), color=RENK["tah"], lw=1.5)
ax[1,1].set_xlabel(mx[:30]); ax[1,1].set_ylabel("hedef (%)")
ax[1,1].set_title(f"En guclu challenger girdisi\nr2={tek.set_index('degisken').r2[mx]:.3f} "
                  f"({GEREKEN_AY[mx]} ay)")
ax[1,1].legend(fontsize=7)

kk = d.loc[tr_idx, CHALLENGER_BUDANMIS[:12] + [HEDEF]].corr()
im = ax[1,2].imshow(kk, cmap="RdBu_r", vmin=-1, vmax=1)
ax[1,2].set_xticks(range(len(kk))); ax[1,2].set_yticks(range(len(kk)))
lbl = [c.replace("KK_TAKIP_", "").replace("_TUTARSAL", "")[:14] for c in kk.columns]
ax[1,2].set_xticklabels(lbl, rotation=90, fontsize=5)
ax[1,2].set_yticklabels(lbl, fontsize=5)
ax[1,2].set_title("Challenger budanmis korelasyon"); ax[1,2].grid(False)
plt.colorbar(im, ax=ax[1,2], shrink=.7)
plt.tight_layout(); plt.show()

print("=" * 102)
print(f"BLOK 3 TAMAM")
print(f"  CHAMPION_BUDANMIS   : {len(CHAMPION_BUDANMIS)}")
print(f"  CHALLENGER_BUDANMIS : {len(CHALLENGER_BUDANMIS)}")
print(f"  TREND_DENE={CONFIG['TREND_DENE']}  MEVSIM_DENE={CONFIG['MEVSIM_DENE']}")
print(f"  Varyant: {eski:,} -> {yeni:,}")
print("=" * 102)
