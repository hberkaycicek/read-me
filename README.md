# ============================================================
# BLOK 15 — İKİ KURGUNUN ADİL KARŞILAŞTIRMASI
#   TEK NOTEBOOK'TA ÇALIŞIR: gelişmiş pipeline'ın olduğu yerde.
#   Diğer çalışmadan yalnızca değişken listesi alındı, kodda sabit.
# ============================================================
import itertools, warnings; warnings.filterwarnings("ignore")

# --- ÖN KONTROL ---------------------------------------------------------
gerekli_nesne = ["d", "HEDEF", "wf_pencereli", "metrikler", "SECILEN"]
eksik_nesne = [n for n in gerekli_nesne if n not in dir()]
if eksik_nesne:
    raise RuntimeError(f"Önce BLOK 12'yi çalıştır. Eksik: {eksik_nesne}")

K20_2, K30_2 = "KK_TAKIP_20_2_TUTARSAL", "KK_TAKIP_30_2_TUTARSAL"
K30_3, K60_3 = "KK_TAKIP_30_3_TUTARSAL", "KK_TAKIP_60_3_TUTARSAL"
K60_6, KKB   = "KK_TAKIP_60_6_TUTARSAL", "KKB_SKOR_O0120"

ARA_I = d.index[(d["BOLGE"].astype(str) == "ARA") & d[HEDEF].notna()]
OOT_I = d.index[(d["BOLGE"].astype(str) == "OOT") & d[HEDEF].notna()]
print(f"ARA {len(ARA_I)} | OOT {len(OOT_I)} kohort")

TEMEL = {
    "A_CHAMPION":   dict(kolonlar=[K20_2, K60_3, KKB], bilgi_ay=3),
    "A_CHALLENGER": dict(kolonlar=[K60_3, KKB],        bilgi_ay=3),
    "B_CHALLENGER": dict(kolonlar=list(SECILEN["CHALLENGER"]["kolonlar"]), bilgi_ay=3),
    "B_CHAMPION":   dict(kolonlar=list(SECILEN["CHAMPION"]["kolonlar"]),   bilgi_ay=6),
}
for ad in list(TEMEL):
    yok = [c for c in TEMEL[ad]["kolonlar"] if c not in d.columns]
    if yok:
        print(f"  [atlandı] {ad}: {yok} kolonu yok")
        TEMEL.pop(ad)
if not TEMEL:
    raise RuntimeError("Hiçbir set kurulamadı — kolon adlarını kontrol et.")
print("değerlendirilecek setler:", list(TEMEL))

EK = {k: v for k, v in {
    "yok": [], "mevsim": ["SIN_AY", "COS_AY"],
    "trend": ["TREND_IDX"], "trend+mevsim": ["TREND_IDX", "SIN_AY", "COS_AY"],
}.items() if all(c in d.columns for c in v)}
print("ek varyantlar:", list(EK))

AILELER, PENCERELER, DUZELTME = ["Linear", "Ridge"], [None, 24, 30, 36], [None, 6]

def paydalar(bilgi_ay):
    ad = [None, K20_2, K30_2, K60_3] + ([K60_6] if bilgi_ay >= 6 else [])
    ix = list(ARA_I) + list(OOT_I)
    return [p for p in ad if p is None or
            (p in d.columns and d.loc[ix, p].notna().all() and (d.loc[ix, p] > 0).all())]

# --- GRID ---------------------------------------------------------------
sat, hata = [], 0
for ad, t in TEMEL.items():
    for ek_ad, ek in EK.items():
        for aile, gec, pay, duz in itertools.product(
                AILELER, PENCERELER, paydalar(t["bilgi_ay"]), DUZELTME):
            kw = dict(kolonlar=t["kolonlar"], aile=aile, bilgi_ay=t["bilgi_ay"],
                      ek_kolonlar=ek or None, max_gecmis=gec,
                      oran_paydasi=pay, duzelt_pencere=duz)
            try:
                a = wf_pencereli(hedef_idx=ARA_I, **kw)
                o = wf_pencereli(hedef_idx=OOT_I, **kw)
            except Exception:
                hata += 1; continue
            if not a or not o or o.get("N", 0) != len(OOT_I):
                continue
            m = metrikler(np.array(o["_ger"], float), np.array(o["_tah"], float))
            sat.append({"SET": ad, "BILGI_AY": t["bilgi_ay"], "EK": ek_ad, "AILE": aile,
                        "PENCERE": gec or "tum",
                        "PAYDA": (pay or "yok").replace("KK_TAKIP_", "").replace("_TUTARSAL", ""),
                        "DUZELT": duz or 0,
                        "ARA_WMAPE": a["WMAPE"], "ARA_SAPMA": a["SAPMA"], **m})

K = pd.DataFrame(sat)
print(f"\n{len(K)} varyant değerlendirildi ({hata} koşum hata verdi, atlandı)\n")
if K.empty:
    raise RuntimeError("Hiç sonuç yok — wf_pencereli parametre adlarını kontrol et.")

# --- BİLEŞİK SKOR -------------------------------------------------------
def nrm(s, ters=False):
    s = s.astype(float); lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or hi == lo: return pd.Series(0.0, index=s.index)
    n = (s - lo) / (hi - lo)
    return 1 - n if ters else n

K["YON_DENGE"] = (K["POZ"] / K["N"] - 0.5).abs()
K["SKOR"] = (0.30*nrm(K["ARA_WMAPE"]) + 0.20*nrm(K["SAPMA"].abs())
           + 0.20*nrm(K["WMAPE_%"])   + 0.15*nrm(K["PEARSON"], ters=True)
           + 0.10*nrm(K["YON_DENGE"]) + 0.05*nrm(K["EN_KOTU"]))
K = K.sort_values("SKOR").reset_index(drop=True)

G = ["SET","BILGI_AY","EK","AILE","PENCERE","PAYDA","DUZELT",
     "ARA_WMAPE","WMAPE_%","MAPE_%","MAE","RMSE","SAPMA","POZ","PEARSON","EN_KOTU","SKOR"]

print("="*135); print("EN İYİ 20 (bileşik skor, düşük = iyi)"); print("="*135)
print(K[G].head(20).to_string(index=False))

print("\n" + "="*135); print("HER SETİN EN İYİSİ"); print("="*135)
print(K.loc[K.groupby("SET")["SKOR"].idxmin(), G].to_string(index=False))

print("\n" + "="*135)
print("TABAN KURGU (ek yok, tüm pencere, oransal yok, düzeltme yok) — çalışmaların aslı")
print("="*135)
t = K[(K.EK=="yok") & (K.PENCERE=="tum") & (K.PAYDA=="yok") & (K.DUZELT==0)]
print(t[G].to_string(index=False) if len(t) else "  (bu kombinasyon üretilemedi)")

print("\n--- MEKANİZMALARIN TEK TEK KATKISI (A kurgusunun değişkenleriyle) ---")
b = K[K.SET == "A_CHAMPION"]
for ad_, msk in [
    ("taban",          (b.PENCERE=="tum") & (b.PAYDA=="yok") & (b.DUZELT==0) & (b.EK=="yok")),
    ("+ kısa pencere", (b.PENCERE!="tum") & (b.PAYDA=="yok") & (b.DUZELT==0) & (b.EK=="yok")),
    ("+ oransal",      (b.PENCERE=="tum") & (b.PAYDA!="yok") & (b.DUZELT==0) & (b.EK=="yok")),
    ("+ düzeltme",     (b.PENCERE=="tum") & (b.PAYDA=="yok") & (b.DUZELT==6) & (b.EK=="yok")),
    ("üçü birlikte",   (b.PENCERE!="tum") & (b.PAYDA!="yok") & (b.DUZELT==6) & (b.EK=="yok")),
]:
    s = b[msk]
    if len(s):
        r_ = s.loc[s["WMAPE_%"].idxmin()]
        print(f"  {ad_:16s} OOT {r_['WMAPE_%']:6.2f} | ARA {r_['ARA_WMAPE']:6.2f} | "
              f"sapma {r_['SAPMA']:+.4f} | poz {int(r_['POZ'])}/12 | "
              f"pearson {r_['PEARSON']:.3f} | payda={r_['PAYDA']} pencere={r_['PENCERE']}")

print("\n--- TREND VE MEVSİMSELLİK İŞE YARIYOR MU ---")
print(K.groupby("EK").agg(EN_IYI_ARA=("ARA_WMAPE","min"), EN_IYI_OOT=("WMAPE_%","min"),
                          EN_IYI_SKOR=("SKOR","min"), N=("SKOR","size")).to_string())

print("\n--- BİLGİ AYINA GÖRE EN İYİ (erkenlik–doğruluk takası) ---")
print(K.loc[K.groupby("BILGI_AY")["SKOR"].idxmin(),
            ["BILGI_AY","SET","ARA_WMAPE","WMAPE_%","SAPMA","PEARSON"]].to_string(index=False))
