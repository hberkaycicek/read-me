# ============================================================
# BLOK 13 — v2c KURGUSUNU GERİ ÇÖZ ve GELECEĞİ ONUNLA TAHMİN ET
#   13.1  v2c parametre araması (bilinen OOT sonucunu yeniden üretme)
#   13.2  şampiyon kurgusuyla gelecek tahminleri
# ============================================================
import numpy as np, pandas as pd, re, itertools, warnings
warnings.filterwarnings("ignore")

HEDEF_SATIR = DOGRU.set_index("MERDIVEN").loc["CHAMPION"]
AMAC = {"WMAPE": float(HEDEF_SATIR["WMAPE"]), "SAPMA": float(HEDEF_SATIR["SAPMA"]),
        "POZ": int(str(HEDEF_SATIR["POZ"]).split("/")[0]),
        "PEARSON": float(HEDEF_SATIR["PEARSON"])}
print("ARANAN SONUÇ:", AMAC)

cfgC  = SECILEN["CHAMPION"]
OOT_I = d.index[(d["BOLGE"].astype(str) == "OOT") & d[HEDEF].notna()]
print(f"OOT hedef_idx: {len(OOT_I)} kohort")

# aday paydalar: 6. aya kadar bilinen öncül oranları
paydalar = [None] + sorted(
    c for c in d.columns
    if re.fullmatch(r"KK_TAKIP_(20|30|60)_[1-6]_TUTARSAL", str(c))
    and d.loc[OOT_I, c].notna().all() and (d.loc[OOT_I, c] > 0).all())
pencereler = [None, 3, 4, 6, 8, 12]
gecmisler  = [None, 24, 30, 36, 42]

print(f"aday payda: {len(paydalar)}  × pencere {len(pencereler)} × geçmiş {len(gecmisler)}"
      f"  = {len(paydalar)*len(pencereler)*len(gecmisler)} kombinasyon\n")

bulgular = []
for pay, pen, gec in itertools.product(paydalar, pencereler, gecmisler):
    try:
        r = wf_pencereli(kolonlar=cfgC["kolonlar"], aile=cfgC["aile"],
                         bilgi_ay=cfgC["bilgi_ay"], hedef_idx=OOT_I,
                         ek_kolonlar=cfgC.get("ek") or None,
                         yarilanma=cfgC.get("yarilanma", 1.0),
                         max_gecmis=gec, oran_paydasi=pay, duzelt_pencere=pen)
    except Exception:
        continue
    if not r or r.get("N", 0) != len(OOT_I):
        continue
    bulgular.append({"payda": pay, "duzelt_pencere": pen, "max_gecmis": gec,
                     "WMAPE": r["WMAPE"], "SAPMA": r["SAPMA"], "POZ": r["POZ"],
                     "PEARSON": r.get("PEARSON", np.nan),
                     "d_WMAPE": abs(r["WMAPE"] - AMAC["WMAPE"])})

B = pd.DataFrame(bulgular).sort_values("d_WMAPE").reset_index(drop=True)
print("=" * 110)
print("13.1  EN YAKIN 15 KOMBİNASYON")
print("=" * 110)
print(B.head(15).to_string(index=False))

TAM = B[(B.d_WMAPE < 0.01) & (abs(B.PEARSON - AMAC["PEARSON"]) < 0.005)
        & (B.POZ == AMAC["POZ"])]
if len(TAM):
    V2C = TAM.iloc[0].to_dict()
    print(f"\n*** TAM EŞLEŞME: payda={V2C['payda']}  duzelt_pencere={V2C['duzelt_pencere']}"
          f"  max_gecmis={V2C['max_gecmis']} ***")
else:
    V2C = B.iloc[0].to_dict()
    print(f"\n!! Tam eşleşme yok; en yakın kullanılıyor (ΔWMAPE={V2C['d_WMAPE']:.4f}).")
    print("   BLOK 10/11'deki senaryo kurulum hücresini paylaşırsan birebir kurarım.")

# ============================================================
# 13.2  GELECEK TAHMİNİ — wf_pencereli ile aynı mantık, hedefsiz
# ============================================================
def gelecek_tahmin(kolonlar, aile, bilgi_ay, tahmin_idx, max_gecmis=None,
                   ek_kolonlar=None, yarilanma=1.0, oran_paydasi=None,
                   duzelt_pencere=None, bugun_bilgisi=False):
    """wf_pencereli'nin birebir aynısı; hedefi bilinmeyen kohortlar için."""
    kol   = list(kolonlar) + list(ek_kolonlar or [])
    olgun = d[d.HEDEF_KULLANILABILIR]
    out   = []
    for i in tahmin_idx:
        c = d.loc[i]
        tahmin_ay = BUGUN if bugun_bilgisi else (c["AY"] + bilgi_ay)
        eg = olgun[olgun["HEDEF_ERISIM"] < tahmin_ay]
        if max_gecmis: eg = eg.tail(max_gecmis)
        kk = kol + ([oran_paydasi] if oran_paydasi and oran_paydasi not in kol else [])
        eg = eg[kk + [HEDEF]].dropna()
        if oran_paydasi: eg = eg[eg[oran_paydasi] > 0]
        if len(eg) < MIN_TRAIN: continue
        x = d.loc[[i], kk]
        if x.isna().any(axis=None): continue
        if oran_paydasi and float(x[oran_paydasi].iloc[0]) <= 0: continue

        pipe = pipeline_kur(aile)
        _e = CONFIG.get("ZAMAN_YARILANMA", 1.0); CONFIG["ZAMAN_YARILANMA"] = yarilanma
        y_fit = (eg[HEDEF]/eg[oran_paydasi]) if oran_paydasi else eg[HEDEF]
        _fit_agirlikli(pipe, eg[kol], y_fit, _agirlik(eg.index), aile)
        CONFIG["ZAMAN_YARILANMA"] = _e

        p = float(pipe.predict(x[kol])[0])
        if oran_paydasi: p *= float(x[oran_paydasi].iloc[0])
        ofs = 0.0
        if duzelt_pencere:
            gec = eg.tail(int(duzelt_pencere))
            if len(gec) >= 3:
                pg = pipe.predict(gec[kol])
                if oran_paydasi: pg = pg*gec[oran_paydasi].values
                ofs = float(np.mean(pg - gec[HEDEF].values))
                p -= ofs
        out.append({"AY": AYS(c["AY"]), "YAS_AY": (BUGUN - c["AY"]).n,
                    "EGITIM_N": len(eg), "HAM_TAHMIN": p + ofs,
                    "DUZELTME": ofs, "TAHMIN": float(np.clip(p, 0, 1))})
    return pd.DataFrame(out)

GEL_I = d.index[d[HEDEF].isna()]
kw = dict(kolonlar=cfgC["kolonlar"], aile=cfgC["aile"], bilgi_ay=cfgC["bilgi_ay"],
          ek_kolonlar=cfgC.get("ek") or None, yarilanma=cfgC.get("yarilanma", 1.0),
          max_gecmis=(None if pd.isna(V2C["max_gecmis"]) else int(V2C["max_gecmis"])),
          oran_paydasi=V2C["payda"],
          duzelt_pencere=(None if pd.isna(V2C["duzelt_pencere"])
                          else int(V2C["duzelt_pencere"])))

G_wf  = gelecek_tahmin(tahmin_idx=GEL_I, bugun_bilgisi=False, **kw)
G_bug = gelecek_tahmin(tahmin_idx=GEL_I, bugun_bilgisi=True,  **kw)

son12 = d[d[HEDEF].notna()].sort_values("AY").tail(12)[HEDEF].mean()
G = (G_wf.rename(columns={"TAHMIN":"TAHMIN_wf","EGITIM_N":"EGITIM_N_wf",
                          "DUZELTME":"DUZELTME_wf"})[["AY","YAS_AY","EGITIM_N_wf",
                                                      "DUZELTME_wf","TAHMIN_wf"]]
     .merge(G_bug.rename(columns={"TAHMIN":"TAHMIN_bugun","EGITIM_N":"EGITIM_N_bugun",
                                  "DUZELTME":"DUZELTME_bugun"})
            [["AY","EGITIM_N_bugun","DUZELTME_bugun","TAHMIN_bugun"]], on="AY", how="outer"))
G["SON12_ORT"] = son12
G["FARK_%"]    = 100*(G.TAHMIN_bugun - son12)/son12

print("\n" + "=" * 110)
print("13.2  GELECEK TAHMİNLERİ — ŞAMPİYON (v2c) KURGUSUYLA")
print("=" * 110)
print(f"kurgu: payda={kw['oran_paydasi']}  duzelt_pencere={kw['duzelt_pencere']}"
      f"  max_gecmis={kw['max_gecmis']}  model={cfgC['aile']}  bilgi_ay={cfgC['bilgi_ay']}")
print(f"girdiler: {list(cfgC['kolonlar']) + list(cfgC.get('ek') or [])}\n")
print(G.to_string(index=False))
print(f"\n  TAHMIN_wf    : kohort {cfgC['bilgi_ay']} aylıkken yapılabilecek tahmin "
      f"(validasyonla aynı mantık)")
print(f"  TAHMIN_bugun : bugünkü ({BUGUN}) tüm bilgiyle yeniden eğitilmiş tahmin")
print(f"\n  son 12 olgun kohort ortalaması : {son12:.4f}")
print(f"  tahmin ortalaması (bugün)      : {G.TAHMIN_bugun.mean():.4f}"
      f"   ({'YUKARI' if G.TAHMIN_bugun.mean() > son12 else 'AŞAĞI'})")

fig, ax = plt.subplots(figsize=(17, 5.5))
g = d[d[HEDEF].notna()].sort_values("AY")
ax.plot(g["_AYS"], g[HEDEF], "o-", lw=2.4, ms=4, color="#16253D", label="Gerçekleşen")
ax.plot(G.AY, G.TAHMIN_bugun, "D--", lw=2.6, ms=9, color="#C25E00",
        label="Gelecek — şampiyon v2c")
ax.plot(G.AY, G.TAHMIN_wf, "x:", lw=1.8, ms=8, color="#8595AF",
        label=f"Gelecek — {cfgC['bilgi_ay']}. ayda yapılan tahmin")
ax.axhline(son12, ls=":", lw=1.5, color="#B4BCC6", label="Son 12 ay ortalaması")
ax.axvline(g["_AYS"].iloc[-1], color="#8A93A0", ls=":", lw=1.5)
ax.set_title("12M90 — gerçekleşen seri ve olgunlaşmamış kohort tahminleri (v2c)", fontsize=13)
ax.set_ylabel("12M90 oranı"); ax.legend(fontsize=9)
plt.xticks(rotation=90, fontsize=7); plt.tight_layout(); plt.show()
