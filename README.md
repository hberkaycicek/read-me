# ============================================================
# BLOK 12 — RAPORLAMA PAKETİ  (tek parça, baştan sona)
#   12.0  hedef kolonu + bölge sınırları + yardımcılar
#   12.1  tam metrik tablosu (tüm basamaklar × TRAIN/ARA/OOT)
#   12.2  şampiyon modelin formülü ve katsayıları
#   12.3  gerçekleşen vs tahmin serisi + grafik
#   12.4  gelecek tahminleri (hedefi bilinmeyen kohortlar)
# ============================================================
import numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)

# ------------------------------------------------------------
# 12.0  HEDEF KOLONU
# ------------------------------------------------------------
def hedef_bul(d, config=None):
    """Hedef kolonunu üç yoldan arar: CONFIG -> isim -> olgunlaşma imzası."""
    # 1) CONFIG içinde doğrudan yazılı mı
    if config:
        for k in ("HEDEF_KOLON", "HEDEF", "HEDEF_AD", "TARGET", "Y_KOLON"):
            v = config.get(k)
            if isinstance(v, str) and v in d.columns:
                return v, f"CONFIG['{k}']"
    # 2) isimden
    anahtar = ("12M90", "L12M90", "12AY90", "_M90", "M90_", "BATIK", "HEDEF", "TARGET")
    aday = [c for c in d.columns
            if any(t in str(c).upper().replace(" ", "").replace("-", "_") for t in anahtar)]
    aday = [c for c in aday if pd.api.types.is_numeric_dtype(d[c])]
    if len(aday) == 1:
        return aday[0], "isim"
    # 3) olgunlaşma imzası: son aylar boş, öncesi dolu
    d2 = d.sort_values("AY")
    imza = [c for c in d2.select_dtypes("number").columns
            if d2[c].tail(4).isna().all() and d2[c].iloc[:-6].notna().mean() > 0.95]
    if len(imza) == 1:
        return imza[0], "olgunlaşma imzası"
    raise RuntimeError(
        "Hedef kolonu otomatik bulunamadı.\n"
        f"  isim adayları : {aday}\n"
        f"  imza adayları : {imza}\n"
        f"  tüm kolonlar  : {sorted(map(str, d.columns))}\n"
        "→ HEDEF = '<kolon adı>' diye elle yaz ve bloğu tekrar çalıştır.")

# Elle yazmak istersen alttaki satırı aç ve otomatik bulmayı atla:
# HEDEF = "L12M90_TUTARSAL"
if "HEDEF" not in dir() or not isinstance(globals().get("HEDEF"), str) or HEDEF not in d.columns:
    HEDEF, _kaynak = hedef_bul(d, CONFIG if "CONFIG" in dir() else None)
    print(f"HEDEF = '{HEDEF}'   (kaynak: {_kaynak})")

print(f"  dolu  : {d[HEDEF].notna().sum()} kohort   "
      f"({d.loc[d[HEDEF].notna(), 'AY'].min()} – {d.loc[d[HEDEF].notna(), 'AY'].max()})")
print(f"  boş   : {d[HEDEF].isna().sum()} kohort   "
      f"({d.loc[d[HEDEF].isna(), 'AY'].min()} – {d.loc[d[HEDEF].isna(), 'AY'].max()})")

# ------------------------------------------------------------
# BÖLGE SINIRLARI  (BLOK 4'teki değerlerle aynı olmalı — gerekirse düzelt)
# ------------------------------------------------------------
P = lambda s: pd.Period(s, freq="M")
SINIR = {
    "TRAIN": (P("2021-01"), P("2024-03")),
    "ARA":   (P("2024-04"), P("2024-09")),
    "OOT":   (P("2024-10"), P("2025-09")),
}
BUGUN = P("2026-10")

def bolge_idx(ad):
    a, b = SINIR[ad]
    m = (d["AY"] >= a) & (d["AY"] <= b) & d[HEDEF].notna()
    return d.index[m]

for b in SINIR:
    print(f"  {b:6s}: {len(bolge_idx(b))} kohort  {SINIR[b][0]} – {SINIR[b][1]}")

# ------------------------------------------------------------
# METRİK SETİ
# ------------------------------------------------------------
def metrikler(y, yhat, w=None):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    w = np.ones(len(y)) if w is None else np.asarray(w, float)[ok]
    if len(y) == 0:
        return {k: np.nan for k in ("N","MAE","MSE","RMSE","MAPE_%","WMAPE_%",
                                    "SAPMA","BIAS_ORANI","PEARSON","SPEARMAN","R2","ARTI")}
    e = yhat - y
    out = {
        "N":          len(y),
        "MAE":        np.mean(np.abs(e)),
        "MSE":        np.mean(e ** 2),
        "RMSE":       np.sqrt(np.mean(e ** 2)),
        "MAPE_%":     100 * np.mean(np.abs(e) / y),
        "WMAPE_%":    100 * np.sum(w * np.abs(e)) / np.sum(w * y),
        "SAPMA":      np.mean(e),                        # + = fazla tahmin
        "BIAS_ORANI": abs(np.mean(e)) / np.mean(np.abs(e)),
        "PEARSON":    stats.pearsonr(y, yhat)[0] if len(y) > 2 else np.nan,
        "SPEARMAN":   stats.spearmanr(y, yhat)[0] if len(y) > 2 else np.nan,
        "R2":         1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2) if len(y) > 2 else np.nan,
        "ARTI":       int((e > 0).sum()),                # kaç ayda fazla tahmin
    }
    return out

def dummy_metrik(idx):
    """Referans: eğitim ortalaması. Karşılaştırma tabanı."""
    y = d.loc[idx, HEDEF].values
    w = d.loc[idx, "BASVURU_ADEDI"].values
    return metrikler(y, np.full(len(y), np.nanmean(y)), w)

# wf_pencereli ne döndürürse döndürsün (DataFrame / Series / tuple / array) normalize et
def _cikar(res, idx):
    if isinstance(res, tuple):
        res = res[0]
    if isinstance(res, pd.DataFrame):
        kol = next((c for c in res.columns
                    if str(c).upper() in ("TAHMIN", "PRED", "YHAT", "Y_PRED")), None)
        yh = res[kol].values if kol else res.iloc[:, -1].values
        return res.index, yh
    if isinstance(res, pd.Series):
        return res.index, res.values
    arr = np.ravel(np.asarray(res, float))
    return pd.Index(list(idx))[:len(arr)], arr

# ------------------------------------------------------------
# 12.1  BASAMAK × BÖLGE METRİK TABLOSU
# ------------------------------------------------------------
# BLOK 11'in seçtiği nihai kurulumları buraya yaz.
# Her değer wf_pencereli'ye aktarılacak argüman sözlüğüdür.
FINAL = {
    # ÖRNEK ŞABLON — gerçek değerlerle değiştir:
    # "ULTRA_ERKEN_1M": dict(kolonlar=["KK_TAKIP_20_2_TUTARSAL"], aile="ULTRA_ERKEN",
    #                        bilgi_ay=1),
    # "ERKEN_2M":       dict(kolonlar=[...], aile="ERKEN", bilgi_ay=2, max_gecmis=30),
    # "ERKEN_3M":       dict(kolonlar=[...], aile="ERKEN", bilgi_ay=3, max_gecmis=30),
    # "ORTA_4M":        dict(kolonlar=[...], aile="ORTA",  bilgi_ay=4),
    # "OLGUN_6M":       dict(kolonlar=["KK_TAKIP_30_6_TUTARSAL","KK_TAKIP_20_2_TUTARSAL_MA3"],
    #                        aile="OLGUN", bilgi_ay=6, max_gecmis=30,
    #                        oran_paydasi="KK_TAKIP_30_2_TUTARSAL", duzelt_pencere=6,
    #                        ek_kolonlar=["MM_SKOR","AY_SIN","AY_COS"]),
}
SAMPIYON = "OLGUN_6M"     # FINAL içindeki şampiyon anahtarı

if not FINAL:
    print("\n!! FINAL boş — BLOK 11 çıktısındaki nihai kurulumları yazman gerekiyor.\n"
          "   12.1–12.4 atlandı.")
else:
    satir = []
    for ad, kw in FINAL.items():
        for bolge in ("TRAIN", "ARA", "OOT"):
            idx = bolge_idx(bolge)
            try:
                res = wf_pencereli(hedef_idx=idx, **kw)
                ridx, yh = _cikar(res, idx)
                gy = d.loc[ridx, HEDEF].values
                w  = d.loc[ridx, "BASVURU_ADEDI"].values
                m  = metrikler(gy, yh, w)
            except Exception as ex:
                print(f"  [hata] {ad} / {bolge}: {type(ex).__name__}: {ex}")
                m = {k: np.nan for k in metrikler([1.0], [1.0]).keys()}
            dm = dummy_metrik(idx)
            m["REF_WMAPE_%"] = dm["WMAPE_%"]
            m["REF_ORANI"]   = m["WMAPE_%"] / dm["WMAPE_%"] if dm["WMAPE_%"] else np.nan
            satir.append({"BASAMAK": ad, "BILGI_AY": kw.get("bilgi_ay"),
                          "BOLGE": bolge, **m})

    TABLO = pd.DataFrame(satir)
    print("\n" + "=" * 110)
    print("12.1  TAM METRİK TABLOSU   (REF_ORANI < 1 = referanstan iyi)")
    print("=" * 110)
    print(TABLO.round(4).to_string(index=False))
    TABLO.to_excel("12_metrik_tablosu.xlsx", index=False)

    # özet: sadece OOT
    print("\n--- Kilitli test (OOT) özeti ---")
    print(TABLO[TABLO.BOLGE == "OOT"][
        ["BASAMAK","BILGI_AY","N","WMAPE_%","MAPE_%","MAE","RMSE","MSE",
         "SAPMA","BIAS_ORANI","PEARSON","R2","REF_WMAPE_%","REF_ORANI"]
    ].round(4).to_string(index=False))

    # overfit oranı: OOT / TRAIN
    piv = TABLO.pivot_table(index="BASAMAK", columns="BOLGE", values="WMAPE_%")
    piv["OVERFIT_ORANI"] = piv.get("OOT") / piv.get("TRAIN")
    print("\n--- Bölgeler arası WMAPE ve overfit oranı ---")
    print(piv.round(4).to_string())

    # ------------------------------------------------------------
    # 12.2  ŞAMPİYONUN KATSAYILARI VE FORMÜLÜ
    # ------------------------------------------------------------
    kw   = FINAL[SAMPIYON]
    kols = list(kw["kolonlar"]) + list(kw.get("ek_kolonlar") or [])
    payda = kw.get("oran_paydasi")

    # pipeline kurucusunu bul; yoksa varsayılan Ridge kur
    _kurucu = next((globals()[n] for n in
                    ("pipeline_kur","pipe_kur","model_kur","kur_pipeline","pipeline_olustur")
                    if n in globals() and callable(globals()[n])), None)
    if _kurucu is not None:
        try:    pipe = _kurucu(kw["aile"])
        except TypeError: pipe = _kurucu()
    else:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV
        pipe = Pipeline([("s", StandardScaler()),
                         ("m", RidgeCV(alphas=np.logspace(-3, 3, 25)))])
        print("\n(uyarı: pipeline kurucusu bulunamadı, varsayılan Ridge kuruldu)")

    tr = d[d[HEDEF].notna()].sort_values("AY")
    tr = tr[tr["AY"] <= SINIR["OOT"][1]]
    if kw.get("max_gecmis"):
        tr = tr.iloc[-int(kw["max_gecmis"]):]
    X = tr[kols]
    y = tr[HEDEF] / tr[payda] if payda else tr[HEDEF]
    w = _agirlik(tr.index) if "_agirlik" in dir() else np.ones(len(tr))
    if "_fit_agirlikli" in dir():
        _fit_agirlikli(pipe, X, y, w, kw["aile"])
    else:
        pipe.fit(X, y)

    def katsayilar(pipe, kols):
        m = getattr(pipe, "regressor_", pipe)
        ns = getattr(m, "named_steps", {})
        sc  = ns.get("s") or ns.get("scaler") or ns.get("std")
        reg = ns.get("m") or ns.get("model") or ns.get("reg") or list(ns.values())[-1]
        b_std = np.ravel(getattr(reg, "coef_", np.full(len(kols), np.nan)))
        if sc is not None and hasattr(sc, "scale_"):
            b_ham = b_std / sc.scale_
            sabit = float(np.ravel(reg.intercept_)[0]) - float(np.sum(b_ham * sc.mean_))
        else:
            b_ham, sabit = b_std, float(np.ravel(reg.intercept_)[0])
        K = pd.DataFrame({"DEGISKEN": kols, "KATSAYI_STD": b_std, "KATSAYI_HAM": b_ham})
        K["ETKI_%"] = 100 * np.abs(K.KATSAYI_STD) / np.abs(K.KATSAYI_STD).sum()
        K["YON"] = np.where(K.KATSAYI_STD >= 0, "↑ artırır", "↓ azaltır")
        return K.sort_values("ETKI_%", ascending=False), sabit, reg

    KATS, SABIT, _reg = katsayilar(pipe, kols)
    print("\n" + "=" * 110)
    print(f"12.2  ŞAMPİYON MODEL: {SAMPIYON}   (eğitim: {tr.AY.min()} – {tr.AY.max()}, "
          f"{len(tr)} kohort)")
    print("=" * 110)
    print(KATS.round(6).to_string(index=False))
    print(f"\nSABIT = {SABIT:.6f}"
          + (f"   |   alpha = {getattr(_reg, 'alpha_', 'n/a')}" if hasattr(_reg, "alpha_") else ""))
    print("\nFORMÜL (ham ölçekte):")
    print(f"   ham_tahmin = {SABIT:+.6f}")
    for k, b in zip(KATS.DEGISKEN, KATS.KATSAYI_HAM):
        print(f"                {'+' if b >= 0 else '-'} {abs(b):.6f} × {k}")
    if payda:
        print(f"   tahmin     = ham_tahmin × {payda}        ← oransal hedef")
    if kw.get("duzelt_pencere"):
        print(f"   tahmin     = tahmin − (son {kw['duzelt_pencere']} kohortun ortalama sapması)")
    KATS.to_excel("12_katsayilar.xlsx", index=False)

    # ------------------------------------------------------------
    # 12.3  GERÇEKLEŞEN vs TAHMİN SERİSİ
    # ------------------------------------------------------------
    parca = []
    for bolge in ("TRAIN", "ARA", "OOT"):
        idx = bolge_idx(bolge)
        res = wf_pencereli(hedef_idx=idx, **kw)
        ridx, yh = _cikar(res, idx)
        parca.append(pd.DataFrame({
            "AY": d.loc[ridx, "AY"].astype(str).values,
            "BOLGE": bolge,
            "GERCEK": d.loc[ridx, HEDEF].values,
            "TAHMIN": yh}))
    SERI = pd.concat(parca).sort_values("AY").reset_index(drop=True)
    SERI["HATA"]   = SERI.TAHMIN - SERI.GERCEK
    SERI["HATA_%"] = 100 * SERI.HATA / SERI.GERCEK
    print("\n" + "=" * 110)
    print("12.3  GERÇEKLEŞEN vs TAHMİN")
    print("=" * 110)
    print(SERI.round(4).to_string(index=False))
    SERI.to_excel("12_seri.xlsx", index=False)

    fig, ax = plt.subplots(figsize=(16, 5.5))
    for b, c, lbl in [("TRAIN", "#FFFFFF", None),
                      ("ARA",   "#FDF1E3", "ARA (hakem)"),
                      ("OOT",   "#E7EFEC", "OOT (kilitli)")]:
        s = SERI[SERI.BOLGE == b]
        if len(s) and lbl:
            ax.axvspan(s.AY.iloc[0], s.AY.iloc[-1], color=c, zorder=0, label=lbl)
    ref = d.loc[bolge_idx("TRAIN"), HEDEF].mean()
    ax.axhline(ref, ls=":", lw=1.5, color="#B4BCC6", label="Referans (eğitim ort.)")
    ax.plot(SERI.AY, SERI.GERCEK, "o-", lw=2.6, ms=5, color="#16253D", label="Gerçekleşen")
    ax.plot(SERI.AY, SERI.TAHMIN, "s--", lw=2.0, ms=4, color="#C25E00", label="Tahmin")
    ax.set_title(f"12M90 — gerçekleşen vs tahmin  ({SAMPIYON})", fontsize=13)
    ax.set_ylabel("12M90 oranı"); ax.legend(ncol=5, fontsize=9)
    plt.xticks(rotation=90, fontsize=7); plt.tight_layout()
    plt.savefig("12_seri.png", dpi=150); plt.show()

    # hata grafiği
    fig, ax = plt.subplots(figsize=(16, 3))
    ax.bar(SERI.AY, SERI.HATA,
           color=np.where(SERI.HATA > 0, "#C25E00", "#2E6B5E"))
    ax.axhline(0, color="#16253D", lw=1)
    ax.set_title("Tahmin hatası (+ fazla tahmin / − eksik tahmin)", fontsize=12)
    plt.xticks(rotation=90, fontsize=7); plt.tight_layout()
    plt.savefig("12_hata.png", dpi=150); plt.show()

    # ------------------------------------------------------------
    # 12.4  GELECEK TAHMİNLERİ
    # ------------------------------------------------------------
    gel = d[d[HEDEF].isna()].sort_values("AY").copy()
    gel = gel.dropna(subset=kols + ([payda] if payda else []))
    if len(gel) == 0:
        print("\n12.4  Tahmin edilecek olgunlaşmamış kohort yok "
              "(ya da gerekli öncüller henüz boş).")
    else:
        yg = np.ravel(pipe.predict(gel[kols]))
        if payda:
            yg = yg * gel[payda].values
        duz = 0.0
        if kw.get("duzelt_pencere"):
            duz = SERI[SERI.BOLGE == "OOT"].tail(int(kw["duzelt_pencere"])).HATA.mean()
            yg = yg - duz
        GELECEK = pd.DataFrame({
            "AY": gel["AY"].astype(str).values,
            "OLGUNLUK_AY": [(BUGUN - a).n for a in gel["AY"]],
            "TAHMIN": yg,
            "HEDEF_ERISIM": [str(a + CONFIG.get("HEDEF_OLGUNLASMA_AY", 12)
                                 + CONFIG.get("RAPOR_GECIKME_AY", 1)) for a in gel["AY"]],
        })
        son12 = d[d[HEDEF].notna()].sort_values("AY").tail(12)[HEDEF]
        GELECEK["SON12_ORT_FARK"] = GELECEK.TAHMIN - son12.mean()
        print("\n" + "=" * 110)
        print(f"12.4  GELECEK TAHMİNLERİ   (sapma düzeltmesi: {duz:+.5f})")
        print("=" * 110)
        print(GELECEK.round(4).to_string(index=False))
        print(f"\n  son 12 gerçekleşen ortalama : {son12.mean():.4f}")
        print(f"  tahmin ortalaması           : {np.mean(yg):.4f}")
        GELECEK.to_excel("12_gelecek.xlsx", index=False)

        fig, ax = plt.subplots(figsize=(16, 5))
        g = d[d[HEDEF].notna()].sort_values("AY")
        ax.plot(g.AY.astype(str), g[HEDEF], "o-", lw=2.4, ms=4,
                color="#16253D", label="Gerçekleşen")
        ax.plot(GELECEK.AY, GELECEK.TAHMIN, "D--", lw=2.4, ms=7,
                color="#C25E00", label="Gelecek tahmini")
        ax.axvline(str(g.AY.iloc[-1]), color="#8A93A0", ls=":", lw=1.5)
        ax.set_title("12M90 — gerçekleşen seri ve olgunlaşmamış kohort tahminleri", fontsize=13)
        ax.legend(); plt.xticks(rotation=90, fontsize=7); plt.tight_layout()
        plt.savefig("12_gelecek.png", dpi=150); plt.show()

print("\nÜretilen dosyalar: 12_metrik_tablosu.xlsx, 12_katsayilar.xlsx, "
      "12_seri.xlsx, 12_gelecek.xlsx, 12_seri.png, 12_hata.png, 12_gelecek.png")
