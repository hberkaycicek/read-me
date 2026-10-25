# ============================================================
# BLOK 12 — RAPORLAMA PAKETİ  (v2 — olgunluk maskesi düzeltildi, dosya yazmıyor)
#   12.0  ayarlar, hedef, olgunluk maskesi, bölgeler, yardımcılar
#   12.0c FINAL yardımcısı (BLOK 11 sonuçlarını bulur)
#   12.1  tam metrik tablosu (tüm basamaklar × TRAIN/ARA/OOT)
#   12.2  şampiyon modelin formülü ve katsayıları
#   12.3  gerçekleşen vs tahmin serisi + grafikler
#   12.4  gelecek tahminleri (olgunlaşmamış kohortlar)
# ============================================================
import numpy as np, pandas as pd, inspect, re
from scipy import stats
import matplotlib.pyplot as plt

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

P = lambda s: pd.Period(s, freq="M")
BUGUN = P("2026-10")                      # içinde bulunduğumuz ay

# ============================================================
# 12.0  HEDEF + OLGUNLUK MASKESİ
# ============================================================
HEDEF_HAM = "KK_TAKIP_90_12_TUTARSAL"
assert HEDEF_HAM in d.columns, f"{HEDEF_HAM} yok — kolon adını düzelt"

if "OLGUN" in d.columns:
    _mask, _kaynak = d["OLGUN"].astype(bool), "OLGUN kolonu"
elif "HEDEF_KULLANILABILIR" in d.columns:
    _mask, _kaynak = d["HEDEF_KULLANILABILIR"].astype(bool), "HEDEF_KULLANILABILIR kolonu"
else:
    _son = BUGUN - (CONFIG.get("HEDEF_OLGUNLASMA_AY", 12) + CONFIG.get("RAPOR_GECIKME_AY", 1))
    _mask, _kaynak = (d["AY"] <= _son), f"takvim kuralı (≤ {_son})"

d["HEDEF_OLGUN"] = d[HEDEF_HAM].where(_mask)   # olgun değilse NaN
HEDEF = "HEDEF_OLGUN"

print("=" * 110)
print("12.0  HEDEF VE OLGUNLUK")
print("=" * 110)
print(f"ham hedef        : {HEDEF_HAM}")
print(f"olgunluk kaynağı : {_kaynak}")
print(f"  olgun          : {d[HEDEF].notna().sum():2d} kohort  "
      f"({d.loc[d[HEDEF].notna(),'AY'].min()} – {d.loc[d[HEDEF].notna(),'AY'].max()})")
print(f"  olgunlaşmamış  : {d[HEDEF].isna().sum():2d} kohort  "
      f"({d.loc[d[HEDEF].isna(),'AY'].min()} – {d.loc[d[HEDEF].isna(),'AY'].max()})")

_olg = d[d[HEDEF].notna()].sort_values("AY")[HEDEF].tail(12).mean()
_gel = d.loc[d[HEDEF].isna(), ["AY", HEDEF_HAM]].copy()
if len(_gel):
    _gel["YAS_AY"]  = [(BUGUN - a).n for a in _gel["AY"]]
    _gel["SANSUR_%"] = 100 * _gel[HEDEF_HAM] / _olg
    _gel["AY"] = _gel["AY"].astype(str)
    print("\nTahmin edilecek kohortlar (ham değerleri sansürlü, kullanılmayacak):")
    print(_gel.round(4).to_string(index=False))
    print(f"  karşılaştırma tabanı = son 12 olgun kohort ortalaması = {_olg:.4f}")

# --- BÖLGELER -----------------------------------------------------------
_BOLGE_VAR = ("BOLGE" in d.columns and
              set(["TRAIN", "ARA", "OOT"]) <= set(d["BOLGE"].dropna().astype(str).unique()))
SINIR = {"TRAIN": (P("2021-01"), P("2024-03")),
         "ARA":   (P("2024-04"), P("2024-09")),
         "OOT":   (P("2024-10"), P("2025-09"))}

def bolge_idx(ad):
    if _BOLGE_VAR:
        m = (d["BOLGE"].astype(str) == ad) & d[HEDEF].notna()
    else:
        a, b = SINIR[ad]
        m = (d["AY"] >= a) & (d["AY"] <= b) & d[HEDEF].notna()
    return d.index[m]

print(f"\nbölge kaynağı: {'d[BOLGE] kolonu' if _BOLGE_VAR else 'SINIR sözlüğü (elle)'}")
for b in ("TRAIN", "ARA", "OOT"):
    ix = bolge_idx(b)
    print(f"  {b:5s}: {len(ix):2d} kohort  {d.loc[ix,'AY'].min()} – {d.loc[ix,'AY'].max()}")

# --- GERİYE DÖNÜK SIZINTI KONTROLÜ --------------------------------------
print("\n--- eğitim evreni nasıl tanımlanmış (notna mı, OLGUN mu) ---")
for fn in ("wf_pencereli", "fold_olustur", "_agirlik"):
    if fn in globals() and callable(globals()[fn]):
        try:
            for l in inspect.getsource(globals()[fn]).split("\n"):
                if re.search(r"notna|isna|dropna|OLGUN|HEDEF_KULLAN", l):
                    print(f"  {fn}: {l.strip()}")
        except Exception:
            pass

# ============================================================
# METRİK SETİ
# ============================================================
def metrikler(y, yhat, w=None):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    w = np.ones(len(y)) if w is None else np.asarray(w, float)[ok]
    anahtar = ("N","MAE","MSE","RMSE","MAPE_%","WMAPE_%","SAPMA",
               "BIAS_ORANI","PEARSON","SPEARMAN","R2","ARTI")
    if len(y) == 0:
        return {k: np.nan for k in anahtar}
    e = yhat - y
    return {
        "N":          len(y),
        "MAE":        np.mean(np.abs(e)),
        "MSE":        np.mean(e ** 2),
        "RMSE":       np.sqrt(np.mean(e ** 2)),
        "MAPE_%":     100 * np.mean(np.abs(e) / y),
        "WMAPE_%":    100 * np.sum(w * np.abs(e)) / np.sum(w * y),
        "SAPMA":      np.mean(e),                       # + = fazla tahmin
        "BIAS_ORANI": abs(np.mean(e)) / np.mean(np.abs(e)),
        "PEARSON":    stats.pearsonr(y, yhat)[0]  if len(y) > 2 else np.nan,
        "SPEARMAN":   stats.spearmanr(y, yhat)[0] if len(y) > 2 else np.nan,
        "R2":         1 - np.sum(e**2) / np.sum((y - y.mean())**2) if len(y) > 2 else np.nan,
        "ARTI":       int((e > 0).sum()),
    }

def dummy_metrik(idx):
    y = d.loc[idx, HEDEF].values
    w = d.loc[idx, "BASVURU_ADEDI"].values
    return metrikler(y, np.full(len(y), np.nanmean(y)), w)

def _cikar(res, idx):
    """wf_pencereli ne döndürürse döndürsün (index, tahmin) ikilisine çevirir."""
    if isinstance(res, tuple):
        res = res[0]
    if isinstance(res, pd.DataFrame):
        kol = next((c for c in res.columns
                    if str(c).upper() in ("TAHMIN","PRED","YHAT","Y_PRED","TAHMIN_%")), None)
        return res.index, (res[kol] if kol else res.iloc[:, -1]).values
    if isinstance(res, pd.Series):
        return res.index, res.values
    arr = np.ravel(np.asarray(res, float))
    return pd.Index(list(idx))[:len(arr)], arr

# ============================================================
# 12.0c  FINAL YARDIMCISI — BLOK 11 sonuçlarını bul
# ============================================================
print("\n" + "=" * 110)
print("12.0c  BLOK 11 SONUÇ ADAYLARI  (FINAL'ı doldurmak için)")
print("=" * 110)
_ad = []
for _n, _v in list(globals().items()):
    if _n.startswith("_") or _n in ("d", "TABLO", "SERI", "GELECEK", "KATS"):
        continue
    if isinstance(_v, pd.DataFrame) and len(_v) and len(_v) < 5000:
        _k = {str(c).upper() for c in _v.columns}
        if _k & {"WMAPE", "WMAPE_%", "BASAMAK", "AILE", "KOLONLAR", "BILGI_AY", "SKOR"}:
            _ad.append((_n, _v.shape, list(_v.columns)[:10]))
    if isinstance(_v, dict) and _v and all(isinstance(x, dict) for x in list(_v.values())[:3]):
        _ad.append((_n, f"dict[{len(_v)}]", list(_v.keys())[:8]))
if _ad:
    for _n, _s, _c in _ad:
        print(f"  {_n:24s} {str(_s):16s} {_c}")
    print("\n→ Bu değişkenlerden hangisi BLOK 11'in nihai seçimiyse içeriğini yazdır:")
    print("     print(<ad>.to_string())   /   <ad>")
else:
    print("  aday bulunamadı — BLOK 11'i yeniden çalıştırman gerekebilir.")

# ============================================================
# FINAL — BLOK 11'in seçtiği nihai kurulumlar
# ============================================================
FINAL = {
    # ŞABLON — gerçek değerlerle doldur:
    # "ULTRA_ERKEN_1M": dict(aile="ULTRA_ERKEN", bilgi_ay=1,
    #                        kolonlar=["KK_TAKIP_20_2_TUTARSAL"]),
    # "ERKEN_2M":       dict(aile="ERKEN", bilgi_ay=2, max_gecmis=30,
    #                        kolonlar=["KK_TAKIP_30_2_TUTARSAL","KK_TAKIP_20_2_TUTARSAL_MA3"]),
    # "ERKEN_3M":       dict(aile="ERKEN", bilgi_ay=3, max_gecmis=30,
    #                        kolonlar=["KK_TAKIP_60_3_TUTARSAL"]),
    # "ORTA_4M":        dict(aile="ORTA",  bilgi_ay=4,
    #                        kolonlar=["KK_TAKIP_30_4_TUTARSAL"]),
    # "OLGUN_6M":       dict(aile="OLGUN", bilgi_ay=6, max_gecmis=30,
    #                        kolonlar=["KK_TAKIP_30_6_TUTARSAL","KK_TAKIP_20_2_TUTARSAL_MA3"],
    #                        ek_kolonlar=["MM_SKOR","SIN_AY","COS_AY"],
    #                        oran_paydasi="KK_TAKIP_30_2_TUTARSAL",
    #                        duzelt_pencere=6),
}
SAMPIYON = "OLGUN_6M"

# ============================================================
if not FINAL:
    print("\n" + "!" * 110)
    print("FINAL boş — 12.1–12.4 atlandı.")
    print("Yukarıdaki aday değişkenlerden nihai seçimi yazdırıp FINAL'ı doldur,")
    print("ya da o çıktıyı paylaş; FINAL hazır halde geri gelsin.")
    print("!" * 110)
else:
    # ========================================================
    # 12.1  BASAMAK × BÖLGE METRİK TABLOSU
    # ========================================================
    _satir = []
    for ad, kw in FINAL.items():
        for bolge in ("TRAIN", "ARA", "OOT"):
            idx = bolge_idx(bolge)
            try:
                ridx, yh = _cikar(wf_pencereli(hedef_idx=idx, **kw), idx)
                m = metrikler(d.loc[ridx, HEDEF].values, yh,
                              d.loc[ridx, "BASVURU_ADEDI"].values)
            except Exception as ex:
                print(f"  [hata] {ad} / {bolge}: {type(ex).__name__}: {ex}")
                m = metrikler([], [])
            dm = dummy_metrik(idx)
            m["REF_WMAPE_%"] = dm["WMAPE_%"]
            m["REF_ORANI"]   = m["WMAPE_%"] / dm["WMAPE_%"] if dm["WMAPE_%"] else np.nan
            _satir.append({"BASAMAK": ad, "BILGI_AY": kw.get("bilgi_ay"),
                           "BOLGE": bolge, **m})
    TABLO = pd.DataFrame(_satir)

    print("\n" + "=" * 110)
    print("12.1  TAM METRİK TABLOSU        (REF_ORANI < 1  →  referanstan iyi)")
    print("=" * 110)
    print(TABLO.to_string(index=False))

    print("\n--- Kilitli test (OOT) özeti ---")
    print(TABLO[TABLO.BOLGE == "OOT"][
        ["BASAMAK","BILGI_AY","N","WMAPE_%","MAPE_%","MAE","RMSE","MSE","SAPMA",
         "BIAS_ORANI","PEARSON","SPEARMAN","R2","ARTI","REF_WMAPE_%","REF_ORANI"]
    ].to_string(index=False))

    _piv = TABLO.pivot_table(index="BASAMAK", columns="BOLGE", values="WMAPE_%")
    _piv = _piv.reindex(columns=["TRAIN","ARA","OOT"])
    _piv["OVERFIT_ORANI"] = _piv["OOT"] / _piv["TRAIN"]
    print("\n--- Bölgeler arası WMAPE ve overfit oranı ---")
    print(_piv.to_string())

    # ========================================================
    # 12.2  ŞAMPİYONUN FORMÜLÜ VE KATSAYILARI
    # ========================================================
    kw    = FINAL[SAMPIYON]
    kols  = list(kw["kolonlar"]) + list(kw.get("ek_kolonlar") or [])
    payda = kw.get("oran_paydasi")

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
        print("\n(uyarı: pipeline kurucusu bulunamadı → varsayılan Ridge kuruldu)")

    tr = d[d[HEDEF].notna()].sort_values("AY")          # artık yalnız OLGUN kohortlar
    if kw.get("max_gecmis"):
        tr = tr.iloc[-int(kw["max_gecmis"]):]
    X = tr[kols]
    y = tr[HEDEF] / tr[payda] if payda else tr[HEDEF]
    w = _agirlik(tr.index) if "_agirlik" in globals() else np.ones(len(tr))
    (_fit_agirlikli(pipe, X, y, w, kw["aile"]) if "_fit_agirlikli" in globals()
     else pipe.fit(X, y))

    def katsayilar(pipe, kols):
        m  = getattr(pipe, "regressor_", pipe)
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
        K["YON"] = np.where(K.KATSAYI_STD >= 0, "artirir", "azaltir")
        return K.sort_values("ETKI_%", ascending=False).reset_index(drop=True), sabit, reg

    KATS, SABIT, _reg = katsayilar(pipe, kols)
    print("\n" + "=" * 110)
    print(f"12.2  ŞAMPİYON: {SAMPIYON}   eğitim: {tr.AY.min()} – {tr.AY.max()}  "
          f"({len(tr)} kohort)")
    print("=" * 110)
    print(KATS.to_string(index=False))
    print(f"\nSABIT = {SABIT:.6f}"
          + (f"   |   alpha = {getattr(_reg,'alpha_','n/a')}" if hasattr(_reg,"alpha_") else ""))
    print("\nFORMÜL (ham ölçekte):")
    print(f"   ham = {SABIT:+.6f}")
    for _k, _b in zip(KATS.DEGISKEN, KATS.KATSAYI_HAM):
        print(f"         {'+' if _b >= 0 else '-'} {abs(_b):.6f} × {_k}")
    if payda:
        print(f"   tahmin = ham × {payda}            ← oransal hedef")
    if kw.get("duzelt_pencere"):
        print(f"   tahmin = tahmin − (son {kw['duzelt_pencere']} kohortun ort. sapması)")

    # ========================================================
    # 12.3  GERÇEKLEŞEN vs TAHMİN
    # ========================================================
    _p = []
    for bolge in ("TRAIN", "ARA", "OOT"):
        idx = bolge_idx(bolge)
        ridx, yh = _cikar(wf_pencereli(hedef_idx=idx, **kw), idx)
        _p.append(pd.DataFrame({"AY": d.loc[ridx, "AY"].astype(str).values,
                                "BOLGE": bolge,
                                "GERCEK": d.loc[ridx, HEDEF].values,
                                "TAHMIN": yh}))
    SERI = pd.concat(_p).sort_values("AY").reset_index(drop=True)
    SERI["HATA"]   = SERI.TAHMIN - SERI.GERCEK
    SERI["HATA_%"] = 100 * SERI.HATA / SERI.GERCEK

    print("\n" + "=" * 110)
    print("12.3  GERÇEKLEŞEN vs TAHMİN (tüm kohortlar)")
    print("=" * 110)
    print(SERI.to_string(index=False))
    print("\n--- bölge özeti ---")
    print(SERI.groupby("BOLGE").agg(
        N=("GERCEK","size"), GERCEK_ORT=("GERCEK","mean"), TAHMIN_ORT=("TAHMIN","mean"),
        SAPMA=("HATA","mean"), MAE=("HATA", lambda s: s.abs().mean()),
        ARTI=("HATA", lambda s: int((s > 0).sum()))
    ).reindex(["TRAIN","ARA","OOT"]).to_string())

    fig, ax = plt.subplots(figsize=(16, 5.5))
    for b, c, lbl in [("ARA", "#FDF1E3", "ARA (hakem)"), ("OOT", "#E7EFEC", "OOT (kilitli)")]:
        s = SERI[SERI.BOLGE == b]
        if len(s):
            ax.axvspan(s.AY.iloc[0], s.AY.iloc[-1], color=c, zorder=0, label=lbl)
    ax.axhline(d.loc[bolge_idx("TRAIN"), HEDEF].mean(), ls=":", lw=1.5,
               color="#B4BCC6", label="Referans (eğitim ort.)")
    ax.plot(SERI.AY, SERI.GERCEK, "o-", lw=2.6, ms=5, color="#16253D", label="Gerçekleşen")
    ax.plot(SERI.AY, SERI.TAHMIN, "s--", lw=2.0, ms=4, color="#C25E00", label="Tahmin")
    ax.set_title(f"12M90 — gerçekleşen vs tahmin  ({SAMPIYON})", fontsize=13)
    ax.set_ylabel("12M90 oranı"); ax.legend(ncol=5, fontsize=9)
    plt.xticks(rotation=90, fontsize=7); plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(figsize=(16, 3))
    ax.bar(SERI.AY, SERI.HATA, color=np.where(SERI.HATA > 0, "#C25E00", "#2E6B5E"))
    ax.axhline(0, color="#16253D", lw=1)
    ax.set_title("Tahmin hatası  (+ fazla tahmin / − eksik tahmin)", fontsize=12)
    plt.xticks(rotation=90, fontsize=7); plt.tight_layout(); plt.show()

    # ========================================================
    # 12.4  GELECEK TAHMİNLERİ
    # ========================================================
    gel = d[d[HEDEF].isna()].sort_values("AY").copy()
    _eksik = gel[kols + ([payda] if payda else [])].isna().any(axis=1)
    if _eksik.any():
        print(f"\n(uyarı: {int(_eksik.sum())} kohortun öncülleri henüz boş, atlandı)")
    gel = gel[~_eksik]

    if len(gel) == 0:
        print("\n12.4  Tahmin edilecek olgunlaşmamış kohort yok.")
    else:
        yg = np.ravel(pipe.predict(gel[kols]))
        if payda:
            yg = yg * gel[payda].values
        duz = 0.0
        if kw.get("duzelt_pencere"):
            duz = SERI[SERI.BOLGE == "OOT"].tail(int(kw["duzelt_pencere"])).HATA.mean()
            yg = yg - duz
        son12 = d[d[HEDEF].notna()].sort_values("AY").tail(12)[HEDEF]
        GELECEK = pd.DataFrame({
            "AY":           gel["AY"].astype(str).values,
            "YAS_AY":       [(BUGUN - a).n for a in gel["AY"]],
            "HAM_SANSURLU": gel[HEDEF_HAM].values,
            "TAHMIN":       yg,
            "SON12_ORT":    son12.mean(),
            "FARK":         yg - son12.mean(),
            "FARK_%":       100 * (yg - son12.mean()) / son12.mean(),
            "HEDEF_ERISIM": [str(a + CONFIG.get("HEDEF_OLGUNLASMA_AY", 12)
                                 + CONFIG.get("RAPOR_GECIKME_AY", 1)) for a in gel["AY"]],
        })
        print("\n" + "=" * 110)
        print(f"12.4  GELECEK TAHMİNLERİ    (sapma düzeltmesi: {duz:+.6f})")
        print("=" * 110)
        print(GELECEK.to_string(index=False))
        print(f"\n  son 12 olgun kohort ortalaması : {son12.mean():.4f}")
        print(f"  tahmin ortalaması              : {np.mean(yg):.4f}")
        print(f"  yön                            : "
              f"{'YUKARI' if np.mean(yg) > son12.mean() else 'AŞAĞI'}")

        fig, ax = plt.subplots(figsize=(16, 5))
        g = d[d[HEDEF].notna()].sort_values("AY")
        ax.plot(g.AY.astype(str), g[HEDEF], "o-", lw=2.4, ms=4,
                color="#16253D", label="Gerçekleşen")
        ax.plot(GELECEK.AY, GELECEK.TAHMIN, "D--", lw=2.4, ms=8,
                color="#C25E00", label="Gelecek tahmini")
        ax.axvline(str(g.AY.iloc[-1]), color="#8A93A0", ls=":", lw=1.5)
        ax.set_title("12M90 — gerçekleşen seri ve olgunlaşmamış kohort tahminleri", fontsize=13)
        ax.set_ylabel("12M90 oranı"); ax.legend()
        plt.xticks(rotation=90, fontsize=7); plt.tight_layout(); plt.show()
