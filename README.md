# %% [1] Konfigürasyon (tüm pipeline buradan yönetilir)
import numpy as np
import pandas as pd

CONFIG = {
    # --- Hedef ---
    "HEDEF_TUR": "TUTARSAL",              # "TUTARSAL" | "ADETSEL"
    "HEDEF_SON": "90_12",

    # --- Bölme (sadece zaman) ---
    # Eski ARA bölgesi geliştirmeye katıldı: CV tabanı 12 -> 20 kohort
    "GELISTIRME_ARALIK": (202101, 202409),   # CV + model seçimi
    "OOT_ARALIK":        (202410, 202509),   # sadece final ölçüm, seçime girmez

    # --- Olgunlaşma ---
    "BILGI_AYI": "2026-10",
    "HEDEF_OLGUNLASMA_AY": 12,
    "RAPOR_GECIKME_AY": 0,                   # '<' karşılaştırması 1 aylık payı zaten içeriyor

    # --- CV ---
    "N_SPLITS": 5,
    "TEST_SIZE": 4,                          # 5 x 4 = son 20 geliştirme kohortu validasyon
    "MIN_TRAIN": 12,
    "IC_CV_SPLITS": 3,                       # Ridge/Lasso/EN alpha seçimi için iç zaman CV

    # --- Aday havuzu ---
    "SON_ONCU_AY": 6,
    "CHALLENGER_MAX_AY": 3,                  # erken model sınırı
    "MAX_K": 3,
    "AYNI_AYDAN_TEK_ONCUL": True,            # 30_4 + 60_4 gibi aynı ay çiftlerini taramaz
    "MODEL_AILELERI": ["Linear", "Ridge", "Lasso", "ElasticNet"],
    "TREND_DENE": False,                     # gelecekte doğrusal ekstrapole eder; önceki turda seçilmedi
    "MEVSIM_DENE": True,
    "TREND_ORIJIN": "2021-01",               # sabit orijin (veri dilimine bağlı değil)

    # --- Gözlem ağırlığı ---
    "SAMPLE_WEIGHT_KOLON": None,             # örn. "BASVURU_ADEDI" (BP p=0.088 sınırda)

    # --- Sıralama: GECERLI modeller içinde yüzdelik sıra, küçük = iyi ---
    "AGIRLIK": {
        "CV_WMAPE": 0.30, "EN_KOTU_WMAPE": 0.12, "FOLD_WMAPE_STD": 0.12,
        "OVERFIT_ORANI": 0.15, "DELTA_MAE": 0.12, "MUTLAK_CV_SAPMA": 0.19,
    },
    # --- Geçerlilik eşikleri (Dummy referansına ve hedef seviyesine bağlı) ---
    "GECERLI": {
        "CV_WMAPE_DUMMY_ORAN": 0.50,         # CV_WMAPE <= 0.50 x Dummy CV_WMAPE
        "EN_KOTU_DUMMY_ORAN": 0.75,          # EN_KOTU_WMAPE <= 0.75 x Dummy EN_KOTU_WMAPE
        "OVERFIT_ORANI_MAX": 2.0,            # CV hatası <= 2 x eğitim hatası
        "SAPMA_HEDEF_ORAN": 0.15,            # |CV_SAPMA| <= 0.15 x ortalama hedef
    },
    "YON_TOLERANSI": "oto",                  # "oto" = 0.5 x std(ardışık hedef farkı) | sayı

    # --- Optuna (opsiyonel) ---
    "OPTUNA_CALISTIR": False,
    "OPTUNA_TRIALS": 60,
    "OPTUNA_SET_SAYISI": 20,

    # --- Sapma düzeltmesi ---
    "SAPMA_K_ADAYLARI": [0, 3, 6, 9, 12],    # 0 = düzeltme yok; k geliştirme döneminde seçilir
    "SAPMA_K_SABIT": None,                   # None = otomatik; sayı verilirse k elle sabitlenir

    # --- Ortalama limit ---
    "LIMIT_KOLON": "TOPLAM_ONAYLANAN_LIMIT",
    "ONAY_ADET_KOLON": "ONAYLANAN_ADET",     # <-- veri setindeki gerçek kolon adıyla değiştir
    "TUFE_KOLON": None,                      # varsa reel limit: limit / TÜFE * 100
}

# Ortak (hedef türünden bağımsız) aday alanlar
# BASVURU_ADEDI ve TOPLAM_ONAYLANAN_LIMIT ölçek değişkeni olduğu için çıkarıldı -> ORTALAMA_LIMIT
ORTAK_ALANLAR = [
    "KKB_SKOR_00120",
    "MM_SKOR",
    "KALIBRE_PD_AGIRLIKLI",
    "ORTALAMA_LIMIT",
]

# Öncül kodları (90+ HARİÇ): gün_ay
ONCU_KODLARI = [
    "20_2", "30_2",
    "30_3", "60_3",
    "30_4", "60_4",
    "30_5", "60_5",
    "30_6", "60_6",
]

def oncul_kolon(kod, tur):
    return f"KK_TAKIP_{kod}_{tur}"

def oncul_ay(kod):
    return int(kod.split("_")[1])

def kisa_ad(kolon):
    return kolon.replace("KK_TAKIP_", "").replace(f"_{CONFIG['HEDEF_TUR']}", "")

print("Ağırlık toplamı:", round(sum(CONFIG["AGIRLIK"].values()), 4))


# %% [2] Veri Hazırlama, Olgunlaşma ve Bölme
def veri_hazirla(df_raw, config):
    tur = config["HEDEF_TUR"]
    hedef = f"KK_TAKIP_{config['HEDEF_SON']}_{tur}"
    onculler = [oncul_kolon(k, tur) for k in ONCU_KODLARI]
    girdiler = onculler + ORTAK_ALANLAR

    # --- Ortalama limit paydası ---
    adet_kol = config["ONAY_ADET_KOLON"]
    if adet_kol not in df_raw.columns:
        print(f"UYARI: '{adet_kol}' yok -> payda olarak BASVURU_ADEDI kullanılıyor "
              "(değişken onay oranını da taşır).")
        adet_kol = "BASVURU_ADEDI"

    ham_ortak = [c for c in ORTAK_ALANLAR if c != "ORTALAMA_LIMIT"]
    ek = [config["LIMIT_KOLON"], adet_kol]
    for k in ["SAMPLE_WEIGHT_KOLON", "TUFE_KOLON"]:
        if config[k] is not None:
            ek.append(config[k])
    gerekli = list(dict.fromkeys(["DONEM_1", hedef, *onculler, *ham_ortak, *ek]))

    eksik = sorted(set(gerekli) - set(df_raw.columns))
    if eksik:
        raise ValueError(f"Eksik kolon: {eksik}")
    if any("_90_" in k for k in girdiler):
        raise ValueError("90+ alanı girdi olarak kullanılamaz.")

    d = df_raw[gerekli].copy()
    d["DONEM_1"] = pd.to_numeric(d["DONEM_1"], errors="raise").astype(int)
    d["AY"] = pd.to_datetime(d["DONEM_1"].astype(str), format="%Y%m").dt.to_period("M")
    d = d.sort_values("AY").reset_index(drop=True)
    if d["AY"].duplicated().any():
        raise ValueError("Tekrarlanan kohort dönemi var.")

    # --- Ay sürekliliği (tüm veri) ---
    tam = pd.period_range(d["AY"].min(), d["AY"].max(), freq="M")
    if len(tam) != len(d):
        raise ValueError(f"Eksik ay: {sorted(set(tam.astype(str)) - set(d['AY'].astype(str)))}")

    sayisal = [c for c in gerekli if c != "DONEM_1"]
    d[sayisal] = d[sayisal].apply(pd.to_numeric, errors="raise")

    # --- Ortalama limit (onaylanan adede bölünmüş, opsiyonel reel) ---
    payda = d[adet_kol].where(d[adet_kol] > 0)
    d["ORTALAMA_LIMIT"] = d[config["LIMIT_KOLON"]] / payda
    if config["TUFE_KOLON"] is not None:
        d["ORTALAMA_LIMIT"] = d["ORTALAMA_LIMIT"] / d[config["TUFE_KOLON"]] * 100

    # --- Mühendislik alanları (sabit orijin) ---
    orijin = pd.Period(config["TREND_ORIJIN"], freq="M")
    d["TREND_IDX"] = d["AY"].apply(lambda p: (p - orijin).n).astype(float)
    d["SIN_AY"] = np.sin(2 * np.pi * d["AY"].dt.month / 12)
    d["COS_AY"] = np.cos(2 * np.pi * d["AY"].dt.month / 12)

    # --- Hedef olgunlaşması ---
    bilgi = pd.Period(config["BILGI_AYI"], freq="M")
    gecikme = config["RAPOR_GECIKME_AY"]
    d["HEDEF_ERISIM"] = d["AY"] + config["HEDEF_OLGUNLASMA_AY"] + gecikme
    d["OLGUN"] = d["HEDEF_ERISIM"] < bilgi
    hedef_gecerli = d[hedef].notna() & np.isfinite(d[hedef]) & d[hedef].between(0, 1)
    d["HEDEF_KULLANILABILIR"] = d["OLGUN"] & hedef_gecerli
    d["GOZLENEN_AY"] = d["AY"].apply(lambda p: (bilgi - p).n) - 1 - gecikme

    # --- Öncül olgunluğu: kısmi değerleri kullanma ---
    kismi = {}
    for k, kol in zip(ONCU_KODLARI, onculler):
        tam_mi = (d["AY"] + oncul_ay(k) + gecikme) < bilgi
        n = int((~tam_mi & d[kol].notna()).sum())
        if n:
            kismi[kol] = n
        d.loc[~tam_mi, kol] = np.nan
    if kismi:
        print("Kısmi öncül değeri NaN yapıldı:", {kisa_ad(k): v for k, v in kismi.items()})

    # --- Oran girdileri 0-1 ---
    kotu = ((d[onculler] < 0) | (d[onculler] > 1)).sum().sum()
    if kotu:
        raise ValueError(f"0-1 dışında {int(kotu)} oran girdisi var.")

    # --- Olgun kohortlarda boş girdi ---
    na = d.loc[d["HEDEF_KULLANILABILIR"], girdiler].isna().sum()
    if na.any():
        print("UYARI - olgun kohortlarda boş girdi:\n", na[na > 0].to_string())

    return d, hedef, onculler, girdiler


def bolme_uygula(d, config):
    """Dönenler: geliştirme ve OOT indeksleri (yalnızca kullanılabilir hedefler)."""
    kullan = d[d["HEDEF_KULLANILABILIR"]]
    maske = lambda ar: kullan["DONEM_1"].between(*ar)
    gel = kullan.index[maske(config["GELISTIRME_ARALIK"])].to_numpy()
    oot = kullan.index[maske(config["OOT_ARALIK"])].to_numpy()
    disarida = kullan.index.difference(np.concatenate([gel, oot]))
    if len(disarida):
        print("UYARI - bölgeye atanmamış olgun kohort:", list(kullan.loc[disarida, "DONEM_1"]))
    if len(set(gel) & set(oot)):
        raise ValueError("Geliştirme ve OOT çakışıyor.")
    if len(gel) == 0 or len(oot) == 0:
        raise ValueError("Geliştirme veya OOT boş.")
    return gel, oot


d, HEDEF, ONCULLER, GIRDILER = veri_hazirla(df_raw, CONFIG)
gel_idx, oot_idx = bolme_uygula(d, CONFIG)

ozet = pd.DataFrame([
    {"BOLGE": "GELISTIRME (CV)", "N": len(gel_idx),
     "BAS": str(d.loc[gel_idx, "AY"].min()), "SON": str(d.loc[gel_idx, "AY"].max())},
    {"BOLGE": "OOT", "N": len(oot_idx),
     "BAS": str(d.loc[oot_idx, "AY"].min()), "SON": str(d.loc[oot_idx, "AY"].max())},
])
print("Hedef:", HEDEF)
print("Toplam kohort:", len(d), "| Olgun hedef:", int(d["HEDEF_KULLANILABILIR"].sum()),
      "| Olgunlaşmamış:", int((~d["HEDEF_KULLANILABILIR"]).sum()))
display(ozet)


# %% [3] Aday Havuzu (tek havuz, her değişken gözlem ayıyla etiketli)
from itertools import combinations

HAVUZ = ONCULLER + ORTAK_ALANLAR
KOLON_AYI = {oncul_kolon(k, CONFIG["HEDEF_TUR"]): oncul_ay(k) for k in ONCU_KODLARI}

def kolon_ayi(kolon):
    return KOLON_AYI.get(kolon, 0)          # ortak alanlar başvuru anında bilinir: 0

def bilgi_ayi_hesapla(kolonlar):
    """Varyantın tahmin üretebileceği en erken ay = kullandığı en geç öncül."""
    return max([kolon_ayi(c) for c in kolonlar], default=0)

def kombinasyon_uygun(alt):
    if not CONFIG["AYNI_AYDAN_TEK_ONCUL"]:
        return True
    aylar = [kolon_ayi(c) for c in alt if kolon_ayi(c) > 0]
    return len(aylar) == len(set(aylar))

havuz_tablo = pd.DataFrame({"KOLON": HAVUZ, "GOZLEM_AYI": [kolon_ayi(c) for c in HAVUZ]})
kombs = [a for k in range(1, CONFIG["MAX_K"] + 1) for a in combinations(HAVUZ, k) if kombinasyon_uygun(a)]
print(f"Havuz: {len(HAVUZ)} değişken | uygun kombinasyon: {len(kombs)}")
print("Bilgi ayı dağılımı:", pd.Series([bilgi_ayi_hesapla(a) for a in kombs]).value_counts().sort_index().to_dict())
display(havuz_tablo)


# %% [4] Olgunlaşma kontrollü CV + metrik motoru
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, ElasticNetCV

MIN_TRAIN = CONFIG["MIN_TRAIN"]
ALPHALAR = np.logspace(-3, 3, 13)        # iç zaman CV maliyeti için 25 -> 13

cv_df = d.loc[gel_idx].sort_values("AY").reset_index(drop=True)
y_cv = cv_df[HEDEF].to_numpy(dtype=float)
HEDEF_ORT = float(np.mean(y_cv))
TOL = (0.5 * float(np.std(np.diff(y_cv))) if CONFIG["YON_TOLERANSI"] == "oto"
       else float(CONFIG["YON_TOLERANSI"]))
print(f"Yön toleransı: {TOL:.5f} | ortalama hedef: {HEDEF_ORT:.5f}")


def model_fabrika(aile, params=None):
    """Grid aileleri iç alpha seçimini zaman sıralı CV ile yapar (LOO/KFold değil)."""
    p = params or {}
    ic = TimeSeriesSplit(n_splits=CONFIG["IC_CV_SPLITS"])
    if aile == "Dummy":      return DummyRegressor(strategy="mean")
    if aile == "Linear":     return LinearRegression()
    if aile == "Ridge":      return RidgeCV(alphas=ALPHALAR, cv=ic)
    if aile == "Lasso":      return LassoCV(alphas=ALPHALAR, cv=ic, max_iter=100000)
    if aile == "ElasticNet": return ElasticNetCV(alphas=ALPHALAR, l1_ratio=[0.2, 0.5, 0.8],
                                                 cv=ic, max_iter=100000)
    return optuna_estimator(aile, p)          # Bölüm 6'da tanımlı


def agirlik_al(df):
    kol = CONFIG["SAMPLE_WEIGHT_KOLON"]
    return None if kol is None else df[kol].to_numpy(dtype=float)


def pipe_fit(estimator, X, y, w=None):
    p = make_pipeline(StandardScaler(), clone(estimator))
    if w is None:
        return p.fit(X, y)
    return p.fit(X, y, **{f"{p.steps[-1][0]}__sample_weight": w})


def fold_olustur(cv_df, bilgi_ayi, config):
    """Walk-forward + olgunlaşma dışlaması. Validasyon blokları bilgi ayından bağımsız (kıyas adil)."""
    n = len(cv_df)
    aylar = pd.PeriodIndex(cv_df["AY"])
    if not np.all(np.diff(aylar.asi8) == 1):
        raise ValueError("CV ayları kesintisiz olmalı.")
    foldlar, ozet = [], []
    tscv = TimeSeriesSplit(n_splits=config["N_SPLITS"], test_size=config["TEST_SIZE"])
    for i, (tr, val) in enumerate(tscv.split(np.arange(n)), 1):
        ilk_tahmin = aylar[val[0]] + bilgi_ayi                     # bloğun İLK kohortuna göre kes
        erisim = aylar[tr] + config["HEDEF_OLGUNLASMA_AY"] + config["RAPOR_GECIKME_AY"]
        tr = tr[erisim < ilk_tahmin]
        if len(tr) < MIN_TRAIN:
            raise ValueError(f"Fold {i} (bilgi ayı {bilgi_ayi}): eğitim {len(tr)} < {MIN_TRAIN}")
        foldlar.append((tr, val))
        ozet.append({"FOLD": i, "TRAIN_N": len(tr), "TRAIN_SON": str(aylar[tr[-1]]),
                     "TEST_BAS": str(aylar[val[0]]), "TEST_SON": str(aylar[val[-1]]),
                     "TEST_N": len(val)})
    return foldlar, pd.DataFrame(ozet)


FOLD_CACHE = {}
def fold_al(bilgi_ayi):
    if bilgi_ayi not in FOLD_CACHE:
        FOLD_CACHE[bilgi_ayi] = fold_olustur(cv_df, bilgi_ayi, CONFIG)[0]
    return FOLD_CACHE[bilgi_ayi]


def temel_metrikler(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    payda = max(np.sum(np.abs(y_true)), 1e-12)
    return {
        "MSE": float(np.mean(err ** 2)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-12)) * 100),
        "WMAPE": float(np.sum(np.abs(err)) / payda * 100),
        "SAPMA": float(np.mean(err)),
    }


def wmape(y, p):
    return float(np.sum(np.abs(np.asarray(p) - np.asarray(y))) / max(np.sum(np.abs(y)), 1e-12) * 100)


def cv_degerlendir(cv_df, kolonlar, estimator, foldlar):
    y = cv_df[HEDEF].to_numpy(dtype=float)
    aylar = pd.PeriodIndex(cv_df["AY"])
    X = cv_df[list(kolonlar)]
    tum_err, tum_true = [], []
    fold_wmape, fold_sapma, overfit_oran, train_mse = [], [], [], []
    d_true, d_pred, aralik_disi = [], [], 0

    for tr, val in foldlar:
        model = pipe_fit(estimator, X.iloc[tr], y[tr], agirlik_al(cv_df.iloc[tr]))
        pred = model.predict(X.iloc[val])
        train_pred = model.predict(X.iloc[tr])
        err = pred - y[val]

        train_mse.append(np.mean((train_pred - y[tr]) ** 2))
        fold_wmape.append(wmape(y[val], pred))
        fold_sapma.append(float(np.mean(err)))
        overfit_oran.append(fold_wmape[-1] / max(wmape(y[tr], train_pred), 1e-12))
        aralik_disi += int(((pred < 0) | (pred > 2 * y[tr].max())).sum())

        bitisik = np.diff(aylar[val].asi8) == 1                 # farklar sadece fold içinde
        d_true.extend(np.diff(y[val])[bitisik]); d_pred.extend(np.diff(pred)[bitisik])
        tum_err.extend(err); tum_true.extend(y[val])

    tum_err, tum_true = np.asarray(tum_err), np.asarray(tum_true)
    d_true, d_pred = np.asarray(d_true), np.asarray(d_pred)
    cv = temel_metrikler(tum_true, tum_true + tum_err)

    delta_mae = float(np.mean(np.abs(d_pred - d_true))) if len(d_true) else np.nan
    delta_ref = float(np.mean(np.abs(d_true))) if len(d_true) else np.nan
    anlamli = np.abs(d_true) > TOL
    dogru = (np.sign(np.where(np.abs(d_pred) <= TOL, 0, d_pred))[anlamli]
             == np.sign(d_true[anlamli]))
    buyukluk = np.abs(d_true[anlamli])

    return {
        "K": len(kolonlar),
        "CV_RMSE": cv["RMSE"], "CV_MAE": cv["MAE"], "CV_MAPE": cv["MAPE"], "CV_WMAPE": cv["WMAPE"],
        "CV_SAPMA": cv["SAPMA"],
        "MUTLAK_CV_SAPMA": abs(cv["SAPMA"]),
        "MUTLAK_FOLD_SAPMA": float(np.mean(np.abs(fold_sapma))),
        "SAPMA_TEK_YON": int(np.all(np.sign(fold_sapma) == np.sign(fold_sapma[0]))),
        "FOLD_WMAPE_STD": float(np.std(fold_wmape)),
        "EN_KOTU_WMAPE": float(np.max(fold_wmape)),
        "OVERFIT_GAP": cv["RMSE"] - float(np.sqrt(np.mean(train_mse))),     # sadece rapor
        "OVERFIT_ORANI": float(np.median(overfit_oran)),                  # fold içi CV/train WMAPE
        "DELTA_MAE": delta_mae,
        "DELTA_IYILESME": (1 - delta_mae / delta_ref) if delta_ref and delta_ref > 0 else np.nan,
        "YON_BASARISI": float(dogru.mean()) if anlamli.any() else np.nan,              # rapor
        "YON_AGIRLIKLI": float((buyukluk * dogru).sum() / buyukluk.sum()) if anlamli.any() else np.nan,
        "YON_N": int(anlamli.sum()),
        "ARALIK_DISI": aralik_disi,
    }


# --- Fold özetleri (erken ve geç bilgi ayı) ---
for b in sorted({CONFIG["CHALLENGER_MAX_AY"], CONFIG["SON_ONCU_AY"]}):
    print(f"=== Bilgi ayı = {b} ===")
    display(fold_olustur(cv_df, b, CONFIG)[1])

# --- Motor doğrulama + Dummy referansı (eşikler buna bağlı) ---
DUMMY = cv_degerlendir(cv_df, HAVUZ[:1], model_fabrika("Dummy"), fold_al(0))
demo = pd.DataFrame([
    {"ADAY": "Referans_Ortalama (Dummy)", **DUMMY},
    {"ADAY": "Ridge_tum_havuz", **cv_degerlendir(cv_df, HAVUZ, Ridge(alpha=1.0),
                                                  fold_al(bilgi_ayi_hesapla(HAVUZ)))},
])
display(demo.set_index("ADAY").T.round(4))


# %% [5] Kombinasyonlu arama + sıralama + otomatik Champion/Challenger seçimi
import time

def varyant_ekleri(config):
    v = [[]]
    if config["TREND_DENE"]:
        v.append(["TREND_IDX"])
    if config["MEVSIM_DENE"]:
        v.append(["SIN_AY", "COS_AY"])
    if config["TREND_DENE"] and config["MEVSIM_DENE"]:
        v.append(["TREND_IDX", "SIN_AY", "COS_AY"])
    return v


def grid_arama(kombs, config):
    kayit, hatalar = [], []
    for alt in kombs:
        b = bilgi_ayi_hesapla(alt)
        foldlar = fold_al(b)
        for ek in varyant_ekleri(config):
            kolonlar = list(alt) + ek
            for aile in config["MODEL_AILELERI"]:
                try:
                    s = cv_degerlendir(cv_df, kolonlar, model_fabrika(aile), foldlar)
                except Exception as e:
                    hatalar.append({"GIRDILER": ", ".join(alt), "MODEL": aile, "HATA": repr(e)[:150]})
                    continue
                s.update({"MODEL": aile, "PARAMS": None, "BILGI_AY": b,
                          "GIRDILER": ", ".join(kisa_ad(c) for c in alt),
                          "KOLONLAR": tuple(kolonlar), "K": len(alt),
                          "TREND": int("TREND_IDX" in kolonlar), "MEVSIM": int("SIN_AY" in kolonlar)})
                kayit.append(s)
    return pd.DataFrame(kayit), pd.DataFrame(hatalar)


def rank_ekle(df, config):
    g = config["GECERLI"]
    df = df.copy()
    df["GECERLI"] = (
        (df["CV_WMAPE"] <= g["CV_WMAPE_DUMMY_ORAN"] * DUMMY["CV_WMAPE"])
        & (df["EN_KOTU_WMAPE"] <= g["EN_KOTU_DUMMY_ORAN"] * DUMMY["EN_KOTU_WMAPE"])
        & (df["OVERFIT_ORANI"] <= g["OVERFIT_ORANI_MAX"])
        & (df["MUTLAK_CV_SAPMA"] <= g["SAPMA_HEDEF_ORAN"] * HEDEF_ORT)
        & (df["ARALIK_DISI"] == 0)
    )
    # Normalizasyon SADECE geçerli modeller içinde, min-max yerine yüzdelik sıra
    df["RANK_SKORU"] = np.nan
    m = df["GECERLI"]
    if m.any():
        skor = 0.0
        for kol, w in config["AGIRLIK"].items():
            s = df.loc[m, kol].astype(float)
            if kol == "OVERFIT_ORANI":
                s = s.clip(lower=1.0)              # CV'nin eğitimden iyi çıkması ödül değil
            skor = skor + w * s.rank(pct=True).fillna(1.0)
        df.loc[m, "RANK_SKORU"] = skor
    return (df.sort_values(["GECERLI", "RANK_SKORU", "CV_WMAPE"], ascending=[False, True, True])
              .reset_index(drop=True))


def spec_yap(satir, rol):
    return {"rol": rol, "kolonlar": list(satir["KOLONLAR"]), "aile": satir["MODEL"],
            "params": satir["PARAMS"], "bilgi_ay": int(satir["BILGI_AY"]),
            "ad": f"{rol} ({satir['GIRDILER']} | {satir['MODEL']} | {int(satir['BILGI_AY'])}. ay)"}


def secim_yap(grid, config):
    """Champion = genel 1.; Challenger = diğer gruptaki (erken <=3 ay / geç >3 ay) en iyi."""
    gec = grid[grid["GECERLI"]]
    if gec.empty:
        raise ValueError("Geçerli model yok: GECERLI eşiklerini kontrol et.")
    champ = gec.iloc[0]
    erken = champ["BILGI_AY"] <= config["CHALLENGER_MAX_AY"]
    diger = gec[(gec["BILGI_AY"] > config["CHALLENGER_MAX_AY"]) if erken
                else (gec["BILGI_AY"] <= config["CHALLENGER_MAX_AY"])]
    if len(diger):
        chal = diger.iloc[0]
    else:
        print("UYARI: diğer grupta (erken/geç) geçerli model yok -> Challenger = genel 2.")
        chal = gec.iloc[1]
    return {s["ad"]: s for s in [spec_yap(champ, "CHAMPION"), spec_yap(chal, "CHALLENGER")]}


t0 = time.time()
grid_ham, grid_hata = grid_arama(kombs, CONFIG)
grid = rank_ekle(grid_ham, CONFIG)
print(f"{len(grid)} varyant | geçerli {int(grid['GECERLI'].sum())} | hata {len(grid_hata)} | "
      f"{time.time() - t0:.0f} sn")
if len(grid_hata):
    display(grid_hata.head(10))

goster = ["GIRDILER", "MODEL", "BILGI_AY", "K", "MEVSIM", "CV_WMAPE", "EN_KOTU_WMAPE",
          "FOLD_WMAPE_STD", "OVERFIT_ORANI", "DELTA_MAE", "CV_SAPMA", "SAPMA_TEK_YON",
          "YON_AGIRLIKLI", "GECERLI", "RANK_SKORU"]
fmt = {c: "{:.4f}" for c in ["OVERFIT_ORANI", "DELTA_MAE", "CV_SAPMA", "YON_AGIRLIKLI", "RANK_SKORU"]}
fmt.update({c: "{:.2f}" for c in ["CV_WMAPE", "EN_KOTU_WMAPE", "FOLD_WMAPE_STD"]})

print("=== Genel ilk 12 ===")
display(grid[goster].head(12).style.format(fmt, decimal=","))
print(f"=== Erken (<= {CONFIG['CHALLENGER_MAX_AY']} ay) ilk 5 ===")
display(grid[grid["BILGI_AY"] <= CONFIG["CHALLENGER_MAX_AY"]][goster].head(5).style.format(fmt, decimal=","))
print(f"=== Geç (> {CONFIG['CHALLENGER_MAX_AY']} ay) ilk 5 ===")
display(grid[grid["BILGI_AY"] > CONFIG["CHALLENGER_MAX_AY"]][goster].head(5).style.format(fmt, decimal=","))

SECIMLER = secim_yap(grid, CONFIG)
for ad in SECIMLER:
    print("Seçilen:", ad)


# %% [6] Optuna ile geniş model aileleri (opsiyonel)
from sklearn.linear_model import Lasso, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

def optuna_estimator(aile, p):
    if aile == "Ridge*":      return Ridge(alpha=p["alpha"])
    if aile == "Lasso*":      return Lasso(alpha=p["alpha"], max_iter=100000)
    if aile == "ElasticNet*": return ElasticNet(alpha=p["alpha"], l1_ratio=p["l1"], max_iter=100000)
    if aile == "Huber":       return HuberRegressor(epsilon=p["eps"], alpha=p["alpha"], max_iter=2000)
    if aile == "SVR":         return SVR(kernel=p["kernel"], C=p["C"], epsilon=p["eps"])
    if aile == "ExtraTrees":  return ExtraTreesRegressor(n_estimators=300, max_depth=p["depth"],
                                                         min_samples_leaf=p["leaf"], random_state=42)
    if aile == "GBoost":      return GradientBoostingRegressor(n_estimators=p["n"], learning_rate=p["lr"],
                                                               max_depth=p["depth"], random_state=42)
    raise ValueError(aile)


if CONFIG["OPTUNA_CALISTIR"]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    setler = (grid[grid["GECERLI"]].drop_duplicates("KOLONLAR")["KOLONLAR"]
              .head(CONFIG["OPTUNA_SET_SAYISI"]).tolist())
    opt_kayit = []

    def amac(trial):
        kolonlar = list(setler[trial.suggest_int("set", 0, len(setler) - 1)])
        aile = trial.suggest_categorical("aile", ["Ridge*", "Lasso*", "ElasticNet*", "Huber",
                                                  "SVR", "ExtraTrees", "GBoost"])
        p = {}
        if aile in ("Ridge*", "Lasso*", "ElasticNet*", "Huber"):
            p["alpha"] = trial.suggest_float(f"{aile}_alpha", 1e-4, 1e2, log=True)
        if aile == "ElasticNet*": p["l1"] = trial.suggest_float("l1", 0.05, 0.95)
        if aile == "Huber":       p["eps"] = trial.suggest_float("h_eps", 1.1, 2.5)
        if aile == "SVR":
            p["kernel"] = trial.suggest_categorical("kernel", ["linear", "rbf"])
            p["C"] = trial.suggest_float("C", 1e-2, 1e2, log=True)
            p["eps"] = trial.suggest_float("svr_eps", 1e-5, 1e-3, log=True)
        if aile == "ExtraTrees":
            p["depth"] = trial.suggest_int("et_depth", 2, 4); p["leaf"] = trial.suggest_int("leaf", 2, 6)
        if aile == "GBoost":
            p["n"] = trial.suggest_int("n", 50, 300); p["lr"] = trial.suggest_float("lr", 0.01, 0.2, log=True)
            p["depth"] = trial.suggest_int("gb_depth", 1, 3)
        b = bilgi_ayi_hesapla([c for c in kolonlar if c in KOLON_AYI])
        s = cv_degerlendir(cv_df, kolonlar, optuna_estimator(aile, p), fold_al(b))
        s.update({"MODEL": aile, "PARAMS": p, "BILGI_AY": b,
                  "GIRDILER": ", ".join(kisa_ad(c) for c in kolonlar if c not in ("SIN_AY", "COS_AY", "TREND_IDX")),
                  "KOLONLAR": tuple(kolonlar), "K": len([c for c in kolonlar if c in HAVUZ]),
                  "TREND": int("TREND_IDX" in kolonlar), "MEVSIM": int("SIN_AY" in kolonlar)})
        opt_kayit.append(s)
        return s["CV_WMAPE"], (s["DELTA_MAE"] if np.isfinite(s["DELTA_MAE"]) else 1.0)

    study = optuna.create_study(directions=["minimize", "minimize"],
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(amac, n_trials=CONFIG["OPTUNA_TRIALS"])

    grid = rank_ekle(pd.concat([grid_ham, pd.DataFrame(opt_kayit)], ignore_index=True), CONFIG)
    print(f"Optuna sonrası: {len(grid)} varyant | geçerli {int(grid['GECERLI'].sum())}")
    display(grid[goster].head(12).style.format(fmt, decimal=","))
    SECIMLER = secim_yap(grid, CONFIG)
    for ad in SECIMLER:
        print("Seçilen:", ad)
else:
    print("Optuna atlandı (CONFIG['OPTUNA_CALISTIR'] = False).")


# %% [7] OOT walk-forward + sapma düzeltmesi + ham formüller
olgun_df = d[d["HEDEF_KULLANILABILIR"]].copy()
OOT_SET = set(oot_idx)

def wf_tahmin(spec):
    """Her olgun kohort, KENDİ tahmin ayında bilinen olgun kohortlarla eğitilen modelle tahmin edilir."""
    kol, b = spec["kolonlar"], spec["bilgi_ay"]
    kayit = []
    for i in olgun_df.index:
        c = olgun_df.loc[i]
        tahmin_ay = c["AY"] + b
        eg = olgun_df[olgun_df["HEDEF_ERISIM"] < tahmin_ay]
        if len(eg) < MIN_TRAIN:
            continue
        m = pipe_fit(model_fabrika(spec["aile"], spec["params"]), eg[kol], eg[HEDEF], agirlik_al(eg))
        kayit.append({"DONEM_1": int(c["DONEM_1"]), "AY": c["AY"],
                      "BOLGE": "OOT" if i in OOT_SET else "GELISTIRME",
                      "TAHMIN_AY": tahmin_ay, "HEDEF_ERISIM": c["HEDEF_ERISIM"],
                      "GERCEK": float(c[HEDEF]),
                      "HAM": float(m.predict(olgun_df.loc[[i], kol])[0]),
                      "DUMMY": float(eg[HEDEF].mean()), "EGITIM_N": len(eg)})
    w = pd.DataFrame(kayit)
    w["ARTIK"] = w["HAM"] - w["GERCEK"]
    return w


def sapma_ekle(w, k):
    """Düzeltme = tahmin anında hedefi belli olan son k kohortun walk-forward artık ortalaması."""
    duz = []
    for _, r in w.iterrows():
        if k == 0:
            duz.append(r["HAM"]); continue
        bilinen = w[w["HEDEF_ERISIM"] < r["TAHMIN_AY"]].tail(k)
        duz.append(r["HAM"] - bilinen["ARTIK"].mean() if len(bilinen) == k else np.nan)
    w[f"DUZ_{k}"] = np.clip(duz, 0, None)
    return w


oot_ozet, k_tablo, WF, K_SECIM = [], [], {}, {}
for ad, spec in SECIMLER.items():
    w = wf_tahmin(spec)
    for k in CONFIG["SAPMA_K_ADAYLARI"]:
        w = sapma_ekle(w, k)
    kk = [f"DUZ_{k}" for k in CONFIG["SAPMA_K_ADAYLARI"]]
    ortak = w[kk].notna().all(axis=1)

    # k SADECE geliştirme döneminde seçilir (OOT'ye bakılmaz)
    dev = w[ortak & (w["BOLGE"] == "GELISTIRME")]
    oot = w[w["BOLGE"] == "OOT"].copy()
    for k in CONFIG["SAPMA_K_ADAYLARI"]:
        k_tablo.append({"MODEL": spec["rol"], "k": k, "GEL_N": len(dev),
                        "GEL_WMAPE": wmape(dev["GERCEK"], dev[f"DUZ_{k}"]) if len(dev) else np.nan,
                        "OOT_WMAPE": wmape(oot["GERCEK"], oot[f"DUZ_{k}"])})
    if CONFIG["SAPMA_K_SABIT"] is not None:
        k_sec = CONFIG["SAPMA_K_SABIT"]
        print(f"{spec['rol']}: k elle sabitlendi = {k_sec} (gerekçesi dokümante edilmeli)")
    elif len(dev) >= 6:
        k_sec = min(CONFIG["SAPMA_K_ADAYLARI"], key=lambda k: wmape(dev["GERCEK"], dev[f"DUZ_{k}"]))
    else:
        k_sec = 0
        print(f"UYARI {spec['rol']}: k seçimi için geliştirme geçmişi yetersiz ({len(dev)}) -> k=0")
    K_SECIM[ad] = k_sec
    w["DUZELTILMIS"] = w[f"DUZ_{k_sec}"]
    oot["DUZELTILMIS"] = oot[f"DUZ_{k_sec}"]
    WF[ad] = w

    dm = temel_metrikler(oot["GERCEK"], oot["DUMMY"])
    for etiket, kol in [("HAM", "HAM"), (f"DUZELTILMIS (k={k_sec})", "DUZELTILMIS")]:
        m = temel_metrikler(oot["GERCEK"], oot[kol])
        oot_ozet.append({"MODEL": spec["rol"], "TAHMIN": etiket, "OOT_N": len(oot), **m,
                         "POZITIF_SAPMA": f"{int((oot[kol] > oot['GERCEK']).sum())}/{len(oot)}",
                         "BECERI": 1 - m["WMAPE"] / dm["WMAPE"]})
    oot_ozet.append({"MODEL": spec["rol"], "TAHMIN": "DUMMY (ortalama)", "OOT_N": len(oot), **dm,
                     "POZITIF_SAPMA": f"{int((oot['DUMMY'] > oot['GERCEK']).sum())}/{len(oot)}",
                     "BECERI": 0.0})

print("=== Sapma düzeltmesi: k seçimi (geliştirmede seçilir, OOT sadece rapor) ===")
display(pd.DataFrame(k_tablo).style.format({"GEL_WMAPE": "{:.2f}", "OOT_WMAPE": "{:.2f}"}, decimal=","))
print("=== OOT (walk-forward, doğru olgunlaşma) ===")
display(pd.DataFrame(oot_ozet).style.format(
    {"MSE": "{:.6f}", "RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE": "{:.2f}", "WMAPE": "{:.2f}",
     "SAPMA": "{:+.4f}", "BECERI": "{:.3f}"}, decimal=","))

# --- Üretim modelleri: tüm olgun veriyle fit, ham uzay formülü, güncel sapma düzeltmesi ---
print("\n=== ÜRETİM FORMÜLLERİ (ham uzay) ===")
URETIM = {}
for ad, spec in SECIMLER.items():
    kol = spec["kolonlar"]
    model = pipe_fit(model_fabrika(spec["aile"], spec["params"]), olgun_df[kol], olgun_df[HEDEF],
                     agirlik_al(olgun_df))
    k = K_SECIM[ad]
    w = WF[ad]
    sapma_simdi = float(w.sort_values("HEDEF_ERISIM")["ARTIK"].tail(k).mean()) if k else 0.0
    URETIM[ad] = {"model": model, "kolonlar": kol, "spec": spec, "k": k, "sapma": sapma_simdi}

    print(f"\n{ad}  (N={len(olgun_df)}, k={k}, güncel düzeltme={sapma_simdi:+.5f})")
    reg, sc = model[-1], model[0]
    if hasattr(reg, "coef_"):
        coef_raw = np.ravel(reg.coef_) / sc.scale_
        b0 = float(np.ravel([reg.intercept_])[0] - np.dot(coef_raw, sc.mean_))
        if hasattr(reg, "alpha_"):
            print(f"  alpha = {reg.alpha_:.4g}")
        print(f"  {HEDEF} = {b0 - sapma_simdi:.8g}  (sapma düzeltmesi sabite dahil)", end="")
        for kk_, cc in zip(kol, coef_raw):
            print(f" {cc:+.8g}*{kk_}", end="")
        print()
        URETIM[ad].update({"coef_raw": coef_raw, "intercept_raw": b0})
    else:
        print("  Doğrusal olmayan model: ham formül yok.")

champ_ad = next(a for a, s in SECIMLER.items() if s["rol"] == "CHAMPION")
display(WF[champ_ad][WF[champ_ad]["BOLGE"] == "OOT"][
    ["DONEM_1", "GERCEK", "HAM", "DUZELTILMIS", "ARTIK", "EGITIM_N"]].style.format(
    {"GERCEK": "{:.2%}", "HAM": "{:.2%}", "DUZELTILMIS": "{:.2%}", "ARTIK": "{:+.2%}"}, decimal=","))


# %% [8] İstatistiksel testler + korelasyon + görseller (Champion)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf

spec = SECIMLER[champ_ad]
KOLONLAR = spec["kolonlar"]
Xh = olgun_df[KOLONLAR].astype(float)
yv = olgun_df[HEDEF].astype(float).to_numpy()
Xc = sm.add_constant(Xh.to_numpy())
pct = FuncFormatter(lambda x, _: f"%{x:.2f}".replace(".", ","))

# 8.1 Katsayı anlamlılığı (OLS proxy, HAC: otokorelasyona dayanıklı)
ols = sm.OLS(yv, Xc).fit()
lag = int(np.ceil(4 * (len(yv) / 100) ** (2 / 9)))            # Newey-West kural lag
ols_hac = sm.OLS(yv, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
print(f"=== KATSAYI ANLAMLILIK (OLS proxy, HAC maxlags={lag}) ===")
print(pd.DataFrame({"degisken": ["const"] + KOLONLAR, "katsayi": ols.params, "se_HAC": ols_hac.bse,
                    "t_HAC": ols_hac.tvalues, "p_HAC": ols_hac.pvalues}).round(6).to_string(index=False))

# 8.2 Artık testleri: KONUŞLANAN modelin artıkları
resid = URETIM[champ_ad]["model"].predict(olgun_df[KOLONLAR]) - yv
rho = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
print(f"\nKonuşlanan model ({spec['aile']}) artıkları:")
print(f"  Durbin-Watson : {durbin_watson(resid):.3f}   (2 = otokorelasyon yok)")
print(f"  Lag-1 rho     : {rho:.3f}  -> efektif n ≈ {len(yv) * (1 - rho) / (1 + rho):.1f} / {len(yv)}")
print("  Ljung-Box:\n", acorr_ljungbox(resid, lags=[1, 3, 6], return_df=True).round(4).to_string())
print(f"  Breusch-Pagan p: {het_breuschpagan(resid, Xc)[1]:.4f}")
jb = jarque_bera(resid)
print(f"  Jarque-Bera p  : {jb[1]:.4f}  (çarpıklık {jb[2]:.2f}, basıklık {jb[3]:.2f})")
print(f"  Ramsey RESET p : {linear_reset(ols, power=2, use_f=True).pvalue:.4f}")
print("\nVIF:")
print(pd.DataFrame({"degisken": KOLONLAR,
                    "VIF": [variance_inflation_factor(Xc, i + 1) for i in range(len(KOLONLAR))]}
                   ).round(2).to_string(index=False))

# 8.3 Girdi kayması: OOT ortalaması geliştirme dağılımında kaç std uzakta
drift = pd.DataFrame([{
    "degisken": kisa_ad(c),
    "GEL_ORT": d.loc[gel_idx, c].mean(), "OOT_ORT": d.loc[oot_idx, c].mean(),
    "DRIFT_Z": (d.loc[oot_idx, c].mean() - d.loc[gel_idx, c].mean()) / d.loc[gel_idx, c].std()}
    for c in KOLONLAR + [HEDEF]])
print("\n=== GİRDİ / HEDEF KAYMASI (|Z| > 1 dikkat) ===")
print(drift.round(4).to_string(index=False))

# 8.4 Korelasyon + artık ACF + Q-Q
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
sns.heatmap(olgun_df[KOLONLAR + [HEDEF]].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax[0],
            xticklabels=[kisa_ad(c) for c in KOLONLAR + [HEDEF]],
            yticklabels=[kisa_ad(c) for c in KOLONLAR + [HEDEF]])
ax[0].set_title("Girdi + hedef korelasyonu")
plot_acf(resid, lags=12, ax=ax[1]); ax[1].set_title("Artık ACF (konuşlanan model)")
sm.qqplot(resid, line="s", ax=ax[2]); ax[2].set_title("Artık Q-Q")
plt.tight_layout(); plt.show()

# 8.5 Tamamı out-of-sample: walk-forward ham + düzeltilmiş tahmin
w = WF[champ_ad].copy()
w["TARIH"] = w["AY"].dt.to_timestamp()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
for bolge, renk in [("GELISTIRME", "#e8f0fe"), ("OOT", "#e6f4ea")]:
    s = w[w["BOLGE"] == bolge]
    for ax in (ax1, ax2):
        ax.axvspan(s["TARIH"].min() - pd.Timedelta(days=15), s["TARIH"].max() + pd.Timedelta(days=15),
                   color=renk, zorder=0)
ax1.plot(w["TARIH"], w["GERCEK"] * 100, "o-", color="#202124", lw=2, label="Gerçekleşen")
ax1.plot(w["TARIH"], w["HAM"] * 100, "o--", color="#c5221f", lw=1.5, label="Tahmin (ham)")
ax1.plot(w["TARIH"], w["DUZELTILMIS"] * 100, "s-", color="#188038", lw=1.5, ms=4,
         label=f"Tahmin (sapma düzeltilmiş, k={K_SECIM[champ_ad]})")
ax1.yaxis.set_major_formatter(pct); ax1.grid(alpha=0.25); ax1.legend(loc="upper left")
ax1.set_ylabel(f"{kisa_ad(HEDEF)}")
ax1.set_title(f"{champ_ad} - walk-forward (tüm noktalar out-of-sample) | mavi: geliştirme, yeşil: OOT",
              fontweight="bold")
ax2.bar(w["TARIH"], w["ARTIK"] * 100, width=20, color=np.where(w["ARTIK"] > 0, "#c5221f", "#1a73e8"))
ax2.axhline(0, color="black", lw=0.8); ax2.grid(alpha=0.25)
ax2.set_ylabel("Ham artık (yp)"); ax2.set_xlabel("Kohort ayı")
plt.tight_layout(); plt.show()


# %% [9] Üretim tahminleri (olgunlaşmamış kohortlar)
gelecek = d[~d["HEDEF_KULLANILABILIR"]].sort_values("AY").copy()
tahmin_tablo = gelecek[["DONEM_1", "GOZLENEN_AY"]].reset_index(drop=True)
tahmin_tablo["KISMI_GERCEK"] = gelecek[HEDEF].to_numpy()

for ad, u in URETIM.items():
    rol = u["spec"]["rol"]
    X = gelecek[u["kolonlar"]]
    eksik = X.isna().any(axis=1).to_numpy()
    ham = np.full(len(X), np.nan)
    if (~eksik).any():
        ham[~eksik] = u["model"].predict(X[~eksik])
    tahmin_tablo[f"{rol}_HAM"] = np.clip(ham, 0, 1)
    tahmin_tablo[f"{rol}_DUZ"] = np.clip(ham - u["sapma"], 0, 1)

print(f"=== {kisa_ad(HEDEF)} TAHMİNLERİ (bilgi ayı {CONFIG['BILGI_AYI']}) ===")
for ad, u in URETIM.items():
    print(f"  {u['spec']['rol']}: {ad} | düzeltme {u['sapma']:+.4%}")
display(tahmin_tablo.style.format(
    {**{c: "{:.2%}" for c in tahmin_tablo.columns if c not in ("DONEM_1", "GOZLENEN_AY")},
     "DONEM_1": "{:.0f}"}, decimal=",", na_rep="-"))

# Olgunlaşmaya yakın kohortlarda tahmin - kısmi gerçek (kısmi değer nihaiden küçük/eşit olur)
yakin = tahmin_tablo[tahmin_tablo["GOZLENEN_AY"] >= 9]
if len(yakin):
    print("Olgunluğa yakın kohortlar (>=9 ay): CHAMPION_HAM - kısmi =",
          (yakin["CHAMPION_HAM"] - yakin["KISMI_GERCEK"]).round(4).tolist(),
          "| CHAMPION_DUZ - kısmi =", (yakin["CHAMPION_DUZ"] - yakin["KISMI_GERCEK"]).round(4).tolist())

hist = olgun_df.sort_values("AY")
gt = pd.to_datetime(tahmin_tablo["DONEM_1"].astype(str), format="%Y%m")
fig, ax = plt.subplots(figsize=(15, 5))
ax.axvspan(gt.min() - pd.Timedelta(days=15), gt.max() + pd.Timedelta(days=15), color="#fce8e6",
           zorder=0, label="Tahmin dönemi (olgunlaşmamış)")
ax.plot(hist["AY"].dt.to_timestamp(), hist[HEDEF] * 100, "o-", color="#202124", lw=2, label="Gerçekleşen (olgun)")
ax.plot(gt, tahmin_tablo["KISMI_GERCEK"] * 100, "x:", color="#5f6368", label="Kısmi gerçek (alt sınır)")
for rol, renk, mk in [("CHAMPION", "#c5221f", "D"), ("CHALLENGER", "#1a73e8", "s")]:
    ax.plot(gt, tahmin_tablo[f"{rol}_DUZ"] * 100, f"{mk}-", color=renk, lw=2, ms=6, label=f"{rol} (düzeltilmiş)")
    ax.plot(gt, tahmin_tablo[f"{rol}_HAM"] * 100, f"{mk}--", color=renk, lw=1, ms=4, alpha=0.5, label=f"{rol} (ham)")
ax.yaxis.set_major_formatter(pct); ax.grid(alpha=0.25); ax.legend(loc="upper left", fontsize=8)
ax.set_title(f"{kisa_ad(HEDEF)} - gerçekleşen + tahminler", fontweight="bold"); ax.set_xlabel("Kohort ayı")
plt.tight_layout(); plt.show()


# %% [10] Sonuçları sakla + hedef türünü değiştir
SONUC = globals().get("SONUC", {})
SONUC[CONFIG["HEDEF_TUR"]] = {
    "SECIMLER": SECIMLER, "URETIM": URETIM, "K_SECIM": K_SECIM,
    "grid": grid.copy(), "oot_ozet": pd.DataFrame(oot_ozet), "tahmin": tahmin_tablo.copy(),
}
print("Saklandı:", list(SONUC.keys()))
# ADETSEL için: CONFIG["HEDEF_TUR"] = "ADETSEL" yap, Bölüm 1'deki CONFIG'i ezmeden
# Bölüm 2 -> 9'u sırayla tekrar çalıştır, sonra bu hücreyi tekrar çalıştır.
