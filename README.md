# ============================================================================
# BLOK 0 — KONFIGURASYON VE SOZLESME
#
# AMAC: Tum akisi tek bir sozlukten yonetmek. Boylece "hedefi ADETSEL'e cevir
#       ve ayni akisi tekrar calistir" tek satirla mumkun olur.
#
# MEVCUT AKISA GORE EKLENENLER (gerekcesiyle):
#   LAG_LISTESI         -> DW=0.608 ve OOT'deki 12/12 yanliligin cozumu
#   KKB_MERKEZLE        -> sabit terim 0.0447 (hedefin 2.9 kati) kaldiraci kiriliyor
#   HACIM_AGIRLIK       -> oran verisinde Var ~ p(1-p)/n; BP p=0.0879 sinirdaydi
#   ARA_HAKEM           -> 9260 varyant arasindan secimin yanliligini kirar
#   AGIRLIK['MUTLAK_CV_SAPMA'] -> OOT hatasinin %82'si yanliliktan geliyordu,
#                                 eski siralamada bu metrik HIC YOKTU
#   ESIK sozlugu        -> sihirli sayilar (20/28/0.0025) tek yerde, gerekceli
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 150)

plt.rcParams.update({"figure.dpi": 110, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})

RENK = {"ger":"#202124", "tah":"#c5221f", "tr":"#1a73e8",
        "ara":"#f29900", "oot":"#188038", "gri":"#80868b"}

# ---------------------------------------------------------------- CONFIG
CONFIG = {
    # ---------- HEDEF ----------
    "HEDEF_TUR":  "TUTARSAL",      # "TUTARSAL" | "ADETSEL"  <- tek satirla gecis
    "HEDEF_SON":  "90_12",

    # ---------- OLGUNLASMA (sizinti kontrolunun kalbi) ----------
    # Kohort t'nin hedefi  t + HEDEF_OLGUNLASMA_AY + RAPOR_GECIKME_AY  aninda bilinir.
    "BILGI_AYI":            "2026-10",   # bugun elimizde ne var
    "HEDEF_OLGUNLASMA_AY":  12,
    "RAPOR_GECIKME_AY":     0,           # veri ambarina dusme gecikmesi varsa artir

    # ---------- BOLME ----------
    "BOLME_TURU":   "zaman",             # "zaman" | "random" (random = sadece deney)
    "TRAIN_ARALIK": (202101, 202403),    # grid + Optuna SADECE burada
    "ARA_ARALIK":   (202404, 202409),    # YENI ROL: aday eleme hakemi
    "OOT_ARALIK":   (202410, 202509),    # KILITLI, tek seferde acilir

    # ---------- CV ----------
    "N_SPLITS":  3,
    "TEST_SIZE": 4,
    "MIN_TRAIN": 10,

    # ---------- CHAMPION / CHALLENGER ----------
    # Champion  : <= SON_ONCU_AY aylik tum onculler serbest
    # Challenger: <= CHALLENGER_MAX_AY  (en erken tahmin, bizim asil hedefimiz)
    "SON_ONCU_AY":        6,
    "CHALLENGER_MAX_AY":  3,

    # ---------- MODEL ARAMASI ----------
    "MODEL_AILELERI": ["Linear", "Ridge", "Lasso", "ElasticNet"],
    "MAX_K":          3,            # n_train=39, fold'lar 21-29 -> 3+1 param guvenli ust sinir
    "ALPHALAR":       np.logspace(-3, 3, 25),
    "OPTUNA_TRIALS":  60,
    "RANDOM_STATE":   42,

    # ---------- MUHENDISLIK VARYANTLARI ----------
    "TREND_DENE":  True,            # TREND_IDX: gelecege dogrusal ekstrapole eder, riskli
    "MEVSIM_DENE": True,            # SIN_AY/COS_AY

    # ---------- YENI: GECIKMELI HEDEF ----------
    # Kohort t'yi t+bilgi_ayi aninda tahmin ediyoruz.
    # Kohort t-k'nin hedefi (t-k)+12'de olgunlasir.
    # Kullanilabilirlik sarti:  k >= HEDEF_OLGUNLASMA_AY - bilgi_ayi
    #   bilgi_ayi=3 -> k>=9      bilgi_ayi=6 -> k>=6
    # lag-12 HER IKI durumda da guvenli, ayrica mevsimselligi de tasir.
    "LAG_KULLAN":   True,
    "LAG_LISTESI":  [12],           # istersen [12, 13] dene; [9] sadece bilgi_ayi>=3 icin

    # ---------- YENI: OLCEK VE AGIRLIK ----------
    "KKB_MERKEZLE":  True,          # sabit terimi yorumlanabilir yapar
    "HACIM_AGIRLIK": True,          # sample_weight = BASVURU_ADEDI
    "TURETME_EKLE":  True,          # roll-rate (60_3/30_3 vb.) -> VIF dusurur

    # ---------- SIRALAMA AGIRLIKLARI ----------
    # Eski kurgu: CV_WMAPE .35 | EN_KOTU .15 | FOLD_STD .15 | OVERFIT .15
    #             DELTA_MAE .12 | YON_BASARISI .08      -> YANLILIK HIC YOKTU
    # Yeni kurgu: yanlilik en buyuk ikinci agirlik, YON siralamadan cikarildi
    #             (7 gozleme dayaniyordu, SVR'yi tek gozlemle 4. siraya itmisti)
    "AGIRLIK": {
        "CV_WMAPE":         0.30,   # ortalama hata buyuklugu
        "MUTLAK_CV_SAPMA":  0.20,   # YANLILIK  <-- yeni, kritik
        "OVERFIT_ORANI":    0.15,   # CV_RMSE / train_RMSE  (mutlak fark degil)
        "EN_KOTU_WMAPE":    0.12,   # en kotu fold
        "FOLD_WMAPE_STD":   0.12,   # fold'lar arasi istikrar
        "DELTA_MAE":        0.11,   # degisim (yon/egim) hatasi
    },

    # ---------- GECERLILIK ESIKLERI ----------
    # Referans: Dummy(ortalama) modelin CV_WMAPE'i ~46.7 cikmisti.
    # Esikleri o referansa gore gerekcelendiriyoruz, sihirli sayi birakmiyoruz.
    "ESIK": {
        "CV_WMAPE_MAX":        20.0,    # ~ Dummy'nin %43'u
        "EN_KOTU_WMAPE_MAX":   28.0,    # ~ Dummy'nin %60'i
        "OVERFIT_ORANI_MAX":    2.00,   # CV hatasi train'in 2 katindan fazla olmasin
        "MUTLAK_SAPMA_MAX":     0.0015, # 0.15 puan; OOT'de olculen yanlilik 0.283 idi
        "ARALIK_DISI_MAX":      0,      # [0,1] disina tahmin yasak
        "ISARET_SERISI_YASAK":  True,   # tum fold sapmalari ayni isaretliyse ELE
    },

    # ---------- YON METRIGI (raporlama, siralama disi) ----------
    "YON_TOLERANSI": 0.0005,

    # ---------- ARA HAKEMLIGI ----------
    "ARA_HAKEM":    True,
    "ARA_ADAY_N":   20,    # TRAIN'de ilk N aday ARA'da sinanir, kazanan OOT'ye gider
}

# ---------------------------------------------------------------- SABITLER
# Oncu kodlari: {gecikme}_{ay}. 90_* HEDEF ile ayni olay -> girdi olarak YASAK.
ONCU_KODLARI = ["20_2", "30_2", "30_3", "60_3",
                "30_4", "60_4", "30_5", "60_5", "30_6", "60_6"]

# Basvuru aninda bilinen, olgunluk gerektirmeyen destek degiskenleri
ORTAK_ALANLAR = ["KKB_SKOR_O0120", "MM_SKOR", "KALIBRE_PD_AGIRLIKLI",
                 "BASVURU_ADEDI", "TOPLAM_ONAYLANAN_LIMIT"]

# ---------------------------------------------------------------- YARDIMCILAR
def oncul_kolon(kod, tur=None):
    """'60_3' -> 'KK_TAKIP_60_3_TUTARSAL'"""
    return f"KK_TAKIP_{kod}_{tur or CONFIG['HEDEF_TUR']}"

def oncul_ay(kod):
    """'60_3' -> 3   (oncul kac ay olgunluk gerektiriyor)"""
    return int(kod.split("_")[1])

def oncul_gecikme(kod):
    """'60_3' -> 60  (gecikme gun sayisi)"""
    return int(kod.split("_")[0])

def hedef_kolon(tur=None):
    return f"KK_TAKIP_{CONFIG['HEDEF_SON']}_{tur or CONFIG['HEDEF_TUR']}"

def ay_no(donem):
    """202403 -> mutlak ay indeksi (karsilastirma icin)"""
    d = int(donem)
    return (d // 100) * 12 + (d % 100)

HEDEF = hedef_kolon()

# ---------------------------------------------------------------- DOGRULAMA
def config_dogrula():
    hata = []
    a = CONFIG
    if a["HEDEF_TUR"] not in ("TUTARSAL", "ADETSEL"):
        hata.append("HEDEF_TUR 'TUTARSAL' veya 'ADETSEL' olmali.")
    if a["CHALLENGER_MAX_AY"] > a["SON_ONCU_AY"]:
        hata.append("CHALLENGER_MAX_AY, SON_ONCU_AY'dan buyuk olamaz.")
    if not (a["TRAIN_ARALIK"][1] < a["ARA_ARALIK"][0] < a["ARA_ARALIK"][1] < a["OOT_ARALIK"][0]):
        hata.append("TRAIN < ARA < OOT kronolojik sirasi bozuk.")
    if abs(sum(a["AGIRLIK"].values()) - 1.0) > 1e-9:
        hata.append(f"AGIRLIK toplami 1.0 degil: {sum(a['AGIRLIK'].values()):.4f}")
    # Gecikmeli hedef sizinti kontrolu: en kucuk lag, en kisa bilgi_ayi ile uyumlu mu
    if a["LAG_KULLAN"]:
        gereken = a["HEDEF_OLGUNLASMA_AY"] - a["CHALLENGER_MAX_AY"]
        kucuk = min(a["LAG_LISTESI"])
        if kucuk < gereken:
            hata.append(f"lag-{kucuk} SIZINTI uretir. "
                        f"CHALLENGER_MAX_AY={a['CHALLENGER_MAX_AY']} icin lag>={gereken} sart.")
    # 90+ oncu yasagi
    if any("90_" in k for k in ONCU_KODLARI):
        hata.append("ONCU_KODLARI icinde 90_* var; hedefle ayni olay, girdi olamaz.")
    return hata

# ---------------------------------------------------------------- OZET
_h = config_dogrula()
print("=" * 94)
print("BLOK 0 — KONFIGURASYON")
print("=" * 94)
if _h:
    for x in _h: print("  !! HATA:", x)
    raise ValueError("CONFIG dogrulamasi basarisiz. Yukaridaki satirlari duzelt.")
print("  Dogrulama: TUM KONTROLLER GECTI\n")

print(f"HEDEF            : {HEDEF}")
print(f"Bolme turu       : {CONFIG['BOLME_TURU']}")
print(f"Bilgi ayi        : {CONFIG['BILGI_AYI']}  "
      f"(kohort t'nin hedefi t+{CONFIG['HEDEF_OLGUNLASMA_AY']}+{CONFIG['RAPOR_GECIKME_AY']}'de bilinir)")
print(f"Bolgeler         : TRAIN {CONFIG['TRAIN_ARALIK']}  |  "
      f"ARA {CONFIG['ARA_ARALIK']}  |  OOT {CONFIG['OOT_ARALIK']}")
print(f"Oncu havuzu      : {len(ONCU_KODLARI)} kod -> {ONCU_KODLARI}")
print(f"Ortak alanlar    : {len(ORTAK_ALANLAR)} -> {ORTAK_ALANLAR}")

print("\n--- OLGUNLASMA ARITMETIGI (sizinti kuralinin tureyisi) ---")
print("  Egitim kohortu t_tr, test kohortu t_val icin:")
print("     t_tr + HEDEF_OLGUNLASMA_AY + RAPOR_GECIKME_AY  <=  t_val + bilgi_ayi")
print("     t_tr  <=  t_val - (HEDEF_OLGUNLASMA_AY + RAPOR_GECIKME_AY - bilgi_ayi)")
for b, ad in [(CONFIG["SON_ONCU_AY"], "CHAMPION"), (CONFIG["CHALLENGER_MAX_AY"], "CHALLENGER")]:
    k = CONFIG["HEDEF_OLGUNLASMA_AY"] + CONFIG["RAPOR_GECIKME_AY"] - b
    print(f"     {ad:11s} bilgi_ayi={b} -> egitim kohortu en az {k:>2d} ay ESKI olmali")

print("\n--- GECIKMELI HEDEF (yeni) ---")
if CONFIG["LAG_KULLAN"]:
    ger = CONFIG["HEDEF_OLGUNLASMA_AY"] - CONFIG["CHALLENGER_MAX_AY"]
    print(f"  Kullanilacak lag'ler : {CONFIG['LAG_LISTESI']}")
    print(f"  Minimum guvenli lag  : {ger}  (en kisa bilgi_ayi={CONFIG['CHALLENGER_MAX_AY']} icin)")
    print(f"  Gerekce              : DW=0.608, Ljung-Box p=0.000, OOT'de 12/12 pozitif sapma")
    print(f"                         -> seviyeyi onculler degil, kohortun kendi gecmisi tasimali")
else:
    print("  KAPALI (karsilastirma deneyi icin)")

print("\n--- SIRALAMA AGIRLIKLARI ---")
_ag = pd.DataFrame({"metrik": list(CONFIG["AGIRLIK"]),
                    "agirlik": list(CONFIG["AGIRLIK"].values())})
_ag["aciklama"] = ["ortalama hata", "YANLILIK (yeni)", "ezber orani",
                   "en kotu fold", "fold istikrari", "degisim hatasi"]
print(_ag.to_string(index=False))
print(f"  TOPLAM = {sum(CONFIG['AGIRLIK'].values()):.2f}")
print("  NOT: YON_BASARISI siralamadan cikarildi (7 gozleme dayaniyor), raporlamada kalacak.")

print("\n--- GECERLILIK ESIKLERI ---")
for k, v in CONFIG["ESIK"].items():
    print(f"  {k:24s} = {v}")

print("\n--- MODEL ARAMASI ---")
nc = len(ONCU_KODLARI) + len(ORTAK_ALANLAR) + (len(CONFIG["LAG_LISTESI"]) if CONFIG["LAG_KULLAN"] else 0)
from math import comb
kmb = sum(comb(nc, k) for k in range(1, CONFIG["MAX_K"] + 1))
vary = 1 + int(CONFIG["TREND_DENE"]) + int(CONFIG["MEVSIM_DENE"]) + int(CONFIG["TREND_DENE"] and CONFIG["MEVSIM_DENE"])
print(f"  Champion havuz buyuklugu (tahmini) : {nc}")
print(f"  Kombinasyon (k=1..{CONFIG['MAX_K']})              : {kmb:,}")
print(f"  Muhendislik varyanti               : {vary}")
print(f"  Model ailesi                       : {len(CONFIG['MODEL_AILELERI'])}")
print(f"  TOPLAM VARYANT (yaklasik)          : {kmb * vary * len(CONFIG['MODEL_AILELERI']):,}")
print(f"\n  >> Bu kadar varyant arasindan secim YANLILIK uretir.")
print(f"  >> Bu yuzden BLOK 7'de ARA bolgesi hakemligi devreye girecek.")
print("=" * 94)
