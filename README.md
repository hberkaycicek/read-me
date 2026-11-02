# Sapma: fold bazında mutlak (fold'lar arası birbirini götürmesin)
CONFIG["AGIRLIK"] = {
    "CV_WMAPE": 0.30, "EN_KOTU_WMAPE": 0.12, "FOLD_WMAPE_STD": 0.12,
    "OVERFIT_ORANI": 0.15, "DELTA_MAE": 0.12, "MUTLAK_FOLD_SAPMA": 0.19,
}

def rank_ekle(df, config):
    g = config["GECERLI"]
    df = df.copy()
    df["GECERLI"] = (
        (df["CV_WMAPE"] <= g["CV_WMAPE_DUMMY_ORAN"] * DUMMY["CV_WMAPE"])
        & (df["EN_KOTU_WMAPE"] <= g["EN_KOTU_DUMMY_ORAN"] * DUMMY["EN_KOTU_WMAPE"])
        & (df["OVERFIT_ORANI"] <= g["OVERFIT_ORANI_MAX"])
        & (df["MUTLAK_FOLD_SAPMA"] <= g["SAPMA_HEDEF_ORAN"] * HEDEF_ORT)
        & (df["ARALIK_DISI"] == 0)
    )
    df["RANK_SKORU"] = np.nan
    m = df["GECERLI"]
    if m.any():
        skor = 0.0
        for kol, w in config["AGIRLIK"].items():
            s = df.loc[m, kol].astype(float)
            if kol == "OVERFIT_ORANI":
                s = s.clip(lower=1.0)
            skor = skor + w * s.rank(pct=True).fillna(1.0)
        df.loc[m, "RANK_SKORU"] = skor
    return (df.sort_values(["GECERLI", "RANK_SKORU", "CV_WMAPE"], ascending=[False, True, True])
              .reset_index(drop=True))

goster = ["GIRDILER", "MODEL", "BILGI_AY", "K", "MEVSIM", "CV_WMAPE", "EN_KOTU_WMAPE",
          "FOLD_WMAPE_STD", "OVERFIT_ORANI", "DELTA_MAE", "CV_SAPMA", "MUTLAK_FOLD_SAPMA",
          "SAPMA_TEK_YON", "YON_AGIRLIKLI", "GECERLI", "RANK_SKORU"]
fmt["MUTLAK_FOLD_SAPMA"] = "{:.4f}"

grid = rank_ekle(grid_ham, CONFIG)
print(f"{len(grid)} varyant | geçerli {int(grid['GECERLI'].sum())}")

# Hangi eşik kaç modeli eliyor
g = CONFIG["GECERLI"]
print("CV_WMAPE geçen      :", int((grid_ham["CV_WMAPE"] <= g["CV_WMAPE_DUMMY_ORAN"] * DUMMY["CV_WMAPE"]).sum()))
print("EN_KOTU geçen       :", int((grid_ham["EN_KOTU_WMAPE"] <= g["EN_KOTU_DUMMY_ORAN"] * DUMMY["EN_KOTU_WMAPE"]).sum()))
print("OVERFIT geçen       :", int((grid_ham["OVERFIT_ORANI"] <= g["OVERFIT_ORANI_MAX"]).sum()))
print("FOLD SAPMA geçen    :", int((grid_ham["MUTLAK_FOLD_SAPMA"] <= g["SAPMA_HEDEF_ORAN"] * HEDEF_ORT).sum()))

print("=== Genel ilk 12 ===")
display(grid[goster].head(12).style.format(fmt, decimal=","))
print("=== Erken (<= 3 ay) ilk 5 ===")
display(grid[grid["BILGI_AY"] <= 3][goster].head(5).style.format(fmt, decimal=","))
print("=== Geç (> 3 ay) ilk 5 ===")
display(grid[grid["BILGI_AY"] > 3][goster].head(5).style.format(fmt, decimal=","))

SECIMLER = secim_yap(grid, CONFIG)
for ad in SECIMLER:
    print("Seçilen:", ad)
