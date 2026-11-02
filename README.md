# Sapma eşiği Dummy'ye bağlı: Dummy fold sapmasının en fazla %35'i
CONFIG["GECERLI"]["SAPMA_DUMMY_ORAN"] = 0.35
SAPMA_ESIK = CONFIG["GECERLI"]["SAPMA_DUMMY_ORAN"] * DUMMY["MUTLAK_FOLD_SAPMA"]
print(f"Fold sapma eşiği: {SAPMA_ESIK:.5f}")

def rank_ekle(df, config):
    g = config["GECERLI"]
    df = df.copy()
    df["GECERLI"] = (
        (df["CV_WMAPE"] <= g["CV_WMAPE_DUMMY_ORAN"] * DUMMY["CV_WMAPE"])
        & (df["EN_KOTU_WMAPE"] <= g["EN_KOTU_DUMMY_ORAN"] * DUMMY["EN_KOTU_WMAPE"])
        & (df["OVERFIT_ORANI"] <= g["OVERFIT_ORANI_MAX"])
        & (df["MUTLAK_FOLD_SAPMA"] <= g["SAPMA_DUMMY_ORAN"] * DUMMY["MUTLAK_FOLD_SAPMA"])
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

grid = rank_ekle(grid_ham, CONFIG)
print(f"{len(grid)} varyant | geçerli {int(grid['GECERLI'].sum())} | "
      f"erken {int((grid['GECERLI'] & (grid['BILGI_AY'] <= 3)).sum())} | "
      f"geç {int((grid['GECERLI'] & (grid['BILGI_AY'] > 3)).sum())}")

print("=== Genel ilk 12 ===")
display(grid[goster].head(12).style.format(fmt, decimal=","))
print("=== Geç (> 3 ay) ilk 5 ===")
display(grid[grid["BILGI_AY"] > 3][goster].head(5).style.format(fmt, decimal=","))

SECIMLER = secim_yap(grid, CONFIG)
for ad in SECIMLER:
    print("Seçilen:", ad)
