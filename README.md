c30, kkb = CHAMP["kolonlar"]
for w_ in [3, 6]:
    d[f"30_2_ORT{w_}"] = d[c30].rolling(w_, min_periods=1).mean()   # sadece geçmiş + kendisi
d["30_2_DEGISIM"] = d[c30].diff().fillna(0)

cv_df = d.loc[gel_idx].sort_values("AY").reset_index(drop=True)   # yeni kolonlarla aynı satırlar
FOLD_CACHE.clear()

adaylar = {
    "Champion: 30_2 + KKB":       [c30, kkb],
    "30_2_ORT3 + KKB":            ["30_2_ORT3", kkb],
    "30_2_ORT6 + KKB":            ["30_2_ORT6", kkb],
    "30_2 + 30_2_ORT6 + KKB":     [c30, "30_2_ORT6", kkb],
    "30_2 + 30_2_DEGISIM + KKB":  [c30, "30_2_DEGISIM", kkb],
}
sonuc = pd.DataFrame([{"MODEL": ad, **cv_degerlendir(k, model_fabrika("Linear"), fold_al(2)),
                       "HAC_MAKS_P": float(np.max(hac_p(k, cv_df)))} for ad, k in adaylar.items()])
display(sonuc[["MODEL", "CV_WMAPE", "EN_KOTU_WMAPE", "FOLD_WMAPE_STD", "OVERFIT_ORANI",
               "DELTA_MAE", "MUTLAK_FOLD_SAPMA", "YON_AGIRLIKLI", "HAC_MAKS_P"]].round(4))
