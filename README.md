def kisa_liste(havuz, baslik, ilk_n=1000):
    """1) CV'de en iyi ilk_n aday içinde TÜM katsayıları anlamlı olanlar (HAC, mevsim dahil)
       2) 1-SE kuralı: en iyi anlamlı modelin CV_WMAPE + standart hatası içindekiler istatistiksel olarak eşit
       3) bunlar arasında sade > erken > basit aile > bileşik skor."""
    h = havuz.nsmallest(ilk_n, "CV_WMAPE").drop_duplicates(["KOLONLAR", "MODEL"]).copy()
    h["HAC_MAKS_P"] = h["KOLONLAR"].apply(lambda k: float(np.max(hac_p(k, cv_df))))
    h["ANLAMLI"] = h["HAC_MAKS_P"] <= CONFIG["HAC_P_ESIK"]
    a = h[h["ANLAMLI"]]
    n_anl = len(a)
    if a.empty:
        print(f"UYARI ({baslik}): ilk {ilk_n} aday içinde tüm katsayıları anlamlı model yok.")
        a = h
    en_iyi = a.nsmallest(1, "CV_WMAPE").iloc[0]
    se = en_iyi["FOLD_WMAPE_STD"] / np.sqrt(CONFIG["N_SPLITS"])
    esik = en_iyi["CV_WMAPE"] + se
    k_ = a[a["CV_WMAPE"] <= esik].copy()
    k_["PARAMETRE"] = k_["KOLONLAR"].apply(len)
    k_["AILE_SIRA"] = k_["MODEL"].map(AILE_SIRA)
    k_ = k_.sort_values(["PARAMETRE", "BILGI_AY", "AILE_SIRA", "RANK_SKORU"]).reset_index(drop=True)
    print(f"=== {baslik}: anlamlı adaylar {n_anl}/{len(h)} | en iyi anlamlı {en_iyi['CV_WMAPE']:.2f} "
          f"+ 1 SE ({se:.2f}) = eşik {esik:.2f} | eşik içinde {len(k_)} model ===")
    print(f"    (bilgi: anlamlılık filtresi olmadan en iyi CV_WMAPE {h['CV_WMAPE'].min():.2f}, "
          f"HAC_MAKS_P {h.iloc[0]['HAC_MAKS_P']:.3f})")
    display(k_[["GIRDILER", "MODEL", "BILGI_AY", "MEVSIM", "PARAMETRE", "CV_WMAPE", "EN_KOTU_WMAPE",
                "FOLD_WMAPE_STD", "HAC_MAKS_P", "RANK_SKORU"]].head(15).style.format(
        {"CV_WMAPE": "{:.2f}", "EN_KOTU_WMAPE": "{:.2f}", "FOLD_WMAPE_STD": "{:.2f}",
         "HAC_MAKS_P": "{:.3f}", "RANK_SKORU": "{:.4f}"}, decimal=","))
    return k_
