# Yakın döneme üstel ağırlık: H ay önceki kohort yarı ağırlık alır
def agirlik_al(df):
    H = CONFIG.get("YARI_OMUR")
    kol = CONFIG["SAMPLE_WEIGHT_KOLON"]
    if kol is None and not H:
        return None
    w = np.ones(len(df)) if kol is None else df[kol].to_numpy(dtype=float)
    if H:
        yas = (df["AY"].max() - df["AY"]).apply(lambda x: x.n).to_numpy(dtype=float)
        w = w * 0.5 ** (yas / H)
    return w

sonuc = []
for H in [None, 12, 18, 24, 36]:
    CONFIG["YARI_OMUR"] = H
    for ad, spec in SECIMLER.items():
        w = wf_tahmin(spec)
        for bolge in ["GELISTIRME", "OOT"]:
            s = w[w["BOLGE"] == bolge]
            sonuc.append({
                "MODEL": spec["rol"], "YARI_OMUR": H if H else "yok", "BOLGE": bolge, "N": len(s),
                "WMAPE": wmape(s["GERCEK"], s["HAM"]),
                "DUMMY_WMAPE": wmape(s["GERCEK"], s["DUMMY"]),
                "SAPMA": float((s["HAM"] - s["GERCEK"]).mean()),
                "KORELASYON": float(np.corrcoef(s["GERCEK"], s["HAM"])[0, 1]),
                "EGIM": float(np.polyfit(s["HAM"], s["GERCEK"], 1)[0]),   # 1 = genlik doğru
            })
CONFIG["YARI_OMUR"] = None

tab = pd.DataFrame(sonuc)
for bolge in ["GELISTIRME", "OOT"]:
    print(f"=== {bolge} {'(H BURADAN SEÇİLİR)' if bolge == 'GELISTIRME' else '(sadece rapor)'} ===")
    display(tab[tab["BOLGE"] == bolge].drop(columns="BOLGE").style.format(
        {"WMAPE": "{:.2f}", "DUMMY_WMAPE": "{:.2f}", "SAPMA": "{:+.4f}",
         "KORELASYON": "{:.3f}", "EGIM": "{:.2f}"}, decimal=","))
