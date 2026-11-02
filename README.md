svr = grid[grid["MODEL"] == "SVR"].sort_values("CV_WMAPE").iloc[0]
print("En iyi SVR:", svr["GIRDILER"], "| mevsim:", svr["MEVSIM"], "| parametreler:", svr["PARAMS"])
spec_svr = {"rol": "SVR", "kolonlar": list(svr["KOLONLAR"]), "aile": "SVR",
            "params": svr["PARAMS"], "bilgi_ay": int(svr["BILGI_AY"])}
w = wf_tahmin(spec_svr)
satir = []
for bolge, m in [("VALIDASYON", (w["BOLGE"] == "GELISTIRME") & (w["AY"] >= VAL_BAS)), ("OOT", w["BOLGE"] == "OOT")]:
    s = w[m]
    satir.append({"BOLGE": bolge, **metrik(s["GERCEK"], s["TAHMIN"])})
display(pd.DataFrame(satir).round(3))
