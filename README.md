bilgi = pd.Period(CONFIG["BILGI_AYI"], "M")
c30 = oncul_kolon("30_2", CONFIG["HEDEF_TUR"])
kkb = [c for c in ORTAK_ALANLAR if c.startswith("KKB")][0]
SADE = {"rol": "SADE", "kolonlar": [c30, kkb], "aile": "Linear", "params": None, "bilgi_ay": 2,
        "ad": f"SADE (30_2, {kkb} | Linear | 2. ay)"}

print("=== Grid'de SADE modelin CV sonucu ===")
display(grid[(grid["GIRDILER"] == f"30_2, {kkb}") & (grid["MEVSIM"] == 0)][goster]
        .style.format(fmt, decimal=","))

# 1) Walk-forward karşılaştırma + overlay + üretim
karsi, tt2 = [], tahmin_tablo[["DONEM_1", "GOZLENEN_AY", "KISMI_GERCEK"]].copy()
for spec in [SECIMLER[champ_ad], SADE]:
    w = wf_tahmin(spec)
    for bolge in ["GELISTIRME", "OOT"]:
        s = w[w["BOLGE"] == bolge]
        dg, dt = np.diff(s["GERCEK"]), np.diff(s["HAM"])
        karsi.append({"MODEL": spec["rol"], "BOLGE": bolge, "N": len(s),
                      "WMAPE": wmape(s["GERCEK"], s["HAM"]),
                      "SAPMA": float((s["HAM"] - s["GERCEK"]).mean()),
                      "KORELASYON": float(np.corrcoef(s["GERCEK"], s["HAM"])[0, 1]),
                      "EGIM": float(np.polyfit(s["HAM"], s["GERCEK"], 1)[0]),
                      "YON": f"{int((np.sign(dg) == np.sign(dt)).sum())}/{len(dg)}"})
    son = w[w["HEDEF_ERISIM"] < bilgi].sort_values("HEDEF_ERISIM").tail(12)
    ov = float(son["ARTIK"].mean())
    m = pipe_fit(model_fabrika(spec["aile"], spec["params"]), olgun_df[spec["kolonlar"]], olgun_df[HEDEF])
    ham = m.predict(gelecek[spec["kolonlar"]])
    tt2[f"{spec['rol']}_HAM"] = np.clip(ham, 0, 1)
    tt2[f"{spec['rol']}_OVERLAY"] = np.clip(ham - ov, 0, 1)
    print(f"{spec['rol']}: overlay {ov:+.4%} ({int((son['ARTIK'] > 0).sum())}/12 pozitif)")

print("=== Walk-forward karşılaştırma (OOT bağımsız değil, teşhis sonrası) ===")
display(pd.DataFrame(karsi).style.format({"WMAPE": "{:.2f}", "SAPMA": "{:+.4f}",
                                          "KORELASYON": "{:.3f}", "EGIM": "{:.2f}"}, decimal=","))

# 2) Chain-ladder: kısmi gerçek / tarihsel olgunlaşma oranı (modelden bağımsız)
tur = CONFIG["HEDEF_TUR"]
k90 = {int(c.split("_")[3]): c for c in df_raw.columns
       if c.startswith("KK_TAKIP_90_") and c.endswith("_" + tur) and c.split("_")[3].isdigit()}
aylar_ = sorted(m for m in k90 if m < 12)
print("Mevcut 90+ olgunluk kolonları:", sorted(k90))
if aylar_:
    ham_df = df_raw.assign(DONEM_1=pd.to_numeric(df_raw["DONEM_1"]).astype(int)).set_index("DONEM_1")
    son_olgun = olgun_df.sort_values("AY").tail(12)["DONEM_1"].astype(int)   # güncel rejim
    oranlar = {m: float((ham_df.loc[son_olgun, k90[m]] / ham_df.loc[son_olgun, HEDEF]).mean()) for m in aylar_}
    oranlar[12] = 1.0
    print("Olgunlaşma oranı (90_m / 90_12, son 12 olgun kohort):", {m: round(v, 3) for m, v in oranlar.items()})
    xs = sorted(oranlar)
    tt2["OLGUNLASMA_ORANI"] = np.interp(tt2["GOZLENEN_AY"], xs, [oranlar[x] for x in xs])
    tt2["CHAIN_LADDER"] = tt2["KISMI_GERCEK"] / tt2["OLGUNLASMA_ORANI"]
    if tt2["GOZLENEN_AY"].min() < min(xs):
        print(f"UYARI: {min(xs)}. aydan genç kohortlarda oran sınır değerle sabitlendi.")
else:
    print("90_m (m<12) kolonu yok, chain-ladder atlandı.")

oran_kol = [c for c in tt2.columns if c not in ("DONEM_1", "GOZLENEN_AY", "OLGUNLASMA_ORANI")]
display(tt2.style.format({**{c: "{:.2%}" for c in oran_kol}, "OLGUNLASMA_ORANI": "{:.3f}",
                          "DONEM_1": "{:.0f}"}, decimal=",", na_rep="-"))
