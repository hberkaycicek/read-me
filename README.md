# 1) Dönüşüm oranı kanıtı (dokümantasyon için)
c30, c60 = oncul_kolon("30_2", CONFIG["HEDEF_TUR"]), oncul_kolon("60_3", CONFIG["HEDEF_TUR"])
r = olgun_df.assign(DONEM=olgun_df["AY"].dt.year.astype(str) + "-H" + ((olgun_df["AY"].dt.month > 6) + 1).astype(str))
oran = r.groupby("DONEM").agg(N=(HEDEF, "size"), HEDEF_ORT=(HEDEF, "mean"),
                              ORT_30_2=(c30, "mean"), ORT_60_3=(c60, "mean"))
oran["HEDEF/30_2"] = oran["HEDEF_ORT"] / oran["ORT_30_2"]
oran["HEDEF/60_3"] = oran["HEDEF_ORT"] / oran["ORT_60_3"]
print("=== Erken gecikmenin batığa dönüşüm oranı (yarıyıl) ===")
display(oran.style.format({"HEDEF_ORT": "{:.2%}", "ORT_30_2": "{:.2%}", "ORT_60_3": "{:.2%}",
                           "HEDEF/30_2": "{:.2f}", "HEDEF/60_3": "{:.2f}"}, decimal=","))

# 2) Overlay: bilgi ayında hedefi belli olan son 12 kohortun walk-forward artık ortalaması
bilgi = pd.Period(CONFIG["BILGI_AYI"], "M")
tt = tahmin_tablo.copy()
for ad, u in URETIM.items():
    rol = u["spec"]["rol"]
    son = WF[ad][WF[ad]["HEDEF_ERISIM"] < bilgi].sort_values("HEDEF_ERISIM").tail(12)
    ov = float(son["ARTIK"].mean())
    print(f"{rol}: overlay {ov:+.4%} | son 12 artığın {int((son['ARTIK'] > 0).sum())}/12'si pozitif "
          f"| dönem {son['DONEM_1'].min()}-{son['DONEM_1'].max()}")
    tt[f"{rol}_OVERLAY"] = (tt[f"{rol}_HAM"] - ov).clip(0, 1)
tt["CHAMP_OVERLAY - KISMI"] = tt["CHAMPION_OVERLAY"] - tt["KISMI_GERCEK"]

goster_t = ["DONEM_1", "GOZLENEN_AY", "KISMI_GERCEK", "CHAMPION_HAM", "CHAMPION_OVERLAY",
            "CHALLENGER_HAM", "CHALLENGER_OVERLAY", "CHAMP_OVERLAY - KISMI"]
display(tt[goster_t].style.format({c: "{:.2%}" for c in goster_t if c not in ("DONEM_1", "GOZLENEN_AY")}
                                  | {"DONEM_1": "{:.0f}"}, decimal=","))
