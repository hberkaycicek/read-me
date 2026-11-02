olgun_df = d[d["HEDEF_KULLANILABILIR"]].copy()          # yeni kolonlar dahil
ADAY = {**CHAMP, "rol": "ADAY", "kolonlar": [c30, "30_2_ORT6", kkb]}

# 1) Katsayı işaretleri (tüm olgun veriyle)
m = pipe_fit(model_fabrika("Linear"), olgun_df[ADAY["kolonlar"]], olgun_df[HEDEF])
coef = m[-1].coef_ / m[0].scale_
print("Katsayılar:", {kisa_ad(k): round(float(v), 6) for k, v in zip(ADAY["kolonlar"], coef)})

# 2) OOT walk-forward: champion ile aynı kohortlarda
karsi = []
for ad, spec in [("Champion", CHAMP), ("30_2 + ORT6 + KKB", ADAY)]:
    o = wf_tahmin(spec).query("BOLGE == 'OOT'")
    karsi.append({"MODEL": ad, **metrik(o["GERCEK"], o["TAHMIN"])})
display(pd.DataFrame(karsi).round(3))
