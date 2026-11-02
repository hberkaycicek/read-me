from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
kol = [oncul_kolon("30_2"), "KKB_SKOR_O0120"]          # v6 champion için kendi kolon listesini yaz
fo = fold_al(bilgi_ayi_hesapla(kol))
Xz = StandardScaler().fit_transform(cv_df[kol])
a_max = np.max(np.abs(Xz.T @ (cv_df[HEDEF] - cv_df[HEDEF].mean()))) / len(cv_df)
print(f"Lasso alpha_max (tüm katsayılar 0): {a_max:.5f}  | eski grid'in en küçüğü: 0.001")

satir = []
denemeler = [("OLS", lambda a: LinearRegression(), [0.0]),
             ("Ridge", lambda a: Ridge(alpha=a), np.logspace(-3, 3, 13)),
             ("Lasso", lambda a: Lasso(alpha=a, max_iter=100000), a_max * np.logspace(-4, 0, 13)),
             ("ElasticNet 0.5", lambda a: ElasticNet(alpha=a, l1_ratio=0.5, max_iter=100000),
              2 * a_max * np.logspace(-4, 0, 13))]
for ad, fab, alphalar in denemeler:
    for a in alphalar:
        s = cv_degerlendir(kol, fab(a), fo)
        m = pipe_fit(fab(a), olgun_df[kol], olgun_df[HEDEF])
        satir.append({"MODEL": ad, "ALPHA": a, "CV_WMAPE": s["CV_WMAPE"], "EN_KOTU": s["EN_KOTU_WMAPE"],
                      **{f"std_b_{kisa_ad(c)}": b for c, b in zip(kol, np.ravel(m[-1].coef_))}})
YOL = pd.DataFrame(satir)
display(YOL.style.format({"ALPHA": "{:.2e}", "CV_WMAPE": "{:.2f}", "EN_KOTU": "{:.2f}",
                          **{c: "{:.5f}" for c in YOL.columns if c.startswith("std_b_")}}, decimal=","))
