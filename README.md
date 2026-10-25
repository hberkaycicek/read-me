# ============================================================================
# BLOK 6.0 — BUG DUZELTMESI: sample_weight parametre adi
#
# HATA: TransformedTargetRegressor.fit(X, y, **fit_params)
#         -> self.regressor_.fit(X, y_trans, **fit_params)
#       regressor_ ZATEN bir Pipeline. Dogru ad: 'm__sample_weight'
#       Ben 'regressor__m__sample_weight' yazdim -> Pipeline step='regressor'
#       ariyor -> KeyError.
# ============================================================================
def _fit_agirlikli(pipe, X, y, w, aile):
    """sample_weight'i dogru parametre adiyla gecirmeyi dene; olmazsa agirliksiz fit."""
    for p in ("m__sample_weight", "regressor__sample_weight", "sample_weight"):
        try:
            pipe.fit(X, y, **{p: w}); return True
        except (TypeError, ValueError, KeyError):
            continue
    pipe.fit(X, y); return False

def cv_degerlendir(kolonlar, aile, bilgi_ay, ek_kolonlar=None, bolge_idx=None, etiket=""):
    kol = list(kolonlar) + list(ek_kolonlar or [])
    bolge_idx = tr_idx if bolge_idx is None else bolge_idx
    foldlar = fold_olustur(bolge_idx, bilgi_ay)
    if not foldlar: return None
    tum_y, tum_p, fold_wmape, fold_sapma, tr_rmse_list, agirlik_oldu = [], [], [], [], [], []
    for tr_i, va_i in foldlar:
        eg = d.loc[tr_i, kol + [HEDEF]].dropna()
        te = d.loc[va_i, kol + [HEDEF]].dropna()
        if len(eg) < MIN_TRAIN or len(te) == 0: return None
        pipe = pipeline_kur(aile)
        w = _agirlik(eg.index)
        agirlik_oldu.append(_fit_agirlikli(pipe, eg[kol], eg[HEDEF], w, aile))
        p = np.asarray(pipe.predict(te[kol]), float); y = te[HEDEF].values
        m = temel_metrikler(y, p)
        fold_wmape.append(m["WMAPE"]); fold_sapma.append(m["SAPMA"])
        tr_rmse_list.append(np.sqrt(mean_squared_error(eg[HEDEF], pipe.predict(eg[kol]))))
        tum_y.extend(y); tum_p.extend(p)
    tum_y, tum_p = np.array(tum_y), np.array(tum_p)
    g = temel_metrikler(tum_y, tum_p); tr_rmse = float(np.mean(tr_rmse_list))
    dy, dp = np.diff(tum_y), np.diff(tum_p)
    yon = float(np.mean(np.sign(dy) == np.sign(dp))) if len(dy) else np.nan
    isaret = bool(len(fold_sapma) >= 2 and
                  (all(s > 0 for s in fold_sapma) or all(s < 0 for s in fold_sapma)))
    try:    mv = maks_vif(d.loc[bolge_idx, kol].dropna())
    except Exception: mv = np.inf
    return {"ETIKET": etiket, "GIRDILER": ", ".join(kol), "K": len(kol), "AILE": aile,
            "BILGI_AY": bilgi_ay, "N_FOLD": len(foldlar),
            "AGIRLIK_UYGULANDI": bool(np.all(agirlik_oldu)),
            "CV_WMAPE": g["WMAPE"], "CV_MAPE": g["MAPE"], "CV_MAE": g["MAE"], "CV_RMSE": g["RMSE"],
            "CV_SAPMA": g["SAPMA"], "MUTLAK_CV_SAPMA": abs(g["SAPMA"]),
            "FOLD_SAPMALAR": [round(s, 6) for s in fold_sapma],
            "EN_KOTU_WMAPE": float(np.max(fold_wmape)), "FOLD_WMAPE_STD": float(np.std(fold_wmape)),
            "ISARET_SERISI": isaret, "TRAIN_RMSE": tr_rmse,
            "OVERFIT_ORANI": float(g["RMSE"]/tr_rmse) if tr_rmse > 1e-12 else np.inf,
            "DELTA_MAE": float(np.mean(np.abs(dp-dy))) if len(dy) else np.nan,
            "YON_BASARISI": yon, "MAKS_VIF": mv,
            "ARALIK_DISI": int(((tum_p < 0) | (tum_p > 1)).sum()), "N_TAHMIN": len(tum_y)}

# Dogrulama: agirlik gercekten uygulandi mi?
print("BUG DUZELTME DOGRULAMASI")
print("-"*70)
for _a in ["Linear", "Ridge", "Oransal", "LOGIT_Linear", "LOGIT_Ridge"]:
    _r = cv_degerlendir([tek.iloc[0]["degisken"]], _a, 6)
    print(f"  {_a:14s} calisti={_r is not None}  "
          f"agirlik_uygulandi={_r['AGIRLIK_UYGULANDI'] if _r else '-'}  "
          f"CV_WMAPE={_r['CV_WMAPE']:.4f}" if _r else f"  {_a}: BASARISIZ")
