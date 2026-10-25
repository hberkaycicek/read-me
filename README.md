# ============================================================
# BLOK 13.3 — ŞAMPİYONUN (v2c) GERÇEK KATSAYILARI + TAM SERİ
# ============================================================
PAY, PEN, GEC = "KK_TAKIP_60_6_TUTARSAL", 6, 30
cfgC = SECILEN["CHAMPION"]
kol  = list(cfgC["kolonlar"]) + list(cfgC.get("ek") or [])

olgun = d[d.HEDEF_KULLANILABILIR]
eg = olgun.tail(GEC)
eg = eg[kol + [PAY, HEDEF]].dropna()
eg = eg[eg[PAY] > 0]

pipe = pipeline_kur(cfgC["aile"])
_fit_agirlikli(pipe, eg[kol], eg[HEDEF]/eg[PAY], _agirlik(eg.index), cfgC["aile"])

sc, reg = pipe.named_steps["s"], pipe.named_steps["m"]
b_std = np.ravel(reg.coef_); b_ham = b_std/sc.scale_
sabit = float(np.ravel(reg.intercept_)[0]) - float(np.sum(b_ham*sc.mean_))
K = pd.DataFrame({"DEGISKEN": kol, "KATSAYI_STD": b_std, "KATSAYI_HAM": b_ham})
K["ETKI_%"] = 100*np.abs(K.KATSAYI_STD)/np.abs(K.KATSAYI_STD).sum()
K["YON"] = np.where(K.KATSAYI_STD >= 0, "artirir", "azaltir")

print("=== ŞAMPİYON v2c KATSAYILARI ===")
print(f"eğitim: {AYS(eg['AY'].min())} – {AYS(eg['AY'].max())}, {len(eg)} kohort")
print(f"hedef : 12M90 / {PAY}   (oransal)\n")
print(K.sort_values("ETKI_%", ascending=False).to_string(index=False))
print(f"\nSABIT = {sabit:.8f}")
print("\nFORMÜL:")
print(f"  oran = {sabit:+.8f}")
for _k, _b in zip(K.DEGISKEN, K.KATSAYI_HAM):
    print(f"         {'+' if _b>=0 else '-'} {abs(_b):.8f} × {_k}")
print(f"  tahmin = oran × {PAY}  −  (son {PEN} kohortun ortalama sapması)")

# v2c'nin tam metrik seti (RMSE/MSE dahil)
r = wf_pencereli(kolonlar=cfgC["kolonlar"], aile=cfgC["aile"], bilgi_ay=cfgC["bilgi_ay"],
                 hedef_idx=OOT_I, ek_kolonlar=cfgC.get("ek") or None,
                 max_gecmis=GEC, oran_paydasi=PAY, duzelt_pencere=PEN)
print("\n=== v2c TAM METRİK SETİ (OOT) ===")
print(pd.Series(metrikler(np.array(r["_ger"]), np.array(r["_tah"]))).to_string())

# tam seri — kopyalanabilir tek satır
g = d[d[HEDEF].notna()].sort_values("AY")
print("\n=== TAM SERİ (57 kohort) ===")
print("AYLAR =", list(g["_AYS"]))
print("GERCEK =", [round(float(v), 6) for v in g[HEDEF]])
