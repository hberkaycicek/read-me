# ============================================================
# BLOK 16 — ŞAMPİYONUN (v2c) AYLIK OOT SERİSİ
#   sunumdaki performans grafiği için
# ============================================================
PAY, PEN, GEC = "KK_TAKIP_60_6_TUTARSAL", 6, 30
cfgC = SECILEN["CHAMPION"]
OOT_I = d.index[(d["BOLGE"].astype(str) == "OOT") & d[HEDEF].notna()]

r = wf_pencereli(kolonlar=cfgC["kolonlar"], aile=cfgC["aile"], bilgi_ay=cfgC["bilgi_ay"],
                 hedef_idx=OOT_I, ek_kolonlar=cfgC.get("ek") or None,
                 max_gecmis=GEC, oran_paydasi=PAY, duzelt_pencere=PEN)

S = pd.DataFrame({"AY": [AYS(t) for t in r["_don"]],
                  "GERCEK_%": [round(v*100, 2) for v in r["_ger"]],
                  "v2c_%":    [round(v*100, 2) for v in r["_tah"]]})
S["HATA_pp"] = (S["v2c_%"] - S["GERCEK_%"]).round(2)
print(S.to_string(index=False))
print("\nkopyalanabilir:")
print("GERCEK =", list(S["GERCEK_%"]))
print("V2C    =", list(S["v2c_%"]))
print(f"\nWMAPE {100*np.sum(np.abs(np.array(r['_tah'])-np.array(r['_ger'])))/np.sum(r['_ger']):.2f}"
      f"   sapma {np.mean(np.array(r['_tah'])-np.array(r['_ger'])):+.4f}"
      f"   fazla tahmin {int((S.HATA_pp>0).sum())}/12")
