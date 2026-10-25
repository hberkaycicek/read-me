s = d["KK_TAKIP_30_2_TUTARSAL"]
d["KK_TAKIP_30_2_TUTARSAL_L1"]     = s.shift(1)                      # bir önceki kohort
d["KK_TAKIP_30_2_TUTARSAL_L2"]     = s.shift(2)                      # iki önceki kohort
d["KK_TAKIP_30_2_TUTARSAL_MA3"]    = s.rolling(3).mean()             # bu + önceki 2 kohort
d["KK_TAKIP_30_2_TUTARSAL_MA3L1"]  = s.rolling(3).mean().shift(1)    # MA3'ün bir gecikmelisi
d["KK_TAKIP_30_2_TUTARSAL_MOM"]    = s.pct_change()                  # bir önceki kohorta göre değişim


k = "KK_TAKIP_30_2_TUTARSAL"
tur = [k, k+"_L1", k+"_L2", k+"_MA3", k+"_MA3L1", k+"_MOM"]
var = [c for c in tur if c in d.columns]
print(d.sort_values("AY")[["_AYS"] + var].tail(10).to_string(index=False))

# doğrulama: türetmeler gerçekten satır kaydırması mı
s = d.sort_values("AY")[k]
print("\nL1 kontrolü  :", bool(np.allclose(d.sort_values("AY")[k+"_L1"].dropna(),
                                           s.shift(1).dropna())))
print("MA3 kontrolü :", bool(np.allclose(d.sort_values("AY")[k+"_MA3"].dropna(),
                                         s.rolling(3).mean().dropna())))
