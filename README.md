s = d["KK_TAKIP_30_2_TUTARSAL"]
d["KK_TAKIP_30_2_TUTARSAL_L1"]     = s.shift(1)                      # bir önceki kohort
d["KK_TAKIP_30_2_TUTARSAL_L2"]     = s.shift(2)                      # iki önceki kohort
d["KK_TAKIP_30_2_TUTARSAL_MA3"]    = s.rolling(3).mean()             # bu + önceki 2 kohort
d["KK_TAKIP_30_2_TUTARSAL_MA3L1"]  = s.rolling(3).mean().shift(1)    # MA3'ün bir gecikmelisi
d["KK_TAKIP_30_2_TUTARSAL_MOM"]    = s.pct_change()                  # bir önceki kohorta göre değişim
