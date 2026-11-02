# Challenger: 1-SE içindeki CHALLENGER_MAX_AY (3.) ay modelleri arasında en istikrarlı olan
# (soru: "1 ay daha beklersek ne kazanırız?")
ch = kisa_c[kisa_c["BILGI_AY"] == CONFIG["CHALLENGER_MAX_AY"]]
ch = ch[ch["KOLONLAR"].apply(lambda k: set(taban(k))) != set(taban(CHAMP_SATIR["KOLONLAR"]))]
if ch.empty:
    raise ValueError("1-SE içinde 3. ay modeli yok.")
CHAL_SATIR = ch.sort_values(["EN_KOTU_WMAPE", "FOLD_WMAPE_STD", "CV_WMAPE"]).iloc[0]
CHAL = spec_yap(CHAL_SATIR, "CHALLENGER")
SECIMLER = {"CHAMPION": CHAMP, "CHALLENGER": CHAL}
print(CHAMP["ad"], "  <- anlamlılık + 1-SE + sadelik")
print(CHAL["ad"], "  <- 1-SE içinde 3. ay, en iyi en kötü fold")
SECIM_TABLO = pd.DataFrame([CHAMP_SATIR, CHAL_SATIR], index=["CHAMPION", "CHALLENGER"])[GOSTER[:-3]]
display(SECIM_TABLO.style.format(FMT, decimal=","))
