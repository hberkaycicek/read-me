print("HEDEF =", HEDEF)
print(d[["AY", HEDEF, "BASVURU_ADEDI"]].sort_values("AY").tail(15).to_string(index=False))
print("\nkolonlar:", sorted(map(str, d.columns))) 


TABLO.to_excel("12_metrik_tablosu.xlsx", index=False)      # sil
KATS.to_excel("12_katsayilar.xlsx", index=False)           # sil
SERI.to_excel("12_seri.xlsx", index=False)                 # sil
GELECEK.to_excel("12_gelecek.xlsx", index=False)           # sil
plt.savefig("12_seri.png", dpi=150); plt.show()            # → plt.show()
plt.savefig("12_hata.png", dpi=150); plt.show()            # → plt.show()
plt.savefig("12_gelecek.png", dpi=150); plt.show()         # → plt.show()


python
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
