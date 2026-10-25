import json, pprint

print("="*100); print("SECILEN"); print("="*100)
for k, v in SECILEN.items():
    print(f"\n--- {k} ---")
    pprint.pprint(v, width=200, sort_dicts=False)

for ad in ["FINAL2", "FIN", "MERDIVEN_DF", "OZET", "best", "DOGRU"]:
    if ad in dir():
        print("\n" + "="*100); print(ad); print("="*100)
        print(globals()[ad].to_string(index=False))

print("\n" + "="*100); print("OOT"); print("="*100)
print(OOT.to_string(index=False))

print("\n" + "="*100); print("OOT_SONUC"); print("="*100)
pprint.pprint(OOT_SONUC, width=200, sort_dicts=False)

print("\n" + "="*100); print("CONFIG"); print("="*100)
pprint.pprint(CONFIG, width=200, sort_dicts=False)

print("\n" + "="*100); print("wf_pencereli imzası"); print("="*100)
import inspect; print(inspect.signature(wf_pencereli))
