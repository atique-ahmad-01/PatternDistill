import json, re
from difflib import SequenceMatcher
rs=json.load(open("data/patterns/patterns.json"))

def norm(s):
    return re.sub(r"[^a-z0-9 ]"," ", s.lower()).split()

dup=0; dup_ex=None
for r in rs:
    st=r["steps"]
    found=False
    for i in range(len(st)):
        for j in range(i+1,len(st)):
            if SequenceMatcher(None," ".join(norm(st[i]))," ".join(norm(st[j]))).ratio()>0.75:
                found=True
                if dup_ex is None: dup_ex=(r["id"],st[i],st[j])
    if found: dup+=1

# anchor-specific labels: single capital letters / point names echoed from the problem
lab=0; lab_ex=None
for r in rs:
    blob=" ".join(r["steps"]+[r["trigger"],r["technique"]])
    plabels=set(re.findall(r"\b([A-Z]{1,3})\b", r["problem"]))
    slabels=set(re.findall(r"\b([A-Z]{1,3})\b", blob))
    shared=(plabels & slabels) - {"A","I","The","If"}
    if shared:
        lab+=1
        if lab_ex is None: lab_ex=(r["id"], sorted(shared)[:4])

n=len(rs)
print("patterns:", n)
print("with near-duplicate steps : %d (%.1f%%)" % (dup, dup/n*100))
if dup_ex: print("   e.g.", dup_ex[0]); print("     -", dup_ex[1][:70]); print("     -", dup_ex[2][:70])
print("echoing anchor labels     : %d (%.1f%%)" % (lab, lab/n*100))
if lab_ex: print("   e.g.", lab_ex[0], "shares labels", lab_ex[1])
