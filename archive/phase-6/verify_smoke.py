import json
rs=[json.loads(l) for l in open("results/results_qwen3-1.7b.jsonl")]
print("rows:", len(rs))
print()
for r in rs:
    print("%-18s %-10s correct=%-5s tok=%-6d budget=%d trunc=%-5s lat=%.0fs" % (
        r["arm"], r["problem_id"], r["correct"], r["tokens"], r["budget"],
        r["truncated"], r["latency_s"]))
inj=[r for r in rs if r["arm"]!="A_vanilla"]
d=[r for r in rs if r["arm"]=="D_new_plus_graph"]
print()
print("=== verifications ===")
print("1. prompt logging  :", all(r["injected_strategy"] for r in inj), "(%d injected rows)" % len(inj))
print("2. escape hatch on :", all(r["escape_hatch"] for r in inj))
print("3. augmentation    :", [r["n_augmentation"] for r in d], "bonus steps in arm D")
print("4. anchors         :", [(r["anchor_id"], round(r["anchor_similarity"],3)) for r in inj])
print("5. tokens/trunc    :", [(r["tokens"], r["truncated"]) for r in rs])
print("6. boxed present   :", [("boxed" in r["raw"]) for r in rs])
print()
print("=== injected prompt, arm C ===")
print([r for r in rs if r["arm"]=="C_new_patterns"][0]["user_prompt"][:1000])
