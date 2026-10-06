"""RoT thought graph -- arXiv:2509.21743 + the authors' reference implementation
(github.com/ahme0599/Retrieval-of-Thought).

    V   node (t,i) = reasoning step i of template t, id "t{t}_s{i}", carrying
        template_type and knowledge_tag metadata alongside the step text.

    E_Sequential  (t,i) -> (t,i+1)                     edge_type "sequential"
    E_Semantic    cos(u, u') >= tau                    edge_type "semantic"

WHERE THE PAPER AND THE CODE DISAGREE, THE CODE WINS. The paper defines a
rescaled similarity sim~=(cos+1)/2 with tau=0.85. The shipped
knowledge_graph.graphml says otherwise: its 230,676 semantic edges have
similarity min 0.8500, median 0.8614, max 0.9944, and every traversal in
`graph_handler.py` scores with sklearn's `cosine_similarity` directly. So tau
is 0.85 on RAW cosine and there is no rescaling anywhere. Implementing the
formula as written would threshold at raw 0.70 and roughly quadruple the edge
count.

The reference caps nothing: 2159 nodes carry ~107 semantic neighbours each.
`max_neighbors: 0` reproduces that; a positive value is a deviation.

MAPPING ONTO THIS PROJECT. The reference's V comes from a curated template
library whose steps are generic method skeletons -- "Standard Transformation:
Rearrange the inequality to the form ax^2+bx+c>0". Only 4 of its 2159 node
texts (0.2%) contain a 4-or-more digit literal. A template t here is instead
one validated teacher solution and its `subtasks` are the steps, and 37% of
those carry a 4+ digit constant belonging to the problem they were written for.
That gap is the main threat to this adaptation and is measured, not assumed --
see `rot_instance_constants` in the build report.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import data_path


def rescale(cos):
    """The paper's sim~=(cos+1)/2. NOT used to build or traverse the graph.

    Kept only so the paper's stated formula can be reproduced on request. The
    reference implementation thresholds raw cosine; see the module docstring.
    """
    return (cos + 1.0) / 2.0


def _shim_transformers_onnx():
    """Make jina-embeddings-v2 loadable under transformers >= 5.

    Its remote `configuration_bert.py` does `from transformers.onnx import
    OnnxConfig` at import time. That module was removed in transformers 5.x, so
    the import raises before a single weight is read -- even though OnnxConfig
    is only ever touched when exporting to ONNX, which we never do.

    Registering an empty stand-in satisfies the import and changes nothing at
    inference. Pinning transformers < 5 instead would be the cleaner fix in
    isolation, but the same environment serves vLLM, and downgrading it there
    to satisfy an embedding model is the larger risk.
    """
    import sys
    if "transformers.onnx" in sys.modules:
        return
    try:
        import transformers.onnx  # noqa: F401
        return
    except ImportError:
        pass
    import types
    mod = types.ModuleType("transformers.onnx")

    class OnnxConfig:              # placeholder; never instantiated at inference
        pass

    mod.OnnxConfig = OnnxConfig
    sys.modules["transformers.onnx"] = mod
    import transformers
    transformers.onnx = mod

    # Second casualty of the same version gap: transformers 5 dropped
    # `find_pruneable_heads_and_indices`, which jina's modeling_bert imports at
    # module scope. It implements attention-HEAD PRUNING and is unreachable on
    # the inference path -- but the import fails before any of that matters.
    # Restored verbatim from transformers 4.x rather than stubbed, so that if
    # anything ever does call it the behaviour is the real one.
    import transformers.pytorch_utils as _pu
    if not hasattr(_pu, "find_pruneable_heads_and_indices"):
        import torch

        def find_pruneable_heads_and_indices(heads, n_heads, head_size,
                                             already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        _pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    # Third: transformers 5 dropped the encoder-decoder defaults from
    # PretrainedConfig, and jina's modeling_bert reads `config.is_decoder` and
    # `config.add_cross_attention` while building its layers. jina's own config
    # never sets them -- under 4.x both came from the base class as False.
    # Restoring exactly those defaults keeps the model the plain encoder it is.
    from transformers import PretrainedConfig
    for _attr in ("is_decoder", "add_cross_attention"):
        if not hasattr(PretrainedConfig(), _attr):
            setattr(PretrainedConfig, _attr, False)

    # Fourth: jina's forward calls `self.get_head_mask(head_mask, n_layers)`,
    # removed from PreTrainedModel in 5.x. Sentence-transformers never passes a
    # head mask, and for None the 4.x method returned one None per layer.
    from transformers import PreTrainedModel
    if not hasattr(PreTrainedModel, "get_head_mask"):
        def get_head_mask(self, head_mask, num_hidden_layers,
                          is_attention_chunked=False):
            if head_mask is not None:
                raise NotImplementedError("head masking is not shimmed")
            return [None] * num_hidden_layers

        PreTrainedModel.get_head_mask = get_head_mask


_ALIBI_SIZE = 512


@contextmanager
def _no_meta_init():
    """Build the model on a real device instead of transformers 5's meta device.

    jina's encoder computes its ALiBi bias in __init__ by multiplying a CPU
    tensor (`torch.Tensor(slopes)`) with an `arange` that lands on meta, which
    raises. Scoped to this one load; every other model keeps the 5.x default.
    """
    import torch
    from transformers import PreTrainedModel
    orig = PreTrainedModel.__dict__.get("get_init_context")
    if orig is None:               # transformers 4.x: no meta init to undo
        yield
        return
    fn = orig.__func__

    def get_init_context(cls, *a, **k):
        return [c for c in fn(cls, *a, **k)
                if not (isinstance(c, torch.device) and c.type == "meta")]

    PreTrainedModel.get_init_context = classmethod(get_init_context)
    try:
        yield
    finally:
        PreTrainedModel.get_init_context = orig


def _restore_alibi(model):
    """Recompute jina's ALiBi buffer after loading.

    It is a non-persistent buffer -- absent from the checkpoint -- and
    transformers 5 zeroes such buffers during from_pretrained. A zero ALiBi
    loads and encodes without error but drops all positional information:
    against the Sep-20 node embeddings it gave mean cosine 0.977 (min 0.93)
    instead of 1.0. Rebuilt here, the match is exact (max |diff| 1.3e-7).

    Rebuilt at 512 positions, not jina's 8192: the full table is 8 heads x
    8192^2 float32 = 2 GiB, which does not fit beside an 8B vLLM server on a
    16 GB card. Entries depend only on (head, i, j), forward slices
    alibi[:, :, :seqlen, :seqlen], and jina's own forward regrows the table
    if an input is longer -- so the values used are identical.
    """
    for mod in model.modules():
        if hasattr(mod, "rebuild_alibi_tensor") and hasattr(mod, "alibi"):
            dev = mod.alibi.device
            del mod.alibi              # drop the zeroed table first
            mod.register_buffer(
                "alibi",
                mod.rebuild_alibi_tensor(size=_ALIBI_SIZE, device=dev),
                persistent=False)


def load_embedder(cfg, log):
    """The paper's embedder: jina-embeddings-v2-small-en.

    Deliberately not `graph.load_embedder`. That one serves nomic-embed, which
    requires `search_document:`/`search_query:` prefixes; jina takes raw text
    and prefixing it would corrupt every score.
    """
    if cfg.get("_mock"):
        from .graph import MockEmbedder
        log.warning("RoT embedder: MOCK (hash-based, not semantic)")
        return MockEmbedder()
    _shim_transformers_onnx()
    from sentence_transformers import SentenceTransformer
    name = cfg["rot"]["embedding"]["model"]
    dev = cfg["rot"]["embedding"]["device"]
    if dev == "auto":
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            dev = "cpu"
    log.info(f"RoT embedder: {name} on {dev}")
    try:
        # Load on CPU so jina's full-size ALiBi table never touches the GPU;
        # see _restore_alibi.
        with _no_meta_init():
            model = SentenceTransformer(name, trust_remote_code=True,
                                        device="cpu")
        _restore_alibi(model)
        return model.to(dev)
    except (ImportError, AttributeError, TypeError) as e:
        import transformers
        raise SystemExit(
            f"could not load {name}: {type(e).__name__}: {e}\n"
            f"\n"
            f"transformers {transformers.__version__} is installed. The paper's\n"
            f"embedder ships custom modelling code written against transformers\n"
            f"4.x, and 5.x removed several APIs it imports at module scope.\n"
            f"_shim_transformers_onnx() patches the known transformers-5 breaks; this\n"
            f"failure is past that point and should not be shimmed further.\n"
            f"\n"
            f"Fix the environment instead:\n"
            f"    pip install 'transformers>=4.40,<5'\n"
            f"This does not affect the vLLM servers -- they run in their own\n"
            f"environment and this package is only used to embed step text.\n"
            f"\n"
            f"Changing rot.embedding.model to something else is a deviation from\n"
            f"arXiv:2509.21743 and must be recorded as one.")


def embed(cfg, model, texts, log=None, label=""):
    """L2-normalised float32 vectors. No prefixes -- see load_embedder."""
    bs = cfg["rot"]["embedding"]["batch_size"]
    out = []
    for i in range(0, len(texts), bs):
        out.append(model.encode(list(texts[i:i + bs]),
                                normalize_embeddings=True,
                                show_progress_bar=False))
    if not out:
        return np.zeros((0, 512), "float32")
    vecs = np.vstack(out).astype(np.float32)
    if log:
        log.info(f"  embedded {label}: {vecs.shape}")
    return vecs


def build(cfg, entries, log):
    """Build V, E_Sequential and E_Semantic from memory-bank entries."""
    if not entries:
        raise ValueError("memory bank is empty -- run `main.py build-memory`")

    rc = cfg["rot"]
    tau = rc["tau_semantic"]
    cap = rc["max_neighbors"]

    # rot.step_text=generic swaps every step (and the approach it is embedded
    # with) for its teacher-written generic rewrite. Same entries, same step
    # counts, so the graph keeps its shape and only the texts change.
    generic = rc.get("step_text", "instance") == "generic"
    if generic:
        from . import generalize as GZ
        gen = GZ.load(cfg)
        missing = [e["entry_id"] for e in entries if e["entry_id"] not in gen]
        if missing:
            raise SystemExit(f"{len(missing)} memory entries have no generic "
                             f"rewrite -- re-run `python main.py generalize`")
        entries = [dict(e, approach=gen[e["entry_id"]]["approach_generic"],
                        subtasks=gen[e["entry_id"]]["subtasks_generic"])
                   for e in entries]

    # ── V ────────────────────────────────────────────────────────────────────
    nodes, seq_next = [], {}
    for e in entries:
        steps = e["subtasks"]
        first = len(nodes)
        for i, text in enumerate(steps):
            nodes.append({
                "node_id": len(nodes),
                "template_id": e["entry_id"],
                "step_index": i,
                "n_steps": len(steps),
                "text": text,
                # metadata, in the paper's sense: what the step is FOR, and
                # which knowledge it draws on. The candidate filter reads these.
                "template_type": e["approach"],
                "knowledge_tag": e.get("group"),
                "problem_id": e["problem_id"],
            })
        # E_Sequential: adjacent steps within the same template, w = 1.
        for i in range(len(steps) - 1):
            seq_next[first + i] = first + i + 1

    log.info(f"RoT graph: {len(nodes)} step nodes from {len(entries)} templates "
             f"({len(seq_next)} sequential edges)")

    # ── embeddings ───────────────────────────────────────────────────────────
    # A step is embedded WITH its template type. "Apply the formula" in
    # isolation is near-contentless and matches everything; the reward R_Q is a
    # query-to-node similarity, so a node that matches everything would be
    # selected for every query and the template would be noise.
    model = load_embedder(cfg, log)
    texts = [f"{n['template_type']} | {n['text']}" for n in nodes]
    vecs = embed(cfg, model, texts, log=log, label="RoT step nodes")

    # ── E_Semantic ───────────────────────────────────────────────────────────
    # All-pairs is fine at this scale (a few thousand nodes). `cap` is the one
    # concession the paper does not state: the definition admits every pair
    # over tau, and at tau=0.85 rescaled that is a dense graph. The cap keeps
    # the neighbour list bounded and is recorded in the metadata.
    sim = vecs @ vecs.T                      # RAW cosine; see module docstring
    adjacency, n_sem = {}, 0
    same_tmpl = np.array([n["template_id"] for n in nodes])
    for a in range(len(nodes)):
        row = sim[a].copy()
        row[a] = -1.0
        if rc["cross_template_only"]:
            row[same_tmpl == nodes[a]["template_id"]] = -1.0
        order = np.argsort(-row)
        if cap:
            order = order[:cap]
        nbrs = [{"to": int(b), "type": "semantic", "w": round(float(row[b]), 4)}
                for b in order if row[b] >= tau]
        if a in seq_next:
            nbrs.append({"to": seq_next[a], "type": "sequential", "w": 1.0})
        if nbrs:
            adjacency[str(a)] = nbrs
        n_sem += sum(1 for x in nbrs if x["type"] == "semantic")

    cross_tag = sum(1 for a, es in adjacency.items() for e in es
                    if e["type"] == "semantic"
                    and nodes[int(a)]["knowledge_tag"] != nodes[e["to"]]["knowledge_tag"])

    # The adaptation's known weak point, measured rather than assumed. In the
    # reference library this is 0.2%; anything near that is fine, anything near
    # ours (~37%) means the template injects another problem's constants into
    # the student's own reasoning voice.
    import re as _re
    _big = _re.compile(r"\d{4,}")
    n_specific = sum(1 for n in nodes if _big.search(n["text"] or ""))

    graph = {
        "metadata": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "paper": "arXiv:2509.21743",
            "template_source": "teacher-generated memory bank (not ReasonFlux-v2)",
            "step_text": "generic" if generic else "instance",
            "n_nodes": len(nodes),
            "n_templates": len(entries),
            "n_sequential_edges": len(seq_next),
            "n_semantic_edges": n_sem,
            "n_cross_tag_semantic_edges": cross_tag,
            "rot_instance_constants": {
                "n_steps_with_4plus_digit_literal": n_specific,
                "share": round(n_specific / max(len(nodes), 1), 4),
                "reference_library_share": 0.0019,
            },
            "tau_semantic_raw_cosine": tau,
            "max_neighbors": cap or None,
            "cross_template_only": rc["cross_template_only"],
            "embed_model": rc["embedding"]["model"],
        },
        "nodes": nodes,
        "adjacency": adjacency,
        "sequential": {str(k): v for k, v in seq_next.items()},
    }

    gpath, epath = _paths(cfg)
    Path(gpath).write_text(json.dumps(graph, ensure_ascii=False))
    np.save(epath, vecs)

    log.info(f"  semantic edges: {n_sem} ({cross_tag} cross-tag); "
             f"sequential: {len(seq_next)}")
    if n_specific / max(len(nodes), 1) > 0.05:
        log.warning(
            f"  {n_specific}/{len(nodes)} step texts "
            f"({n_specific / len(nodes):.0%}) contain a 4+ digit literal. The "
            f"reference template library is at 0.2%. These steps name another "
            f"problem's constants, and Thinking Intervention asserts them in "
            f"the student's own voice.")
    if n_sem == 0:
        log.warning(f"  no semantic edges at tau={tau} (rescaled) -- traversal "
                    f"will only ever follow sequential edges, which reduces RoT "
                    f"to 'replay one template verbatim'.")
    log.info(f"  -> {gpath}")
    return graph, vecs, dict(graph["metadata"])


def _paths(cfg):
    """Instance and generic graphs live side by side, so switching
    rot.step_text never silently evaluates against the other one."""
    sfx = ".generic" if cfg["rot"].get("step_text", "instance") == "generic" else ""
    return (data_path(cfg, f"rot_thought_graph{sfx}.json"),
            data_path(cfg, f"rot_node_embeddings{sfx}.npy"))


def load(cfg):
    gpath, epath = _paths(cfg)
    g = json.loads(Path(gpath).read_text())
    vecs = np.load(epath)
    return g, vecs
