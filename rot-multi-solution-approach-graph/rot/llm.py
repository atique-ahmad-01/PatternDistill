"""OpenAI-compatible chat client for vLLM-served models.

Both the teacher (Qwen3-8B) and the student (Qwen3-1.7B) are reached over HTTP,
so nothing in this pipeline loads model weights. That keeps the GPU concern
entirely inside the container and lets stages run, resume, and be debugged
without touching CUDA.

Token counts come from the server's `usage` block rather than a local
tokenizer, so the input/output token figures reported in results are the ones
the server actually billed -- not an approximation that could drift from it.
"""

from __future__ import annotations

import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


def split_thinking_tokens(text, completion_tokens):
    """(thinking_tokens, answer_tokens) estimated by character share.

    The server reports one completion_tokens for the whole reply, so the two
    parts are apportioned by character count. Approximate, and labelled as such
    -- but it is the quantity the experiment is actually about: whether a
    retrieved approach collapses the SEARCH for a strategy, which lives in the
    reasoning trace, not in the final answer.

    THE UNTERMINATED CASE IS NOT A FALLBACK, IT IS THE COMMON ONE. A reply that
    hit max_tokens mid-trace has an opening <think> and no closing tag. It is
    reasoning from end to end -- the model never got as far as an answer. An
    earlier version returned (0, completion_tokens) for it, i.e. booked a pure
    reasoning trace as pure answer, which inverted the headline measurement on
    exactly the longest-reasoning rows. Worse, truncation rates differ by arm,
    so the bias differed by arm too and the arm ordering it produced was an
    artifact of truncation rather than of reasoning length.

    Callers should still treat a truncated row's count as CENSORED (the true
    trace is >= completion_tokens, length unknown) and report token means over
    finished rows only.
    """
    import re
    text = text or ""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        think_chars = len(m.group(1))
        share = min(think_chars / max(len(text), 1), 1.0)
        think_tok = int(round(completion_tokens * share))
        return think_tok, completion_tokens - think_tok
    if "<think>" in text:
        return completion_tokens, 0      # unterminated: all reasoning
    return 0, completion_tokens          # no trace at all: all answer

def split_thinking_continued(text, completion_tokens):
    """(thinking, answer) for a reply whose <think> was opened by a PREFILL.

    Thinking Intervention hands the model an already-open reasoning block, so
    the generated text carries no opening tag -- it starts mid-trace. Passing it
    to split_thinking_tokens would find no <think>, conclude "no trace at all",
    and book the entire reasoning trace as answer tokens. That is the same
    inversion documented above, and it would hit every single RoT row.

    The prefill itself is NOT counted here. It is part of the prompt, so the
    server bills it as prompt_tokens -- which is the honest place for it, since
    the template is a cost the arm pays on the input side.
    """
    import re
    text = text or ""
    i = text.find("</think>")
    if i < 0:
        # Qwen3 often answers straight out of an open block without closing it.
        # The contract line is the boundary: everything before it was the
        # reasoning, and booking it all as reasoning would zero out
        # answer_tokens on every such reply.
        m = re.search(r"^\**\s*(?:FINAL\s+)?ANSWER\s*\**\s*[:=]",
                      text, re.IGNORECASE | re.MULTILINE)
        if not m:
            return completion_tokens, 0      # genuinely never reached an answer
        i = m.start()
    share = min(i / max(len(text), 1), 1.0)
    think_tok = int(round(completion_tokens * share))
    return think_tok, completion_tokens - think_tok


@dataclass
class Reply:
    """One completion plus everything the evaluation needs to account for it."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str = ""
    model: str = ""
    error: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    def split_thinking(self):
        """(thinking_tokens, answer_tokens) -- see split_thinking_tokens."""
        return split_thinking_tokens(self.text, self.completion_tokens)


@dataclass
class ChatClient:
    """Thin, retrying wrapper over an OpenAI-compatible /v1/chat/completions."""

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    max_tokens: int = 3072
    timeout: float = 900.0
    max_retries: int = 3
    concurrency: int = 8
    enable_thinking: bool = False
    seed: int | None = None
    _client: object = field(default=None, repr=False)
    _lock: object = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        from openai import OpenAI
        self._client = OpenAI(base_url=self.base_url,
                              api_key=self.api_key or "EMPTY",
                              timeout=self.timeout,
                              max_retries=0)  # retries handled here, with backoff

    @classmethod
    def from_config(cls, section: dict, seed: int | None = None):
        return cls(
            model=section["model"],
            base_url=section.get("base_url", "http://localhost:8000/v1"),
            api_key=section.get("api_key", "EMPTY"),
            temperature=section.get("temperature", 0.0),
            top_p=section.get("top_p", 1.0),
            top_k=section.get("top_k"),
            max_tokens=section.get("max_tokens", 3072),
            timeout=section.get("timeout", 900),
            max_retries=section.get("max_retries", 3),
            concurrency=section.get("concurrency", 8),
            enable_thinking=section.get("enable_thinking", False),
            seed=seed,
        )

    def health(self) -> tuple[bool, str]:
        """Check the server is up and serving the configured model."""
        try:
            served = [m.id for m in self._client.models.list().data]
        except Exception as e:
            return False, f"cannot reach {self.base_url}: {type(e).__name__}: {e}"
        if self.model not in served:
            return False, (f"{self.base_url} is serving {served}, "
                           f"not {self.model!r}")
        return True, f"{self.model} @ {self.base_url}"

    def chat(self, system: str, user: str, n: int = 1, prefill: str | None = None,
             **overrides) -> list[Reply]:
        """One request, `n` sampled completions. Returns one Reply per sample.

        vLLM shares the prefill across the n samples, so n=5 costs far less
        than five separate requests -- and the samples stay independent.

        `prefill` opens the assistant turn with fixed text the model then
        continues, which is how Thinking Intervention places a template INSIDE
        the reasoning block. Without it the template can only be described to
        the model in the user turn, which is a different intervention and not
        the one arXiv:2509.21743 measures.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        if prefill:
            messages.append({"role": "assistant", "content": prefill})

        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=overrides.get("temperature", self.temperature),
            top_p=overrides.get("top_p", self.top_p),
            max_tokens=overrides.get("max_tokens", self.max_tokens),
            n=n,
        )
        if self.seed is not None:
            kwargs["seed"] = self.seed
        # Qwen3 gates reasoning through the chat template. top_k has no field
        # in the OpenAI schema, so vLLM takes it through extra_body alongside.
        extra = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if self.top_k is not None:
            extra["top_k"] = self.top_k
        if prefill:
            # vLLM renders the final assistant message verbatim and generates
            # from its end. Both flags are required together; setting only one
            # is a 400 from the server.
            extra["add_generation_prompt"] = False
            extra["continue_final_message"] = True
        kwargs["extra_body"] = extra

        last_err = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                dt = time.perf_counter() - t0
                usage = resp.usage
                # The server reports prompt tokens once for the whole request;
                # attribute them to each sample so per-sample accounting stays
                # additive, and note that n>1 shares one prefill.
                p_tok = getattr(usage, "prompt_tokens", 0) or 0
                c_tot = getattr(usage, "completion_tokens", 0) or 0
                out = []
                for ch in resp.choices:
                    out.append(Reply(
                        text=(ch.message.content or ""),
                        prompt_tokens=p_tok,
                        completion_tokens=c_tot // max(len(resp.choices), 1),
                        latency_s=dt,
                        finish_reason=ch.finish_reason or "",
                        model=self.model,
                    ))
                return out
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    back = min(2 ** attempt + random.random(), 30)
                    time.sleep(back)

        return [Reply(text="", latency_s=0.0, model=self.model,
                      error=last_err or "unknown") for _ in range(n)]

    def map(self, jobs, fn, desc="", log=None):
        """Run `fn(job)` over jobs concurrently, preserving input order."""
        results = [None] * len(jobs)
        done = {"n": 0}
        lock = threading.Lock()

        def _run(i_job):
            i, job = i_job
            out = fn(job)
            with lock:
                done["n"] += 1
                if log and (done["n"] % 10 == 0 or done["n"] == len(jobs)):
                    log.info(f"  {desc} {done['n']}/{len(jobs)}")
            return i, out

        with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as ex:
            for i, out in ex.map(_run, list(enumerate(jobs))):
                results[i] = out
        return results


# ── Offline backend ──────────────────────────────────────────────────────────

class MockChatClient(ChatClient):
    """Deterministic fake server, for testing the pipeline without a GPU.

    It exists so the wiring -- prompt construction, parsing, memory, graph,
    retrieval, scoring, reporting -- can be verified end to end before any GPU
    time is spent. It answers in the exact contract format the real prompts
    ask for, and it is never a substitute for a real run: results produced with
    it are stamped mock=true.
    """

    def __post_init__(self):
        self._client = None
        self._answers = {}
        self._by_text = {}
        self._error_rate = 0.0

    def set_answers(self, mapping, problems=None, error_rate=0.0):
        """problem_id -> integer the mock should 'solve' it as.

        `problems` lets the mock recognise which problem a prompt refers to.
        The prompt contains only the problem TEXT (ids never appear in it, and
        must not -- an id in the prompt would be a leak), so the mock matches
        on text.
        """
        self._answers = dict(mapping)
        # Deterministic share of deliberately-wrong replies. The teacher gets a
        # non-zero rate so the correctness filter and the "family the teacher
        # cannot solve" path are both exercised; the student gets 0 so arm
        # differences stay attributable to the prompt.
        self._error_rate = error_rate
        if problems:
            self._by_text = {self._key(p["problem"]): p["problem_id"]
                             for p in problems}

    @staticmethod
    def _norm(text):
        import re
        return re.sub(r"\s+", " ", str(text).strip().lower())

    @classmethod
    def _key(cls, text):
        # Full normalised text, NOT a prefix. RB-DERIV families contain a dozen
        # near-identical instances differing only in their trailing numbers, so
        # any prefix key collides across siblings and the mock answers the
        # wrong question.
        return cls._norm(text)

    def health(self):
        return True, f"MOCK {self.model} (no server)"

    def chat(self, system: str, user: str, n: int = 1, prefill: str | None = None,
             **overrides) -> list[Reply]:
        import hashlib
        key = hashlib.sha256(user.encode()).hexdigest()
        pid = None
        # The query problem is always LAST in the prompt; retrieved memories
        # come before it and may contain their own problem text (arm C). Take
        # the latest match so the mock answers the question being asked.
        haystack = self._norm(user)
        best = -1
        for text_key, candidate in self._by_text.items():
            pos = haystack.rfind(text_key)
            if pos > best:
                best, pid = pos, candidate
        if best < 0:
            pid = None
        base = self._answers.get(pid)

        out = []
        for i in range(n):
            if base is None:
                val = int(key[:6], 16) % 1000
            elif self._error_rate and (int(key[:8], 16) % 100) < self._error_rate * 100:
                val = base + 7 + i          # deterministically wrong
            else:
                val = base
            if prefill:
                # Continue the caller's open block instead of opening a new one,
                # so the RoT accounting path is what gets exercised offline.
                text = ("The template above gives the structure; applying it.\n"
                        "</think>\n"
                        f"ANSWER: {val}\n")
                out.append(Reply(text=text, prompt_tokens=(len(user) + len(prefill)) // 4,
                                 completion_tokens=len(text) // 4,
                                 latency_s=0.001, finish_reason="stop",
                                 model=self.model + "(mock)"))
                continue
            think = ("<think>\nLet me look for a structural shortcut here. "
                     "Brute force is impossible at this size, so I need a "
                     "closed form or a cycle. Trying the standard identity for "
                     "this family.\n</think>\n")
            text = (think
                    + f"APPROACH: mock strategy variant {i + 1}\n"
                    f"STEP 1: Identify the structure of the problem.\n"
                    f"STEP 2: Apply the closed form for this family.\n"
                    f"STEP 3: Reduce and evaluate.\n"
                    f"ANSWER: {val}\n")
            out.append(Reply(text=text, prompt_tokens=len(user) // 4,
                             completion_tokens=len(text) // 4,
                             latency_s=0.001, finish_reason="stop",
                             model=self.model + "(mock)"))
        return out


def build_client(section: dict, seed=None, mock=False) -> ChatClient:
    cls = MockChatClient if mock else ChatClient
    return cls.from_config(section, seed=seed)
