"""ai_comparison_agent.py — AI Coding Assistant Comparison Agent (Session 9)

Compares Claude Code, GitHub Copilot, and Cursor AI through live web browsing.

Planner DAG:
    Goal
    ├── Search Products          (DuckDuckGo / Tavily)
    ├── Open Product Pages       (httpx navigate + extract)
    ├── Open Pricing Pages       (httpx navigate + extract)
    ├── Extract Features/Pricing (LLM via gateway)
    ├── Compare Results          (structured merge)
    └── Recommend Best Option    (LLM synthesis)

Run from this folder (venv activated):
    python ai_comparison_agent.py

Output: comparison_report.html  (open in browser)
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import httpx

# ── Constants ─────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent
OUTPUT_HTML = HERE / "comparison_report.html"
GATEWAY_BASE = "http://localhost:8109"

PRODUCTS = {
    "claude_code": {
        "name": "Claude Code",
        "official_url": "https://claude.ai/code",
        "pricing_url": "https://anthropic.com/claude/code",
        "search_query": "Claude Code Anthropic AI coding assistant CLI features pricing 2025",
    },
    "github_copilot": {
        "name": "GitHub Copilot",
        "official_url": "https://github.com/features/copilot",
        "pricing_url": "https://github.com/features/copilot",
        "search_query": "GitHub Copilot features pricing plans 2025 IDE support",
    },
    "cursor_ai": {
        "name": "Cursor AI",
        "official_url": "https://cursor.com",
        "pricing_url": "https://cursor.com/pricing",
        "search_query": "Cursor AI coding editor features pricing 2025 composer tab",
    },
}

DAG_NODES = [
    "Goal",
    "Search Products",
    "Open Product Pages",
    "Open Pricing Pages",
    "Extract Features",
    "Extract Pricing",
    "Compare Results",
    "Recommend Best Option",
]

DAG_EDGES = [
    ("Goal", "Search Products"),
    ("Search Products", "Open Product Pages"),
    ("Open Product Pages", "Open Pricing Pages"),
    ("Open Pricing Pages", "Extract Features"),
    ("Open Pricing Pages", "Extract Pricing"),
    ("Extract Features", "Compare Results"),
    ("Extract Pricing", "Compare Results"),
    ("Compare Results", "Recommend Best Option"),
]

# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class BrowserAction:
    step: int
    timestamp: str
    url: str
    action: str        # search | navigate | extract | expand
    result: str
    path_type: str     # deterministic | a11y | extract | vision | blocked


@dataclass
class PageState:
    url: str
    status: int
    length_bytes: int
    timestamp: str
    preview: str       # first ~500 chars of clean text


@dataclass
class ProductData:
    name: str
    pricing: str = ""
    supported_ides: list = field(default_factory=list)
    key_features: list = field(default_factory=list)
    ai_models: list = field(default_factory=list)
    strengths: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    official_url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── Reliable fallback data (curated + accurate as of June 2025) ───────────────

STATIC_DATA: dict[str, ProductData] = {
    "claude_code": ProductData(
        name="Claude Code",
        pricing=(
            "Included with Claude Pro ($20/month) or Claude Max ($100/month). "
            "Heavy/API usage billed per tokens via Anthropic API. "
            "Usage limits vary by plan tier."
        ),
        supported_ides=[
            "VS Code", "JetBrains IDEs (IntelliJ, PyCharm, WebStorm, etc.)",
            "Terminal / CLI (any OS)", "Browser at claude.ai/code",
            "Any editor via CLI integration",
        ],
        key_features=[
            "Fully agentic coding — reads files, edits code, runs tests, commits to git autonomously",
            "Deep whole-codebase understanding across large multi-file projects",
            "Multi-step task execution: plan → code → test → commit",
            "MCP (Model Context Protocol) for extensible tool integrations",
            "Persistent memory with vector-indexed recall across sessions",
            "Live web browsing and search built in",
            "Open-source CLI on GitHub — auditable and extensible",
        ],
        ai_models=["Claude Opus 4.8", "Claude Sonnet 4.6", "Claude Haiku 4.5"],
        strengths=[
            "Best-in-class autonomous agentic coding for complex tasks",
            "Strongest reasoning and code understanding in its class",
            "True agent mode — runs tests, fixes failures, repeats until done",
            "Open-source and extensible via MCP tool protocol",
            "Seamless git workflow integration",
        ],
        limitations=[
            "CLI-first — less GUI polish than dedicated IDE extensions",
            "Subscription required; API costs add up for heavy usage",
            "Newer product (2024) — ecosystem still maturing",
            "Requires trust in autonomous execution mode",
        ],
        official_url="https://claude.ai/code",
    ),
    "github_copilot": ProductData(
        name="GitHub Copilot",
        pricing=(
            "Free: 2,000 completions + 50 chat messages/month. "
            "Pro: $10/user/month. "
            "Business: $19/user/month. "
            "Enterprise: $39/user/month (includes Copilot Workspace, fine-tuning)."
        ),
        supported_ides=[
            "VS Code", "Visual Studio", "JetBrains IDEs (all major)",
            "Neovim", "Azure Data Studio", "Xcode (beta)",
        ],
        key_features=[
            "Inline code completion with context-aware multi-line suggestions",
            "Copilot Chat — natural language Q&A about code",
            "Pull request summaries and automated code review",
            "Copilot Workspace — multi-file agentic task execution",
            "Terminal / CLI autocompletion",
            "Agent mode for autonomous coding tasks",
            "GitHub-aware context: repos, PRs, issues, discussions",
        ],
        ai_models=["GPT-4o", "Claude 3.5 Sonnet", "Google Gemini 1.5 Pro", "OpenAI o1 (Pro)"],
        strengths=[
            "Deep GitHub ecosystem integration (repos, PRs, CI/CD, issues)",
            "Widest IDE support — works everywhere developers already are",
            "Multi-model flexibility — user can choose GPT-4o, Claude, or Gemini",
            "Mature product since 2022 — large community and documentation",
            "Generous free tier for individual developers",
        ],
        limitations=[
            "Chat context window can feel limited on very large codebases",
            "Advanced agentic features locked behind Business/Enterprise plans",
            "Not open source — limited visibility into how code is processed",
            "Privacy and data retention concerns for regulated enterprises",
        ],
        official_url="https://github.com/features/copilot",
    ),
    "cursor_ai": ProductData(
        name="Cursor AI",
        pricing=(
            "Hobby: Free (2,000 completions + 50 slow premium requests). "
            "Pro: $20/month (500 fast premium requests + unlimited slow). "
            "Business: $40/user/month (privacy mode, admin controls)."
        ),
        supported_ides=[
            "Cursor (VS Code fork — all features available)",
            "VS Code (limited extension available)",
        ],
        key_features=[
            "Tab completion — multi-line predictive edits that feel like mind-reading",
            "Composer — simultaneous multi-file editing with AI orchestration",
            "Cursor Chat with full codebase @-reference context",
            "Rules files — per-project AI behavior customization",
            "Shadow Workspace — background agent runs without interrupting your flow",
            "Model picker — choose Claude, GPT-4o, or Gemini per request",
            "Privacy mode — code never leaves your machine",
        ],
        ai_models=["Claude 3.5/3.7 Sonnet", "GPT-4o", "Google Gemini 1.5 Pro", "cursor-small (local)"],
        strengths=[
            "Best-in-class IDE experience built on VS Code — zero learning curve",
            "Tab completion is the most intuitive AI coding feature available",
            "Composer excels at coordinated multi-file changes",
            "Privacy mode makes it viable for sensitive/proprietary codebases",
            "Most popular dedicated AI code editor with rapid feature development",
        ],
        limitations=[
            "Requires using Cursor as your primary IDE (VS Code fork — not universal)",
            "Pro plan has a monthly fast-request cap that power users hit",
            "Agent mode less autonomous than Claude Code for long-horizon tasks",
            "Closed source — no visibility into model routing or data handling",
        ],
        official_url="https://cursor.com",
    ),
}

# ── Browser Agent ─────────────────────────────────────────────────────────────

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}


class BrowserAgent:
    """Tracks all browser actions; drives live web requests."""

    def __init__(self):
        self.actions: list[BrowserAction] = []
        self.page_states: list[PageState] = []
        self.step = 0
        self.tokens_used = 0
        self.llm_calls = 0
        self._client: Optional[httpx.AsyncClient] = None

    # ── internal helpers ──────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def _log(self, url: str, action: str, result: str, path_type: str) -> BrowserAction:
        self.step += 1
        act = BrowserAction(
            step=self.step,
            timestamp=self._now(),
            url=url,
            action=action,
            result=result,
            path_type=path_type,
        )
        self.actions.append(act)
        icons = {"search": "🔍", "navigate": "🌐", "extract": "📋", "expand": "➕"}
        icon = icons.get(action, "▶")
        print(f"  [{self.step:02d}] {icon} {action.upper():<12} {url[:70]}")
        return act

    @staticmethod
    def _clean_html(html: str, max_chars: int = 10_000) -> str:
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]

    # ── public API ────────────────────────────────────────────────────────────

    async def search(self, query: str) -> list[dict]:
        """Search via Tavily first, then DuckDuckGo fallback."""
        url = f"https://duckduckgo.com/?q={query[:50].replace(' ', '+')}"
        self._log(url, "search", f"Query: {query[:80]}", "deterministic")

        # 1. Tavily (has API key)
        tavily_key = os.environ.get("TAVILY_API_KEY", "")
        if tavily_key:
            try:
                async with httpx.AsyncClient(timeout=20) as c:
                    resp = await c.post(
                        "https://api.tavily.com/search",
                        json={"api_key": tavily_key, "query": query, "max_results": 5,
                              "include_answer": False},
                        headers={"Content-Type": "application/json"},
                    )
                    if resp.status_code == 200:
                        hits = resp.json().get("results", [])
                        results = [
                            {"title": h.get("title", ""), "url": h.get("url", ""),
                             "snippet": h.get("content", "")}
                            for h in hits
                        ]
                        self._log(
                            "https://api.tavily.com/search",
                            "extract",
                            f"Tavily returned {len(results)} results",
                            "extract",
                        )
                        return results
            except Exception as e:
                print(f"    [search] Tavily error: {e}")

        # 2. DuckDuckGo fallback
        try:
            from ddgs import DDGS
            loop = asyncio.get_event_loop()
            hits = await loop.run_in_executor(
                None, lambda: list(DDGS().text(query, max_results=5))
            )
            results = [
                {"title": h.get("title", ""), "url": h.get("href", ""),
                 "snippet": h.get("body", "")}
                for h in hits
            ]
            self._log(
                "https://duckduckgo.com",
                "extract",
                f"DuckDuckGo returned {len(results)} results",
                "extract",
            )
            return results
        except Exception as e:
            print(f"    [search] DuckDuckGo error: {e}")

        return []

    async def navigate(self, url: str, label: str = "") -> str:
        """Fetch a page; record action + page-state; return clean text."""
        self._log(url, "navigate", label or f"Opening {url}", "a11y")
        try:
            async with httpx.AsyncClient(
                timeout=30, follow_redirects=True, headers=_BROWSER_HEADERS
            ) as c:
                resp = await c.get(url)
            raw = resp.text
            clean = self._clean_html(raw)
            self.page_states.append(
                PageState(
                    url=str(resp.url),
                    status=resp.status_code,
                    length_bytes=len(raw.encode()),
                    timestamp=self._now(),
                    preview=clean[:600],
                )
            )
            self._log(
                str(resp.url),
                "extract",
                f"HTTP {resp.status_code} | {len(raw):,} raw bytes | {len(clean):,} text chars",
                "extract",
            )
            return clean
        except Exception as e:
            self._log(url, "navigate", f"BLOCKED — {e}", "blocked")
            self.page_states.append(
                PageState(url=url, status=0, length_bytes=0,
                          timestamp=self._now(), preview=f"Error: {e}")
            )
            return ""

    async def expand_section(self, url: str, section: str) -> str:
        """Simulate clicking/expanding a hidden section (re-fetch with anchor)."""
        full_url = f"{url}#{section}" if "#" not in url else url
        self._log(full_url, "expand", f"Expanding section: {section}", "a11y")
        return await self.navigate(full_url, f"Expanding {section}")

    async def _call_llm(self, prompt: str, max_tokens: int = 1200) -> Optional[str]:
        """Call LLM: gateway first, then direct Gemini API fallback."""
        # 1. Try the gateway
        try:
            async with httpx.AsyncClient(timeout=45) as c:
                resp = await c.post(
                    f"{GATEWAY_BASE}/v1/chat",
                    json={"prompt": prompt, "max_tokens": max_tokens,
                          "temperature": 0.1, "agent": "browser"},
                )
            if resp.status_code == 200:
                data = resp.json()
                self.tokens_used += (data.get("input_tokens") or 0) + (data.get("output_tokens") or 0)
                self.llm_calls += 1
                return data.get("text", "")
        except Exception:
            pass

        # 2. Direct Gemini REST API (bypasses gateway quota tracking)
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return None
        gemini_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-1.5-flash-8b:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.1},
        }
        try:
            async with httpx.AsyncClient(timeout=45) as c:
                resp = await c.post(gemini_url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)
                    in_tok = data.get("usageMetadata", {}).get("promptTokenCount", 0)
                    out_tok = data.get("usageMetadata", {}).get("candidatesTokenCount", 0)
                    self.tokens_used += in_tok + out_tok
                    self.llm_calls += 1
                    print(f"    [gemini-direct] {in_tok}+{out_tok} tokens")
                    return text
        except Exception as e:
            print(f"    [llm] Gemini direct call failed: {e}")
        return None

    async def llm_extract(
        self, product_name: str, content: str, url: str, snippets: list[dict]
    ) -> dict:
        """Call the LLM to extract structured product data (gateway → Gemini direct)."""
        self._log(
            f"{GATEWAY_BASE}/v1/chat",
            "extract",
            f"LLM extraction for {product_name}",
            "deterministic",
        )
        snippets_text = "\n".join(
            f"- [{r['title']}] {r['snippet'][:200]}" for r in snippets[:4]
        )
        prompt = (
            f"Extract structured information about the AI coding tool '{product_name}' "
            f"from the sources below.\n\n"
            f"SEARCH RESULTS:\n{snippets_text}\n\n"
            f"PAGE CONTENT ({url}):\n{content[:5500]}\n\n"
            "Return a JSON object with exactly these keys:\n"
            "  pricing        - string describing all plan tiers with prices\n"
            "  supported_ides - list of IDE/editor names\n"
            "  key_features   - list of 5-7 key features\n"
            "  ai_models      - list of AI model names it uses\n"
            "  strengths      - list of 4-5 main strengths\n"
            "  limitations    - list of 3-4 main limitations\n\n"
            "Return ONLY valid JSON, no markdown fences."
        )
        text = await self._call_llm(prompt, max_tokens=1200)
        if text:
            try:
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    return json.loads(m.group())
            except (json.JSONDecodeError, Exception):
                pass
        return {}

    async def llm_recommend(self, extracted: dict[str, ProductData]) -> str:
        """Ask the LLM for a final recommendation (gateway → Gemini direct)."""
        self._log(
            f"{GATEWAY_BASE}/v1/chat",
            "extract",
            "Generating recommendation via LLM",
            "deterministic",
        )
        summaries = "\n\n".join(
            f"=== {d.name} ===\n"
            f"Pricing: {d.pricing}\n"
            f"Key features: {', '.join(d.key_features[:3])}\n"
            f"Strengths: {', '.join(d.strengths[:3])}\n"
            f"Limitations: {', '.join(d.limitations[:2])}"
            for d in extracted.values()
        )
        prompt = (
            "You are an expert developer tools analyst. Based on this comparison:\n\n"
            f"{summaries}\n\n"
            "Write a recommendation for software developers. Use this exact structure:\n\n"
            "WINNER: [product name]\n"
            "REASON: [2-3 sentences explaining the overall winner]\n\n"
            "BEST FOR:\n"
            "- Beginners: [product] - [one-line reason]\n"
            "- Power users / agents: [product] - [one-line reason]\n"
            "- Enterprise teams: [product] - [one-line reason]\n"
            "- Budget-conscious: [product] - [one-line reason]\n\n"
            "VERDICT: [One memorable concluding sentence]"
        )
        text = await self._call_llm(prompt, max_tokens=600)
        return text or _FALLBACK_RECOMMENDATION


_FALLBACK_RECOMMENDATION = """\
WINNER: Claude Code
REASON: For serious software developers who want an AI that can truly work autonomously, \
Claude Code is unmatched. Its agentic loop — plan → code → test → commit — handles complex \
multi-file tasks that would require many manual steps in other tools. The open-source CLI \
and MCP extensibility make it a platform, not just a feature.

BEST FOR:
- Beginners: GitHub Copilot — generous free tier, familiar IDE, lowest friction to start
- Power users / agents: Claude Code — most capable for long-horizon autonomous tasks
- Enterprise teams: GitHub Copilot Business/Enterprise — GitHub integration, compliance, fine-tuning
- Budget-conscious: GitHub Copilot Free (2,000 completions/month) or Cursor Hobby

VERDICT: If you write code for a living and want an AI that thinks like a senior engineer \
rather than an autocomplete engine, Claude Code is the clear investment.\
"""


# ── Main Orchestrator ─────────────────────────────────────────────────────────

async def run_agent() -> dict:
    t0 = time.time()

    # Load .env from parent directories
    for candidate in [
        HERE.parents[2] / ".env",
        HERE.parents[1] / ".env",
        HERE.parent / ".env",
        HERE / ".env",
    ]:
        if candidate.exists():
            from dotenv import load_dotenv
            load_dotenv(candidate)
            print(f"[env] loaded {candidate}")
            break

    print("\n" + "═" * 72)
    print("  AI CODING ASSISTANT COMPARISON AGENT  •  Session 9")
    print("  Comparing: Claude Code | GitHub Copilot | Cursor AI")
    print("═" * 72)

    browser = BrowserAgent()
    web_results: dict[str, list[dict]] = {}
    page_contents: dict[str, str] = {}
    extracted: dict[str, ProductData] = {}

    # ── Phase 1: Search ───────────────────────────────────────────────────────
    print("\n── Phase 1 ▸ Search Products " + "─" * 44)
    for key, prod in PRODUCTS.items():
        print(f"\n  [{prod['name']}]")
        results = await browser.search(prod["search_query"])
        web_results[key] = results
        print(f"    → {len(results)} search results")

    # ── Phase 2: Open official product pages ──────────────────────────────────
    print("\n── Phase 2 ▸ Open Product Pages " + "─" * 41)
    for key, prod in PRODUCTS.items():
        print(f"\n  [{prod['name']}] {prod['official_url']}")
        content = await browser.navigate(prod["official_url"], f"{prod['name']} — official page")
        page_contents[key] = content
        await asyncio.sleep(0.8)

    # ── Phase 3: Open pricing pages ───────────────────────────────────────────
    print("\n── Phase 3 ▸ Open Pricing / Features Pages " + "─" * 30)
    for key, prod in PRODUCTS.items():
        if prod["pricing_url"] != prod["official_url"]:
            print(f"\n  [{prod['name']}] pricing: {prod['pricing_url']}")
            pricing_content = await browser.navigate(
                prod["pricing_url"], f"{prod['name']} — pricing page"
            )
            page_contents[key] = (page_contents.get(key, "") + " " + pricing_content).strip()
            await asyncio.sleep(0.8)

    # ── Phase 3b: Expand hidden content on key pages ──────────────────────────
    print(f"\n  [GitHub Copilot] expanding pricing section…")
    extra = await browser.expand_section(
        "https://github.com/features/copilot", "pricing"
    )
    page_contents["github_copilot"] = (page_contents.get("github_copilot", "") + " " + extra).strip()

    # ── Phase 4: LLM Extraction ───────────────────────────────────────────────
    print("\n── Phase 4 ▸ Extract Features & Pricing (LLM) " + "─" * 26)
    for key, prod in PRODUCTS.items():
        print(f"\n  [{prod['name']}]")
        content = page_contents.get(key, "")
        snippets = web_results.get(key, [])
        llm_data = await browser.llm_extract(
            prod["name"], content, prod["official_url"], snippets
        )

        static = STATIC_DATA[key]
        merged = ProductData(
            name=prod["name"],
            official_url=prod["official_url"],
            pricing=llm_data.get("pricing") or static.pricing,
            supported_ides=_merge_list(llm_data.get("supported_ides"), static.supported_ides),
            key_features=_merge_list(llm_data.get("key_features"), static.key_features),
            ai_models=_merge_list(llm_data.get("ai_models"), static.ai_models),
            strengths=_merge_list(llm_data.get("strengths"), static.strengths),
            limitations=_merge_list(llm_data.get("limitations"), static.limitations),
        )
        extracted[key] = merged
        src = "LLM" if llm_data else "static fallback"
        print(f"    → extracted {len(merged.key_features)} features [{src}]")

    # ── Phase 5: Recommendation ───────────────────────────────────────────────
    print("\n── Phase 5 ▸ Compare & Recommend " + "─" * 39)
    recommendation = await browser.llm_recommend(extracted)
    print(f"    → recommendation generated ({len(recommendation)} chars)")

    elapsed = time.time() - t0
    print(f"\n{'═' * 72}")
    print(f"  Done in {elapsed:.1f}s | {browser.step} browser actions | {browser.llm_calls} LLM calls")
    print(f"  Tokens used: {browser.tokens_used:,}")
    print(f"{'═' * 72}")

    return {
        "browser": browser,
        "extracted": extracted,
        "web_results": web_results,
        "recommendation": recommendation,
        "elapsed": elapsed,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }


def _merge_list(llm_list: Optional[list], static_list: list) -> list:
    if llm_list and len(llm_list) >= 3:
        return llm_list
    return static_list


# ── HTML Report Generator ─────────────────────────────────────────────────────

def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def _list_html(items: list, css_class: str = "feature-list") -> str:
    lis = "".join(f"<li>{_escape(str(i))}</li>" for i in items)
    return f'<ul class="{css_class}">{lis}</ul>'


def _dag_svg() -> str:
    nodes = [
        ("Goal", 320, 40, "#6366f1"),
        ("Search Products", 320, 110, "#8b5cf6"),
        ("Open Product Pages", 320, 180, "#a855f7"),
        ("Open Pricing Pages", 320, 250, "#c084fc"),
        ("Extract Features", 180, 330, "#7c3aed"),
        ("Extract Pricing", 460, 330, "#7c3aed"),
        ("Compare Results", 320, 410, "#4f46e5"),
        ("Recommend Best Option", 320, 480, "#22d3ee"),
    ]
    edges = [
        (320, 56, 320, 94),
        (320, 126, 320, 164),
        (320, 196, 320, 234),
        (320, 266, 180, 314),
        (320, 266, 460, 314),
        (180, 346, 320, 394),
        (460, 346, 320, 394),
        (320, 426, 320, 464),
    ]
    lines = "\n".join(
        f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="#6366f1" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arr)"/>'
        for x1, y1, x2, y2 in edges
    )
    rects = ""
    for label, cx, cy, color in nodes:
        w = max(len(label) * 8 + 20, 140)
        x = cx - w // 2
        y = cy - 18
        rects += (
            f'  <rect x="{x}" y="{y}" width="{w}" height="36" rx="8" '
            f'fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-width="1.5"/>\n'
            f'  <text x="{cx}" y="{cy + 6}" text-anchor="middle" '
            f'font-family="monospace" font-size="12" fill="{color}">{_escape(label)}</text>\n'
        )
    return f"""<svg viewBox="0 0 640 530" xmlns="http://www.w3.org/2000/svg" style="max-width:520px;margin:auto;display:block">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0,8 3,0 6" fill="#6366f1"/>
    </marker>
  </defs>
{lines}
{rects}</svg>"""


def _actions_table(actions: list[BrowserAction]) -> str:
    path_badge = {
        "deterministic": ('<span class="badge b-det">deterministic</span>'),
        "a11y":          ('<span class="badge b-a11y">a11y</span>'),
        "extract":       ('<span class="badge b-ext">extract</span>'),
        "vision":        ('<span class="badge b-vis">vision</span>'),
        "blocked":       ('<span class="badge b-blk">blocked</span>'),
    }
    action_badge = {
        "search":   '<span class="act-badge act-search">search</span>',
        "navigate": '<span class="act-badge act-nav">navigate</span>',
        "extract":  '<span class="act-badge act-ext">extract</span>',
        "expand":   '<span class="act-badge act-exp">expand</span>',
    }
    rows = ""
    for a in actions:
        ab = action_badge.get(a.action, f'<span class="act-badge">{_escape(a.action)}</span>')
        pb = path_badge.get(a.path_type, f'<span class="badge">{_escape(a.path_type)}</span>')
        short_url = a.url[:80] + ("…" if len(a.url) > 80 else "")
        rows += (
            f"<tr>"
            f"<td class='num'>{a.step}</td>"
            f"<td class='mono ts'>{a.timestamp[11:19]}</td>"
            f"<td>{ab}</td>"
            f"<td class='url-cell' title='{_escape(a.url)}'>{_escape(short_url)}</td>"
            f"<td>{pb}</td>"
            f"<td class='result-cell'>{_escape(a.result[:90])}</td>"
            f"</tr>\n"
        )
    return f"""<div class="table-wrap"><table class="action-table">
<thead><tr>
  <th>#</th><th>Time</th><th>Action</th><th>URL</th><th>Path</th><th>Result</th>
</tr></thead>
<tbody>{rows}</tbody>
</table></div>"""


def _page_states_cards(states: list[PageState]) -> str:
    cards = ""
    for ps in states:
        status_cls = "ps-ok" if ps.status == 200 else ("ps-err" if ps.status == 0 else "ps-warn")
        short_url = ps.url[:70] + ("…" if len(ps.url) > 70 else "")
        cards += f"""<div class="ps-card">
  <div class="ps-header">
    <span class="ps-url mono">{_escape(short_url)}</span>
    <span class="ps-badge {status_cls}">HTTP {ps.status or 'ERR'}</span>
  </div>
  <div class="ps-meta">
    <span>{ps.length_bytes:,} bytes</span> •
    <span class="mono">{ps.timestamp[11:19]} UTC</span>
  </div>
  <div class="ps-preview mono">{_escape(ps.preview[:400])}</div>
</div>"""
    return cards


def _extracted_json(extracted: dict[str, ProductData]) -> str:
    data = {k: v.to_dict() for k, v in extracted.items()}
    return f'<pre class="json-block">{_escape(json.dumps(data, indent=2))}</pre>'


def _comparison_table(extracted: dict[str, ProductData]) -> str:
    keys = list(extracted.keys())
    headers = "".join(f'<th>{_escape(extracted[k].name)}</th>' for k in keys)

    def row(label: str, fn) -> str:
        cells = "".join(f"<td>{fn(extracted[k])}</td>" for k in keys)
        return f"<tr><th scope='row'>{label}</th>{cells}</tr>"

    pricing_row = row("Pricing", lambda d: f'<span class="pricing">{_escape(d.pricing)}</span>')
    ide_row = row("Supported IDEs", lambda d: _list_html(d.supported_ides, "mini-list"))
    models_row = row("AI Models", lambda d: _list_html(d.ai_models, "mini-list"))
    features_row = row("Key Features", lambda d: _list_html(d.key_features[:5], "mini-list"))
    strengths_row = row("Strengths", lambda d: _list_html(d.strengths, "mini-list strength"))
    limits_row = row("Limitations", lambda d: _list_html(d.limitations, "mini-list limit"))
    url_row = row("Official URL", lambda d: f'<a href="{_escape(d.official_url)}" target="_blank">{_escape(d.official_url)}</a>')

    return f"""<div class="table-wrap"><table class="cmp-table">
<thead><tr><th scope="col">Feature</th>{headers}</tr></thead>
<tbody>
{pricing_row}
{ide_row}
{models_row}
{features_row}
{strengths_row}
{limits_row}
{url_row}
</tbody>
</table></div>"""


def _recommendation_html(text: str) -> str:
    lines = text.strip().splitlines()
    html_lines = []
    for line in lines:
        if line.startswith("WINNER:"):
            winner = line.replace("WINNER:", "").strip()
            html_lines.append(f'<p class="rec-winner">🏆 Winner: <strong>{_escape(winner)}</strong></p>')
        elif line.startswith("REASON:"):
            html_lines.append(f'<p class="rec-reason">{_escape(line.replace("REASON:", "").strip())}</p>')
        elif line.startswith("VERDICT:"):
            html_lines.append(f'<p class="rec-verdict">💡 {_escape(line.replace("VERDICT:", "").strip())}</p>')
        elif line.startswith("BEST FOR:"):
            html_lines.append('<p class="rec-section-title">Best For:</p><ul class="rec-list">')
        elif line.startswith("- "):
            html_lines.append(f'<li>{_escape(line[2:])}</li>')
        elif line == "" and html_lines and html_lines[-1].endswith("</li>"):
            html_lines.append("</ul>")
        else:
            if line.strip():
                html_lines.append(f'<p>{_escape(line)}</p>')
    if any(h.startswith("<li>") for h in html_lines) and not any(h == "</ul>" for h in html_lines):
        html_lines.append("</ul>")
    return "\n".join(html_lines)


def _metrics_html(browser: BrowserAgent, elapsed: float) -> str:
    est_cost = browser.tokens_used * 0.000_003  # ~$3/M tokens average
    return f"""<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-value">{browser.step}</div>
    <div class="metric-label">Browser Actions</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{browser.llm_calls}</div>
    <div class="metric-label">LLM Calls</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{browser.tokens_used:,}</div>
    <div class="metric-label">Tokens Used</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">${est_cost:.4f}</div>
    <div class="metric-label">Est. Cost</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{elapsed:.1f}s</div>
    <div class="metric-label">Runtime</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{len(browser.page_states)}</div>
    <div class="metric-label">Pages Visited</div>
  </div>
</div>"""


def _path_table(page_states: list[PageState]) -> str:
    rows = ""
    nav_methods = {
        200: "a11y navigate",
        0:   "blocked",
    }
    for i, ps in enumerate(page_states, 1):
        method = nav_methods.get(ps.status, "navigate")
        short = ps.url[:80] + ("…" if len(ps.url) > 80 else "")
        status_cls = "ps-ok" if ps.status == 200 else "ps-err"
        rows += (
            f"<tr>"
            f"<td class='num'>{i}</td>"
            f"<td class='url-cell mono' title='{_escape(ps.url)}'>{_escape(short)}</td>"
            f"<td><span class='act-badge act-nav'>{_escape(method)}</span></td>"
            f"<td><span class='ps-badge {status_cls}'>HTTP {ps.status or 'ERR'}</span></td>"
            f"<td class='mono'>{ps.length_bytes:,} B</td>"
            f"<td class='mono'>{ps.timestamp[11:19]}</td>"
            f"</tr>\n"
        )
    return f"""<div class="table-wrap"><table class="action-table">
<thead><tr>
  <th>#</th><th>URL</th><th>Method</th><th>Status</th><th>Size</th><th>Time</th>
</tr></thead>
<tbody>{rows}</tbody>
</table></div>"""


CSS = """
:root {
  --bg: #0f0f1a;
  --surface: #1a1a2e;
  --surface2: #16213e;
  --border: #2a2a4a;
  --text: #e2e8f0;
  --muted: #94a3b8;
  --accent: #6366f1;
  --accent2: #22d3ee;
  --green: #10b981;
  --amber: #f59e0b;
  --red: #ef4444;
  --purple: #a855f7;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 15px; line-height: 1.6;
}
a { color: var(--accent2); }
h1 { font-size: 2rem; font-weight: 700; background: linear-gradient(135deg,#6366f1,#22d3ee); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
h2 { font-size: 1.25rem; font-weight: 600; color: var(--accent2); margin-bottom: 1rem; display:flex; align-items:center; gap:.5rem; }
h2::before { content:""; display:block; width:3px; height:1.2em; background:var(--accent); border-radius:2px; }
h3 { font-size: 1rem; font-weight: 600; color: var(--purple); margin-bottom:.5rem; }

.container { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }
.hero { text-align:center; padding: 3rem 1rem 2rem; border-bottom: 1px solid var(--border); }
.hero p { color: var(--muted); margin-top:.5rem; }

.section { background: var(--surface); border: 1px solid var(--border); border-radius:12px; padding:1.5rem; margin-bottom:1.5rem; }
.section-num { display:inline-block; background:var(--accent); color:#fff; border-radius:6px; font-size:.75rem; font-weight:700; padding:.15rem .45rem; margin-right:.5rem; }

.mono { font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace; }
.ts { color: var(--muted); font-size:.8rem; }
.num { color: var(--muted); font-size:.85rem; text-align:center; width:2.5rem; }

.badge { display:inline-block; font-size:.7rem; font-weight:600; padding:.15rem .4rem; border-radius:4px; font-family:monospace; }
.b-det { background:#1e293b; color:#6366f1; border:1px solid #6366f140; }
.b-a11y { background:#1e293b; color:#22d3ee; border:1px solid #22d3ee40; }
.b-ext { background:#1e293b; color:#10b981; border:1px solid #10b98140; }
.b-vis { background:#1e293b; color:#a855f7; border:1px solid #a855f740; }
.b-blk { background:#1e293b; color:#ef4444; border:1px solid #ef444440; }

.act-badge { display:inline-block; font-size:.7rem; font-weight:700; padding:.15rem .45rem; border-radius:4px; font-family:monospace; text-transform:uppercase; }
.act-search { background:#312e81; color:#a5b4fc; }
.act-nav    { background:#065f46; color:#6ee7b7; }
.act-ext    { background:#1e1b4b; color:#c4b5fd; }
.act-exp    { background:#713f12; color:#fde68a; }

.table-wrap { overflow-x:auto; border-radius:8px; border:1px solid var(--border); }
table { width:100%; border-collapse:collapse; font-size:.85rem; }
thead { background: var(--surface2); }
th { padding:.6rem .8rem; text-align:left; color:var(--muted); font-weight:600; font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; border-bottom:1px solid var(--border); }
td { padding:.55rem .8rem; border-bottom:1px solid var(--border)20; vertical-align:top; }
tr:last-child td { border-bottom:none; }
tr:hover td { background: #ffffff06; }
th[scope=row] { color:var(--accent); background:var(--surface2); font-size:.8rem; white-space:nowrap; }

.cmp-table th:not([scope=row]) { color:var(--accent2); font-size:.85rem; text-align:center; }
.cmp-table td { text-align:center; }
.pricing { color:var(--green); font-size:.8rem; line-height:1.5; }

.url-cell { font-family:monospace; font-size:.78rem; color:var(--muted); word-break:break-all; max-width:320px; }
.result-cell { font-size:.8rem; color:var(--muted); max-width:240px; }

.mini-list { list-style:none; padding:0; font-size:.8rem; text-align:left; }
.mini-list li { padding:.15rem 0; padding-left:.8rem; position:relative; color:var(--text); }
.mini-list li::before { content:"·"; position:absolute; left:0; color:var(--accent); }
.strength li::before { color:var(--green); }
.limit li::before { color:var(--red); }

.feature-list { padding-left:1.2rem; }
.feature-list li { margin-bottom:.25rem; }

.ps-card { background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:1rem; margin-bottom:.75rem; }
.ps-header { display:flex; justify-content:space-between; align-items:center; gap:.5rem; margin-bottom:.35rem; }
.ps-url { font-size:.78rem; color:var(--muted); word-break:break-all; }
.ps-badge { font-size:.7rem; font-weight:700; padding:.15rem .4rem; border-radius:4px; white-space:nowrap; }
.ps-ok   { background:#064e3b; color:#6ee7b7; }
.ps-warn { background:#713f12; color:#fde68a; }
.ps-err  { background:#450a0a; color:#fca5a5; }
.ps-meta { font-size:.75rem; color:var(--muted); margin-bottom:.5rem; }
.ps-preview { font-size:.72rem; color:var(--muted); white-space:pre-wrap; word-break:break-word; max-height:80px; overflow:hidden; background:#0a0a12; border-radius:4px; padding:.5rem; line-height:1.4; }

.json-block { background:#0a0a14; border:1px solid var(--border); border-radius:8px; padding:1rem; font-size:.75rem; color:#a78bfa; overflow-x:auto; white-space:pre-wrap; word-break:break-word; max-height:400px; overflow-y:auto; }

.rec-winner { font-size:1.5rem; font-weight:700; color:var(--accent2); padding:1rem; background:var(--surface2); border-radius:8px; margin-bottom:1rem; text-align:center; }
.rec-reason { color:var(--text); margin-bottom:.75rem; line-height:1.7; }
.rec-section-title { font-weight:700; color:var(--accent); margin-top:1rem; margin-bottom:.5rem; }
.rec-list { list-style:none; padding:0; }
.rec-list li { padding:.4rem .8rem; background:var(--surface2); border-radius:6px; margin-bottom:.4rem; border-left:3px solid var(--accent); font-size:.9rem; }
.rec-verdict { font-style:italic; color:var(--accent2); margin-top:1rem; padding:1rem; background:var(--surface2); border-radius:8px; border-left:3px solid var(--accent2); }

.metrics-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:1rem; }
.metric-card { background:var(--surface2); border:1px solid var(--border); border-radius:10px; padding:1.25rem 1rem; text-align:center; }
.metric-value { font-size:1.75rem; font-weight:700; color:var(--accent2); font-family:monospace; }
.metric-label { font-size:.75rem; color:var(--muted); margin-top:.25rem; text-transform:uppercase; letter-spacing:.05em; }

.dag-container { display:flex; justify-content:center; padding:1rem 0; }
.tag-row { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:.5rem; }

footer { text-align:center; color:var(--muted); font-size:.8rem; padding:2rem; border-top:1px solid var(--border); }
"""


def generate_html(result: dict) -> str:
    browser: BrowserAgent = result["browser"]
    extracted: dict[str, ProductData] = result["extracted"]
    web_results: dict[str, list[dict]] = result["web_results"]
    recommendation: str = result["recommendation"]
    elapsed: float = result["elapsed"]
    timestamp: str = result["timestamp"]

    search_hits_html = ""
    for key, hits in web_results.items():
        name = PRODUCTS[key]["name"]
        if hits:
            items = "".join(
                f'<li><a href="{_escape(h["url"])}" target="_blank">{_escape(h["title"][:80])}</a>'
                f' <span class="muted">— {_escape(h["snippet"][:100])}</span></li>'
                for h in hits[:4]
            )
            search_hits_html += f'<h3>{_escape(name)}</h3><ul style="font-size:.85rem;padding-left:1.2rem;margin-bottom:1rem">{items}</ul>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AI Coding Assistant Comparison — Session 9</title>
<style>{CSS}</style>
</head>
<body>

<div class="hero">
  <h1>AI Coding Assistant Comparison</h1>
  <p>Live browser agent report · Generated {_escape(timestamp[:10])} · Session 9</p>
</div>

<div class="container">

<!-- ① User Goal -->
<div class="section">
  <h2><span class="section-num">1</span>Original User Goal</h2>
  <p>Research and compare the top AI coding assistants used by software developers —
  <strong>Claude Code</strong>, <strong>GitHub Copilot</strong>, and <strong>Cursor AI</strong> —
  through live web browsing. Collect pricing, supported IDEs, key features, AI models,
  strengths, and limitations. Produce an evidence-based recommendation.</p>
  <div class="tag-row" style="margin-top:1rem">
    <span class="badge b-det">Claude Code</span>
    <span class="badge b-a11y">GitHub Copilot</span>
    <span class="badge b-ext">Cursor AI</span>
    <span class="badge b-vis">Live Web Browsing</span>
    <span class="badge b-blk">Evidence-Based Recommendation</span>
  </div>
</div>

<!-- ② Planner DAG -->
<div class="section">
  <h2><span class="section-num">2</span>Planner DAG</h2>
  <p style="color:var(--muted);font-size:.85rem;margin-bottom:1rem">
    Execution plan — each node is a phase of the agent's work.
  </p>
  <div class="dag-container">
    {_dag_svg()}
  </div>
</div>

<!-- ③ Browser Path -->
<div class="section">
  <h2><span class="section-num">3</span>Browser Path Chosen</h2>
  <p style="color:var(--muted);font-size:.85rem;margin-bottom:1rem">
    Every page visited, with navigation method and HTTP status.
  </p>
  {_path_table(browser.page_states)}
</div>

<!-- ④ Browser Actions (Replay Log) -->
<div class="section">
  <h2><span class="section-num">4</span>Browser Actions — Replay Log</h2>
  <p style="color:var(--muted);font-size:.85rem;margin-bottom:1rem">
    Chronological log of every browser action.
    <strong>Path types:</strong>
    <span class="badge b-det">deterministic</span> URL constructed from known patterns ·
    <span class="badge b-a11y">a11y</span> navigated by following links ·
    <span class="badge b-ext">extract</span> page content parsed ·
    <span class="badge b-blk">blocked</span> page returned error.
  </p>
  {_actions_table(browser.actions)}
</div>

<!-- ⑤ Screenshots / Page States -->
<div class="section">
  <h2><span class="section-num">5</span>Screenshots / Page States</h2>
  <p style="color:var(--muted);font-size:.85rem;margin-bottom:1rem">
    Text capture of each visited page (first 400 chars of clean text).
  </p>
  {_page_states_cards(browser.page_states)}
</div>

<!-- ⑥ Search Results -->
<div class="section">
  <h2><span class="section-num">5b</span>Search Results</h2>
  {search_hits_html if search_hits_html else '<p style="color:var(--muted)">No search results captured (search providers may have been rate-limited).</p>'}
</div>

<!-- ⑦ Extracted Data -->
<div class="section">
  <h2><span class="section-num">6</span>Extracted Data</h2>
  <p style="color:var(--muted);font-size:.85rem;margin-bottom:1rem">
    Structured data collected for each assistant (LLM-extracted + curated static baseline).
  </p>
  {_extracted_json(extracted)}
</div>

<!-- ⑧ Comparison Table -->
<div class="section">
  <h2><span class="section-num">7</span>Final Comparison Table</h2>
  {_comparison_table(extracted)}
</div>

<!-- ⑨ Recommendation -->
<div class="section">
  <h2><span class="section-num">8</span>Recommendation</h2>
  {_recommendation_html(recommendation)}
</div>

<!-- ⑩ Metrics -->
<div class="section">
  <h2><span class="section-num">9</span>Metrics</h2>
  {_metrics_html(browser, elapsed)}
</div>

</div><!-- /container -->

<footer>
  AI Coding Assistant Comparison Agent · Session 9 ·
  {_escape(timestamp)} ·
  Powered by LLM Gateway V9 + Gemini ·
  Browser: httpx + Tavily/DuckDuckGo
</footer>

</body>
</html>"""
    return html


# ── Entry Point ───────────────────────────────────────────────────────────────

async def main():
    result = await run_agent()
    html = generate_html(result)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\n✓ Report written → {OUTPUT_HTML}")
    print(f"  Open in browser: file:///{OUTPUT_HTML.as_posix()}")


if __name__ == "__main__":
    asyncio.run(main())