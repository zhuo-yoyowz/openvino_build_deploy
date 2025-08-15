# multiagents_workflow.py
from __future__ import annotations

import asyncio
import json
import re
import base64
from io import BytesIO
from typing import AsyncGenerator, Dict, List, Tuple, Any, Optional

from PIL import Image

from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.agent.workflow import ReActAgent, AgentWorkflow
from llama_index.core.agent.workflow import AgentOutput, ToolCallResult
from llama_index.core.workflow.events import StopEvent
from llama_index.llms.openvino import OpenVINOLLM


# ────────────────────────────────────────────────────────────────────────────────
# LLM (OpenVINO)
# ────────────────────────────────────────────────────────────────────────────────
def _build_llm():
    return OpenVINOLLM(
        model_id_or_path="./qwen2.5-7b-instruct-int4-ov",
        context_window=8192,
        max_new_tokens=1024,
        generate_kwargs={"do_sample": True},  # streaming-friendly
        device_map="AUTO:GPU,CPU",
    )


def _tune_openvino_streaming(llm):
    """Best-effort tweaks to make token streaming snappier."""
    try:
        if hasattr(llm, "_streamer") and hasattr(llm._streamer, "tokens_len"):
            llm._streamer.tokens_len = 1
    except Exception:
        pass
    try:
        cfg = getattr(llm, "config", None)
        if cfg:
            cfg.max_new_tokens = getattr(cfg, "max_new_tokens", 512) or 512
            if getattr(cfg, "do_sample", None) is None:
                cfg.do_sample = True
            if getattr(cfg, "temperature", None) is None:
                cfg.temperature = 0.7
            if getattr(cfg, "top_k", None) is None:
                cfg.top_k = 50
            if getattr(cfg, "top_p", None) is None:
                cfg.top_p = 0.95
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────────────────────────
_SEARCH_KEYWORDS = (
    "search", "google", "duckduckgo", "find", "look up", "online shopping",
    "buy", "price", "prices", "review", "reviews", "compare", "product",
    "amazon", "walmart", "target", "shop"
)
_CART_KEYWORDS = (
    "cart", "add to cart", "add", "remove", "checkout", "view cart", "clear cart",
    "shopping cart"
)
_VIDEO_KEYWORDS = ("video", "frame", "clip", "timestamp", "scene", "shot", "from the video")


def _classify_intent(text: str) -> str:
    q = (text or "").lower()
    if any(k in q for k in _VIDEO_KEYWORDS):
        return "video"
    if any(k in q for k in _CART_KEYWORDS):
        return "cart"
    if any(k in q for k in _SEARCH_KEYWORDS):
        return "search"
    return "general"


def _parse_mcp_result(res: Any) -> Dict[str, Any]:
    """Parse FastMCP tool result into a dict or fall back to concatenated text."""
    # 1) structuredContent.result
    sc = getattr(res, "structuredContent", None)
    if isinstance(sc, dict) and "result" in sc:
        try:
            data = json.loads(sc["result"])
            return data if isinstance(data, dict) else {"text": str(data)}
        except Exception:
            pass

    # 2) content list of dicts/blocks with .text
    buf = ""
    content = getattr(res, "content", None)
    if isinstance(content, list):
        for blk in content:
            t = getattr(blk, "text", None) or (blk.get("text") if isinstance(blk, dict) else None)
            if t:
                buf += t

    # 3) older: res.blocks with .text
    if not buf:
        blocks = getattr(res, "blocks", None)
        if isinstance(blocks, list):
            for b in blocks:
                t = getattr(b, "text", None)
                if t:
                    buf += t

    return {"text": buf} if buf else {}


def _images_from_payload(data: Dict[str, Any]) -> List[Image.Image]:
    """Prefer image_b64 → PIL. image_paths are ignored to avoid Windows long-path issues."""
    imgs: List[Image.Image] = []
    b64_list = data.get("image_b64")
    if isinstance(b64_list, list):
        for s in b64_list:
            try:
                if isinstance(s, str) and s.startswith("data:image"):
                    _, enc = s.split(",", 1)
                    raw = base64.b64decode(enc)
                elif isinstance(s, str):
                    raw = base64.b64decode(s)
                else:
                    continue
                img = Image.open(BytesIO(raw)).convert("RGB")
                imgs.append(img)
            except Exception:
                pass
    return imgs


# ────────────────────────────────────────────────────────────────────────────────
# Workflow
# ────────────────────────────────────────────────────────────────────────────────
class MultiAgentWorkflow:
    def __init__(self):
        self.agent_workflow: AgentWorkflow | None = None
        self.mcp_clients: Dict[str, BasicMCPClient] = {}
        self.tool_map: Dict[str, List[Any]] = {}
        self.llm = _build_llm()
        _tune_openvino_streaming(self.llm)

        self.last_search_results: list[dict] = []

    # ────────────────────────────────────────────────────────────────────────
    # Init all agents & MCP tools
    # ────────────────────────────────────────────────────────────────────────
    async def initialize(self):
        urls = {
            "ShoppingCart": "http://127.0.0.1:8000/sse",
            "Search":       "http://127.0.0.1:8001/sse",
            "Video":        "http://127.0.0.1:8002/sse",
        }

        self.tool_map.clear()
        self.mcp_clients.clear()

        # Connect to MCP servers and fetch tools
        for name, url in urls.items():
            for attempt in range(3):
                try:
                    client = BasicMCPClient(url)
                    self.mcp_clients[name] = client

                    tools = await asyncio.wait_for(
                        McpToolSpec(client=client).to_tool_list_async(), timeout=5
                    )
                    self.tool_map[name] = tools

                    print(f"✅ {name} MCP - {len(tools)} tools ready")
                    break
                except Exception as e:
                    print(f"❌ {name} MCP attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(1.0)
            else:
                raise RuntimeError(f"{name} MCP could not connect — ensure uvicorn is running")

        # Define agents (we keep them for cart flows; other intents use direct MCP/LLM)
        self.agent_workflow = AgentWorkflow(
            agents=[
                ReActAgent(
                    "ShoppingCartAgent",
                    "Manage cart operations via shopping_cart tools.",
                    tools=self.tool_map["ShoppingCart"],
                    llm=self.llm,
                ),
                ReActAgent(
                    "SearchAgent",
                    "Search web content via search tools.",
                    tools=self.tool_map["Search"],
                    llm=self.llm,
                ),
                ReActAgent(
                    "VideoSearchAgent",
                    "Extract information and images from video via video tools.",
                    tools=self.tool_map["Video"],
                    llm=self.llm,
                ),
                ReActAgent(
                    "RouterAgent",
                    "Route user queries (Video / Search / ShoppingCart / General).",
                    tools=[],
                    llm=self.llm,
                    can_handoff_to=[
                        "ShoppingCartAgent",
                        "SearchAgent",
                        "VideoSearchAgent",
                    ],
                ),
            ],
            root_agent="RouterAgent",
        )

    # ────────────────────────────────────────────────────────────────────────
    # Cart render & tool output parsing
    # ────────────────────────────────────────────────────────────────────────
    def _render_cart_table(self, cart_like) -> str:
        """Render cart as a Markdown table. Accepts list or dict with 'cart' key."""
        items = []
        if isinstance(cart_like, list):
            items = cart_like
        elif isinstance(cart_like, dict):
            for key in ("cart", "items", "data", "result"):
                if isinstance(cart_like.get(key), list):
                    items = cart_like[key]
                    break
        if not items:
            return "### 🛒 Your Shopping Cart is Empty"

        lines = [
            "### 🛒 Your Shopping Cart",
            "",
            "| # | Product | Qty | Unit Price | Line Total |",
            "|---:|---|---:|---:|---:|",
        ]
        total = 0.0
        for i, item in enumerate(items, 1):
            name = str(item.get("product_name") or item.get("name") or item.get("product") or "Unknown")
            try:
                qty = int(item.get("quantity") or item.get("qty") or 1)
            except Exception:
                qty = 1
            try:
                unit = float(item.get("price_per_unit") or item.get("unit_price") or item.get("unitPrice") or item.get("price") or 0.0)
            except Exception:
                unit = 0.0
            try:
                line_total = float(item.get("total_price") or (unit * qty))
            except Exception:
                line_total = unit * qty
            total += line_total
            lines.append(f"| {i} | {name} | {qty} | ${unit:.2f} | ${line_total:.2f} |")
        lines.append(f"\n**Total: ${total:.2f}**")
        return "\n".join(lines)

    def _data_from_tool_output(self, tool_out: Any) -> Dict[str, Any] | List[Dict[str, Any]] | None:
        """Try to robustly extract dict/list payloads from FastMCP tool output."""
        if tool_out is None:
            return {}
        # 1) structuredContent.result
        sc = getattr(tool_out, "structuredContent", None)
        if isinstance(sc, dict) and "result" in sc:
            try:
                return json.loads(sc["result"])
            except Exception:
                pass
        # 2) content blocks (newer FastMCP)
        text_buf = ""
        content = getattr(tool_out, "content", None)
        if isinstance(content, list):
            for blk in content:
                t = getattr(blk, "text", None) or (blk.get("text") if isinstance(blk, dict) else None)
                if t:
                    text_buf += t
        # 3) blocks (older)
        if not text_buf:
            blocks = getattr(tool_out, "blocks", None)
            if isinstance(blocks, list):
                for b in blocks:
                    t = getattr(b, "text", None)
                    if t:
                        text_buf += t
        if text_buf:
            # try whole string then JSON substring
            try:
                return json.loads(text_buf)
            except Exception:
                m = re.search(r"(\{.*\}|\[.*\])", text_buf, flags=re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(1))
                    except Exception:
                        return {}
        return {}

    async def _fetch_cart_state(self) -> str:
        client = self.mcp_clients.get("ShoppingCart")
        if not client:
            return "🛒 **Cart unavailable (no ShoppingCart MCP client).**"

        res = None
        # Some FastMCP tools with no params require `None` (not `{}`)
        for args in (None, {}):
            try:
                res = await asyncio.wait_for(client.call_tool("view_cart", args), timeout=8)
                break
            except Exception:
                continue

        if res is None:
            return "🛒 **Cart error (view_cart failed).**"

        data = self._data_from_tool_output(res)
        if data:
            return self._render_cart_table(data)

        # Fallback: dump raw text for visibility
        raw_text = ""
        content = getattr(res, "content", None)
        if isinstance(content, list):
            for blk in content:
                raw_text += (blk.get("text") if isinstance(blk, dict) else getattr(blk, "text", "")) or ""
        blocks = getattr(res, "blocks", None)
        if isinstance(blocks, list):
            raw_text += "".join(getattr(b, "text", "") or "" for b in blocks)

        return "🛒 Cart\n\n```\n" + (raw_text.strip() or "{}") + "\n```"

    # ────────────────────────────────────────────────────────────────────────
    # Direct MCP calls (fast path)
    # ────────────────────────────────────────────────────────────────────────
    async def _call_video_direct(self, user_msg: str) -> tuple[Optional[str], List[Image.Image]]:
        client = self.mcp_clients.get("Video")
        if not client:
            return None, []
        try:
            res = await asyncio.wait_for(
                client.call_tool("search_from_video", {"query": user_msg}),
                timeout=25,
            )
        except Exception:
            return None, []
        data = _parse_mcp_result(res)
        answer = data.get("answer") or data.get("text")
        images = _images_from_payload(data)
        return (answer, images)

    async def _call_search_direct(self, user_msg: str) -> Optional[str]:
        client = self.mcp_clients.get("Search")
        if not client:
            return None
        try:
            res = await asyncio.wait_for(
                client.call_tool("search", {"query": user_msg}),
                timeout=25,
            )
        except Exception:
            return None

        data = _parse_mcp_result(res)
        # If MCP returns structured "results" list → pretty markdown bullets
        if isinstance(data, dict) and isinstance(data.get("results"), list) and data["results"]:
            self.last_search_results = []  # reset
            lines = ["### 🔎 Search Results", ""]
            for idx, item in enumerate(data["results"][:10], 1):
                title = item.get("title") or item.get("heading") or "Result"
                link = item.get("url") or item.get("href") or ""
                snippet = item.get("snippet") or item.get("description") or ""
                if link:
                    lines.append(f"{idx}. [{title}]({link})")
                    if snippet:
                        lines.append(f"   - {snippet.strip()}")
                    self.last_search_results.append({"title": title, "url": link, "snippet": snippet})
                else:
                    lines.append(f"{idx}. {title}")
                    if snippet:
                        lines.append(f"   - {snippet.strip()}")
                    self.last_search_results.append({"title": title, "url": "", "snippet": snippet})
            lines.append("")
            lines.append("_Tip: click a URL **or** type `open 3` to select result #3._")
            return "\n".join(lines)

        # Else fall back to the plain text
        txt = data.get("text") if isinstance(data, dict) else None
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
        return None

    # ────────────────────────────────────────────────────────────────────────
    # Product selection & price helpers
    # ────────────────────────────────────────────────────────────────────────
    def _fuzzy_match(self, a: str, b: str) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

    async def _select_product_by_index(self, idx: int):
        """Select product by last search results index via SearchAgent.select_product."""
        client = self.mcp_clients.get("Search")
        if not client:
            return
        if not self.last_search_results or idx < 1 or idx > len(self.last_search_results):
            return
        url = self.last_search_results[idx - 1].get("url") or ""
        if not url:
            return
        try:
            await asyncio.wait_for(client.call_tool("select_product", {"url": url}), timeout=12)
        except Exception:
            pass

    async def _select_product_by_url(self, url: str, name_hint: str | None = None):
        client = self.mcp_clients.get("Search")
        if not client or not url:
            return
        try:
            payload = {"url": url}
            if name_hint:
                payload["name_hint"] = name_hint
            await asyncio.wait_for(client.call_tool("select_product", payload), timeout=12)
        except Exception:
            pass

    async def _resolve_price_for(self, product_name: str) -> Optional[float]:
        """Ask SearchAgent for last selected product and return price_per_unit if names match closely."""
        client = self.mcp_clients.get("Search")
        if not client:
            return None
        try:
            res = await asyncio.wait_for(client.call_tool("get_selected_product", {}), timeout=8)
            data = _parse_mcp_result(res)
        except Exception:
            return None

        if isinstance(data, dict):
            sel_name = data.get("product_name") or ""
            price = data.get("price_per_unit")
            try:
                if price is not None and (self._fuzzy_match(product_name, sel_name) > 0.6 or not product_name):
                    return float(price)
            except Exception:
                return None
        return None

    # ────────────────────────────────────────────────────────────────────────
    # Chat streaming helpers
    # ────────────────────────────────────────────────────────────────────────
    async def _stream_llm_answer(self, prompt: str,
                                 user_msg: str,
                                 log_str: str,
                                 cart_state: str,
                                 images: List[Image.Image]):
        """Stream assistant bubble token-by-token (LLM), robust to delta vs. cumulative events."""
        chat_stream: List[Dict[str, str]] = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": ""},
        ]

        if hasattr(self.llm, "stream_complete"):
            acc = ""                 # what we've actually shown to user
            accum_total = ""         # last seen cumulative text from model (if any)
            prefer_cumulative = False

            for ev in self.llm.stream_complete(prompt):
                new_raw: Optional[str] = None

                # 1) If we haven't switched to cumulative yet and a delta exists → use it.
                if not prefer_cumulative and isinstance(getattr(ev, "delta", None), str) and ev.delta:
                    new_raw = ev.delta
                    # Track a plausible cumulative to compare against later
                    accum_total += ev.delta

                # 2) If cumulative text exists → switch to cumulative and compute only the suffix.
                elif isinstance(getattr(ev, "text", None), str) and ev.text:
                    total = ev.text
                    prefer_cumulative = True
                    if total.startswith(accum_total):
                        new_raw = total[len(accum_total):]
                    else:
                        # longest common prefix fallback
                        i = 0
                        m = min(len(total), len(accum_total))
                        while i < m and total[i] == accum_total[i]:
                            i += 1
                        new_raw = total[i:]
                    accum_total = total

                if not new_raw:
                    continue
                if not isinstance(new_raw, str) or not new_raw:
                    continue

                acc += new_raw
                chat_stream[-1]["content"] = acc

                # Replace chat & log each tick
                yield log_str, chat_stream, cart_state, images
                await asyncio.sleep(0.02)
            return

        # Fallback (non-streaming)
        resp = self.llm.complete(prompt)
        chat_stream[-1]["content"] = (resp.text or "").strip()
        yield log_str, chat_stream, cart_state, images
        await asyncio.sleep(0.02)



    async def _stream_fixed_text(self, text: str,
                                 user_msg: str,
                                 log_str: str,
                                 cart_state: str,
                                 images: List[Image.Image],
                                 chunk_chars: int = 200):
        """Stream a prebuilt string in chunks to the assistant bubble (for MCP/static results)."""
        chat_stream: List[Dict[str, str]] = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": ""},
        ]
        t = (text or "").strip()
        if not t:
            yield log_str, chat_stream, cart_state, images
            await asyncio.sleep(0.02)
            return
        i, n = 0, len(t)
        while i < n:
            j = min(n, i + chunk_chars)
            while j < n and not t[j].isspace():
                j += 1
            chat_stream[-1]["content"] = t[:j]
            yield log_str, chat_stream, cart_state, images
            await asyncio.sleep(0.02)
            i = j


    # ────────────────────────────────────────────────────────────────────────
    # Streaming query (main)
    # Emits (full_log:str, assistant_msgs:list[dict], cart_markdown:str, images:list[PIL.Image])
    # ────────────────────────────────────────────────────────────────────────
    async def query(
        self, user_msg: str, memory
    ) -> AsyncGenerator[Tuple[str, List[Dict[str, str]], str, List[Image.Image]], None]:
        if self.agent_workflow is None:
            raise RuntimeError("initialize() must be awaited first")

        # One authoritative reasoning buffer (front-end replaces value each tick)
        log_parts: List[str] = []
        def set_log(s: str) -> str:
            if s:
                log_parts.append(s)
            return "".join(log_parts)

        cart_state = ""
        images: List[Image.Image] = []

        # 1) classify & show first line
        intent = _classify_intent(user_msg)
        yield set_log(f"[DEBUG] Intent classified as: {intent}\n"), [], cart_state, images
        await asyncio.sleep(0.03)

        # 2) STREAM RouterAgent reasoning into the log (dedup-aware)
        try:
            yield set_log("[RouterAgent] "), [], cart_state, images
            await asyncio.sleep(0.03)

            def _normalize_chunk(chunk: str, prev_last: str) -> str:
                # Replace hard breaks with spaces; collapse whitespace
                chunk = chunk.replace("\r", " ").replace("\n", " ")
                chunk = re.sub(r"\s+", " ", chunk)
                # If previous char and next char are both alnum, insert one space
                if prev_last and prev_last.isalnum() and chunk[:1].isalnum():
                    chunk = " " + chunk
                # Remove spaces before punctuation
                chunk = re.sub(r"\s+([,.;:!?])", r"\1", chunk)
                return chunk

            last_char = " "     # track last emitted char
            accum_total = ""    # everything we've seen so far (for cumulative streams)
            prefer_cumulative = False  # once cumulative is detected, ignore deltas

            if hasattr(self.llm, "stream_complete"):
                for ev in self.llm.stream_complete(
                    "You are the RouterAgent.\n"
                    "Given the USER QUERY below, briefly explain which route you would take and why.\n"
                    "Choose one of: Video, Search, ShoppingCart, General.\n"
                    "Keep it to 1–3 short sentences. Do NOT call tools.\n"
                    f"USER QUERY: {user_msg}\n"
                    "ROUTER THOUGHT: "
                ):
                    new_raw = None

                    # 1) If not in cumulative mode and a delta exists, use it
                    if not prefer_cumulative and isinstance(getattr(ev, "delta", None), str) and ev.delta:
                        new_raw = ev.delta
                        accum_total += ev.delta

                    # 2) If a cumulative text exists, switch and compute suffix
                    elif isinstance(getattr(ev, "text", None), str) and ev.text:
                        total = ev.text
                        prefer_cumulative = True
                        if total.startswith(accum_total):
                            new_raw = total[len(accum_total):]
                        else:
                            # longest common prefix fallback
                            i = 0
                            m = min(len(total), len(accum_total))
                            while i < m and total[i] == accum_total[i]:
                                i += 1
                            new_raw = total[i:]
                        accum_total = total

                    if not new_raw:
                        continue

                    cleaned = _normalize_chunk(new_raw, last_char)
                    if cleaned:
                        last_char = cleaned[-1]
                        yield set_log(cleaned), [], cart_state, images
                        await asyncio.sleep(0.03)

                # finish RouterAgent line neatly
                yield set_log("\n"), [], cart_state, images
                await asyncio.sleep(0.03)
            else:
                # non-streaming fallback
                resp = self.llm.complete(
                    "You are the RouterAgent.\n"
                    "Given the USER QUERY below, briefly explain which route you would take and why.\n"
                    "Choose one of: Video, Search, ShoppingCart, General.\n"
                    "Keep it to 1–3 short sentences. Do NOT call tools.\n"
                    f"USER QUERY: {user_msg}\n"
                    "ROUTER THOUGHT: "
                )
                text = (resp.text or "").strip()
                text = _normalize_chunk(text, " ")
                yield set_log(text + "\n"), [], cart_state, images
                await asyncio.sleep(0.03)

        except Exception as e:
            yield set_log(f"[DEBUG] RouterAgent reasoning stream failed: {e}\n"), [], cart_state, images
            await asyncio.sleep(0.03)

        # 3) Clicked URL → implicitly select that product
        url_match = re.search(r'(https?://\S+)', user_msg, re.I)
        if url_match:
            await self._select_product_by_url(url_match.group(1))
            # stream a tiny acknowledgement bubble
            async for step in self._stream_fixed_text(
                "Selected the product from the link. You can now say, e.g., `Add 2 <name> to cart`.",
                user_msg, "".join(log_parts), cart_state, images, chunk_chars=40
            ):
                yield step
            return

        # 4) "open N" → select from last results
        m = re.search(r'\bopen(?:\s+link)?\s+(\d+)\b', user_msg, re.I)
        if m:
            await self._select_product_by_index(int(m.group(1)))
            async for step in self._stream_fixed_text(
                "Opened result and captured product details.",
                user_msg, "".join(log_parts), cart_state, images, chunk_chars=40
            ):
                yield step
            return

        # 5) Video
        if intent == "video":
            ans, imgs = await self._call_video_direct(user_msg)
            if imgs:
                images.extend(imgs)
            if ans:
                async for step in self._stream_fixed_text(ans, user_msg, "".join(log_parts), cart_state, images):
                    yield step
                return
            async for step in self._stream_llm_answer(user_msg, user_msg, "".join(log_parts), cart_state, images):
                yield step
            return

        # 6) Search
        if intent == "search":
            ans = await self._call_search_direct(user_msg)
            if ans:
                async for step in self._stream_fixed_text(ans, user_msg, "".join(log_parts), cart_state, images):
                    yield step
                return
            async for step in self._stream_llm_answer(user_msg, user_msg, "".join(log_parts), cart_state, images):
                yield step
            return

        # 7) Cart
        if intent == "cart":
            # Early direct add-to-cart with price auto-fill (bypass agent if explicit)
            m_add = re.search(r"add\s+(\d+)\s+(.+?)\s+(?:to|into)\s+(?:my\s+)?cart\b", user_msg, re.I)
            if m_add:
                qty = int(m_add.group(1))
                prod = m_add.group(2).strip()
                price = await self._resolve_price_for(prod)
                payload = {
                    "product_name": prod,
                    "quantity": qty,
                    "price_per_unit": float(price) if price is not None else 0.0,
                }
                client = self.mcp_clients.get("ShoppingCart")
                if client:
                    try:
                        await asyncio.wait_for(client.call_tool("add_to_cart", payload), timeout=10)
                    except Exception as e:
                        yield set_log(f"[DEBUG] add_to_cart failed: {e}\n"), [], cart_state, images
                        await asyncio.sleep(0.03)
                cart_state = await self._fetch_cart_state()
                # stream a tiny confirmation bubble
                async for step in self._stream_fixed_text(
                    "Cart updated.",
                    user_msg, "".join(log_parts), cart_state, images, chunk_chars=20
                ):
                    yield step
                return

            # Otherwise, stream the agent's events and keep refreshing the cart panel
            handler = self.agent_workflow.run(user_msg)

            # Warm cart panel so it's never empty
            try:
                cart_state = await self._fetch_cart_state()
                yield "".join(log_parts), [], cart_state, images
                await asyncio.sleep(0.03)
            except Exception as e:
                yield set_log(f"[DEBUG] initial cart refresh failed: {e}\n"), [], cart_state, images
                await asyncio.sleep(0.03)

            tool_tries = 0
            MAX_TOOL_TRIES = 5
            PER_EVENT_TIMEOUT = 6.0
            OVERALL_TIMEOUT = 40.0

            start_ts = asyncio.get_event_loop().time()
            aiter = handler.stream_events()

            while True:
                # Overall timeout
                if asyncio.get_event_loop().time() - start_ts > OVERALL_TIMEOUT:
                    yield set_log("[DEBUG] Cart overall timeout; final refresh.\n"), [], cart_state, images
                    await asyncio.sleep(0.03)
                    cart_state = await self._fetch_cart_state()
                    async for step in self._stream_fixed_text(
                        "Cart updated.",
                        user_msg, "".join(log_parts), cart_state, images, chunk_chars=20
                    ):
                        yield step
                    return

                try:
                    ev = await asyncio.wait_for(aiter.__anext__(), timeout=PER_EVENT_TIMEOUT)
                except asyncio.TimeoutError:
                    yield set_log("[DEBUG] Cart per-event timeout; refreshing cart.\n"), [], cart_state, images
                    await asyncio.sleep(0.03)
                    cart_state = await self._fetch_cart_state()
                    yield "".join(log_parts), [], cart_state, images
                    await asyncio.sleep(0.03)
                    break
                except StopAsyncIteration:
                    break

                if isinstance(ev, ToolCallResult):
                    tname = (getattr(ev, "tool_name", "") or "").lower()

                    # If a cart tool ran, try to render from its payload immediately
                    if tname in {"add_to_cart", "view_cart", "clear_cart", "calculate_paint_cost", "calculate_paint_gallons"}:
                        data = self._data_from_tool_output(getattr(ev, "tool_output", None))
                        cart_like = None
                        if isinstance(data, dict):
                            # common locations
                            for key in ("cart", "items", "result", "data"):
                                if isinstance(data.get(key), list):
                                    cart_like = data[key]
                                    break
                            # sometimes the dict *is* the cart list shape already
                            if cart_like is None and isinstance(data.get("0"), dict):
                                cart_like = list(data.values())
                        elif isinstance(data, list):
                            cart_like = data

                        if cart_like is not None:
                            cart_state = self._render_cart_table(cart_like)
                            yield "".join(log_parts), [], cart_state, images
                            await asyncio.sleep(0.03)
                        else:
                            # fallback to calling view_cart if payload didn't include items
                            try:
                                cart_state = await self._fetch_cart_state()
                                yield "".join(log_parts), [], cart_state, images
                                await asyncio.sleep(0.03)
                            except Exception as e:
                                yield set_log(f"[DEBUG] cart refresh after {tname} failed: {e}\n"), [], cart_state, images
                                await asyncio.sleep(0.03)

                    yield set_log(f"[DEBUG] ToolCallResult: {tname}\n"), [], cart_state, images
                    await asyncio.sleep(0.03)
                    tool_tries += 1

                    # Refresh the cart after each cart tool call
                    try:
                        cart_state = await self._fetch_cart_state()
                    except Exception as e:
                        yield set_log(f"[DEBUG] cart refresh after {tname} failed: {e}\n"), [], cart_state, images
                        await asyncio.sleep(0.03)
                    yield "".join(log_parts), [], cart_state, images
                    await asyncio.sleep(0.03)

                    if tool_tries >= MAX_TOOL_TRIES:
                        yield set_log("[DEBUG] Max cart tool attempts reached.\n"), [], cart_state, images
                        await asyncio.sleep(0.03)
                        cart_state = await self._fetch_cart_state()
                        async for step in self._stream_fixed_text(
                            "Cart updated.",
                            user_msg, "".join(log_parts), cart_state, images, chunk_chars=20
                        ):
                            yield step
                        return

                elif isinstance(ev, AgentOutput):
                    # If a textual reply appears from the agent, show it and exit cart flow
                    resp = getattr(ev, "response", None)
                    content = getattr(resp, "content", None) if resp else None
                    if isinstance(content, str) and content.strip():
                        cart_state = await self._fetch_cart_state()
                        # log + streamed bubble
                        async for step in self._stream_fixed_text(
                            content.strip(),
                            user_msg,
                            set_log(f"[RouterAgent] {content.strip()}\n"),
                            cart_state,
                            images,
                            chunk_chars=60
                        ):
                            yield step
                        return

                elif isinstance(ev, StopEvent):
                    break

                yield "".join(log_parts), [], cart_state, images
                await asyncio.sleep(0.03)

            # End of stream: final cart show
            cart_state = await self._fetch_cart_state()
            async for step in self._stream_fixed_text(
                "Cart updated.",
                user_msg, "".join(log_parts), cart_state, images, chunk_chars=20
            ):
                yield step
            return

        # 8) General → stream LLM
        async for step in self._stream_llm_answer(user_msg, user_msg, "".join(log_parts), cart_state, images):
            yield step
        return

    # ────────────────────────────────────────────────────────────────────────
    # Public run (not used by gradio path)
    # ────────────────────────────────────────────────────────────────────────
    def run(self, user_msg: str):
        if self.agent_workflow is None:
            raise RuntimeError("initialize() must be awaited first")
        return self.agent_workflow.run(user_msg)