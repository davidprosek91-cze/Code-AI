#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama kódový asistent – optimalizovaná verze
==============================================
GUI aplikace (tkinter) pro psaní dlouhého kódu s lokálním modelem Ollama.

  Výchozí server: http://localhost:11434
  Výchozí model:  qwen3.5:0.8b   (obě hodnoty lze změnit v liště nahoře)

Optimalizace oproti původní verzi:
  - inkrementální StatTracker místo O(n) skenu celého výstupu při každém refreshi
  - batchované vkládání chunků do widgetu (1 insert + 1 yview na poll cyklus)
  - queue.SimpleQueue (nižší režie)
  - time.monotonic() / perf_counter() (monotónní, přesnější)
  - word count přes str.split() bez regexu
  - throttlovaný refresh statistik (0.5 s při streamu, okamžitý po dokončení)

Novinky v této verzi (dlouhé generování):
  - num_predict = -1 (neomezená délka odpovědi) a větší num_ctx (kontext)
  - tlačítko „Pokračovat“ – po dokončení odpovědi (nebo při dosažení limitu
    num_predict / kontextu) umožní poslat modelu výzvu k pokračování
  - automatická detekce done_reason == "length" → upozornění v status baru
  - udržování pozice kurzoru (mark) na konci widgetu i pro velmi dlouhé výstupy

Ovládání:
  Ctrl+Enter   odešle požadavek (z pole Vstup)
  Ctrl+Enter   v režimu pokračování (tlačítko Pokračovat) – pošle výzvu k pokračování
  Ctrl+C       zkopíruje označený text (funguje ve výstupu i ve vstupu)
  Pravé tlačítko myši  kontextové menu
  Tlačítko     Zastavit přeruší běžící generování
  Tlačítko     Pokračovat pošle modelu výzvu k pokračování tam, kde skončil
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request

import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, scrolledtext, ttk

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:0.8b"

SYSTEM_PROMPT = (
    "Jsi zkušený expert programátor a kopilot v psani kodu. "
    "Když píšeš delší kód, rozděl ho do menších logických bloků. "
    "Nikdy neříkej, že nemůžeš pokračovat – vždy pokračuj tam, kde jsi skončil."
)

# Pokyn, který se posílá modelu při stisku tlačítka „Pokračovat“.
CONTINUE_PROMPT = (
    "Pokračuj přesně tam, kde jsi skončil. "
    "Neopakuj již vygenerovaný text ani kód. "
    "Nepiš žádné úvody ani omluvy – jen navazuj dalším obsahem."
)

# ---------------------------------------------------------------------------
# Volby generování – neomezená délka + větší kontextové okno
# ---------------------------------------------------------------------------

GEN_OPTIONS = {
    "temperature": 0.7,
    # -1 = bez limitu (Ollama přestane až na EOS tokenu nebo při zaplnění kontextu)
    "num_predict": -1,
    # větší kontextové okno (ve tokenech); zvyšte, pokud máte dost RAM/VRAM
    "num_ctx": 8192,
    # ponechat kontext mezi voláními (užitečné pro navazování)
    "num_keep": 32,
}

# ---------------------------------------------------------------------------
# Mapování jazyka bloku kódu na příponu souboru
# ---------------------------------------------------------------------------

LANG_EXT = {
    "python": ".py", "py": ".py", "python3": ".py",
    "javascript": ".js", "js": ".js", "node": ".js",
    "typescript": ".ts", "ts": ".ts",
    "jsx": ".jsx", "tsx": ".tsx",
    "html": ".html", "htm": ".html",
    "css": ".css", "scss": ".scss", "sass": ".sass", "less": ".less",
    "json": ".json", "json5": ".json5",
    "yaml": ".yaml", "yml": ".yaml",
    "toml": ".toml", "ini": ".ini", "cfg": ".cfg",
    "xml": ".xml", "svg": ".svg",
    "markdown": ".md", "md": ".md",
    "bash": ".sh", "sh": ".sh", "shell": ".sh", "zsh": ".sh",
    "powershell": ".ps1", "ps1": ".ps1", "bat": ".bat", "cmd": ".bat",
    "sql": ".sql",
    "java": ".java",
    "kotlin": ".kt", "kt": ".kt",
    "scala": ".scala",
    "c": ".c", "h": ".h",
    "cpp": ".cpp", "c++": ".cpp", "cxx": ".cpp", "cc": ".cpp", "hpp": ".hpp",
    "csharp": ".cs", "c#": ".cs", "cs": ".cs",
    "go": ".go", "golang": ".go",
    "rust": ".rs", "rs": ".rs",
    "ruby": ".rb", "rb": ".rb",
    "php": ".php",
    "perl": ".pl", "pl": ".pl",
    "lua": ".lua",
    "r": ".r",
    "swift": ".swift",
    "dart": ".dart",
    "objective-c": ".m", "objc": ".m",
    "haskell": ".hs", "hs": ".hs",
    "elixir": ".ex", "ex": ".ex",
    "erlang": ".erl",
    "clojure": ".clj",
    "lisp": ".lisp",
    "vim": ".vim",
    "dockerfile": ".dockerfile", "docker": ".dockerfile",
    "makefile": ".mk", "make": ".mk",
    "text": ".txt", "txt": ".txt", "plaintext": ".txt",
}


def ext_for_lang(lang: str) -> str:
    """Vrátí vhodnou příponu souboru pro daný jazyk (nebo .txt)."""
    return LANG_EXT.get((lang or "").strip().lower(), ".txt")


def extract_code_blocks(content: str) -> list[dict]:
    """Vrátí seznam bloků kódu ve tvaru {'lang': ..., 'code': ...}."""
    blocks: list[dict] = []
    lines = content.split("\n")
    n = len(lines)
    i = 0
    while i < n:
        if lines[i].lstrip().startswith("```"):
            info = lines[i].lstrip()[3:].strip()
            lang = info.split()[0] if info else ""
            code_lines: list[str] = []
            i += 1
            while i < n and not lines[i].lstrip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            blocks.append({"lang": lang, "code": "\n".join(code_lines)})
        i += 1
    return blocks


def unique_path(directory: str, base: str, ext: str) -> str:
    """Vrátí cestu tak, aby nedocházelo k přepisu existujících souborů."""
    candidate = os.path.join(directory, f"{base}{ext}")
    if not os.path.exists(candidate):
        return candidate
    n = 2
    while True:
        candidate = os.path.join(directory, f"{base}_{n}{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1


# ---------------------------------------------------------------------------
# Statistika textu – jednorázová (používá se jen v modálním dialogu)
# ---------------------------------------------------------------------------

def analyze_text(content: str) -> dict:
    """Základní statistika textu: znaky, slova, řádky a bloky kódu dle jazyka."""
    lines = content.count("\n") + (1 if content else 0)
    words = len(content.split())
    blocks = 0
    in_block = False
    langs: dict[str, int] = {}
    for line in content.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            if in_block:
                in_block = False
            else:
                in_block = True
                blocks += 1
                lang = s[3:].strip() or "(bez jazyka)"
                langs[lang] = langs.get(lang, 0) + 1
    return {
        "chars": len(content),
        "words": words,
        "lines": lines,
        "blocks": blocks,
        "langs": langs,
    }


# ---------------------------------------------------------------------------
# Inkrementální tracker statistik – O(chunk) místo O(celý výstup)
# ---------------------------------------------------------------------------

class StatTracker:
    """Sleduje znaky/slova/řádky/bloky přírůstkově, jak přicházejí chunky.

    Umožňuje zobrazovat živou statistiku bez opakovaného skenování celého
    textu widgetu (což je u dlouhých výstupů výrazně dražší).
    """

    __slots__ = (
        "chars", "words", "newlines", "blocks", "langs",
        "_last_was_space", "_in_block", "_partial",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.chars = 0
        self.words = 0
        self.newlines = 0
        self.blocks = 0
        self.langs: dict[str, int] = {}
        self._last_was_space = True   # byl poslední znak bílé místo?
        self._in_block = False        # jsme uvnitř ``` bloku?
        self._partial = ""            # nedokončený řádek (bez \n na konci)

    def feed(self, text: str) -> None:
        if not text:
            return

        self.chars += len(text)

        # --- slova (přes C-optimized str.split) ---
        wc = len(text.split())
        if wc and not self._last_was_space and not text[0].isspace():
            wc -= 1  # první „slovo" navazuje na předchozí chunk
        self.words += wc
        self._last_was_space = text[-1].isspace()

        # --- řádky + fence ``` ---
        if "\n" in text:
            combined = self._partial + text
            parts = combined.split("\n")
            self._partial = parts[-1]
            for line in parts[:-1]:
                self.newlines += 1
                self._check_line(line)
        else:
            self._partial += text

    def _check_line(self, line: str) -> None:
        s = line.lstrip()
        if s.startswith("```"):
            if self._in_block:
                self._in_block = False
            else:
                self._in_block = True
                self.blocks += 1
                lang = s[3:].strip() or "(bez jazyka)"
                self.langs[lang] = self.langs.get(lang, 0) + 1

    def snapshot(self) -> tuple[int, int, int, int, dict[str, int]]:
        """Vrátí (chars, words, lines, blocks, langs) jako jednorázový snapshot."""
        lines = self.newlines + (1 if self.chars else 0)
        return self.chars, self.words, lines, self.blocks, self.langs


# ---------------------------------------------------------------------------
# Klient Ollama
# ---------------------------------------------------------------------------

class OllamaClient:
    """Tenký klient nad POST /api/chat se streamováním (NDJSON)."""

    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def stream(self, messages, stop_event, on_thinking, on_chunk) -> dict:
        """Postupně volá on_thinking/on_chunk.

        Vrací slovník se statistikami z poslední (hotové) zprávy serveru
        včetně 'done_reason' (např. 'stop' nebo 'length').
        """
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": dict(GEN_OPTIONS),
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.host + "/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        stats: dict = {}
        with urllib.request.urlopen(req) as resp:
            for raw in resp:
                if stop_event.is_set():
                    break
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("error"):
                    raise RuntimeError(obj["error"])
                msg = obj.get("message") or {}
                thinking = msg.get("thinking")
                if thinking:
                    on_thinking(thinking)
                content = msg.get("content")
                if content:
                    on_chunk(content)
                if obj.get("done"):
                    for key in (
                        "eval_count",
                        "eval_duration",
                        "prompt_eval_count",
                        "prompt_eval_duration",
                        "total_duration",
                        "load_duration",
                    ):
                        val = obj.get(key)
                        if isinstance(val, (int, float)):
                            stats[key] = val
                    dr = obj.get("done_reason")
                    if dr:
                        stats["done_reason"] = dr
                    break
        return stats


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App:
    POLL_MS = 60
    STATS_THROTTLE = 0.5   # s – jak často obnovovat řádek statistik při streamu

    def __init__(self, root: tk.Tk):
        self.root = root
        self.history: list[dict] = []
        self.reply_buf: list[str] = []
        self.last_reply_text: str = ""
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.q: queue.SimpleQueue = queue.SimpleQueue()
        self.show_thinking = tk.BooleanVar(value=False)

        # ukládání bloků kódu
        self.auto_save_code = tk.BooleanVar(value=True)
        self.save_dir_var = tk.StringVar(value=os.getcwd())
        self.dir_label_var = tk.StringVar()
        self.code_counter = 0

        # statistika
        self.last_stats: dict = {}
        self.totals: dict = {
            "replies": 0,
            "reply_chars": 0,
            "gen_s": 0.0,
            "eval": 0,
            "prompt_eval": 0,
        }
        self._stats_fresh = False
        self._tracker = StatTracker()
        self._last_stats_update = 0.0
        self._last_perf = 0.0

        # příznak, že poslední odpověď skončila dosažením limitu (length)
        self._last_done_reason: str = ""
        # příznak, že běží režim „pokračovat“ (aby se historie nezdvojovala)
        self._continuing: bool = False

        self._build_ui()
        self._update_dir_label()
        self.root.after(self.POLL_MS, self._poll)

    # -- UI -----------------------------------------------------------------

    def _build_ui(self):
        root = self.root
        root.title(f"Ollama kódový asistent – {DEFAULT_MODEL}")
        root.geometry("1040x760")
        root.minsize(720, 480)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        mono = tkfont.nametofont("TkFixedFont").copy()
        mono.configure(size=11)
        self._mono = mono

        # ---- horní lišta (1. řada: server, model) ------------------------
        bar = ttk.Frame(root, padding=(8, 6))
        bar.pack(side="top", fill="x")

        ttk.Label(bar, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Entry(bar, textvariable=self.model_var, width=16).pack(
            side="left", padx=(3, 10)
        )

        ttk.Label(bar, text="Server:").pack(side="left")
        self.host_var = tk.StringVar(value=DEFAULT_HOST)
        ttk.Entry(bar, textvariable=self.host_var, width=26).pack(
            side="left", padx=(3, 10)
        )

        ttk.Checkbutton(
            bar, text="Zobrazovat úvahy modelu (thinking)", variable=self.show_thinking
        ).pack(side="left", padx=(0, 10))

        ttk.Button(bar, text="Uložit výstup…", command=self._save_output).pack(
            side="right", padx=3
        )
        ttk.Button(bar, text="Nový dialog", command=self._new_dialog).pack(
            side="right", padx=3
        )

        # ---- horní lišta (2. řada: ukládání bloků kódu) ------------------
        bar2 = ttk.Frame(root, padding=(8, 0, 8, 6))
        bar2.pack(side="top", fill="x")

        ttk.Checkbutton(
            bar2,
            text="Automaticky ukládat bloky kódu z odpovědí",
            variable=self.auto_save_code,
        ).pack(side="left")

        ttk.Button(bar2, text="Složka…", command=self._choose_dir).pack(
            side="left", padx=(10, 4)
        )
        ttk.Button(
            bar2, text="Uložit bloky kódu…", command=self._manual_save_blocks
        ).pack(side="left", padx=3)

        ttk.Label(
            bar2, textvariable=self.dir_label_var, foreground="#606060"
        ).pack(side="left", padx=(8, 0))

        # ---- dělicí okno: výstup nahoře, vstup dole ----------------------
        paned = ttk.Panedwindow(root, orient="vertical")
        paned.pack(side="top", fill="both", expand=True)

        top = ttk.Frame(paned)
        bottom = ttk.Frame(paned)
        paned.add(top, weight=3)
        paned.add(bottom, weight=1)

        ttk.Label(top, text="Výstup asistenta").pack(anchor="w", padx=2, pady=(0, 2))
        self.out_text = scrolledtext.ScrolledText(
            top, wrap="none", font=mono, undo=False, state="normal"
        )
        self.out_text.pack(fill="both", expand=True)
        self.out_text.tag_configure(
            "user", font=(mono.actual("family"), 11, "bold"),
            foreground="#0b3d91", spacing3=4
        )
        self.out_text.tag_configure("bot", spacing1=1)
        self.out_text.tag_configure(
            "think", foreground="#8a8a8a",
            font=("TkFixedFont", 10, "italic")
        )
        self.out_text.tag_configure("err", foreground="#b00000")
        self.out_text.tag_configure(
            "sys", foreground="#606060", font=("TkFixedFont", 10, "italic")
        )
        self.out_text.tag_configure(
            "cont", foreground="#006060", font=("TkFixedFont", 10, "italic")
        )

        # Značka na konci – pomáhá udržet pozici i při velmi dlouhém výstupu
        self.out_text.mark_set("end_mark", "end-1c")
        self.out_text.mark_gravity("end_mark", "right")

        # ---- statistika + kopírování (řádek pod textem) ------------------
        srow = ttk.Frame(top)
        srow.pack(side="bottom", fill="x", pady=(3, 0))
        self.stats_var = tk.StringVar(value="Statistika se zobrazí po první odpovědi.")
        ttk.Button(
            srow, text="Kopírovat označené", command=self._copy_selection
        ).pack(side="right", padx=(6, 0))
        ttk.Button(srow, text="Kopírovat výstup", command=self._copy_output).pack(
            side="right", padx=(6, 0)
        )
        ttk.Button(srow, text="Statistika…", command=self._show_stats).pack(
            side="right", padx=(6, 0)
        )
        ttk.Label(srow, textvariable=self.stats_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )

        ttk.Label(
            bottom, text="Vstup (Ctrl+Enter odešle, Enter nový řádek)"
        ).pack(anchor="w", padx=2, pady=(4, 2))
        self.in_text = scrolledtext.ScrolledText(
            bottom, wrap="word", font=mono, height=7, undo=True
        )
        self.in_text.pack(fill="both", expand=True)
        self.in_text.bind("<Control-Return>", self._on_send_key)
        self.in_text.focus_set()

        # ---- kontextová menu (pravé tlačítko myši) ------------------------
        self.out_menu = tk.Menu(root, tearoff=0)
        self.out_menu.add_command(
            label="Kopírovat označené    Ctrl+C", command=self._copy_selection
        )
        self.out_menu.add_command(
            label="Označit vše",
            command=lambda: self._select_all(self.out_text),
        )
        self.out_menu.add_separator()
        self.out_menu.add_command(
            label="Kopírovat celý výstup", command=self._copy_output
        )
        self.out_menu.add_command(label="Uložit výstup…", command=self._save_output)
        self.out_menu.add_command(
            label="Uložit bloky kódu…", command=self._manual_save_blocks
        )
        self.out_menu.add_command(label="Statistika…", command=self._show_stats)
        self.out_menu.add_separator()
        self.out_menu.add_command(
            label="Pokračovat v generování", command=self._continue_generation
        )

        self.in_menu = tk.Menu(root, tearoff=0)
        self.in_menu.add_command(
            label="Kopírovat    Ctrl+C",
            command=lambda: self._copy_selection(self.in_text),
        )
        self.in_menu.add_command(
            label="Vložit    Ctrl+V",
            command=lambda: self._paste_to(self.in_text),
        )
        self.in_menu.add_separator()
        self.in_menu.add_command(
            label="Označit vše", command=lambda: self._select_all(self.in_text)
        )
        self.in_menu.add_command(label="Odeslat    Ctrl+Enter", command=self._send)

        # zkratky a menu i ve výstupu/vstupu
        for w in (self.out_text, self.in_text):
            w.bind("<Control-c>", self._on_copy_key)
            w.bind("<Control-C>", self._on_copy_key)
        self.out_text.bind("<Button-3>", lambda e: self._popup(self.out_menu, e))
        self.in_text.bind("<Button-3>", lambda e: self._popup(self.in_menu, e))

        # ---- spodní lišta ------------------------------------------------
        foot = ttk.Frame(root, padding=(8, 5))
        foot.pack(side="bottom", fill="x")

        self.send_btn = ttk.Button(foot, text="Odeslat", command=self._send)
        self.send_btn.pack(side="left")
        self.stop_btn = ttk.Button(
            foot, text="Zastavit", command=self._stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=(6, 0))
        self.continue_btn = ttk.Button(
            foot,
            text="Pokračovat v generování",
            command=self._continue_generation,
            state="disabled",
        )
        self.continue_btn.pack(side="left", padx=(6, 0))

        self.status_var = tk.StringVar(
            value="Připraveno – napište požadavek do vstupu."
        )
        ttk.Label(foot, textvariable=self.status_var, anchor="w").pack(
            side="left", padx=14
        )

    # -- odesílání ---------------------------------------------------------

    def _on_send_key(self, _event=None):
        self._send()
        return "break"

    def _send(self):
        if self.worker and self.worker.is_alive():
            return
        prompt = self.in_text.get("1.0", "end").rstrip()
        if not prompt:
            self.status_var.set("Vstup je prázdný.")
            return

        self._append("user", f"\n{'─' * 8}\nVSTUP\n{'─' * 8}\n{prompt}\n\n")
        self.history.append({"role": "user", "content": prompt})
        self.in_text.delete("1.0", "end")
        self._continuing = False
        self._start_generation()

    # -- pokračování v generování -----------------------------------------

    def _continue_generation(self):
        """Pošle modelu výzvu k pokračování tam, kde skončil."""
        if self.worker and self.worker.is_alive():
            return
        # Musí existovat předchozí odpověď asistenta
        if not self.history or self.history[-1]["role"] != "assistant":
            self.status_var.set("Není co pokračovat – nejprve pošlete dotaz.")
            return

        self._append(
            "cont",
            f"\n{'─' * 8}\nPOKRAČOVÁNÍ\n{'─' * 8}\n{CONTINUE_PROMPT}\n\n",
        )
        self.history.append({"role": "user", "content": CONTINUE_PROMPT})
        self._continuing = True
        self._start_generation()

    def _start_generation(self):
        """Spustí worker vlákno s aktuální historií."""
        self.reply_buf = []
        self.stop_event.clear()
        self.thinking_seen = False
        self._stats_fresh = False
        self._last_done_reason = ""

        host = self.host_var.get().strip() or DEFAULT_HOST
        model = self.model_var.get().strip() or DEFAULT_MODEL
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self.history]

        self.send_btn.configure(state="disabled")
        self.continue_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        if self._continuing:
            self.status_var.set(f"Pokračuji v generování (model {model})…")
        else:
            self.status_var.set(f"Odesláno modelu {model}…")

        self._last_perf = time.perf_counter()
        self.worker = threading.Thread(
            target=self._run, args=(host, model, messages), daemon=True
        )
        self.worker.start()

    def _run(self, host: str, model: str, messages):
        t0 = time.perf_counter()
        put = self.q.put  # lokální reference (mikro-optimalizace)
        try:
            client = OllamaClient(host, model)
            raw_stats = client.stream(
                messages,
                self.stop_event,
                lambda t: put(("thinking", t)),
                lambda c: put(("chunk", c)),
            )
            stats = dict(raw_stats or {})
            stats["gen_s"] = time.perf_counter() - t0
            if self.stop_event.is_set():
                put(("status", f"Generování zastaveno po {stats['gen_s']:.1f} s."))
            else:
                msg = f"Hotovo · {stats['gen_s']:.1f} s"
                tps = self._toks_per_s(stats)
                if stats.get("eval_count"):
                    msg += f" · {stats['eval_count']} tokenů"
                    if tps:
                        msg += f" ({tps:.1f} tok/s)"
                if stats.get("done_reason") == "length":
                    msg += " · dosažen limit, lze pokračovat tlačítkem „Pokračovat“"
                put(("status", msg))
            put(("stats", stats))
        except urllib.error.URLError as e:
            put(("error", f"Chyba spojení se serverem {host}: {e.reason}\n"
                          "Zkontrolujte, zda běží `ollama serve`."))
        except Exception as e:  # noqa: BLE001
            put(("error", f"Chyba: {e}"))
        finally:
            put(("finished", None))

    @staticmethod
    def _toks_per_s(stats: dict) -> float:
        """Rychlost generování v Tok/s (eval_count / eval_duration)."""
        n = stats.get("eval_count")
        d = stats.get("eval_duration")
        if n and d:
            return n / (d / 1e9)
        return 0.0

    def _stop(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.status_var.set("Zastavuji…")

    # -- příjem streamu ----------------------------------------------------

    def _poll(self):
        q_get = self.q.get_nowait
        chunk_buf: list[str] = []
        forced_stats = False

        try:
            while True:
                kind, payload = q_get()

                if kind == "chunk":
                    chunk_buf.append(payload)
                    continue

                # vyprázdnit nasbírané chunky před zpracováním jiné zprávy
                if chunk_buf:
                    self._flush_chunks(chunk_buf)
                    chunk_buf = []

                if kind == "thinking":
                    if self.show_thinking.get():
                        if not self.thinking_seen:
                            self._append("think", "\n— úvahy modelu —\n")
                        self._append("think", payload)
                    elif not self.thinking_seen:
                        self.status_var.set("Model přemýšlí…")
                    self.thinking_seen = True

                elif kind == "status":
                    self.status_var.set(payload)

                elif kind == "stats":
                    self.last_stats = payload
                    self._stats_fresh = True
                    forced_stats = True
                    dr = payload.get("done_reason") if isinstance(payload, dict) else None
                    if dr:
                        self._last_done_reason = dr

                elif kind == "error":
                    self._append("err", f"\n{payload}\n")
                    self.status_var.set("Chyba – viz výstup.")
                    forced_stats = True

                elif kind == "finished":
                    if self.reply_buf:
                        content = "".join(self.reply_buf)
                        self.history.append(
                            {"role": "assistant", "content": content}
                        )
                        self.last_reply_text = content
                        if self._stats_fresh:
                            g = self.last_stats
                            self.totals["replies"] += 1
                            self.totals["reply_chars"] += len(content)
                            self.totals["gen_s"] += float(g.get("gen_s") or 0.0)
                            self.totals["eval"] += int(g.get("eval_count") or 0)
                            self.totals["prompt_eval"] += int(
                                g.get("prompt_eval_count") or 0
                            )
                        self._stats_fresh = False

                        if self.auto_save_code.get():
                            self._save_blocks_from_text(content, auto=True)

                    self.reply_buf = []
                    self.send_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    # Tlačítko Pokračovat zapneme vždy, když existuje odpověď
                    if self.history and self.history[-1]["role"] == "assistant":
                        self.continue_btn.configure(state="normal")
                        if self._last_done_reason == "length":
                            self._append(
                                "cont",
                                "\n[Model dosáhl limitu délky odpovědi – "
                                "můžete pokračovat tlačítkem „Pokračovat v generování“. ]\n",
                            )
                    forced_stats = True

        except queue.Empty:
            pass

        if chunk_buf:
            self._flush_chunks(chunk_buf)

        # --- throttlovaný refresh statistik ---
        now = time.monotonic()
        if forced_stats or (
            chunk_buf and now - self._last_stats_update >= self.STATS_THROTTLE
        ):
            self._refresh_stats()
            self._last_stats_update = now

        self.root.after(self.POLL_MS, self._poll)

    def _flush_chunks(self, chunks: list[str]):
        """Vloží nasbírané chunky jedním insertem + jedním yview checkem."""
        text = "".join(chunks)
        self.reply_buf.append(text)

        # detekce, zda uživatel je na konci (1 yview call místo N)
        at_bottom = self.out_text.yview()[1] >= 0.999
        self.out_text.insert("end", text, "bot")
        if at_bottom:
            # Pro velmi dlouhé výstupy je see("end") dražší – používejeme mark
            self.out_text.mark_set("insert", "end-1c")
            self.out_text.see("end")

        # inkrementální tracker (žádný full-scan)
        self._tracker.feed(text)
        self.status_var.set("Generuji odpověď…")

    # -- pomocné akce ------------------------------------------------------

    def _append(self, tag: str, text: str):
        at_bottom = self.out_text.yview()[1] >= 0.999
        self.out_text.insert("end", text, tag)
        if at_bottom:
            self.out_text.see("end")
        self._tracker.feed(text)

    # -- kopírování označeného textu --------------------------------------

    def _popup(self, menu: tk.Menu, event: tk.Event):
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _on_copy_key(self, event: tk.Event):
        self._copy_selection(event.widget)
        return "break"

    def _copy_selection(self, widget: tk.Text | None = None):
        w = widget if isinstance(widget, tk.Text) else self.out_text
        try:
            sel = w.get("sel.first", "sel.last")
        except tk.TclError:
            sel = ""
        if not sel:
            self.status_var.set(
                "Nic není označeno – text označte tažením myši "
                "(nebo pravé tlačítko → „Označit vše“)."
            )
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(sel)
        self.status_var.set(
            f"Zkopírováno označené: {len(sel)} znaků, {len(sel.split())} slov."
        )

    @staticmethod
    def _select_all(widget: tk.Text):
        widget.tag_remove("sel", "1.0", "end")
        widget.tag_add("sel", "1.0", "end-1c")

    @staticmethod
    def _paste_to(widget: tk.Text):
        widget.event_generate("<<Paste>>")

    # -- ukládání bloků kódu ----------------------------------------------

    def _update_dir_label(self):
        path = self.save_dir_var.get() or os.getcwd()
        shown = path if len(path) <= 60 else "…" + path[-57:]
        self.dir_label_var.set(f"→ {shown}")

    def _choose_dir(self):
        chosen = filedialog.askdirectory(
            parent=self.root,
            title="Cílová složka pro bloky kódu",
            initialdir=self.save_dir_var.get() or os.getcwd(),
            mustexist=False,
        )
        if chosen:
            self.save_dir_var.set(chosen)
            self._update_dir_label()
            self.status_var.set(f"Cílová složka: {chosen}")

    def _manual_save_blocks(self):
        if not self.last_reply_text:
            messagebox.showinfo(
                "Uložit bloky kódu",
                "Zatím není uložená žádná odpověď, ze které by šly bloky vzít.",
                parent=self.root,
            )
            return
        saved = self._save_blocks_from_text(self.last_reply_text, auto=False)
        if not saved:
            messagebox.showinfo(
                "Uložit bloky kódu",
                "V poslední odpovědi nebyly nalezeny žádné bloky kódu (```…```).",
                parent=self.root,
            )

    def _save_blocks_from_text(self, content: str, auto: bool = False) -> list[str]:
        """Uloží všechny ```bloky``` z textu jako samostatné soubory."""
        blocks = extract_code_blocks(content)
        if not blocks:
            if not auto:
                self.status_var.set("V odpovědi nejsou žádné bloky kódu.")
            return []

        directory = self.save_dir_var.get().strip() or os.getcwd()
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            messagebox.showerror(
                "Chyba ukládání",
                f"Není možné vytvořit složku:\n{directory}\n\n{e}",
                parent=self.root,
            )
            return []

        saved: list[str] = []
        for block in blocks:
            ext = ext_for_lang(block["lang"])
            self.code_counter += 1
            base = f"blok_{self.code_counter}"
            path = unique_path(directory, base, ext)
            code = block["code"]
            if not code.endswith("\n"):
                code += "\n"
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(code)
                saved.append(path)
            except OSError as e:
                self._append(
                    "err", f"\nNelze uložit blok kódu do {path}: {e}\n"
                )
                self.status_var.set("Chyba při ukládání bloku kódu.")

        if saved:
            names = ", ".join(os.path.basename(p) for p in saved)
            prefix = "Automaticky uloženo" if auto else "Uloženo"
            self.status_var.set(f"{prefix} {len(saved)} blok(ů) kódu: {names}")
            if not auto:
                messagebox.showinfo(
                    "Bloky kódu uloženy",
                    "Uložené soubory:\n" + "\n".join(saved),
                    parent=self.root,
                )
        return saved

    # -- statistika --------------------------------------------------------

    def _refresh_stats(self):
        """Obnoví kompaktní řádek statistik z inkrementálního trackeru."""
        chars, words, lines, blocks, _langs = self._tracker.snapshot()
        parts = [
            f"Znaky {chars}",
            f"Slova {words}",
            f"Řádky {lines}",
            f"Bloků kódu {blocks}",
        ]
        g = self.last_stats
        if g.get("eval_count") or g.get("gen_s"):
            seg = f"Poslední odpověď {float(g.get('gen_s') or 0):.1f} s"
            if g.get("eval_count"):
                seg += f" · {g['eval_count']} tok"
                tps = self._toks_per_s(g)
                if tps:
                    seg += f" · {tps:.1f} tok/s"
            parts.append(seg)
        elif self.worker and self.worker.is_alive():
            parts.append(f"Generuji… {chars} znaků")
        self.stats_var.set("  ·  ".join(parts))

    def _show_stats(self):
        """Detailní statistika – používá jeden full-scan (volané zřídka)."""
        g = self.last_stats
        t = self.totals
        st = analyze_text(self.out_text.get("1.0", "end-1c"))

        lines = ["— Poslední odpověď —"]
        if g.get("gen_s") is not None or g.get("eval_count"):
            if g.get("gen_s") is not None:
                lines.append(f"Doba generování:  {float(g['gen_s']):.2f} s")
            if g.get("eval_count"):
                lines.append(f"Tokeny odpovědi (eval_count):  {g['eval_count']}")
            tps = self._toks_per_s(g)
            if tps:
                lines.append(
                    f"Rychlost:  {tps:.1f} tok/s  "
                    f"(eval_duration {float(g['eval_duration']) / 1e9:.2f} s)"
                )
            if g.get("prompt_eval_count"):
                lines.append(
                    f"Tokeny vstupu (prompt_eval_count):  {g['prompt_eval_count']}"
                )
            if g.get("prompt_eval_duration"):
                lines.append(
                    f"Zpracování vstupu:  "
                    f"{float(g['prompt_eval_duration']) / 1e9:.2f} s"
                )
            if g.get("load_duration"):
                lines.append(
                    f"Načtení modelu:  {float(g['load_duration']) / 1e9:.2f} s"
                )
            if g.get("total_duration"):
                lines.append(
                    f"Celková doba volání API:  "
                    f"{float(g['total_duration']) / 1e9:.2f} s"
                )
            if g.get("done_reason"):
                lines.append(f"Důvod ukončení:  {g['done_reason']}")
        else:
            lines.append("Zatím žádná odpověď.")

        lines += [
            "", "— Text ve výstupu —",
            f"Znaky:  {st['chars']}   Slova:  {st['words']}   "
            f"Řádky:  {st['lines']}",
            f"Bloky kódu:  {st['blocks']}",
        ]
        if st["langs"]:
            lines.append(
                "Jazyky:  "
                + ", ".join(f"{k}: {v}" for k, v in sorted(st["langs"].items()))
            )

        lines += [
            "", "— Celý dialog —",
            f"Odpovědí:  {t['replies']}   "
            f"Znaků odpovědí:  {t['reply_chars']}",
            f"Celkový čas generování:  {t['gen_s']:.1f} s",
            f"Tokeny odpovědí celkem:  {t['eval']}   "
            f"Vstupní tokeny celkem:  {t['prompt_eval']}",
        ]
        if t["replies"] and t["gen_s"]:
            lines.append(
                f"Průměr na odpověď:  {t['gen_s'] / t['replies']:.1f} s, "
                f"{t['eval'] / t['replies']:.0f} tokenů"
            )

        lines += [
            "", "— Ukládání kódu —",
            f"Cílová složka:  {self.save_dir_var.get() or os.getcwd()}",
            f"Automatické ukládání:  "
            f"{'zapnuto' if self.auto_save_code.get() else 'vypnuto'}",
            f"Uložených souborů v této relaci:  {self.code_counter}",
            "",
            "— Generování —",
            f"num_predict:  {GEN_OPTIONS.get('num_predict')} (=-1 neomezeně)",
            f"num_ctx:      {GEN_OPTIONS.get('num_ctx')} tokenů",
        ]

        messagebox.showinfo(
            "Statistika výstupu", "\n".join(lines), parent=self.root
        )

    def _new_dialog(self):
        content = self.out_text.get("1.0", "end").strip()
        if content and not messagebox.askyesno(
            "Nový dialog", "Smazat celý výstup a začít nový dialog?",
            parent=self.root,
        ):
            return
        self.out_text.delete("1.0", "end")
        self.history.clear()
        self.reply_buf.clear()
        self.last_reply_text = ""
        self.last_stats = {}
        self.totals = {"replies": 0, "reply_chars": 0, "gen_s": 0.0,
                       "eval": 0, "prompt_eval": 0}
        self._stats_fresh = False
        self._tracker.reset()          # <-- reset inkrementálního trackeru
        self._last_done_reason = ""
        self._continuing = False
        self.continue_btn.configure(state="disabled")
        self._refresh_stats()
        self.status_var.set("Nový dialog – historie byla smazána.")

    def _copy_output(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.out_text.get("1.0", "end-1c"))
        self.status_var.set("Výstup zkopírován do schránky.")

    def _save_output(self):
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Uložit výstup",
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("Vše", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.out_text.get("1.0", "end-1c"))
            self.status_var.set(f"Uloženo: {path}")
        except OSError as e:
            messagebox.showerror("Chyba ukládání", str(e), parent=self.root)

    def _on_close(self):
        self.stop_event.set()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
