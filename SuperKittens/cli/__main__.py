"""SuperKittens CLI — `sk launch`, `sk models`, `sk <model>`, `sk <model> run`.

Thin user-facing driver over the inference registry + api.load. Weights live in
a single `weights/` directory at the repo root (override with SK_WEIGHTS_ROOT);
`sk <model>` downloads the model there (GGUF + tokenizer); `sk <model> run`
opens an interactive chat with live tok/s. No loader changes: weights are passed
to api.load via the adapter's `snapshot=` override.

    sk launch              banner + commands
    sk models              list models, status, and weight paths
    sk <model>             download a model's weights (if not already local)
    sk <model> run         chat with the model (live tok/s)
    sk talk [<A> <B>]      two models converse (default qwen3-1.7b + qwen3-8b-q4km)
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

from .banner import print_banner

REPO_ROOT = Path(__file__).resolve().parents[2]

# Keep the CLI output clean: silence huggingface_hub's unauthenticated-request
# warning + info chatter (downloads still work fine without a token).
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import logging as _logging
_logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)


def weights_root() -> Path:
    return Path(os.environ.get("SK_WEIGHTS_ROOT") or (REPO_ROOT / "weights"))


# GGUF artifacts live in a separate HF repo from the canonical (safetensors)
# repo. Convention: <canonical>-GGUF for first-party repos (Qwen); the
# exceptions below point at community repos. Everything else derives from
# registry.hf_repo + "-GGUF".
GGUF_REPO_OVERRIDE = {
    "mistral-7b-v0.3":  "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    "nemotron-nano-8b": "bartowski/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-GGUF",
}

_AUX_FILES = ["tokenizer.json", "tokenizer_config.json", "tokenizer.model",
              "config.json", "generation_config.json", "special_tokens_map.json"]


# ---- small ANSI helper (no deps; strips when not a TTY / NO_COLOR) ----------
class _C:
    def __init__(self, stream):
        on = bool(getattr(stream, "isatty", lambda: False)()) and not os.environ.get("NO_COLOR")
        self.pink = "\x1b[38;2;236;72;153m" if on else ""
        self.violet = "\x1b[38;2;168;85;247m" if on else ""
        self.cyan = "\x1b[38;2;110;200;220m" if on else ""
        self.dim = "\x1b[2m" if on else ""
        self.bold = "\x1b[1m" if on else ""
        self.reset = "\x1b[0m" if on else ""


def _specs():
    from SuperKittens.inference import registry
    return registry.SPECS


def _dest(spec) -> Path:
    return weights_root() / spec.weight_dir


def _is_downloaded(spec) -> bool:
    dest = _dest(spec)
    if spec.gguf_name:
        return (dest / spec.gguf_name).exists()
    return dest.exists() and any(dest.glob("*.safetensors"))


# ---- commands ---------------------------------------------------------------
def cmd_launch(_):
    print_banner()
    c = _C(sys.stdout)
    specs = _specs()
    have = sum(1 for s in specs.values() if _is_downloaded(s))
    print()
    print(f"  {c.bold}Commands{c.reset}")
    print(f"    {c.pink}sk get{c.reset}               browse models by family (↑/↓) and download")
    print(f"    {c.pink}sk models{c.reset}            list models, status, paths")
    print(f"    {c.pink}sk <model>{c.reset}           download a model")
    print(f"    {c.pink}sk <model> run{c.reset}       chat with the model (live tok/s)")
    print(f"    {c.pink}sk talk{c.reset}              watch two models converse")
    print(f"\n  {c.dim}weights: {weights_root()}  ·  {have}/{len(specs)} downloaded{c.reset}\n")


def cmd_models(_):
    c = _C(sys.stdout)
    wr = weights_root()
    print(f"\n  {c.bold}{'MODEL':<22}{'STATUS':<16}PATH{c.reset}")
    for name, s in _specs().items():
        down = _is_downloaded(s)
        mark = f"{c.pink}✓{c.reset}" if down else f"{c.dim}·{c.reset}"
        status = "downloaded" if down else "not local"
        col = "" if down else c.dim
        print(f"  {mark} {col}{name:<20}{status:<16}{wr / s.weight_dir}{c.reset}")
    print()


# Family display order + friendly labels for the picker.
_FAMILY_ORDER = ["qwen3", "qwen2", "gemma4", "gemma4_unified",
                 "mamba2", "deepseek", "nemotron", "mistral"]
_FAMILY_LABEL = {"qwen3": "Qwen3", "qwen2": "Qwen2.5", "gemma4": "Gemma 4",
                 "gemma4_unified": "Gemma 4 (unified)", "mamba2": "Mamba-2",
                 "deepseek": "DeepSeek", "nemotron": "Nemotron", "mistral": "Mistral"}


def _quant_of(spec) -> str:
    if spec.gguf_name:
        m = re.search(r"(Q\d[_A-Za-z0-9]*)", spec.gguf_name)
        return m.group(1) if m else "GGUF"
    return "safetensors"


def _grouped():
    """OrderedDict family -> [(name, spec)], families in _FAMILY_ORDER."""
    from collections import OrderedDict
    specs = _specs()
    groups = OrderedDict()
    fams = _FAMILY_ORDER + [f for f in {s.family for s in specs.values()}
                            if f not in _FAMILY_ORDER]
    for fam in fams:
        members = [(n, s) for n, s in specs.items() if s.family == fam]
        if members:
            groups[fam] = members
    return groups


def _picker():
    """Arrow-key model browser grouped by family. Returns a model name or None."""
    import curses

    groups = _grouped()
    rows = []  # (kind, label, spec)
    for fam, members in groups.items():
        rows.append(("hdr", _FAMILY_LABEL.get(fam, fam), None))
        for name, s in members:
            rows.append(("mdl", name, s))
    selectable = [i for i, r in enumerate(rows) if r[0] == "mdl"]
    if not selectable:
        return None

    def _run(stdscr):
        curses.curs_set(0)
        try:
            curses.start_color(); curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_MAGENTA, -1)
        except Exception:
            pass
        cur = 0
        while True:
            stdscr.erase()
            maxy, maxx = stdscr.getmaxyx()
            title = "Select a model   ↑/↓ move · Enter download · q quit"
            stdscr.addnstr(0, 0, title, maxx - 1, curses.A_BOLD)
            y = 2
            for i, (kind, label, s) in enumerate(rows):
                if y >= maxy - 1:
                    break
                if kind == "hdr":
                    try:
                        stdscr.addnstr(y, 2, label, maxx - 3,
                                       curses.A_BOLD | curses.color_pair(1))
                    except Exception:
                        stdscr.addnstr(y, 2, label, maxx - 3, curses.A_BOLD)
                else:
                    is_cur = selectable[cur] == i
                    down = _is_downloaded(s)
                    mark = "✓" if down else " "
                    text = f"{mark} {label:<22}{_quant_of(s)}"
                    attr = curses.A_REVERSE if is_cur else (curses.A_DIM if down else curses.A_NORMAL)
                    stdscr.addnstr(y, 5, text, maxx - 6, attr)
                y += 1
            stdscr.refresh()
            k = stdscr.getch()
            if k in (ord("q"), ord("Q"), 27):
                return None
            if k in (curses.KEY_UP, ord("k")):
                cur = (cur - 1) % len(selectable)
            elif k in (curses.KEY_DOWN, ord("j")):
                cur = (cur + 1) % len(selectable)
            elif k in (curses.KEY_ENTER, 10, 13):
                return rows[selectable[cur]][1]

    try:
        return curses.wrapper(_run)
    except Exception as e:
        print(f"  (picker unavailable: {e}; use `sk models` + `sk <model>`)", file=sys.stderr)
        return None


def cmd_get(args):
    if args:
        return cmd_download(args)
    sel = _picker()
    if not sel:
        return
    _ensure(sel)
    try:
        ans = input(f"\n  run {sel} now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"
    if ans in ("", "y", "yes"):
        cmd_run([sel])


def _fmt_size(b: float) -> str:
    return f"{b/1e9:.2f} GB" if b >= 1e9 else f"{b/1e6:.0f} MB"


def _render_bar(label: str, done: int, total: int, elapsed: float):
    c = _C(sys.stdout)
    width = 30
    frac = (done / total) if total else 0.0
    fill = int(frac * width)
    bar = "█" * fill + "·" * (width - fill)
    spd = (done / elapsed) if elapsed > 0 else 0.0
    eta = ((total - done) / spd) if (spd > 0 and total) else 0.0
    sys.stdout.write(
        f"\r  {label}  {c.pink}{bar}{c.reset} {frac*100:5.1f}%  "
        f"{_fmt_size(done)}/{_fmt_size(total)}  {c.dim}{spd/1e6:5.1f} MB/s · ETA {eta:3.0f}s{c.reset}   ")
    sys.stdout.flush()


def _stream_download(url: str, dest_path: Path, label: str):
    """Stream a public file with a themed live progress bar (%, size, MB/s, ETA)."""
    import urllib.request
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(dest_path) + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "superkittens-cli"})
    with urllib.request.urlopen(req) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        t0 = last = time.time()
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last >= 0.1 or done == total:
                    last = now
                    _render_bar(label, done, total or done, now - t0)
    sys.stdout.write("\n")
    os.replace(tmp, dest_path)


def _download(name: str) -> Path:
    from huggingface_hub import hf_hub_download, list_repo_files
    spec = _specs()[name]
    dest = _dest(spec)
    dest.mkdir(parents=True, exist_ok=True)

    if spec.gguf_name:
        repo = GGUF_REPO_OVERRIDE.get(name, f"{spec.hf_repo}-GGUF")
        target = spec.gguf_name
        try:
            files = list_repo_files(repo)
        except Exception as e:
            print(f"  ! could not list {repo}: {e}", file=sys.stderr)
            files = []
        if files and target not in files:
            m = re.search(r"(Q\d[_A-Za-z0-9]*)", target)
            tag = m.group(1).lower() if m else ""
            cand = [f for f in files if f.endswith(".gguf") and tag in f.lower()]
            if cand:
                target = cand[0]
                print(f"  (resolved GGUF filename -> {target})")
        print(f"==> downloading {name}  {repo}/{target}")
        url = f"https://huggingface.co/{repo}/resolve/main/{target}"
        try:
            _stream_download(url, dest / target, name)
        except Exception as e:
            print(f"  (stream failed: {e}; using hf fallback)", file=sys.stderr)
            hf_hub_download(repo, target, local_dir=str(dest))
    else:
        from huggingface_hub import snapshot_download
        print(f"==> downloading {name}: full repo {spec.hf_repo}")
        snapshot_download(spec.hf_repo, local_dir=str(dest),
                          allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"])

    for f in _AUX_FILES:
        if (dest / f).exists():
            continue
        try:
            hf_hub_download(spec.hf_repo, f, local_dir=str(dest))
        except Exception:
            pass
    print(f"  -> {dest}")
    return dest


def _ensure(name: str) -> Path:
    specs = _specs()
    if name not in specs:
        print(f"unknown model {name!r}. run `sk models`.", file=sys.stderr)
        sys.exit(2)
    if not _is_downloaded(specs[name]):
        _download(name)
    else:
        print(f"  {name} already downloaded: {_dest(specs[name])}")
    return _dest(specs[name])


def cmd_download(args):
    if not args:
        print("usage: sk <model>", file=sys.stderr); sys.exit(2)
    _ensure(args[0])


def _load(name: str):
    import SuperKittens.api as api
    snap = _ensure(name)
    print(f"  loading {name} ...", flush=True)
    return api.load(name, snapshot=str(snap))


def _strip_think(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    return re.sub(r"[ \t]+", " ", s).strip()


def _read_line(prompt: str):
    """Read one line. In a TTY, bare Esc (or Ctrl-C/Ctrl-D) returns None so the
    caller can leave the chat; arrow-key escape sequences are ignored. Falls
    back to plain readline when stdin is not a TTY (piped input)."""
    import termios, tty, select
    sys.stdout.write(prompt); sys.stdout.flush()
    if not sys.stdin.isatty():
        ln = sys.stdin.readline()
        if not ln:
            return None
        sys.stdout.write(ln if ln.endswith("\n") else ln + "\n")
        return ln.rstrip("\n")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    buf = bytearray()
    try:
        tty.setcbreak(fd)
        while True:
            ch = os.read(fd, 1)
            if ch == b"\x1b":                 # Esc
                r, _, _ = select.select([fd], [], [], 0.06)
                if r:                          # arrow / escape sequence -> ignore
                    os.read(fd, 2)
                    continue
                sys.stdout.write("\n"); return None   # bare Esc -> leave chat
            if ch in (b"\x03", b"\x04"):       # Ctrl-C / Ctrl-D
                sys.stdout.write("\n"); return None
            if ch in (b"\r", b"\n"):
                sys.stdout.write("\n"); return buf.decode("utf-8", "replace")
            if ch in (b"\x7f", b"\x08"):       # backspace
                if buf:
                    buf.pop()
                    sys.stdout.write("\b \b"); sys.stdout.flush()
                continue
            buf += ch
            os.write(fd, ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def cmd_run(args):
    import numpy as np
    import termios, tty, select
    name = args[0]
    m = _load(name)
    c = _C(sys.stdout)
    tok = m.tokenizer
    rng = np.random.default_rng()
    stops = set()
    for x in (getattr(tok, "eos_ids", None) or []):
        stops.add(int(x))
    if getattr(tok, "eos_id", None) is not None:
        stops.add(int(tok.eos_id))
    istty = sys.stdin.isatty()
    print(f"\n  {c.violet}╭─ {c.pink}{c.bold}{name}{c.reset}{c.violet} ready ─ type a message · Esc to leave · Esc mid-reply to stop ─╮{c.reset}\n")
    history = []
    while True:
        line = _read_line(f"{c.cyan}you ❯ {c.reset}")
        if line is None:
            break
        line = line.strip()
        if not line:
            continue
        if line in ("exit", "quit"):
            break
        history.append({"role": "user", "content": line + " /no_think"})
        ids = np.array(tok.chat(history, bos=True), dtype=np.int32)
        # Prefill (TTFT) and decode are timed separately so tok/s reflects the
        # steady-state decode rate, not prompt prefill.
        m.reset()
        t_pre = time.time()
        m._forward(ids)
        ttft = time.time() - t_pre
        first = m._sample(m._last_logits(), 0.7, 0.9, None, rng)
        out = [first]
        last = first
        interrupted = False
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd) if istty else None
        t_dec = time.time()
        try:
            if old is not None:
                tty.setcbreak(fd)             # poll for Esc between decode steps
            if first not in stops:
                for _ in range(255):
                    m._forward(np.array([last], dtype=np.int32))
                    last = m._sample(m._last_logits(), 0.7, 0.9, None, rng)
                    out.append(last)
                    if last in stops:
                        break
                    if istty and select.select([fd], [], [], 0)[0]:
                        if os.read(fd, 1) in (b"\x1b", b"\x03"):
                            interrupted = True
                            break
        finally:
            if old is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        dec_dt = time.time() - t_dec
        n_dec = max(len(out) - 1, 1)
        tps = n_dec / dec_dt if dec_dt > 0 else 0.0
        text = _strip_think(tok.decode(out, skip_special=True))
        history.append({"role": "assistant", "content": text})
        tag = f"{c.dim} (stopped){c.reset}" if interrupted else ""
        print(f"{c.pink}{name} ❯{c.reset} {text}{tag}")
        print(f"{c.dim}      {len(out)} tok · {tps:.1f} tok/s · {ttft*1000:.0f}ms to first token{c.reset}\n")
    print(f"  {c.dim}bye 🐾{c.reset}")


def cmd_talk(args):
    a_name = args[0] if len(args) > 0 else "qwen3-1.7b"
    b_name = args[1] if len(args) > 1 else "qwen3-8b-q4km"
    A, B = _load(a_name), _load(b_name)
    cast = [(f"🐾 Pixel ({a_name})", A,
             "You are Pixel, a playful curious kitten. Reply in ONE short sentence. /no_think"),
            (f"🐱 Sage  ({b_name})", B,
             "You are Sage, a wise old cat. Reply in ONE short sentence. /no_think")]
    topic = "What is the best thing about running on a Mac mini?"
    print(f"\nTOPIC: {topic}\n" + "-" * 60)
    last = topic
    for turn in range(6):
        nm, model, persona = cast[turn % 2]
        prompt = f'{persona}\n\nThe other cat just said: "{last}"\nYour reply:'
        txt = _strip_think(model.chat(prompt, max_new_tokens=64,
                                      temperature=0.7, top_p=0.9, seed=turn))
        print(f"{nm}: {txt}", flush=True)
        last = txt
    print("-" * 60)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("launch", "help", "-h", "--help"):
        return cmd_launch(argv)
    cmd = argv[0]
    if cmd in ("models", "ls", "list"):
        return cmd_models(argv)
    if cmd == "talk":
        return cmd_talk(argv[1:])
    if cmd in ("get", "browse", "download"):
        return cmd_get(argv[1:])
    if cmd in _specs():
        if len(argv) > 1 and argv[1] == "run":
            return cmd_run([cmd])
        return cmd_download([cmd])
    print(f"unknown command/model {cmd!r}. try `sk models`.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
