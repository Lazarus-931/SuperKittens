"""SuperKittens CLI banner.

Purple->pink gradient wordmark for the SK CLI (e.g. ``sk launch``).
The colored ASCII art was rendered with oh-my-logo and embedded here so the
banner works offline (no Node/npx at runtime). See cli/README_logo.md for the
exact command + palette and how to regenerate.

Public API:
    print_banner(stream=None, force_color=None, width=None)
    BANNER          -- full "SuperKittens" wordmark (ANSI color, ~59 cols)
    BANNER_COMPACT  -- compact "SK" mark (ANSI color, ~18 cols)
"""

from __future__ import annotations

import os
import re
import shutil
import sys

# Rendered by: npx oh-my-logo "SuperKittens" \
#   --palette-colors "#7c3aed,#a855f7,#d946ef,#ec4899" --color -d horizontal
BANNER = (
    "  \x1b[38;2;124;58;237m_\x1b[39m\x1b[38;2;139;67;240m_\x1b[39m\x1b[38;2;153;76;244m_\x1b[39m\x1b[38;2;168;85;247m_\x1b[39m                        \x1b[38;2;184;80;244m_\x1b[39m  \x1b[38;2;201;75;242m_\x1b[39m\x1b[38;2;217;70;239m_\x1b[39m\x1b[38;2;223;71;210m_\x1b[39m \x1b[38;2;230;71;182m_\x1b[39m   \x1b[38;2;236;72;153m_\x1b[39m                 \n \x1b[38;2;124;58;237m/\x1b[39m \x1b[38;2;128;60;238m_\x1b[39m\x1b[38;2;132;63;239m_\x1b[39m\x1b[38;2;136;65;240m_\x1b[39m\x1b[38;2;140;68;241m|\x1b[39m \x1b[38;2;144;70;242m_\x1b[39m   \x1b[38;2;148;73;242m_\x1b[39m \x1b[38;2;152;75;243m_\x1b[39m \x1b[38;2;156;78;244m_\x1b[39m\x1b[38;2;160;80;245m_\x1b[39m   \x1b[38;2;164;83;246m_\x1b[39m\x1b[38;2;168;85;247m_\x1b[39m\x1b[38;2;172;84;246m_\x1b[39m \x1b[38;2;176;83;246m_\x1b[39m \x1b[38;2;180;81;245m_\x1b[39m\x1b[38;2;184;80;244m_\x1b[39m\x1b[38;2;188;79;244m|\x1b[39m \x1b[38;2;193;78;243m|\x1b[39m\x1b[38;2;197;76;242m/\x1b[39m \x1b[38;2;201;75;242m(\x1b[39m\x1b[38;2;205;74;241m_\x1b[39m\x1b[38;2;209;73;240m)\x1b[39m \x1b[38;2;213;71;240m|\x1b[39m\x1b[38;2;217;70;239m_\x1b[39m\x1b[38;2;219;70;232m|\x1b[39m \x1b[38;2;220;70;225m|\x1b[39m\x1b[38;2;222;71;218m_\x1b[39m \x1b[38;2;223;71;210m_\x1b[39m\x1b[38;2;225;71;203m_\x1b[39m\x1b[38;2;227;71;196m_\x1b[39m \x1b[38;2;228;71;189m_\x1b[39m \x1b[38;2;230;71;182m_\x1b[39m\x1b[38;2;231;72;175m_\x1b[39m  \x1b[38;2;233;72;167m_\x1b[39m\x1b[38;2;234;72;160m_\x1b[39m\x1b[38;2;236;72;153m_\x1b[39m \n \x1b[38;2;124;58;237m\\\x1b[39m\x1b[38;2;127;60;238m_\x1b[39m\x1b[38;2;131;62;239m_\x1b[39m\x1b[38;2;134;64;239m_\x1b[39m \x1b[38;2;138;66;240m\\\x1b[39m\x1b[38;2;141;68;241m|\x1b[39m \x1b[38;2;144;70;242m|\x1b[39m \x1b[38;2;148;73;242m|\x1b[39m \x1b[38;2;151;75;243m|\x1b[39m \x1b[38;2;154;77;244m'\x1b[39m\x1b[38;2;158;79;245m_\x1b[39m \x1b[38;2;161;81;245m\\\x1b[39m \x1b[38;2;165;83;246m/\x1b[39m \x1b[38;2;168;85;247m_\x1b[39m \x1b[38;2;172;84;246m\\\x1b[39m \x1b[38;2;176;83;246m'\x1b[39m\x1b[38;2;180;81;245m_\x1b[39m\x1b[38;2;184;80;244m_\x1b[39m\x1b[38;2;188;79;244m|\x1b[39m \x1b[38;2;193;78;243m'\x1b[39m \x1b[38;2;197;76;242m/\x1b[39m\x1b[38;2;201;75;242m|\x1b[39m \x1b[38;2;205;74;241m|\x1b[39m \x1b[38;2;209;73;240m_\x1b[39m\x1b[38;2;213;71;240m_\x1b[39m\x1b[38;2;217;70;239m|\x1b[39m \x1b[38;2;219;70;232m_\x1b[39m\x1b[38;2;220;70;225m_\x1b[39m\x1b[38;2;222;71;218m/\x1b[39m \x1b[38;2;223;71;210m_\x1b[39m \x1b[38;2;225;71;203m\\\x1b[39m \x1b[38;2;227;71;196m'\x1b[39m\x1b[38;2;228;71;189m_\x1b[39m \x1b[38;2;230;71;182m\\\x1b[39m\x1b[38;2;231;72;175m/\x1b[39m \x1b[38;2;233;72;167m_\x1b[39m\x1b[38;2;234;72;160m_\x1b[39m\x1b[38;2;236;72;153m|\x1b[39m\n  \x1b[38;2;124;58;237m_\x1b[39m\x1b[38;2;128;60;238m_\x1b[39m\x1b[38;2;132;63;239m_\x1b[39m\x1b[38;2;136;65;240m)\x1b[39m \x1b[38;2;140;68;241m|\x1b[39m \x1b[38;2;144;70;242m|\x1b[39m\x1b[38;2;148;73;242m_\x1b[39m\x1b[38;2;152;75;243m|\x1b[39m \x1b[38;2;156;78;244m|\x1b[39m \x1b[38;2;160;80;245m|\x1b[39m\x1b[38;2;164;83;246m_\x1b[39m\x1b[38;2;168;85;247m)\x1b[39m \x1b[38;2;172;84;246m|\x1b[39m  \x1b[38;2;176;83;246m_\x1b[39m\x1b[38;2;180;81;245m_\x1b[39m\x1b[38;2;184;80;244m/\x1b[39m \x1b[38;2;188;79;244m|\x1b[39m  \x1b[38;2;193;78;243m|\x1b[39m \x1b[38;2;197;76;242m.\x1b[39m \x1b[38;2;201;75;242m\\\x1b[39m\x1b[38;2;205;74;241m|\x1b[39m \x1b[38;2;209;73;240m|\x1b[39m \x1b[38;2;213;71;240m|\x1b[39m\x1b[38;2;217;70;239m_\x1b[39m\x1b[38;2;219;70;232m|\x1b[39m \x1b[38;2;220;70;225m|\x1b[39m\x1b[38;2;222;71;218m|\x1b[39m  \x1b[38;2;223;71;210m_\x1b[39m\x1b[38;2;225;71;203m_\x1b[39m\x1b[38;2;227;71;196m/\x1b[39m \x1b[38;2;228;71;189m|\x1b[39m \x1b[38;2;230;71;182m|\x1b[39m \x1b[38;2;231;72;175m\\\x1b[39m\x1b[38;2;233;72;167m_\x1b[39m\x1b[38;2;234;72;160m_\x1b[39m \x1b[38;2;236;72;153m\\\x1b[39m\n \x1b[38;2;124;58;237m|\x1b[39m\x1b[38;2;127;60;238m_\x1b[39m\x1b[38;2;129;61;238m_\x1b[39m\x1b[38;2;132;63;239m_\x1b[39m\x1b[38;2;134;64;239m_\x1b[39m\x1b[38;2;137;66;240m/\x1b[39m \x1b[38;2;140;68;241m\\\x1b[39m\x1b[38;2;142;69;241m_\x1b[39m\x1b[38;2;145;71;242m_\x1b[39m\x1b[38;2;147;72;242m,\x1b[39m\x1b[38;2;150;74;243m_\x1b[39m\x1b[38;2;152;75;243m|\x1b[39m \x1b[38;2;155;77;244m.\x1b[39m\x1b[38;2;158;79;245m_\x1b[39m\x1b[38;2;160;80;245m_\x1b[39m\x1b[38;2;163;82;246m/\x1b[39m \x1b[38;2;165;83;246m\\\x1b[39m\x1b[38;2;168;85;247m_\x1b[39m\x1b[38;2;171;84;247m_\x1b[39m\x1b[38;2;174;83;246m_\x1b[39m\x1b[38;2;177;82;246m|\x1b[39m\x1b[38;2;180;81;245m_\x1b[39m\x1b[38;2;182;81;245m|\x1b[39m  \x1b[38;2;185;80;244m|\x1b[39m\x1b[38;2;188;79;244m_\x1b[39m\x1b[38;2;191;78;243m|\x1b[39m\x1b[38;2;194;77;243m\\\x1b[39m\x1b[38;2;197;76;242m_\x1b[39m\x1b[38;2;200;75;242m\\\x1b[39m\x1b[38;2;203;74;241m_\x1b[39m\x1b[38;2;205;74;241m|\x1b[39m\x1b[38;2;208;73;240m\\\x1b[39m\x1b[38;2;211;72;240m_\x1b[39m\x1b[38;2;214;71;239m_\x1b[39m\x1b[38;2;217;70;239m|\x1b[39m\x1b[38;2;218;70;234m\\\x1b[39m\x1b[38;2;219;70;229m_\x1b[39m\x1b[38;2;220;70;224m_\x1b[39m\x1b[38;2;221;70;219m\\\x1b[39m\x1b[38;2;223;71;214m_\x1b[39m\x1b[38;2;224;71;209m_\x1b[39m\x1b[38;2;225;71;204m_\x1b[39m\x1b[38;2;226;71;199m|\x1b[39m\x1b[38;2;227;71;193m_\x1b[39m\x1b[38;2;228;71;188m|\x1b[39m \x1b[38;2;229;71;183m|\x1b[39m\x1b[38;2;230;71;178m_\x1b[39m\x1b[38;2;232;72;173m|\x1b[39m\x1b[38;2;233;72;168m_\x1b[39m\x1b[38;2;234;72;163m_\x1b[39m\x1b[38;2;235;72;158m_\x1b[39m\x1b[38;2;236;72;153m/\x1b[39m\n             \x1b[38;2;124;58;237m|\x1b[39m\x1b[38;2;168;85;247m_\x1b[39m\x1b[38;2;217;70;239m|\x1b[39m                                           "
)

# Rendered by: npx oh-my-logo "SK" \
#   --palette-colors "#7c3aed,#a855f7,#d946ef,#ec4899" --color -d horizontal \
#   --filled --block-font block
BANNER_COMPACT = (
    "\x1b[38;2;124;58;237m \x1b[38;2;133;63;239m█\x1b[38;2;142;69;241m█\x1b[38;2;150;74;243m█\x1b[38;2;159;80;245m█\x1b[38;2;168;85;247m█\x1b[38;2;176;83;246m█\x1b[38;2;184;80;244m█\x1b[38;2;193;78;243m╗\x1b[38;2;201;75;242m \x1b[38;2;209;73;240m█\x1b[38;2;217;70;239m█\x1b[38;2;220;70;225m╗\x1b[38;2;223;71;210m \x1b[38;2;227;71;196m \x1b[38;2;230;71;182m█\x1b[38;2;233;72;167m█\x1b[38;2;236;72;153m╗\x1b[39m\n\x1b[38;2;124;58;237m \x1b[38;2;133;63;239m█\x1b[38;2;142;69;241m█\x1b[38;2;150;74;243m╔\x1b[38;2;159;80;245m═\x1b[38;2;168;85;247m═\x1b[38;2;176;83;246m═\x1b[38;2;184;80;244m═\x1b[38;2;193;78;243m╝\x1b[38;2;201;75;242m \x1b[38;2;209;73;240m█\x1b[38;2;217;70;239m█\x1b[38;2;220;70;225m║\x1b[38;2;223;71;210m \x1b[38;2;227;71;196m█\x1b[38;2;230;71;182m█\x1b[38;2;233;72;167m╔\x1b[38;2;236;72;153m╝\x1b[39m\n\x1b[38;2;124;58;237m \x1b[38;2;133;63;239m█\x1b[38;2;142;69;241m█\x1b[38;2;150;74;243m█\x1b[38;2;159;80;245m█\x1b[38;2;168;85;247m█\x1b[38;2;176;83;246m█\x1b[38;2;184;80;244m█\x1b[38;2;193;78;243m╗\x1b[38;2;201;75;242m \x1b[38;2;209;73;240m█\x1b[38;2;217;70;239m█\x1b[38;2;220;70;225m█\x1b[38;2;223;71;210m█\x1b[38;2;227;71;196m█\x1b[38;2;230;71;182m╔\x1b[38;2;233;72;167m╝\x1b[38;2;236;72;153m \x1b[39m\n\x1b[38;2;124;58;237m \x1b[38;2;133;63;239m╚\x1b[38;2;142;69;241m═\x1b[38;2;150;74;243m═\x1b[38;2;159;80;245m═\x1b[38;2;168;85;247m═\x1b[38;2;176;83;246m█\x1b[38;2;184;80;244m█\x1b[38;2;193;78;243m║\x1b[38;2;201;75;242m \x1b[38;2;209;73;240m█\x1b[38;2;217;70;239m█\x1b[38;2;220;70;225m╔\x1b[38;2;223;71;210m═\x1b[38;2;227;71;196m█\x1b[38;2;230;71;182m█\x1b[38;2;233;72;167m╗\x1b[38;2;236;72;153m \x1b[39m\n\x1b[38;2;124;58;237m \x1b[38;2;133;63;239m█\x1b[38;2;142;69;241m█\x1b[38;2;150;74;243m█\x1b[38;2;159;80;245m█\x1b[38;2;168;85;247m█\x1b[38;2;176;83;246m█\x1b[38;2;184;80;244m█\x1b[38;2;193;78;243m║\x1b[38;2;201;75;242m \x1b[38;2;209;73;240m█\x1b[38;2;217;70;239m█\x1b[38;2;220;70;225m║\x1b[38;2;223;71;210m \x1b[38;2;227;71;196m \x1b[38;2;230;71;182m█\x1b[38;2;233;72;167m█\x1b[38;2;236;72;153m╗\x1b[39m\n\x1b[38;2;124;58;237m \x1b[38;2;133;63;239m╚\x1b[38;2;142;69;241m═\x1b[38;2;150;74;243m═\x1b[38;2;159;80;245m═\x1b[38;2;168;85;247m═\x1b[38;2;176;83;246m═\x1b[38;2;184;80;244m═\x1b[38;2;193;78;243m╝\x1b[38;2;201;75;242m \x1b[38;2;209;73;240m╚\x1b[38;2;217;70;239m═\x1b[38;2;220;70;225m╝\x1b[38;2;223;71;210m \x1b[38;2;227;71;196m \x1b[38;2;230;71;182m╚\x1b[38;2;233;72;167m═\x1b[38;2;236;72;153m╝\x1b[39m"
)

# Minimum terminal width before falling back to the compact mark.
_FULL_MIN_WIDTH = 60

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

_TAGLINE = "Maximizing Efficient Operations per Watt \u2014 Metal kernels for Apple Silicon"


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _supports_color(stream, force_color) -> bool:
    if force_color is not None:
        return force_color
    # Respect the de-facto NO_COLOR / FORCE_COLOR conventions.
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def _term_width(width, stream) -> int:
    if width is not None:
        return width
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def render_banner(force_color: bool | None = None, width: int | None = None,
                  stream=None) -> str:
    """Return the banner string (compact or full) honoring width + color.

    Colors are stripped when the destination is not a TTY (or NO_COLOR is set),
    so piped / redirected output stays clean.
    """
    stream = stream if stream is not None else sys.stdout
    cols = _term_width(width, stream)
    art = BANNER if cols >= _FULL_MIN_WIDTH else BANNER_COMPACT
    if not _supports_color(stream, force_color):
        art = _strip_ansi(art)
    return art


def print_banner(stream=None, force_color: bool | None = None,
                 width: int | None = None) -> None:
    """Print the SK banner to ``stream`` (default stdout)."""
    stream = stream if stream is not None else sys.stdout
    stream.write(render_banner(force_color=force_color, width=width, stream=stream))
    stream.write("\n")
    try:
        stream.flush()
    except Exception:
        pass


if __name__ == "__main__":
    print_banner()
