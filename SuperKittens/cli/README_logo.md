# SK CLI logo

The SuperKittens CLI banner (`SuperKittens/cli/banner.py`) is a purple -> pink
gradient ASCII wordmark, rendered once with [oh-my-logo](https://github.com/shinshin86/oh-my-logo)
and embedded as ANSI strings so the CLI prints it offline (no Node/npx needed at
runtime).

## Palette

A custom violet -> magenta -> pink gradient that mirrors SuperKitty's purple
bodysuit (see `meow.png` in the repo root) flowing into the README's pink accent:

```
#7c3aed  (violet)
#a855f7  (purple)
#d946ef  (magenta)
#ec4899  (pink)
```

## Regenerate

The full wordmark (Standard figlet font, horizontal gradient):

```bash
npx oh-my-logo "SuperKittens" \
  --palette-colors "#7c3aed,#a855f7,#d946ef,#ec4899" \
  --color -d horizontal
```

The compact `SK` mark for narrow terminals (filled block font):

```bash
npx oh-my-logo "SK" \
  --palette-colors "#7c3aed,#a855f7,#d946ef,#ec4899" \
  --color -d horizontal --filled --block-font block
```

`--color` forces the ANSI truecolor escapes even when stdout is a pipe, so the
captured output keeps its gradient. Pipe each command into the
`BANNER` / `BANNER_COMPACT` literals in `banner.py` (escape `ESC` as `\x1b`,
trim the blank padding lines that `--filled` adds).

## Runtime behavior

`print_banner()` (in `banner.py`):

- picks the full wordmark when the terminal is >= 60 cols, else the compact `SK`;
- strips color when the destination is not a TTY, or when `NO_COLOR` is set
  (and honors `FORCE_COLOR` / the `force_color=` arg the other way);
- appends the MEOW tagline.

```bash
python -c "from SuperKittens.cli.banner import print_banner; print_banner()"
```

## Preview

Full wordmark (`BANNER`, ~59 cols) -- colored in a truecolor terminal:

```
  ____                        _  ___ _   _
 / ___| _   _ _ __   ___ _ __| |/ (_) |_| |_ ___ _ __  ___
 \___ \| | | | '_ \ / _ \ '__| ' /| | __| __/ _ \ '_ \/ __|
  ___) | |_| | |_) |  __/ |  | . \| | |_| ||  __/ | | \__ \
 |____/ \__,_| .__/ \___|_|  |_|\_\_|\__|\__\___|_| |_|___/
             |_|
Maximizing Efficient Operations per Watt — Metal kernels for Apple Silicon
```

Compact mark (`BANNER_COMPACT`, ~18 cols) for narrow terminals:

```
 ███████╗ ██╗  ██╗
 ██╔════╝ ██║ ██╔╝
 ███████╗ █████╔╝
 ╚════██║ ██╔═██╗
 ███████║ ██║  ██╗
 ╚══════╝ ╚═╝  ╚═╝
MEOW
```
