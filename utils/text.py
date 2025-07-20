import re
from enum import Enum

_033 = '\033'
RESET = f"{_033}[0m"


class Styles(Enum):
    RESET = 0
    BOLD = 1
    DIM = 2
    ITALIC = 3
    UNDERLINE = 4
    BLINK_SLOW = 5
    BLINK_RAPID = 6
    REVERSE = 7
    CONCEAL = 8
    CROSSED_OUT = 9
    DOUBLE_UNDERLINE = 21
    NORMAL_INTENSITY = 22
    NOT_ITALIC = 23
    NOT_UNDERLINED = 24
    NOT_BLINKING = 25
    REVEAL = 28
    NOT_CROSSED_OUT = 29
    FRAMED = 51
    ENCIRCLED = 52
    OVERLINED = 53
    NOT_FRAMED_OR_ENCIRCLED = 54
    NOT_OVERLINED = 55


class TextColors(Enum):
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    BRIGHT_BLACK = 90
    BRIGHT_RED = 91
    BRIGHT_GREEN = 92
    BRIGHT_YELLOW = 93
    BRIGHT_BLUE = 94
    BRIGHT_MAGENTA = 95
    BRIGHT_CYAN = 96
    BRIGHT_WHITE = 97

    def color_text(self, s: str) -> str:
        return f"{_033}[{self.value}m{s}{RESET}"


class BgColors(Enum):
    BLACK = 40
    RED = 41
    GREEN = 42
    YELLOW = 43
    BLUE = 44
    MAGENTA = 45
    CYAN = 46
    WHITE = 47
    BRIGHT_BLACK = 100
    BRIGHT_RED = 101
    BRIGHT_GREEN = 102
    BRIGHT_YELLOW = 103
    BRIGHT_BLUE = 104
    BRIGHT_MAGENTA = 105
    BRIGHT_CYAN = 106
    BRIGHT_WHITE = 107


def style_text(
        text: str,
        text_color: TextColors = None,
        bg_color: BgColors = None,
        styles: list[Styles] = None
) -> str:
    text = ''.join(text.rsplit(RESET, 1))
    styles = list(set(styles)) if styles else None
    codes = []

    if styles:
        codes.extend([_.value for _ in styles])

    if text_color:
        codes.append(text_color.value)

    if bg_color:
        codes.append(bg_color.value)

    # Build the ANSI escape sequence
    codes_str = ';'.join(map(str, codes))
    ansi_sequence = f"{_033}[{codes_str}m" if codes else ''

    return f"{ansi_sequence}{text}{RESET}"


def frame_highlight_text(text: str,
                         text_color: TextColors = TextColors.BLUE,
                         bg_color: BgColors = None,
                         styles: list[Styles] = None
                         ) -> str:
    text = f" {text} "

    if styles is None:
        styles = [Styles.FRAMED]
    else:
        styles.append(Styles.FRAMED)

    return style_text(text=text,
                      text_color=text_color,
                      bg_color=bg_color,
                      styles=styles)


def superscript(s: str) -> str:
    superscript_map = {
        '0': '\u2070', '1': '\u00B9', '2': '\u00B2', '3': '\u00B3',
        '4': '\u2074', '5': '\u2075', '6': '\u2076', '7': '\u2077',
        '8': '\u2078', '9': '\u2079', '+': '\u207A', '-': '\u207B',
        '=': '\u207C', '(': '\u207D', ')': '\u207E', 'a': '\u1D43',
        'b': '\u1D47', 'c': '\u1D9C', 'd': '\u1D48', 'e': '\u1D49',
        'f': '\u1DA0', 'g': '\u1D4D', 'h': '\u02B0', 'i': '\u2071',
        'j': '\u02B2', 'k': '\u1D4F', 'l': '\u02E1', 'm': '\u1D50',
        'n': '\u207F', 'o': '\u1D52', 'p': '\u1D56', 'r': '\u02B3',
        's': '\u02E2', 't': '\u1D57', 'u': '\u1D58', 'v': '\u1D5B',
        'w': '\u02B7', 'x': '\u02E3', 'y': '\u02B8', 'z': '\u1DBB'
    }
    return ''.join(superscript_map[char] if char in superscript_map else char for char in s)


def subscript(s: str) -> str:
    subscript_map = {
        '0': '\u2080', '1': '\u2081', '2': '\u2082', '3': '\u2083',
        '4': '\u2084', '5': '\u2085', '6': '\u2086', '7': '\u2087',
        '8': '\u2088', '9': '\u2089', '+': '\u208A', '-': '\u208B',
        '=': '\u208C', '(': '\u208D', ')': '\u208E', 'a': '\u2090',
        'e': '\u2091', 'h': '\u2095', 'i': '\u1D62', 'j': '\u2C7C',
        'k': '\u2096', 'l': '\u2097', 'm': '\u2098', 'n': '\u2099',
        'o': '\u2092', 'p': '\u209A', 'r': '\u1D63', 's': '\u209B',
        't': '\u209C', 'u': '\u1D64', 'v': '\u1D65', 'x': '\u2093'
    }
    return ''.join(subscript_map[char] if char in subscript_map else char for char in s)


def real_len(s: str) -> int:
    ansi_escape = re.compile(r'\033\[[0-9;]*m')
    return len(ansi_escape.sub('', s))
