import random
from enum import Enum
from functools import wraps
import hashlib
import os
import pickle
from textwrap import dedent

import pandas as pd


class TextEffects(Enum):
    RESET = "0"

    BOLD = "1"
    DIM = "2"
    ITALIC = "3"
    UNDERLINE = "4"
    BLINK = "5"
    REVERSE = "7"
    HIDDEN = "8"
    STRIKETHROUGH = "9"

    def __str__(self):
        return f"\033[{self.value}m"

    def apply(self, text) -> str:
        return f"{self}{str(text)}{TextEffects.RESET}"


class TextColors(Enum):
    RESET = "0"

    BLACK = "30"
    RED = "31"
    GREEN = "32"
    YELLOW = "33"
    BLUE = "34"
    MAGENTA = "35"
    CYAN = "36"
    WHITE = "37"

    BRIGHT_BLACK = "90"
    BRIGHT_RED = "91"
    BRIGHT_GREEN = "92"
    BRIGHT_YELLOW = "93"
    BRIGHT_BLUE = "94"
    BRIGHT_MAGENTA = "95"
    BRIGHT_CYAN = "96"
    BRIGHT_WHITE = "97"

    def __str__(self):
        return f"\033[{self.value}m"

    def apply(self, text) -> str:
        return f"{self}{text}{TextColors.RESET}"


class Evaluations(Enum):
    C = TextColors.BRIGHT_GREEN  # Correct
    M = TextColors.BRIGHT_YELLOW  # Miss
    W = TextColors.BRIGHT_RED  # Wrong


def get_header(
        text: str,
        width: int = 50,
        fill_char: str = "~"
) -> str:
    return f" {text.strip()} ".center(width, fill_char)


def clean_str(text: str) -> str:
    return dedent(text).strip()


def cache_to_disk(
        _func=None,
        cache_dir=".cache",
        overwrite: bool = False
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache directory for function if not exists
            func_cache_dir: str = os.path.join(cache_dir, func.__name__)
            os.makedirs(func_cache_dir, exist_ok=True)

            # Generate file name
            arg_hash = hashlib.md5(pickle.dumps((args, kwargs))).hexdigest()
            cache_file = os.path.join(func_cache_dir, f"{arg_hash}.pkl")

            # Check cache
            if os.path.exists(cache_file) and not overwrite:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            # Run function
            result = func(*args, **kwargs)

            # Cache result to disk
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            return result

        return wrapper

    if callable(_func):
        return decorator(_func)

    return decorator


def read_file(file_name: str):
    with open(file_name, 'r') as file:
        for line in file:
            yield line.strip()


@cache_to_disk(overwrite=True)
def get_upper_words_of_n_length(
        n: int = 5
) -> dict[str, int]:
    df = pd.read_csv(
        "unigram_freq.csv",
        dtype={"word": str, "count": int}
    )
    word_counts: dict[str, int] = dict(zip(df["word"], df["count"]))

    filtered_word_counts: dict[str, int] = {
        word.strip().upper(): word_count
        for word, word_count in word_counts.items()
        if (
                str(word).isalpha()
                and len(str(word)) == n
        )
    }

    return filtered_word_counts


def get_guess_evaluation(
        guess: str,
        solution: str
) -> str:
    guess_evaluation = ""
    for g, s in zip(guess, solution):
        if g == s:
            guess_evaluation += Evaluations.C.name
        elif g in solution:
            guess_evaluation += Evaluations.M.name
        elif g not in solution:
            guess_evaluation += Evaluations.W.name

    return guess_evaluation


def process_guess(
        guess: str,
        guess_evaluation: str,
        possible_solutions: list[str] = None
):
    if possible_solutions is None:
        possible_solutions = list(get_upper_words_of_n_length().keys())

    seen_this_turn = set()
    for i, (letter, evaluation) in enumerate(zip(guess, guess_evaluation)):
        match evaluation:
            case Evaluations.C.name:
                possible_solutions = [
                    word
                    for word in possible_solutions
                    if word[i] == letter
                ]

            case Evaluations.M.name:
                possible_solutions = [
                    word
                    for word in possible_solutions
                    if letter in word and word[i] != letter
                ]

            case Evaluations.W.name:
                if letter not in seen_this_turn:
                    possible_solutions = [
                        word
                        for word in possible_solutions
                        if letter not in word
                    ]
        seen_this_turn.add(letter)
    return possible_solutions


def get_random_word() -> str:
    word_counts = get_upper_words_of_n_length()

    return random.choice(word_counts.keys())
