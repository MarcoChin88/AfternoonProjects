from __future__ import annotations

import concurrent.futures
import csv
import string
from collections import Counter
from functools import cache

from common import cache_to_disk


def read_file(file_name: str):
    with open(file_name, 'r') as file:
        for line in file:
            yield line.strip()


@cache_to_disk
def get_words(word_length: int = 5) -> set[str]:
    file_gen = read_file(file_name="word_bank.txt")
    return {w.upper() for w in file_gen if len(w) == word_length}


@cache_to_disk
def get_word_frequencies(word_length: int = 5) -> dict[str, float]:
    with open("unigram_freq.csv") as csv_file:
        csv_rows: list[dict] = list(csv.DictReader(csv_file))

        word_counts: dict[str, int] = {
            word.upper(): int(row["count"])
            for row in csv_rows
            if len(word := row["word"]) == word_length
        }

        total_count: int = sum(word_counts.values())

        all_words = get_words()
        for w in all_words:
            if w not in word_counts:
                word_counts[w] = 1

        return {
            word: word_1_in_N
            for word_1_in_N, word in
            sorted(
                (total_count / cnt, word)
                for word, cnt in word_counts.items()
            )
        }


@cache_to_disk(overwrite=True)
def get_nth_letter_counts():
    word_frequencies = get_word_frequencies()

    counts = list(map(Counter, zip(*word_frequencies.keys())))

    return counts


letter_counts = get_nth_letter_counts()


@cache
def get_letter_info(idx: int, letter: str) -> float:
    return letter_counts[idx][letter]


def get_word_information_amount(_word: str) -> float:
    return sum(get_letter_info(_i, l) for _i, l in enumerate(_word))


def get_guess() -> str:
    while len(user_input := input("Guess: ").strip().upper()) != 5:
        print("Guess must be 5 letters long.")

    return user_input


def get_guess_evaluation() -> str:
    while (
            len(user_input := input("Guess evaluation: ").strip().upper()) != 5
            or not set(user_input).issubset(set("CMW"))
    ):
        print(f"Guess must be 5 letters from {set("CMW")}.")

    return user_input


WORD_FREQUENCIES: dict[str, float] = get_word_frequencies()


class Wordle:
    def __init__(self):
        self.possible_letters: list[set[str]] = [set(string.ascii_uppercase) for _ in range(5)]
        self.possible_solutions: list[str] = list(WORD_FREQUENCIES.keys())

    def set_correct(self, idx: int, letter: str):
        self.possible_letters[idx] = {letter[0].upper()}

    def set_incorrect(self, idx: int, letter: str):
        self.possible_letters[idx].discard(letter[0].upper())

    def print_state(self):
        suggestions = self.get_suggestions()
        print("Suggestions:")
        print('\n'.join(suggestions))
        state = [
            list(possible_letters)[0] if len(possible_letters) == 1 else "_"
            for i, possible_letters in enumerate(self.possible_letters)
        ]
        print(' '.join(state))

        for p in self.possible_letters:
            print(', '.join(sorted(p)))

    def process_guess(self, letter_to_result_dict: list[tuple[[str, str]]]):
        contained = set()
        for i, (letter, result) in enumerate(letter_to_result_dict):
            match result:
                case "C":
                    self.possible_letters[i] = {letter}
                    contained.add(letter)
                case "M":
                    self.possible_letters[i].discard(letter)
                    contained.add(letter)
                case "W":
                    for j in range(5):
                        self.possible_letters[j].discard(letter[0].upper())

        self.possible_solutions = [
            word
            for word in self.possible_solutions
            if all((word[i] in p for i, p in enumerate(self.possible_letters))) and all((s in word for s in contained))
        ]

    def get_suggestions(self) -> list[str]:
        s = sorted(
            self.possible_solutions,
            key=lambda _w: WORD_FREQUENCIES[_w],
        )
        return s[:10]

    def play(self):
        for turn in range(1, 7):
            print(f" Turn {turn} ".center(50, "="))
            self.print_state()

            guess = get_guess()
            guess_result = get_guess_evaluation()

            letter_to_result_dict = zip(guess, guess_result)
            self.process_guess(letter_to_result_dict)

            print()

            print()

            print("".center(50, "="))


def get_ranks():
    ranks = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        all_words = list(WORD_FREQUENCIES.keys())
        for word in all_words:
            future = executor.submit(get_word_information_amount, word)
            futures[future] = word
        for future in concurrent.futures.as_completed(futures):
            word = futures[future]
            ranks[word] = future.result()

    ranks = dict(sorted(ranks.items(), key=lambda r: r[1], reverse=True))

    print(list(ranks.items())[:20])

    return ranks


if __name__ == "__main__":
    w = Wordle()
    w.play()
