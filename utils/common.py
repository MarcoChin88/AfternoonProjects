import inspect
import shutil
import time
from functools import wraps
from inspect import BoundArguments
from pathlib import Path
from pprint import pprint, pformat
from textwrap import dedent

from utils.text import style_text, TextColors, Styles, frame_highlight_text


def create_or_replace_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)


def clean_str(s: str) -> str:
    return dedent(s).strip()


def trunc_str(s: str, l: int = 10) -> str:
    if len(s) > l:
        return s[:l] + '...'
    return s


def get_func_call_str(func, *args, **kwargs):
    bound_args: BoundArguments = inspect.signature(func).bind(*args, **kwargs)
    bound_args.apply_defaults()

    args_str = ", ".join([f"{k}={pformat(v)}" for k, v in bound_args.arguments.items()])

    return f"{func.__name__}({args_str})"


def as_header(s: str, length: int = 100, lvl: int = 1) -> str:
    lvl_fills = ['=', '-', '.']

    max_lvl = len(lvl_fills)
    if not 1 <= lvl <= max_lvl:
        raise Exception(f"Header {lvl=} must in [{1}, {max_lvl}]")

    return f" {s} ".center(length, lvl_fills[lvl - 1])


def logit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log call
        print()
        func_call_str = get_func_call_str(func, *args, **kwargs)
        print(f"Calling: {func_call_str}")

        # Run it
        result = func(*args, **kwargs)

        print(f"Returned: {pformat(result)}")
        print()

        # Return
        return result

    return wrapper


def timeit(n=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            total_time = 0
            for _ in range(n):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                total_time += (end_time - start_time)
            avg_time = total_time / n

            func_call_str = get_func_call_str(func, *args, **kwargs)

            if n == 1:
                print(f"{func_call_str}: {avg_time:0.6f}s")
            else:
                print(f"{func_call_str}: Avg({n=}) = {avg_time:0.6f}s")

            return result

        return wrapper

    return decorator


def get_denominated_amount_str(amount: float,
                               denominations: dict[float, str],
                               fmt: str = ',.2f'):
    label_value_dict = {label: value for value, label in sorted(denominations.items(), reverse=True)}
    denom_amounts = {value: 0 for value, label in label_value_dict.items()}

    remainder = amount
    for label, value in label_value_dict.items():
        denom_amount: int = int(remainder // value)
        remainder = remainder - denom_amount * value

        denom_amounts[label] = denom_amount

    parts = []
    for label, amount in denom_amounts.items():
        if amount == 0:
            continue
        amount_str = style_text(text=format(amount, fmt), text_color=TextColors.MAGENTA, styles=[Styles.BOLD])
        label_str = style_text(text=label, text_color=TextColors.BLUE, styles=[Styles.BOLD])

        s = f"{amount_str} {label_str}"
        parts.append(s)

    res = ', '.join(parts[:1]) if parts else '0 anything'
    return res

def cache_to_disk(_func=None, cache_dir=".cache", overwrite: bool = False):
    import hashlib
    import os
    import pickle
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache directory for function if not exists
            func_cache_dir = os.path.join(cache_dir, func.__name__)
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

            # Return result
            return result

        return wrapper

    # This lets you call it without ()
    return decorator(_func) if _func and callable(_func) else decorator
