import re

 
def find_first_repeated_ngram(
    tokens: list[str],
    min_n: int = 3,
    max_n: int = 8,
):
    n_tokens = len(tokens)
    for n in range(max_n, min_n - 1, -1):
        seen: dict[tuple[str, ...], int] = {}
        for i in range(n_tokens - n + 1):
            gram = tuple(tokens[i : i + n])
            if gram in seen:
                return (seen[gram], n)   # first occurrence, gram length
            seen[gram] = i
    return None
 
 
def remove_trailing_ngrams(
    tokens: list[str],
    min_n: int = 1,
    max_n: int = 8,
) -> list[str]:
    if not tokens:
        return tokens
 
    changed = True
    while changed:
        changed = False
        n = len(tokens)
        for k in range(min_n, min(max_n + 1, n)):  # k = suffix length to test
            tail = tuple(tokens[n - k:])            # last k tokens
            body = tokens[: n - k]                  # everything before the tail
 
            # Collect all k-length prefixes of any run in body
            body_kprefixes: set[tuple[str, ...]] = set()
            for i in range(len(body) - k + 1):
                body_kprefixes.add(tuple(body[i : i + k]))
 
            if tail in body_kprefixes:
                tokens = body       # drop the trailing fragment
                changed = True
                break               # restart from k=min_n with shorter sequence
 
    return tokens
 
 
def remove_repeated_ngrams(
    tokenizer,
    text: str,
    min_n: int = 3,
    max_n: int = 8,
    max_passes: int = 10,
) -> str:
    tokens = tokenizer.tokenize(text)
 
    for _ in range(max_passes):
        before = tokens[:]
        # Step B: strip trailing partial n-gram fragment
        tokens = remove_trailing_ngrams(tokens, min_n=min_n, max_n=max_n)
 
        if tokens == before:
            break  # nothing changed → stable
 
    cleaned = tokenizer.convert_tokens_to_string(tokens).strip()
    
    return cleaned