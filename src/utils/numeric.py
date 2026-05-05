def intr(x):
    """Float to int with rounding (instead of int(x) which does a floor)"""
    return int(round(x))


def get_square_row_col(n):
    """Returns (n_rows, n_cols) for a grid close to square with n_rows * n_cols >= n."""
    n_rows = int(n**0.5)
    n_cols = n // n_rows
    if n_rows * n_cols < n:
        n_rows += 1
    return n_rows, n_cols


def suggest_ids(ids):
    """[1,3,4] -> [0,2,5]"""
    if not ids:
        return [0]
    return sorted(set(range(max(ids) + 2)) - set(ids))
