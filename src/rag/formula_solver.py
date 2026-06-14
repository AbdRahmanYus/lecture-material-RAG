# from sympy import symbols, factor, solve

# s = symbols('s')
# expr = s**2 + 3*s + 2      # this is just algebra, not a number
# roots = solve(expr, s)      # finds s = -1 and s = -2
# print(roots)                # [-2, -1]


"""
formula_solver.py

Symbolic Control Engineering computations using SymPy.
Handles transfer function analysis, stability checking,
Routh-Hurwitz arrays, and state space conversion.
"""

import logging
from sympy import (
    symbols, Poly, solve, simplify,
    Matrix, eye, zeros, factor
)
from sympy import re as sym_re   # real part — renamed to avoid clash with Python built-in

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Shared symbol ──────────────────────────────────────────────────────────────
s = symbols('s')   # the Laplace variable used across all functions


# ── Function 1: Find poles ─────────────────────────────────────────────────────
def find_poles(denominator_expr) -> list:
    """
    Finds the poles of a transfer function by solving for the
    roots of the denominator polynomial.

    Args:
        denominator_expr: A SymPy expression in terms of s.
                          Example: s**2 + 3*s + 2

    Returns:
        A list of SymPy values representing the poles.

    Example:
        >>> from sympy import symbols
        >>> s = symbols('s')
        >>> find_poles(s**2 + 3*s + 2)
        [-2, -1]
    """
    if denominator_expr is None:
        raise ValueError("Denominator expression cannot be None.")

    logger.info(f"Finding poles of: {denominator_expr}")
    poles = solve(denominator_expr, s)
    logger.info(f"Poles found: {poles}")
    return poles


# ── Function 2: Check stability ────────────────────────────────────────────────
def check_stability(poles: list) -> dict:
    """
    Checks whether a system is stable based on its poles.

    A continuous-time LTI system is stable if and only if
    ALL poles have strictly negative real parts.

    Args:
        poles: List of poles returned by find_poles().

    Returns:
        A dict with keys:
            'stable'  : bool — True if all poles have Re < 0
            'poles'   : list of (pole, real_part) tuples
            'verdict' : human-readable string

    Example:
        >>> check_stability([-1, -2])
        {'stable': True, 'verdict': 'STABLE — all poles in left half-plane'}
    """
    if not poles:
        raise ValueError("Poles list is empty — cannot assess stability.")

    pole_analysis = []
    all_stable = True

    for pole in poles:
        real_part = sym_re(pole)  # extract the real part of each pole

        # evalf() forces SymPy to compute a numerical value
        # so we can compare it to zero reliably
        real_part_numerical = float(real_part.evalf())

        is_stable = real_part_numerical < 0
        if not is_stable:
            all_stable = False

        pole_analysis.append({
            "pole":      pole,
            "real_part": real_part_numerical,
            "stable":    is_stable,
        })

    if all_stable:
        verdict = "STABLE — all poles are in the left half-plane (Re < 0)"
    else:
        unstable_poles = [p for p in pole_analysis if not p["stable"]]
        verdict = (
            f"UNSTABLE — {len(unstable_poles)} pole(s) are in the "
            f"right half-plane or on the imaginary axis"
        )

    logger.info(verdict)
    return {
        "stable":  all_stable,
        "poles":   pole_analysis,
        "verdict": verdict,
    }


# ── Function 3: Routh-Hurwitz array ───────────────────────────────────────────
def routh_hurwitz(coefficients: list) -> dict:
    """
    Constructs the Routh-Hurwitz array for a characteristic polynomial
    and counts sign changes in the first column to determine stability.

    Args:
        coefficients: List of polynomial coefficients in DESCENDING order.
                      For s³ + 2s² + 3s + 8, pass [1, 2, 3, 8].

    Returns:
        A dict with keys:
            'array'          : the full Routh array as a list of lists
            'first_column'   : values in the first column
            'sign_changes'   : number of sign changes (= RHP poles)
            'stable'         : bool
            'verdict'        : human-readable string

    Example:
        >>> routh_hurwitz([1, 2, 3, 8])
        # Returns array with sign changes indicating instability
    """
    if not coefficients or len(coefficients) < 2:
        raise ValueError("Need at least 2 coefficients to build Routh array.")

    n = len(coefficients)  # order of polynomial + 1

    # ── Build the first two rows from the coefficients ─────────────────────────
    # Row 0 gets the coefficients at even indices: a_n, a_{n-2}, a_{n-4}, ...
    # Row 1 gets the coefficients at odd indices:  a_{n-1}, a_{n-3}, ...
    num_cols = (n + 1) // 2  # number of columns in the array

    # Pad coefficients with a trailing zero if n is even
    padded = coefficients + ([0] if n % 2 == 0 else [])

    row0 = padded[0::2]  # every other element starting at index 0
    row1 = padded[1::2]  # every other element starting at index 1

    # Pad rows to equal length
    while len(row0) < num_cols:
        row0.append(0)
    while len(row1) < num_cols:
        row1.append(0)

    array = [row0, row1]

    # ── Fill remaining rows ────────────────────────────────────────────────────
    for i in range(2, n):
        prev      = array[i - 1]   # row directly above
        prev_prev = array[i - 2]   # row two above

        pivot = prev[0]  # first element of the row above

        new_row = []
        for j in range(num_cols - 1):
            if pivot == 0:
                # Special case: zero in first column
                # Replace with a small epsilon — full handling is an extension
                logger.warning(
                    "Zero pivot encountered in Routh array. "
                    "System may have poles on imaginary axis."
                )
                pivot = 1e-9

            entry = (pivot * prev_prev[j + 1] - prev_prev[0] * prev[j + 1]) / pivot
            new_row.append(entry)

        new_row.append(0)  # pad to correct length
        array.append(new_row)

    # ── Count sign changes in the first column ─────────────────────────────────
    first_column = [row[0] for row in array]
    sign_changes = 0

    for i in range(1, len(first_column)):
        if first_column[i - 1] * first_column[i] < 0:
            sign_changes += 1

    stable = sign_changes == 0

    if stable:
        verdict = "STABLE — no sign changes in first column of Routh array"
    else:
        verdict = (
            f"UNSTABLE — {sign_changes} sign change(s) in first column "
            f"= {sign_changes} pole(s) in the right half-plane"
        )

    logger.info(verdict)

    return {
        "array":        array,
        "first_column": first_column,
        "sign_changes": sign_changes,
        "stable":       stable,
        "verdict":      verdict,
    }


# ── Function 4: Transfer function to state space ───────────────────────────────
def to_state_space(numerator_coeffs: list, denominator_coeffs: list) -> dict:
    """
    Converts a transfer function to state space form using
    the controllable canonical form.

    For a transfer function b(s)/a(s) where a(s) is monic (leading
    coefficient = 1), returns matrices A, B, C, D.

    Args:
        numerator_coeffs:   Coefficients of numerator in descending order.
                            For 2s + 1, pass [2, 1].
        denominator_coeffs: Coefficients of denominator in descending order.
                            For s² + 3s + 2, pass [1, 3, 2].

    Returns:
        A dict with keys 'A', 'B', 'C', 'D' as SymPy Matrix objects.

    Example:
        >>> to_state_space([1], [1, 3, 2])
        # Returns controllable canonical form matrices for 1/(s²+3s+2)
    """
    if not numerator_coeffs or not denominator_coeffs:
        raise ValueError("Numerator and denominator coefficients cannot be empty.")

    n = len(denominator_coeffs) - 1  # system order

    if n < 1:
        raise ValueError("System order must be at least 1.")

    # Normalise so leading coefficient of denominator is 1
    lead = denominator_coeffs[0]
    den  = [c / lead for c in denominator_coeffs]
    num  = [c / lead for c in numerator_coeffs]

    # Pad numerator to length n+1 if needed
    while len(num) < n + 1:
        num = [0] + num

    # ── A matrix: companion matrix (controllable canonical form) ───────────────
    A = zeros(n, n)

    # Top-right block: identity shifted one column right
    for i in range(n - 1):
        A[i, i + 1] = 1

    # Last row: negative of denominator coefficients (excluding leading 1)
    for j in range(n):
        A[n - 1, j] = -den[n - j]

    # ── B matrix ───────────────────────────────────────────────────────────────
    B = zeros(n, 1)
    B[n - 1, 0] = 1

    # ── C matrix ───────────────────────────────────────────────────────────────
    C = zeros(1, n)
    for j in range(n):
        C[0, j] = num[n - j] - num[0] * den[n - j]

    # ── D matrix ───────────────────────────────────────────────────────────────
    D = Matrix([[num[0]]])

    logger.info(f"State space conversion complete — system order: {n}")

    return {"A": A, "B": B, "C": C, "D": D}


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*60)
    print("TEST 1 — Poles and stability")
    print("="*60)
    # Transfer function: 1 / (s^2 + 3s + 2)
    # Denominator: s^2 + 3s + 2
    # denom = input("Enter the denominator of the transfer function (in terms of s): ")
    denom = s**2 + 3*s + 2
    poles = find_poles(denom)
    result = check_stability(poles)
    print(f"Denominator : {denom}")
    print(f"Poles       : {poles}")
    print(f"Verdict     : {result['verdict']}")

    print("\n" + "="*60)
    print("TEST 2 — Routh-Hurwitz array")
    print("="*60)
    # From your lecture note: s^3 + 2s^2 + 3s + 8
    # Expected result: UNSTABLE (sign change in first column)
    routh = routh_hurwitz([1, 2, 3, 8])
    print("Characteristic equation: s³ + 2s² + 3s + 8")
    print("Routh Array:")
    for i, row in enumerate(routh["array"]):
        print(f"  Row {i}: {[round(float(x), 4) for x in row]}")
    print(f"First column : {[round(float(x), 4) for x in routh['first_column']]}")
    print(f"Sign changes : {routh['sign_changes']}")
    print(f"Verdict      : {routh['verdict']}")

    print("\n" + "="*60)
    print("TEST 3 — Transfer function to state space")
    print("="*60)
    # G(s) = 1 / (s^2 + 3s + 2)
    ss = to_state_space([1], [1, 3, 2])
    print("Transfer function: 1 / (s² + 3s + 2)")
    print(f"A =\n{ss['A']}")
    print(f"B =\n{ss['B']}")
    print(f"C =\n{ss['C']}")
    print(f"D =\n{ss['D']}")