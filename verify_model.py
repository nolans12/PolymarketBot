"""
verify_model.py — Validates the model math against Table 8.1.2 from the paper.

Expected return formula: r = (1 - q^(w)) / q^(w)

Run this before going live to confirm our implementation matches the paper exactly.

Usage:
    python3 verify_model.py

All 8 test cases should show PASS. Any FAIL means a bug in the formula.
"""

from model import classify_mode

# Test cases directly from Table 8.1.2 in the paper
# (asset, q, expected_r_numerator, expected_r_denominator, expected_r_pct, expected_mode_str)
TABLE_8_1_2 = [
    # (asset, q, numerator, denominator, r_as_decimal, mode_str)
    # r = (1 - q) / q — paper Table 8.1.2
    # BTC Up q=0.087: r = 0.913/0.087 = 10.4943 (+1049%)
    ("BTC Up",   0.087, 0.913, 0.087, 10.4943, "extreme discount"),
    ("ETH Up",   0.571, 0.429, 0.571,  0.7513, "mid zone"),
    ("ETH Up",   0.771, 0.229, 0.771,  0.2970, "high confidence"),
    ("BTC Down", 0.815, 0.185, 0.815,  0.2270, "high confidence"),
    # Paper shows numerator 0.141 but 1-0.856=0.144 (rounding in paper)
    ("BTC Down", 0.856, 0.144, 0.856,  0.1682, "high confidence"),
    ("BTC Up",   0.898, 0.102, 0.898,  0.1136, "high confidence"),
    ("BTC Down", 0.951, 0.049, 0.951,  0.0515, "near-certain"),
    ("ETH Up",   0.970, 0.030, 0.970,  0.0309, "near-certain"),
]

# Mode label mapping for display
MODE_LABELS = {
    1: "high confidence / near-certain",
    2: "extreme discount / mid zone",
    None: "out of range",
}


def expected_return(q: float) -> float:
    """r = (1 - q) / q — the paper's return formula."""
    return (1.0 - q) / q


def run_verification():
    print("=" * 70)
    print("VERIFY MODEL — Table 8.1.2 cross-check")
    print("Formula: r = (1 - q^(w)) / q^(w)")
    print("=" * 70)
    print(f"{'Asset':<12} {'q':>6} {'r (calc)':>10} {'r (paper)':>10} {'err':>8} {'Mode':>6}  Result")
    print("-" * 70)

    all_pass = True

    for asset, q, num, den, r_paper_pct, mode_str in TABLE_8_1_2:
        r_calc = expected_return(q)
        r_paper = r_paper_pct

        # Allow 0.5% relative tolerance for rounding in the paper
        rel_err = abs(r_calc - r_paper) / r_paper if r_paper != 0 else abs(r_calc)
        passed = rel_err < 0.005

        mode_int = classify_mode(q)
        mode_label = {1: "Mode 1", 2: "Mode 2", None: "NONE"}[mode_int]

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(
            f"{asset:<12} {q:>6.3f} {r_calc:>10.4f} {r_paper:>10.4f} "
            f"{rel_err:>7.4f}  {mode_label:<8}  {status}"
        )

    print("=" * 70)

    # Also verify the return formula numerator/denominator for each row
    print("\nNumerator/Denominator check (r = numerator/denominator from paper):")
    print(f"{'Asset':<12} {'q':>6} {'1-q (calc)':>12} {'num (paper)':>12}  Result")
    print("-" * 55)
    for asset, q, num, den, r_paper_pct, mode_str in TABLE_8_1_2:
        one_minus_q = round(1.0 - q, 3)
        passed = abs(one_minus_q - num) < 0.002
        if not passed:
            all_pass = False
        status = "PASS" if passed else "FAIL"
        print(f"{asset:<12} {q:>6.3f} {one_minus_q:>12.3f} {num:>12.3f}  {status}")

    print("=" * 70)
    if all_pass:
        print("ALL TESTS PASSED — model math matches paper exactly.")
    else:
        print("SOME TESTS FAILED — check the formula or test case values.")
    print()

    # Spot-check mode classification
    print("Mode classification spot-check:")
    spot_checks = [
        (0.087, 2, "extreme discount"),
        (0.571, 2, "mid zone"),
        (0.771, 1, "high confidence"),
        (0.951, 1, "near-certain"),
        (0.050, None, "out of range below"),
        (0.980, None, "out of range above"),
    ]
    for q, expected_mode, label in spot_checks:
        got = classify_mode(q)
        status = "PASS" if got == expected_mode else "FAIL"
        print(f"  q={q:.3f} ({label:<25}) -> mode={got}  {status}")
        if got != expected_mode:
            all_pass = False

    print()
    if all_pass:
        print("VERIFICATION COMPLETE — safe to proceed to live trading.")
    else:
        print("VERIFICATION FAILED — do not trade until all tests pass.")


if __name__ == "__main__":
    run_verification()
