"""Exact CAN acceptance-filter solver.

A filter is an 11-bit ``(code, mask)`` pair.  Identifier ``x`` is accepted
iff ``(x & mask) == (code & mask)``; the bits of ``code`` where ``mask`` is
zero must themselves be zero ("code must clear un-compared bits").

Every valid filter therefore corresponds to a sub-cube of the 11-bit
identifier space -- a length-11 pattern of ``0`` / ``1`` / ``x`` -- with an
11-bit ``mask`` and a canonical ``code`` (zero outside the mask).  We
generate the relevant sub-cubes by bucketing allowed/forbidden identifiers
per mask (all 2**11 masks).

The solver searches for at most ``limit`` filters that, in order of priority

  1. accept every allowed identifier,
  2. accept no forbidden identifier,
  3. minimise the number of filters,
  4. among those, minimise the sum of accepted-identifier counts
     (2 ** (# zero bits of mask)) of the chosen filters,
  5. among those, minimise the sorted ``(mask, code)`` sequence in
     lexicographic order.

It returns ``None`` when no feasible cover exists within the limit.  The
search is an exact memoised set-cover DP: branching only over filters that
cover a pivot (least uncovered) identifier keeps the search exhaustive
without enumerating permutations, a set-packing lower bound prunes
branches that cannot become feasible within the remaining budget, and
memoisation over (uncovered set, remaining budget) makes the dynamic
program exact -- every feasible combination is reachable, so the returned
cover is globally optimal on objectives 3-5, not merely a good beam
search incumbent.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

ID_BITS = 11
ID_MAX = 1 << ID_BITS  # 2048
FULL_ID_MASK = ID_MAX - 1

# A canonical filter: (mask, code) with code & ~mask == 0.
Pattern = Tuple[int, int]


def build_candidates(
    allowed: Sequence[int], forbidden: Sequence[int]
) -> Tuple[List[Pattern], List[int], List[int]]:
    """Return ``(patterns, cover_bits, costs)`` for every useful filter.

    A candidate accepts at least one allowed identifier and no forbidden
    identifier.  For each distinct subset of covered allowed identifiers we
    keep only the filter with minimum accepted-identifier cost, breaking
    ties toward the lexicographically smaller ``(mask, code)``: every other
    filter with the same coverage is worse on objective 4 (or 5).
    """
    allowed = sorted(allowed)
    index_of = {x: i for i, x in enumerate(allowed)}
    n_allowed = len(allowed)

    # covered-bitmask -> (minimum cost, smallest pattern at that cost)
    best: Dict[int, Tuple[int, Pattern]] = {}

    for mask in range(ID_MAX):
        forbidden_buckets = {y & mask for y in forbidden}
        buckets: Dict[int, int] = {}
        for x in allowed:
            b = x & mask
            buckets[b] = buckets.get(b, 0) | (1 << index_of[x])
        cost = ID_MAX >> mask.bit_count()  # identifiers the filter accepts
        for code, cover in buckets.items():
            if code in forbidden_buckets:
                continue  # the filter would also accept a forbidden id
            pat = (mask, code)
            old = best.get(cover)
            if old is None or (cost, pat) < old:
                best[cover] = (cost, pat)

    items = sorted(
        ((pat, cover, cost) for cover, (cost, pat) in best.items()),
        key=lambda t: (t[2], t[0]),
    )
    patterns = [t[0] for t in items]
    cover_bits = [t[1] for t in items]
    costs = [t[2] for t in items]
    return patterns, cover_bits, costs


def solve(
    allowed: Sequence[int],
    forbidden: Sequence[int],
    limit: int,
) -> Optional[List[Pattern]]:
    """Return the optimal filter list in ascending ``(mask, code)`` order.

    ``None`` means the feasible space within ``limit`` was exhausted with no
    cover.
    """
    allowed = sorted(set(allowed))
    n_allowed = len(allowed)
    patterns, cover_bits, costs = build_candidates(allowed, forbidden)
    full = (1 << n_allowed) - 1

    # Candidate indices covering each allowed bit (cost/pattern ordered).
    owners: List[List[int]] = [[] for _ in range(n_allowed)]
    for i, cov in enumerate(cover_bits):
        b = cov
        while b:
            bit = b & -b
            owners[bit.bit_length() - 1].append(i)
            b ^= bit
    if any(not o for o in owners):
        # An allowed id is only co-accepted with a forbidden id: infeasible.
        return None

    # compat[i]: bitmask of allowed ids that share at least one candidate
    # filter with id i (i included).  Ids that are pairwise incompatible
    # each need a dedicated filter, so any such set is an admissible
    # set-packing lower bound on the number of filters still required.
    compat = [0] * n_allowed
    for cov in cover_bits:
        b = cov
        while b:
            bit = b & -b
            compat[bit.bit_length() - 1] |= cov
            b ^= bit

    max_cover = max(cov.bit_count() for cov in cover_bits)

    packing_cache: Dict[int, int] = {}

    def packing_lower_bound(need: int) -> int:
        """Greedy maximal pairwise-incompatible subset of ``need``."""
        cached = packing_cache.get(need)
        if cached is not None:
            return cached
        count = 0
        avail = need
        while avail:
            bit = avail & -avail
            count += 1
            avail &= ~compat[bit.bit_length() - 1]
        packing_cache[need] = count
        return count

    # Exact memoised DP.  best_cover(need, budget) returns the optimal
    # (total cost, sorted pattern tuple) among covers of ``need`` using at
    # most ``budget`` filters, or None when no such cover exists.  It is
    # queried only for budgets below the first feasible filter count, so
    # every cover it weighs uses exactly ``budget`` filters; combining the
    # pivot filter with the optimal sub-cover (costs add, sorted tuples
    # merge monotonically at equal length) therefore preserves the global
    # objective order.
    memo: Dict[Tuple[int, int], Optional[Tuple[int, Tuple[Pattern, ...]]]] = {}

    def best_cover(need: int, budget: int) -> Optional[Tuple[int, Tuple[Pattern, ...]]]:
        if need == 0:
            return (0, ())
        if budget == 0:
            return None
        key = (need, budget)
        if key in memo:
            return memo[key]
        result: Optional[Tuple[int, Tuple[Pattern, ...]]] = None
        if (
            need.bit_count() <= budget * max_cover
            and packing_lower_bound(need) <= budget
        ):
            pivot = (need & -need).bit_length() - 1
            for i in owners[pivot]:
                sub = best_cover(need & ~cover_bits[i], budget - 1)
                if sub is None:
                    continue
                value = (
                    costs[i] + sub[0],
                    tuple(sorted(sub[1] + (patterns[i],))),
                )
                if result is None or value < result:
                    result = value
        memo[key] = result
        return result

    # Objective 3 (fewest filters) first: the smallest feasible budget is
    # found before any cost/lexicographic comparison takes place.
    for budget in range(1, limit + 1):
        completed = best_cover(full, budget)
        if completed is not None:
            return list(completed[1])
    return None


def accepts(mask: int, code: int, x: int) -> bool:
    """Identifier ``x`` is accepted by filter ``(mask, code)``."""
    return (x & mask) == (code & mask)


def accepted_count(mask: int) -> int:
    """Number of 11-bit identifiers accepted by a filter with this mask."""
    return ID_MAX >> mask.bit_count()


def covered_allowed(mask: int, code: int, allowed: Sequence[int]) -> List[int]:
    """Allowed identifiers accepted by the filter, ascending."""
    return sorted(x for x in allowed if accepts(mask, code, x))
