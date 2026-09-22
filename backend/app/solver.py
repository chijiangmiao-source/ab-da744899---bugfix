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
search is exact: candidate filters are generated exhaustively, then an
iterative-deepening branch-and-bound set-cover search proves the minimum
filter count (objective 3), after which a second exhaustive pass over that
many filters optimises objectives 4 and 5.  Both passes branch only over
filters covering a pivot uncovered identifier, use a set-packing lower
bound on the remaining filter count, memoise infeasible remainder states,
and collapse sibling branches whose filters leave the same remainder.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

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

    # Candidate indices covering each allowed bit.
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

    # Per-bit conflict masks: bits j that share at least one candidate with
    # bit i can be covered in the same filter as i; others cannot.
    shares = [0] * n_allowed
    for cov in cover_bits:
        b = cov
        while b:
            bit = b & -b
            j = bit.bit_length() - 1
            shares[j] |= cov
            b ^= bit

    min_cost = min(costs)

    def packing_lower_bound(need: int) -> int:
        """Greedy set-packing bound on filters still required.

        Pick an uncovered bit, count one filter, then discard every bit that
        some single candidate could cover together with it.  The remaining
        bits are pairwise un-coverable with the chosen ones, so each needs a
        distinct additional filter.
        """
        count = 0
        while need:
            bit = need & -need
            j = bit.bit_length() - 1
            count += 1
            need &= ~shares[j]
        return count

    def choose_pivot(need: int) -> int:
        """Uncovered bit with the fewest candidates that actually cover it."""
        best_bit = -1
        best_count = 0
        b = need
        while b:
            bit = b & -b
            j = bit.bit_length() - 1
            count = sum(1 for i in owners[j] if cover_bits[i] & need)
            if best_bit == -1 or count < best_count:
                best_bit, best_count = j, count
                if count <= 1:
                    break
            b ^= bit
        return best_bit

    # ------------------------------------------------------------------
    # Pass 1: minimum feasible filter count via iterative deepening.
    # ------------------------------------------------------------------
    def feasible_within(k: int) -> bool:
        dead: Set[Tuple[int, int]] = set()

        def dfs(need: int, slots: int) -> bool:
            if need == 0:
                return True
            if slots == 0:
                return False
            key = (need, slots)
            if key in dead:
                return False
            if packing_lower_bound(need) > slots:
                dead.add(key)
                return False

            pivot = choose_pivot(need)
            # Filters leaving the same remainder are interchangeable for
            # feasibility: explore each distinct remainder once.
            seen_remainders: Set[int] = set()
            for i in owners[pivot]:
                rest = need & ~cover_bits[i]
                if rest == need or rest in seen_remainders:
                    continue
                seen_remainders.add(rest)
                if dfs(rest, slots - 1):
                    return True
            dead.add(key)
            return False

        return dfs(full, k)

    min_filters: Optional[int] = None
    for k in range(1, limit + 1):
        if feasible_within(k):
            min_filters = k
            break
    if min_filters is None:
        return None

    # ------------------------------------------------------------------
    # Pass 2: among covers using exactly min_filters, minimise total
    # exposure, then the sorted (mask, code) sequence.
    # ------------------------------------------------------------------
    best_cost: Optional[int] = None
    best_seq: Optional[Tuple[Pattern, ...]] = None
    dead: Set[Tuple[int, int]] = set()
    # Best (cost-so-far, chosen sequence) reaching each (need, slots).  A
    # state reached with a smaller pair always completes no worse: equal
    # remainder/slots give identical continuations, and merging the same
    # future multiset preserves sorted-sequence lexicographic order.
    reached: Dict[Tuple[int, int], Tuple[int, Tuple[Pattern, ...]]] = {}

    def dfs_optimise(
        need: int,
        slots: int,
        cost_so_far: int,
        seq: Tuple[Pattern, ...],
    ) -> None:
        nonlocal best_cost, best_seq
        if need == 0:
            if best_cost is None or (cost_so_far, seq) < (best_cost, best_seq):
                best_cost = cost_so_far
                best_seq = seq
            return
        if slots == 0:
            return
        key = (need, slots)
        if key in dead:
            return
        prior = reached.get(key)
        if prior is not None and prior <= (cost_so_far, seq):
            return
        reached[key] = (cost_so_far, seq)
        bound = packing_lower_bound(need)
        if bound > slots:
            dead.add(key)
            return
        # Every additional filter costs at least min_cost.  Only prune on a
        # strictly larger cost: ties must survive for lexicographic tie-break.
        if best_cost is not None and cost_so_far + bound * min_cost > best_cost:
            return

        pivot = choose_pivot(need)
        # Same remainder -> identical sub-problem; keep only the branch with
        # the smallest (cost, pattern), since the continuation is the same.
        best_by_remainder: Dict[int, Tuple[int, Pattern]] = {}
        for i in owners[pivot]:
            if not (cover_bits[i] & need):
                continue
            rest = need & ~cover_bits[i]
            if rest == need:
                continue
            value = (costs[i], patterns[i])
            old = best_by_remainder.get(rest)
            if old is None or value < old:
                best_by_remainder[rest] = value

        branches = sorted(
            ((value[0], value[1], rest) for rest, value in best_by_remainder.items()),
        )
        for cand_cost, pattern, rest in branches:
            dfs_optimise(
                rest,
                slots - 1,
                cost_so_far + cand_cost,
                tuple(sorted(seq + (pattern,))),
            )

    dfs_optimise(full, min_filters, 0, ())
    assert best_seq is not None  # pass 1 proved feasibility
    return list(best_seq)


def accepts(mask: int, code: int, x: int) -> bool:
    """Identifier ``x`` is accepted by filter ``(mask, code)``."""
    return (x & mask) == (code & mask)


def accepted_count(mask: int) -> int:
    """Number of 11-bit identifiers accepted by a filter with this mask."""
    return ID_MAX >> mask.bit_count()


def covered_allowed(mask: int, code: int, allowed: Sequence[int]) -> List[int]:
    """Allowed identifiers accepted by the filter, ascending."""
    return sorted(x for x in allowed if accepts(mask, code, x))
