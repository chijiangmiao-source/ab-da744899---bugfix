"""Tests for the exact solver objectives."""
import itertools
import random

import pytest

from app.solver import (
    ID_MAX,
    accepted_count,
    accepts,
    build_candidates,
    solve,
)


def brute_opt(allowed, forbidden, limit):
    pats, covs, costs = build_candidates(allowed, forbidden)
    full = (1 << len(allowed)) - 1
    best = None
    for r in range(min(len(pats), limit) + 1):
        for combo in itertools.combinations(range(len(pats)), r):
            cb = 0
            for i in combo:
                cb |= covs[i]
            if cb != full:
                continue
            key = (
                r,
                sum(costs[i] for i in combo),
                tuple(sorted(pats[i] for i in combo)),
            )
            if best is None or key < best:
                best = key
        if best:
            break
    return None if best is None else list(best[2])


def assert_sound(allowed, forbidden, limit, result):
    assert result is not None
    assert len(result) <= limit
    assert result == sorted(result)
    for mask, code in result:
        assert 0 <= mask < ID_MAX
        assert 0 <= code < ID_MAX
        assert code & ~mask == 0  # code clears un-compared bits
        for y in forbidden:
            assert not accepts(mask, code, y)
    for x in allowed:
        assert any(accepts(mask, code, x) for mask, code in result)


@pytest.mark.parametrize(
    "allowed,forbidden,limit",
    [
        ([0, 1], [], 8),
        ([0, 2], [], 8),
        ([0, 3], [], 8),
        ([0, 3], [1], 1),
        ([0, 3], [1], 2),
        ([0, 500, 1000, 2047], [], 8),
        ([0, 2, 4, 6], [1, 3, 5, 7], 8),
        (list(range(2, 22)), [0, 1], 8),
        (list(range(0, 40, 2)), list(range(1, 40, 2)), 8),
    ],
)
def test_fixed_cases(allowed, forbidden, limit):
    result = solve(allowed, forbidden, limit)
    expected = brute_opt(allowed, forbidden, limit)
    assert result == expected
    if result is not None:
        assert_sound(allowed, forbidden, limit, result)


def test_exhaustion_is_complete():
    # Allowed 0 (00) and 3 (11) differ in two bits: any one filter
    # accepting both must ignore both low bits, which also accepts the
    # forbidden id 1 (01).  Limit 1 is infeasible; two exact masks work.
    allowed, forbidden = [0, 3], [1]
    assert solve(allowed, forbidden, 1) is None
    result = solve(allowed, forbidden, 2)
    assert_sound(allowed, forbidden, 2, result)
    assert result == [(2047, 0), (2047, 3)]


def test_production_batch_prefers_fewest_filters_over_exposure():
    # Production batch from the audit station: 15 allowed telemetry ids,
    # the contiguous forbidden block 438..485, filter limit 8.
    # A 5-filter cover with total exposure 704 exists, but the ruling
    # objectives minimise the filter count first: the global optimum is
    # the canonical 4-filter sequence with total exposure 1088.
    allowed = [143, 148, 341, 393, 563, 594, 689, 730, 742, 830, 979,
               1045, 1221, 1522, 1610]
    forbidden = list(range(438, 486))
    result = solve(allowed, forbidden, 8)
    assert result == [(224, 64), (608, 0), (1536, 512), (1736, 1216)]
    assert_sound(allowed, forbidden, 8, result)
    assert sum(accepted_count(mask) for mask, _ in result) == 1088
    # every allowed id is hit at least once, every forbidden id 438..485
    # is rejected by each of the four filters
    for x in allowed:
        assert any(accepts(m, c, x) for m, c in result)
    for y in forbidden:
        assert all(not accepts(m, c, y) for m, c in result)


def test_no_forbidden_has_single_wildcard_filter():
    result = solve([3, 17, 900, 2047], [], 8)
    assert result == [(0, 0)]  # one filter, minimum cost 2048... see below
    # A single filter covering all 2048 ids is (mask=0, code=0); it is the
    # unique 1-filter solution and therefore optimal on every objective.


def test_random_equivalence_with_brute_force():
    rng = random.Random(20260920)
    for _ in range(80):
        na = rng.randint(2, 8)
        nf = rng.randint(0, 12)
        ids = rng.sample(range(ID_MAX), na + nf)
        allowed = sorted(ids[:na])
        forbidden = sorted(ids[na:])
        limit = rng.randint(1, 8)
        result = solve(allowed, forbidden, limit)
        assert result == brute_opt(allowed, forbidden, limit)
        if result is not None:
            assert_sound(allowed, forbidden, limit, result)


def test_full_size_inputs_are_timely():
    rng = random.Random(42)
    allowed = rng.sample(range(ID_MAX), 20)
    forbidden = rng.sample(
        [x for x in range(ID_MAX) if x not in allowed], 128
    )
    import time

    t0 = time.time()
    result = solve(allowed, forbidden, 8)
    assert time.time() - t0 < 10
    if result is not None:
        assert_sound(allowed, forbidden, 8, result)


def test_accepted_count_matches_enumeration():
    for mask in (0, 1, 0x7FF, 0x555, 0x2AA):
        assert accepted_count(mask) == sum(
            1 for x in range(ID_MAX) if accepts(mask, 0, x)
        )
