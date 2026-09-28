"""Tests for the headless DAG identification layer (plan 6.16, 7.0).

Run standalone:
    PYTHONPATH="engines/python;sidecar" .venv/Scripts/python.exe tests/python/test_dag.py

The named cases below are the textbook graphs -- the classic collider, M-bias,
a mediator chain, the front door, the instrument, the napkin -- and each one
asserts the *exact* adjustment sets, not just that something came back. On top
of that, `test_dsep_matches_path_enumeration` and
`test_minimal_sets_match_brute_force` check the fast algorithms against a slow,
independent implementation that enumerates every path and every subset, so a
regression in the reachability search cannot hide behind a plausible answer.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "engines" / "python"))
sys.path.insert(0, str(_ROOT / "sidecar"))

from capy_sidecar import dag  # noqa: E402
from capy_sidecar.dag import CAPTION, SpecError  # noqa: E402


# ---------------------------------------------------------------------------
# Building graphs the way the editor serialises them
# ---------------------------------------------------------------------------


def G(nodes, edges, **extra):
    """`nodes` are ids (or ('id', latent) pairs); `edges` are (from, to[, kind])."""
    ns = []
    for n in nodes:
        if isinstance(n, tuple):
            nid, latent = n
        else:
            nid, latent = n, False
        ns.append({"id": nid, "variable": None if latent else nid, "label": nid,
                   "latent": latent, "role": "latent" if latent else "other"})
    es = [{"from": e[0], "to": e[1], "kind": (e[2] if len(e) > 2 else "directed")} for e in edges]
    out = {"schema": "capy.dag", "version": 1, "id": "test_dag", "nodes": ns, "edges": es}
    out.update(extra)
    return out


def sets_of(result):
    return [sorted(s) for s in result["backdoor"]["minimal_sets"]]


def kinds(result):
    return {b["node"]: b["kind"] for b in result["bad_controls"]}


def is_sentence(text):
    return isinstance(text, str) and len(text) > 20 and text.strip()[-1] in ".!?"


# ---------------------------------------------------------------------------
# d-separation: the three structures every textbook opens with
# ---------------------------------------------------------------------------


def test_chain_fork_collider():
    chain = G(["A", "B", "C"], [("A", "B"), ("B", "C")])
    assert not dag.d_separated(chain, "A", "C"), "a chain leaves A and C dependent"
    assert dag.d_separated(chain, "A", "C", ["B"]), "conditioning on the middle of a chain blocks it"

    fork = G(["A", "B", "C"], [("B", "A"), ("B", "C")])
    assert not dag.d_separated(fork, "A", "C"), "a common cause makes A and C dependent"
    assert dag.d_separated(fork, "A", "C", ["B"]), "conditioning on the common cause blocks the fork"

    collider = G(["A", "B", "C"], [("A", "B"), ("C", "B")])
    assert dag.d_separated(collider, "A", "C"), "a collider is closed while you leave it alone"
    assert not dag.d_separated(collider, "A", "C", ["B"]), "conditioning on a collider opens it"

    # ... and conditioning on a *descendant* of the collider opens it too.
    with_child = G(["A", "B", "C", "D"], [("A", "B"), ("C", "B"), ("B", "D")])
    assert dag.d_separated(with_child, "A", "C")
    assert not dag.d_separated(with_child, "A", "C", ["D"])
    print("chain / fork / collider OK")


def test_classic_collider_graph():
    """X -> C <- Y. C is a descendant of the treatment and a collider: never adjust."""
    g = G(["X", "Y", "C"], [("X", "Y"), ("X", "C"), ("Y", "C")])
    r = dag.identify(g, "X", "Y")
    assert r["valid"]
    assert r["backdoor"]["identified"]
    assert sets_of(r) == [[]], f"nothing needs adjusting here, got {sets_of(r)}"
    assert kinds(r) == {"C": "descendant_of_both"}, kinds(r)
    assert r["descendants_of_treatment"] == ["Y", "C"]
    assert "C" in [c["node"] for c in r["colliders"]]
    assert is_sentence(r["bad_controls"][0]["why"])
    # and the user is told what conditioning on it would do
    assert not dag.d_separated(g, "X", "Y", ["C"]) is None
    print("classic collider OK")


def test_m_bias():
    """U1 -> X, U1 -> M, U2 -> M, U2 -> Y, X -> Y.

    The famous trap: M looks like a covariate, is measured, is not caused by the
    treatment -- and adjusting for it *creates* the bias it looks like it fixes.
    """
    g = G(["X", "Y", "M", ("U1", True), ("U2", True)],
          [("X", "Y"), ("U1", "X"), ("U1", "M"), ("U2", "M"), ("U2", "Y")])
    r = dag.identify(g, "X", "Y")
    assert r["valid"]
    assert r["backdoor"]["identified"]
    assert sets_of(r) == [[]], f"the empty set is the only minimal set in M-bias, got {sets_of(r)}"
    assert kinds(r) == {"M": "collider"}, kinds(r)
    assert "opens a path" in r["bad_controls"][0]["why"]
    assert dag.d_separated(g, "X", "Y", []) is False  # X -> Y is a real arrow
    # The collider is what makes {M} wrong, and the module says so structurally too.
    assert [c["on_backdoor_path"] for c in r["colliders"] if c["node"] == "M"] == [True]
    print("M-bias OK")


def test_mediator_chain():
    """C -> X -> M -> Y, C -> Y. Adjust for C; never for M."""
    g = G(["X", "M", "Y", "C"], [("X", "M"), ("M", "Y"), ("C", "X"), ("C", "Y")])
    r = dag.identify(g, "X", "Y")
    assert sets_of(r) == [["C"]], f"exactly one minimal set here, got {sets_of(r)}"
    assert r["mediators"] == ["M"]
    assert kinds(r) == {"M": "mediator"}, kinds(r)
    assert "total effect" in r["bad_controls"][0]["why"]
    assert r["backdoor"]["minimal_sets_variables"] == [["C"]]
    print("mediator chain OK")


def test_two_minimal_sets_are_both_listed():
    """C -> X, C -> M, M -> Y, X -> Y: either C or M closes the back door.

    This is the case the plan cares about -- show *sets*, not "the" set.
    """
    g = G(["X", "Y", "C", "M"], [("X", "Y"), ("C", "X"), ("C", "M"), ("M", "Y")])
    r = dag.identify(g, "X", "Y")
    assert sets_of(r) == [["C"], ["M"]], sets_of(r)
    assert not r["backdoor"]["truncated"]
    assert r["backdoor"]["canonical_set"] == ["C", "M"]
    assert r["bad_controls"] == [], "neither variable is a bad control here"
    print("two minimal sets OK")


def test_front_door():
    """X -> M -> Y with unmeasured confounding X <-> Y.

    Back-door fails; the front door still identifies the effect.
    """
    g = G(["X", "M", "Y"], [("X", "M"), ("M", "Y"), ("X", "Y", "bidirected")])
    r = dag.identify(g, "X", "Y")
    assert r["backdoor"]["identified"] is False
    assert sets_of(r) == []
    assert "not measured" in r["backdoor"]["reason"] or "unmeasured" in r["backdoor"]["reason"]
    assert r["frontdoor"]["identified"] is True
    assert [sorted(s) for s in r["frontdoor"]["sets"]] == [["M"]], r["frontdoor"]["sets"]
    assert r["mediators"] == ["M"]
    assert is_sentence(r["frontdoor"]["reason"])
    # a direct arrow shuts the front door
    g2 = G(["X", "M", "Y"], [("X", "M"), ("M", "Y"), ("X", "Y"), ("X", "Y", "bidirected")])
    r2 = dag.identify(g2, "X", "Y")
    assert r2["frontdoor"]["identified"] is False
    assert "straight to" in r2["frontdoor"]["reason"]
    print("front door OK")


def test_instrument():
    """Z -> X -> Y with X <-> Y: Z is an instrument. Add Z -> Y and it is not."""
    g = G(["Z", "X", "Y"], [("Z", "X"), ("X", "Y"), ("X", "Y", "bidirected")])
    r = dag.identify(g, "X", "Y")
    assert [i["node"] for i in r["instruments"]] == ["Z"]
    assert r["instruments"][0]["conditional_on"] == []
    assert "LATE" in r["instruments"][0]["estimand"]
    assert r["backdoor"]["identified"] is False

    leaky = G(["Z", "X", "Y"], [("Z", "X"), ("X", "Y"), ("Z", "Y"), ("X", "Y", "bidirected")])
    assert dag.identify(leaky, "X", "Y")["instruments"] == [], "an arrow Z -> Y disqualifies Z"

    # a variable the treatment causes is never an instrument
    downstream = G(["X", "Y", "W"], [("X", "Y"), ("X", "W"), ("X", "Y", "bidirected")])
    assert dag.identify(downstream, "X", "Y")["instruments"] == []
    print("instrument OK")


def test_conditional_instrument():
    """A -> Z, A -> Y, Z -> X: Z only works as an instrument once A is held fixed."""
    g = G(["A", "Z", "X", "Y"], [("A", "Z"), ("A", "Y"), ("Z", "X"), ("X", "Y"),
                                 ("X", "Y", "bidirected")])
    r = dag.identify(g, "X", "Y")
    found = {i["node"]: i["conditional_on"] for i in r["instruments"]}
    assert found == {"Z": ["A"]}, found
    assert "'A'" in r["instruments"][0]["why"]
    print("conditional instrument OK")


def test_napkin_is_honest():
    """The napkin graph: no adjustment set exists, and the module says so plainly."""
    g = G(["W", "Z", "X", "Y", ("U1", True), ("U2", True)],
          [("W", "Z"), ("Z", "X"), ("X", "Y"), ("U1", "W"), ("U1", "X"), ("U2", "W"), ("U2", "Y")])
    r = dag.identify(g, "X", "Y")
    assert r["backdoor"]["identified"] is False
    assert sets_of(r) == []
    assert r["frontdoor"]["identified"] is False
    # It is honest about the route that *does* exist here.
    assert {i["node"]: i["conditional_on"] for i in r["instruments"]} == {"Z": ["W"]}
    assert "different question" in r["summary"]
    print("napkin OK")


def test_bidirected_edge_is_unmeasured_confounding():
    g = G(["X", "Y"], [("X", "Y"), ("X", "Y", "bidirected")])
    r = dag.identify(g, "X", "Y")
    assert r["backdoor"]["identified"] is False
    assert len(r["latents"]) == 1 and r["latents"][0]["synthetic"] is True
    assert r["latents"][0]["between"] == ["X", "Y"]
    # a declared latent node behaves the same way
    g2 = G(["X", "Y", ("U", True)], [("X", "Y"), ("U", "X"), ("U", "Y")])
    r2 = dag.identify(g2, "X", "Y")
    assert r2["backdoor"]["identified"] is False
    assert "U" not in r2["backdoor"]["candidates"], "you cannot adjust for what you did not measure"
    print("bidirected = unmeasured OK")


def test_bad_controls_by_kind():
    """Descendant, mediator, descendant-of-both, and a collider each get named."""
    g = G(["X", "Y", "C", "M", "P", "S", "K"],
          [("X", "Y"), ("C", "X"), ("C", "Y"),      # honest confounder
           ("X", "M"), ("M", "Y"),                  # mediator
           ("X", "P"),                              # pure post-treatment
           ("X", "S"), ("Y", "S"),                  # selection node
           ("K", "X"), ("K", "Y")])                 # second confounder
    r = dag.identify(g, "X", "Y")
    assert kinds(r) == {"M": "mediator", "P": "descendant_of_treatment", "S": "descendant_of_both"}
    assert sorted(r["descendants_of_treatment"]) == ["M", "P", "S", "Y"]
    assert sets_of(r) == [["C", "K"]]
    # the standalone helper the UI calls needs no outcome to flag the descendants
    b = dag.bad_controls(g, "X")
    assert sorted(b["descendants_of_treatment"]) == ["M", "P", "S", "Y"]
    assert b["caption"] == CAPTION
    assert {row["node"] for row in b["bad_controls"]} == {"M", "P", "S", "Y"}
    # with an outcome it also sees the colliders
    b2 = dag.bad_controls(g, "X", "Y")
    assert b2["outcome"] == "Y"
    assert {row["node"] for row in b2["bad_controls"]} == {"M", "P", "S"}
    print("bad controls OK")


def test_cycles_are_explained_never_a_crash():
    g = G(["A", "B", "C"], [("A", "B"), ("B", "C"), ("C", "A")])
    r = dag.identify(g, "A", "C")
    assert r["valid"] is False
    assert len(r["cycles"]) == 1
    assert r["cycles"][0]["nodes"] == ["A", "B", "C"]
    assert "circle" in r["cycles"][0]["explain"] and is_sentence(r["cycles"][0]["explain"])
    assert r["backdoor"]["identified"] is False
    assert r["backdoor"]["minimal_sets"] == []
    assert r["instruments"] == [] and r["bad_controls"] == []
    assert is_sentence(r["summary"])
    json.dumps(r)  # a cyclic graph still returns a serialisable object

    v = dag.validate(g)
    assert v["ok"] is False and v["cycles"][0]["labels"] == ["A", "B", "C"]

    two = G(["A", "B"], [("A", "B"), ("B", "A")])
    assert dag.validate(two)["ok"] is False
    self_loop = G(["A", "B"], [("A", "A"), ("A", "B")])
    assert dag.validate(self_loop)["ok"] is False
    assert any("itself" in e for e in dag.validate(self_loop)["errors"])
    print("cycles OK")


def test_validate_explains_broken_pictures():
    v = dag.validate({"schema": "capy.dag", "version": 1, "id": "b",
                      "nodes": [{"id": "A"}, {"id": "A"}, {"id": ""}],
                      "edges": [{"from": "A", "to": "ghost"}, {"from": "A"}]})
    assert v["ok"] is False
    assert all(is_sentence(e) for e in v["errors"]), v["errors"]
    assert any("unique" in e for e in v["errors"])
    assert any("no id" in e for e in v["errors"])
    assert any("ghost" in w for w in v["warnings"])
    assert v["caption"] == CAPTION

    empty = dag.validate({"schema": "capy.dag", "version": 1, "id": "e", "nodes": [], "edges": []})
    assert empty["ok"] is True and empty["warnings"], "an empty canvas is not an error, but say so"
    assert dag.validate(None)["ok"] is False
    print("validate OK")


def test_missing_treatment_is_a_sentence_not_a_traceback():
    g = G(["X", "Y"], [("X", "Y")])
    for bad in ("nope", None):
        try:
            dag.identify(g, bad, "Y")
        except SpecError as exc:
            assert is_sentence(str(exc)), str(exc)
            assert "Traceback" not in str(exc)
        else:
            raise AssertionError(f"identify({bad!r}) should have raised SpecError")
    # roles fill in when the caller passes nothing
    roled = {"schema": "capy.dag", "version": 1, "id": "r",
             "nodes": [{"id": "d", "variable": "d", "role": "treatment"},
                       {"id": "y", "variable": "y", "role": "outcome"}],
             "edges": [{"from": "d", "to": "y"}]}
    assert dag.identify(roled)["treatment"] == "d"
    # column names resolve as well as node ids
    assert dag.identify(roled, "d", "y")["outcome"] == "y"
    try:
        dag.identify(roled, "d", "d")
    except SpecError as exc:
        assert "two things" in str(exc)
    else:
        raise AssertionError("treatment == outcome should raise")
    print("spec errors OK")


def test_selected_set_is_checked():
    g = G(["X", "Y", "C", "M"], [("X", "Y"), ("C", "X"), ("C", "Y"), ("X", "M")])
    good = dag.identify({**g, "selected_adjustment_set": ["C"]}, "X", "Y")["selected"]
    assert good["sufficient"] is True and is_sentence(good["reason"])
    bad = dag.identify({**g, "selected_adjustment_set": ["M"]}, "X", "Y")["selected"]
    assert bad["sufficient"] is False and "after the treatment" in bad["reason"]
    none = dag.identify({**g, "selected_adjustment_set": []}, "X", "Y")["selected"]
    assert none is None
    print("selected set OK")


def test_truncation_is_announced_not_silent():
    """Twelve confounders, no small sufficient set: say what was not searched."""
    n = 12
    nodes = ["X", "Y"] + [f"c{i}" for i in range(n)]
    edges = [("X", "Y")] + [e for i in range(n) for e in ((f"c{i}", "X"), (f"c{i}", "Y"))]
    r = dag.identify(G(nodes, edges), "X", "Y")
    assert r["backdoor"]["identified"] is True, "the full set works, so the answer is yes"
    assert r["backdoor"]["truncated"] is True
    assert r["backdoor"]["minimal_sets"] == []
    assert len(r["backdoor"]["canonical_set"]) == n
    assert is_sentence(r["backdoor"]["truncation_note"])
    assert r["backdoor"]["truncation_note"] in r["notes"]
    assert r["backdoor"]["searched_up_to"] >= 1

    # small graphs are searched exhaustively and say so
    small = dag.identify(G(["X", "Y", "C"], [("X", "Y"), ("C", "X"), ("C", "Y")]), "X", "Y")
    assert small["backdoor"]["truncated"] is False
    assert sets_of(small) == [["C"]]
    print("truncation OK")


def test_caption_on_every_object():
    g = G(["X", "Y", "C"], [("X", "Y"), ("C", "X"), ("C", "Y")])
    roles = {"treatment": "X", "outcome": "Y", "confounders": ["C"]}
    for name, obj in (("identify", dag.identify(g, "X", "Y")),
                      ("validate", dag.validate(g)),
                      ("bad_controls", dag.bad_controls(g, "X", "Y")),
                      ("suggest_from_roles", dag.suggest_from_roles(roles))):
        assert obj["caption"] == CAPTION, f"{name} lost the caption"
    assert dag.identify(G(["A", "B"], [("A", "B"), ("B", "A")]), "A", "B")["caption"] == CAPTION
    print("caption OK")


def test_suggest_from_roles():
    schema_path = _ROOT / "schemas" / "capy.dag.v1.json"
    roles = {"treatment": "d", "outcome": "y", "confounders": ["age", "sex"],
             "mediator": ["m"], "forbidden": ["post"], "unit": "id", "time": "yr"}
    d = dag.suggest_from_roles(roles)
    assert d["schema"] == "capy.dag" and d["version"] == 1
    try:
        import jsonschema
        jsonschema.validate(d, json.loads(schema_path.read_text(encoding="utf-8")))
    except ImportError:  # pragma: no cover
        print("  (jsonschema not installed; skipped the schema check)")

    ids = [n["id"] for n in d["nodes"]]
    assert ids == ["d", "y", "age", "sex", "m", "post"], ids
    assert all(n["x"] is not None and n["y"] is not None for n in d["nodes"]), "give the canvas a layout"
    assert all(e.get("note") for e in d["edges"]), "every drawn arrow says which role drew it"
    arrows = {(e["from"], e["to"]) for e in d["edges"]}
    assert ("d", "y") in arrows and ("age", "d") in arrows and ("age", "y") in arrows
    assert ("d", "m") in arrows and ("m", "y") in arrows and ("d", "post") in arrows
    assert d["selected_adjustment_set"] == ["age", "sex"]

    # the starter graph has to say what it implies
    r = dag.identify(d)
    assert sets_of(r) == [["age", "sex"]], sets_of(r)
    assert r["selected"]["sufficient"] is True
    assert kinds(r) == {"m": "mediator", "post": "descendant_of_treatment"}
    assert any("design fact" in note for note in d["notes"]), d["notes"]
    assert any("dashed" in note for note in d["notes"])

    # naming an instrument means admitting unmeasured confounding
    iv = dag.suggest_from_roles({"treatment": "d", "outcome": "y", "instruments": ["z"]})
    assert any(e["kind"] == "bidirected" for e in iv["edges"])
    assert iv["selected_adjustment_set"] is None
    riv = dag.identify(iv)
    assert riv["backdoor"]["identified"] is False
    assert [i["node"] for i in riv["instruments"]] == ["z"]

    # an empty role set is not a crash
    blank = dag.suggest_from_roles({})
    assert blank["nodes"] == [] and blank["edges"] == [] and blank["notes"]
    assert dag.suggest_from_roles(None)["caption"] == CAPTION
    # a whole spec is tolerated in place of its roles block
    spec = {"design": "observational", "roles": roles}
    assert [n["id"] for n in dag.suggest_from_roles(spec)["nodes"]] == ids
    print("suggest_from_roles OK")


def test_results_are_json_and_deterministic():
    g = G(["X", "Y", "C", "M", "Z"], [("X", "Y"), ("C", "X"), ("C", "Y"), ("X", "M"),
                                      ("Z", "X"), ("X", "Y", "bidirected")])
    first = json.dumps(dag.identify(g, "X", "Y"), sort_keys=True)
    for _ in range(3):
        assert json.dumps(dag.identify(g, "X", "Y"), sort_keys=True) == first
    assert "NaN" not in first and "Infinity" not in first

    # the same picture drawn in a different order gives the same answer
    shuffled = {**g, "nodes": list(reversed(g["nodes"])), "edges": list(reversed(g["edges"]))}
    a, b = dag.identify(g, "X", "Y"), dag.identify(shuffled, "X", "Y")
    assert {tuple(sorted(s)) for s in a["backdoor"]["minimal_sets"]} == \
           {tuple(sorted(s)) for s in b["backdoor"]["minimal_sets"]}
    assert kinds(a) == kinds(b)
    print("json / determinism OK")


# ---------------------------------------------------------------------------
# The fast algorithms against slow, independent ones
# ---------------------------------------------------------------------------


def _expand(d):
    """Bidirected edges become an explicit latent parent, as the module does."""
    parents, children = {}, {}

    def add(n):
        parents.setdefault(n, set())
        children.setdefault(n, set())

    for n in d["nodes"]:
        add(n["id"])
    k = 0
    for e in d["edges"]:
        a, b = e["from"], e["to"]
        add(a)
        add(b)
        if e.get("kind") == "bidirected":
            k += 1
            u = f"__u{k}"
            add(u)
            children[u] |= {a, b}
            parents[a].add(u)
            parents[b].add(u)
        else:
            children[a].add(b)
            parents[b].add(a)
    return parents, children


def _reach(adj, start):
    seen, stack = set(), list(start)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj[n])
    return seen


def _all_paths(parents, children, x, y):
    nbr = {n: children[n] | parents[n] for n in children}
    out, stack = [], [(x, [x], {x})]
    while stack:
        n, path, seen = stack.pop()
        for m in nbr[n]:
            if m in seen:
                continue
            if m == y:
                out.append(path + [m])
            else:
                stack.append((m, path + [m], seen | {m}))
    return out


def slow_dsep(d, x, y, z):
    """d-separation by enumerating every path and blocking it by hand."""
    parents, children = _expand(d)
    z = set(z)
    anc_z = _reach(parents, z)
    for path in _all_paths(parents, children, x, y):
        for i in range(1, len(path) - 1):
            w = path[i]
            collider = (w in children[path[i - 1]]) and (w in children[path[i + 1]])
            if (collider and w not in anc_z) or (not collider and w in z):
                break
        else:
            return False
    return True


def slow_minimal_sets(d, x, y):
    """Every sufficient back-door set, by testing all of them."""
    parents, children = _expand(d)
    latent = {n["id"] for n in d["nodes"] if n.get("latent")}
    de_x = _reach(children, {x})
    pool = [n["id"] for n in d["nodes"]
            if n["id"] not in de_x and n["id"] != y and n["id"] not in latent]
    bd = {**d, "edges": [e for e in d["edges"]
                         if not (e.get("kind", "directed") == "directed" and e["from"] == x)]}
    ok = [frozenset(c) for k in range(len(pool) + 1)
          for c in itertools.combinations(pool, k) if slow_dsep(bd, x, y, c)]
    return ok, [s for s in ok if not any(t < s for t in ok)]


def _random_dag(rng, n=6):
    names = [chr(ord("a") + i) for i in range(n)]
    latents = set(rng.sample(names, k=rng.choice([0, 1])))
    nodes = [{"id": v, "variable": None if v in latents else v, "latent": v in latents}
             for v in names]
    edges = [{"from": names[i], "to": names[j], "kind": "directed"}
             for i in range(n) for j in range(i + 1, n) if rng.random() < 0.35]
    for _ in range(rng.randint(0, 1)):
        a, b = rng.sample(names, 2)
        edges.append({"from": a, "to": b, "kind": "bidirected"})
    return {"schema": "capy.dag", "version": 1, "id": "f", "nodes": nodes, "edges": edges}


def test_dsep_matches_path_enumeration():
    rng = random.Random(11)
    checked = 0
    for _ in range(40):
        d = _random_dag(rng)
        names = [n["id"] for n in d["nodes"]]
        for x, y in itertools.combinations(names, 2):
            rest = [n for n in names if n not in (x, y)]
            for k in (0, 1, 2):
                for z in itertools.combinations(rest, k):
                    checked += 1
                    fast = dag.d_separated(d, x, y, list(z))
                    slow = slow_dsep(d, x, y, z)
                    assert fast == slow, f"d-sep disagreement on {d['edges']} for {x},{y}|{z}"
    assert checked > 5000
    print(f"d-separation matches path enumeration on {checked} checks OK")


def test_minimal_sets_match_brute_force():
    rng = random.Random(23)
    tested = 0
    for _ in range(60):
        d = _random_dag(rng)
        names = [n["id"] for n in d["nodes"]]
        x, y = names[0], names[-1]
        r = dag.identify(d, x, y)
        assert r["valid"], "random DAGs are acyclic by construction"
        ok, minimal = slow_minimal_sets(d, x, y)
        assert r["backdoor"]["identified"] == bool(ok), (
            f"identified={r['backdoor']['identified']} but brute force found {len(ok)} sets "
            f"on {d['edges']}")
        if not r["backdoor"]["truncated"]:
            got = sorted(tuple(sorted(s)) for s in r["backdoor"]["minimal_sets"])
            want = sorted(tuple(sorted(s)) for s in minimal)
            assert got == want, f"{got} != {want} on {d['edges']}"
            tested += 1
    assert tested > 30
    print(f"minimal back-door sets match brute force on {tested} graphs OK")


def test_frontdoor_and_instruments_are_sane():
    """Every front-door set and instrument the module reports must survive its own criterion."""
    rng = random.Random(5)
    found_fd = found_iv = 0
    for _ in range(60):
        d = _random_dag(rng)
        names = [n["id"] for n in d["nodes"]]
        x, y = names[0], names[-1]
        r = dag.identify(d, x, y)
        parents, children = _expand(d)
        de_x = _reach(children, {x}) - {x}
        anc_y = _reach(parents, {y})
        for s in r["frontdoor"]["sets"]:
            found_fd += 1
            assert s, "an empty front-door set is not a front-door argument"
            for m in s:
                assert m in de_x and m in anc_y, f"{m} is not on a path from {x} to {y}"
            # every directed path from x to y must be intercepted
            blocked = set(s)
            reachable = {x}
            stack = [x]
            while stack:
                n = stack.pop()
                for c in children[n]:
                    if c in blocked or c in reachable:
                        continue
                    reachable.add(c)
                    stack.append(c)
            assert y not in reachable, f"front-door set {s} misses a path to {y}"
        for iv in r["instruments"]:
            found_iv += 1
            z, w = iv["node"], iv["conditional_on"]
            assert z not in de_x, "an instrument cannot be caused by the treatment"
            gx = {**d, "edges": [e for e in d["edges"]
                                 if not (e.get("kind", "directed") == "directed" and e["from"] == x)]}
            assert not slow_dsep(gx, z, x, w), f"{z} does not move {x}"
            assert slow_dsep(gx, z, y, w), f"{z} reaches {y} by another route"
    print(f"front-door ({found_fd}) and instrument ({found_iv}) claims all check out OK")


def main() -> int:
    tests = [
        ("chain, fork, collider", test_chain_fork_collider),
        ("classic collider graph", test_classic_collider_graph),
        ("M-bias", test_m_bias),
        ("mediator chain", test_mediator_chain),
        ("two minimal sets", test_two_minimal_sets_are_both_listed),
        ("front door", test_front_door),
        ("instrument", test_instrument),
        ("conditional instrument", test_conditional_instrument),
        ("napkin", test_napkin_is_honest),
        ("bidirected = unmeasured", test_bidirected_edge_is_unmeasured_confounding),
        ("bad controls", test_bad_controls_by_kind),
        ("cycles", test_cycles_are_explained_never_a_crash),
        ("validate", test_validate_explains_broken_pictures),
        ("spec errors", test_missing_treatment_is_a_sentence_not_a_traceback),
        ("selected set", test_selected_set_is_checked),
        ("truncation is announced", test_truncation_is_announced_not_silent),
        ("caption everywhere", test_caption_on_every_object),
        ("suggest from roles", test_suggest_from_roles),
        ("json and determinism", test_results_are_json_and_deterministic),
        ("d-sep vs path enumeration", test_dsep_matches_path_enumeration),
        ("minimal sets vs brute force", test_minimal_sets_match_brute_force),
        ("front door / IV sanity", test_frontdoor_and_instruments_are_sane),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n== {name} ==")
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {exc}")
    print("\n" + ("ALL DAG TESTS PASSED" if not failed else f"{failed} TEST GROUP(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
