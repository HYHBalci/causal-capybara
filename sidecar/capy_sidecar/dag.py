"""DAG identification -- the headless half of the DAG editor (plan 6.16 and 7.0).

A graph is not evidence. It is a picture of what the analyst already believes,
and this module only reads that picture back to them: which doors are open,
which sets of measured variables close them, and which variables would do
damage if they were dropped into "adjust for".

Everything here is plain Python -- dicts, sets and one BFS. No R, no dagitty,
no networkx, no dowhy. The algorithms are the textbook ones:

* A dashed **bidirected** edge means "something I did not measure moves both of
  these". It is expanded into an explicit latent common cause, so m-separation
  in the drawn graph is ordinary d-separation in the expanded graph.
* **d-separation** is the two-phase reachability search (ancestors of the
  conditioning set, then a walk that remembers which way it arrived) -- Koller
  & Friedman, Algorithm 3.1; equivalently Shachter's Bayes-Ball.
* The **back-door criterion** is Pearl (1995): the set may contain no
  descendant of the treatment, and it must d-separate treatment from outcome
  once the arrows *out of* the treatment are cut.
* Whether *any* back-door set exists is decided once and for all by the
  canonical set ``An({X, Y}) \\ ({X, Y} u de(X))``: if adjusting for that fails,
  no set works. So the headline answer never depends on how far the enumeration
  of small sets happened to get -- and when the enumeration is cut short, the
  result says so out loud instead of quietly capping the list.
* The **front-door criterion** is Pearl (1995) as well, and the instrument
  criterion is the graphical one: relevant to the treatment and separated from
  the outcome once the treatment's outgoing arrows are cut.

Nothing in here is a test of the graph. The data cannot tell you that an arrow
is missing. Hence the caption, which every object this module returns carries:

    This graph is an assumption you are making.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# The sidecar bootstraps the Python engine onto sys.path in app.py; do it here
# too so this module can be imported on its own (tests, a REPL, a script).
_PY_ENGINE_DIR = Path(__file__).resolve().parents[2] / "engines" / "python"
if _PY_ENGINE_DIR.is_dir() and str(_PY_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PY_ENGINE_DIR))

try:  # pragma: no cover - exercised whenever the engine is present
    from capy_py.contracts import SpecError
except Exception:  # pragma: no cover - keeps the DAG editor alive without the engine
    class SpecError(Exception):  # type: ignore[no-redef]
        """The spec cannot identify what it claims to. Fixable in the UI."""

        kind = "spec_error"

        def __init__(self, message: str, detail: str | None = None) -> None:
            super().__init__(message)
            self.message = message
            self.detail = detail


CAPTION = "This graph is an assumption you are making."
SCHEMA = "capy.dag"
VERSION = 1

#: A node id that never collides with a user's, used for the latent common
#: cause standing behind a bidirected edge.
LATENT_PREFIX = "u::"

# Bounds. Enumerating adjustment sets is combinatorial; these keep the editor
# responsive while a user drags nodes around. Whenever one of them bites, the
# result says `truncated: true` and a note explains what was not searched.
MAX_SET_SIZE = 4          # largest back-door set the enumeration will look for
MAX_SETS = 12             # how many minimal sets to list
MAX_TRIALS = 12_000       # subsets tested before we stop and say so
MAX_FRONTDOOR_SIZE = 3
MAX_INSTRUMENT_CONDITION_SIZE = 2
MAX_CYCLES = 8

ROLE_VALUES = {"treatment", "outcome", "confounder", "mediator", "instrument",
               "collider", "latent", "other", "running", "unit", "time"}

_UP = 0    # arrived at this node from one of its children
_DOWN = 1  # arrived at this node from one of its parents


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


@dataclass
class Graph:
    """A directed graph, with bidirected edges already expanded into latents.

    ``order`` keeps declaration order so every list this module returns comes
    back in the order the user drew things, which is also what makes the output
    deterministic.
    """

    order: list[str] = field(default_factory=list)
    node: dict[str, dict[str, Any]] = field(default_factory=dict)
    parents: dict[str, set[str]] = field(default_factory=dict)
    children: dict[str, set[str]] = field(default_factory=dict)
    latent: set[str] = field(default_factory=set)
    synthetic: set[str] = field(default_factory=set)
    bidirected: list[tuple[str, str]] = field(default_factory=list)

    # -- construction ----------------------------------------------------
    def add_node(self, nid: str, **info: Any) -> None:
        if nid not in self.node:
            self.order.append(nid)
            self.node[nid] = {"id": nid}
            self.parents[nid] = set()
            self.children[nid] = set()
        self.node[nid].update({k: v for k, v in info.items() if v is not None})

    def add_edge(self, a: str, b: str) -> None:
        self.add_node(a)
        self.add_node(b)
        self.children[a].add(b)
        self.parents[b].add(a)

    # -- views -----------------------------------------------------------
    @property
    def observed(self) -> list[str]:
        return [n for n in self.order if n not in self.latent]

    def index(self, nid: str) -> int:
        try:
            return self.order.index(nid)
        except ValueError:
            return len(self.order)

    def sort(self, nodes: Iterable[str]) -> list[str]:
        return sorted(set(nodes), key=self.index)

    def label(self, nid: str) -> str:
        info = self.node.get(nid, {})
        return str(info.get("label") or info.get("variable") or nid)

    def variable(self, nid: str) -> str | None:
        info = self.node.get(nid, {})
        var = info.get("variable")
        return str(var) if var else None

    # -- reachability ----------------------------------------------------
    def ancestors(self, nodes: Iterable[str]) -> set[str]:
        """Ancestors, inclusive of the starting nodes. Safe on cyclic graphs."""
        seen: set[str] = set()
        stack = [n for n in nodes if n in self.node]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(self.parents[n] - seen)
        return seen

    def descendants(self, nodes: Iterable[str]) -> set[str]:
        """Descendants, inclusive of the starting nodes. Safe on cyclic graphs."""
        seen: set[str] = set()
        stack = [n for n in nodes if n in self.node]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(self.children[n] - seen)
        return seen

    def directed_path_exists(self, src: str, dst: str, blocked: Iterable[str] = ()) -> bool:
        """Is there a directed path src -> ... -> dst avoiding `blocked`?"""
        stop = set(blocked)
        if src == dst:
            return True
        seen = {src}
        stack = [src]
        while stack:
            n = stack.pop()
            for c in self.children[n]:
                if c == dst:
                    return True
                if c in stop or c in seen:
                    continue
                seen.add(c)
                stack.append(c)
        return False

    def without_out_edges(self, nodes: Iterable[str]) -> "Graph":
        """The graph with every arrow *out of* `nodes` removed."""
        cut = set(nodes)
        g = Graph(order=list(self.order), node=self.node,
                  latent=set(self.latent), synthetic=set(self.synthetic),
                  bidirected=list(self.bidirected))
        g.parents = {n: set(p) for n, p in self.parents.items()}
        g.children = {n: set(c) for n, c in self.children.items()}
        for n in cut:
            for c in self.children.get(n, set()):
                g.parents[c].discard(n)
            g.children[n] = set()
        return g


# ---------------------------------------------------------------------------
# Reading a capy.dag.v1 object
# ---------------------------------------------------------------------------


def _is_latent(raw: Mapping[str, Any]) -> bool:
    """A node is unmeasured if it says so, or if it is bound to no column.

    The schema documents ``variable: null`` as "latent"; an explicit
    ``latent: false`` always wins, because a half-drawn node in the canvas is
    not a claim about the world.
    """
    if raw.get("latent") is True:
        return True
    if raw.get("latent") is False:
        return False
    if str(raw.get("role") or "") == "latent":
        return True
    return "variable" in raw and raw.get("variable") in (None, "")


def _parse(dag: Mapping[str, Any] | None) -> tuple[Graph, list[str], list[str]]:
    """Build the working graph. Returns (graph, errors, warnings).

    Never raises on a malformed picture: the editor calls this on every
    keystroke, and a half-drawn graph must come back explained, not thrown.
    """
    errors: list[str] = []
    warnings: list[str] = []
    g = Graph()
    if not isinstance(dag, Mapping):
        errors.append("A graph object was expected here, with a list of nodes and a list of edges.")
        return g, errors, warnings

    schema = dag.get("schema")
    if schema not in (None, SCHEMA):
        warnings.append(f"This object says it is a '{schema}', not a '{SCHEMA}'. I read it as a graph anyway.")
    if dag.get("version") not in (None, VERSION):
        warnings.append(f"This graph was written for version {dag.get('version')}; I read it as version {VERSION}.")

    raw_nodes = dag.get("nodes")
    if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes)):
        errors.append("The graph has no list of nodes.")
        raw_nodes = []
    for i, raw in enumerate(raw_nodes):
        if not isinstance(raw, Mapping):
            errors.append(f"Node {i + 1} is not a node object.")
            continue
        nid = str(raw.get("id") or "").strip()
        if not nid:
            errors.append(f"Node {i + 1} has no id, so no edge can point at it.")
            continue
        if nid in g.node:
            errors.append(f"Two nodes share the id '{nid}'. Ids have to be unique or the arrows are ambiguous.")
            continue
        role = raw.get("role")
        if role is not None and str(role) not in ROLE_VALUES:
            warnings.append(f"'{nid}' has the role '{role}', which is not one I know. I treated it as 'other'.")
            role = "other"
        g.add_node(nid, variable=raw.get("variable"), label=raw.get("label"),
                   role=role, x=raw.get("x"), y=raw.get("y"))
        if _is_latent(raw):
            g.latent.add(nid)

    raw_edges = dag.get("edges")
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
        if raw_edges is not None:
            errors.append("The graph's edges are not a list.")
        raw_edges = []

    seen_edges: set[tuple[str, str, str]] = set()
    for i, raw in enumerate(raw_edges):
        if not isinstance(raw, Mapping):
            errors.append(f"Edge {i + 1} is not an edge object.")
            continue
        a = str(raw.get("from") or "").strip()
        b = str(raw.get("to") or "").strip()
        kind = str(raw.get("kind") or "directed")
        if not a or not b:
            errors.append(f"Edge {i + 1} is missing one of its ends.")
            continue
        if kind not in ("directed", "bidirected"):
            warnings.append(f"Edge {a} -> {b} has kind '{kind}'; I drew it as a directed arrow.")
            kind = "directed"
        for end in (a, b):
            if end not in g.node:
                warnings.append(
                    f"An edge points at '{end}', which is not in the node list. "
                    "I added it as an unmeasured node so the picture still makes sense."
                )
                g.add_node(end, label=end)
                g.latent.add(end)
        if a == b:
            errors.append(f"'{g.label(a)}' has an arrow to itself. Nothing can cause itself.")
            continue
        key = (a, b, kind) if kind == "directed" else tuple(sorted((a, b))) + ("bidirected",)  # type: ignore[assignment]
        if key in seen_edges:
            warnings.append(f"The edge between '{g.label(a)}' and '{g.label(b)}' is drawn twice; I used it once.")
            continue
        seen_edges.add(key)  # type: ignore[arg-type]
        if kind == "directed":
            g.add_edge(a, b)
        else:
            g.bidirected.append((a, b))

    # Expand every bidirected edge into the unmeasured common cause it stands for.
    for a, b in g.bidirected:
        uid = f"{LATENT_PREFIX}{a}<->{b}"
        n = 2
        while uid in g.node:
            uid = f"{LATENT_PREFIX}{a}<->{b}#{n}"
            n += 1
        g.add_node(uid, label=f"unmeasured cause of {g.label(a)} and {g.label(b)}", role="latent")
        g.latent.add(uid)
        g.synthetic.add(uid)
        g.add_edge(uid, a)
        g.add_edge(uid, b)

    for a in g.order:  # sorted, so the warnings come out in the same order every run
        for b in g.sort(g.children[a]):
            if a in g.children.get(b, set()):
                warnings.append(
                    f"'{g.label(a)}' and '{g.label(b)}' point at each other. "
                    "That is a loop, not a causal order."
                )
                break

    return g, errors, warnings


# ---------------------------------------------------------------------------
# d-separation
# ---------------------------------------------------------------------------


def _reachable(g: Graph, sources: Iterable[str], given: Iterable[str]) -> set[str]:
    """Nodes d-connected to `sources` given `given` (Koller & Friedman Alg. 3.1)."""
    z = {n for n in given if n in g.node}
    anc_z = g.ancestors(z)  # inclusive: a collider is open if it, or a descendant, is in z
    visited: set[tuple[str, int]] = set()
    reached: set[str] = set()
    stack: list[tuple[str, int]] = [(s, _UP) for s in sources if s in g.node]
    while stack:
        y, d = stack.pop()
        if (y, d) in visited:
            continue
        visited.add((y, d))
        if y not in z:
            reached.add(y)
        if d == _UP and y not in z:
            for p in g.parents[y]:
                stack.append((p, _UP))
            for c in g.children[y]:
                stack.append((c, _DOWN))
        elif d == _DOWN:
            if y not in z:
                for c in g.children[y]:
                    stack.append((c, _DOWN))
            if y in anc_z:  # an open collider: the trail may turn around here
                for p in g.parents[y]:
                    stack.append((p, _UP))
    return reached


def _dsep(g: Graph, xs: Iterable[str], ys: Iterable[str], given: Iterable[str]) -> bool:
    xs, ys, z = set(xs), set(ys), set(given)
    if xs & ys:
        return False
    if (xs | ys) & z:
        return False
    return not (_reachable(g, xs, z) & ys)


def d_separated(dag: Mapping[str, Any], x: str | Iterable[str], y: str | Iterable[str],
                given: Iterable[str] = ()) -> bool:
    """Public d-separation test on a capy.dag.v1 object (ids or column names).

    Bidirected edges count as unmeasured common causes, so this is m-separation
    on the drawn graph.
    """
    g, errors, _ = _parse(dag)
    if errors:
        raise SpecError(errors[0])
    xs = [_resolve(g, n, what="node") for n in ([x] if isinstance(x, str) else list(x))]
    ys = [_resolve(g, n, what="node") for n in ([y] if isinstance(y, str) else list(y))]
    zs = [_resolve(g, n, what="node") for n in ([given] if isinstance(given, str) else list(given))]
    return _dsep(g, xs, ys, zs)


# ---------------------------------------------------------------------------
# Cycles
# ---------------------------------------------------------------------------


def _cycles(g: Graph) -> list[list[str]]:
    """Distinct simple cycles (up to MAX_CYCLES), found by an iterative DFS."""
    found: list[list[str]] = []
    seen_keys: set[tuple[str, ...]] = set()
    colour: dict[str, int] = {n: 0 for n in g.order}  # 0 white, 1 grey, 2 black
    for root in g.order:
        if colour[root] != 0:
            continue
        path: list[str] = []
        on_path: set[str] = set()
        stack: list[tuple[str, Any]] = [(root, iter(g.sort(g.children[root])))]
        colour[root] = 1
        path.append(root)
        on_path.add(root)
        while stack:
            node, it = stack[-1]
            advanced = False
            for child in it:
                if child in on_path:
                    cyc = path[path.index(child):]
                    key = _cycle_key(cyc)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        found.append(list(cyc))
                        if len(found) >= MAX_CYCLES:
                            return found
                    continue
                if colour.get(child, 0) == 0:
                    colour[child] = 1
                    path.append(child)
                    on_path.add(child)
                    stack.append((child, iter(g.sort(g.children[child]))))
                    advanced = True
                    break
            if not advanced:
                colour[node] = 2
                stack.pop()
                path.pop()
                on_path.discard(node)
    return found


def _cycle_key(cycle: Sequence[str]) -> tuple[str, ...]:
    if not cycle:
        return ()
    i = min(range(len(cycle)), key=lambda k: cycle[k])
    return tuple(cycle[i:] + cycle[:i])


# ---------------------------------------------------------------------------
# Back door
# ---------------------------------------------------------------------------


def _backdoor_ok(g: Graph, gbd: Graph, x: str, y: str, z: Iterable[str]) -> bool:
    """Pearl's back-door criterion for a single treatment."""
    zs = set(z)
    if zs & g.descendants({x}):     # inclusive, so this also rejects x itself
        return False
    if zs & g.latent:               # you cannot adjust for what you did not measure
        return False
    if y in zs:
        return False
    return _dsep(gbd, {x}, {y}, zs)


def _backdoor_candidates(g: Graph, gbd: Graph, x: str, y: str) -> list[str]:
    """Every minimal back-door set lives inside the ancestors of X and Y."""
    anc = gbd.ancestors({x, y})
    bad = g.descendants({x}) | {y} | g.latent
    return [n for n in g.order if n in anc and n not in bad]


def _minimal_backdoor_sets(g: Graph, gbd: Graph, x: str, y: str, cands: Sequence[str],
                           max_size: int, max_sets: int) -> tuple[list[list[str]], bool, int]:
    """Enumerate minimal sufficient sets by increasing size.

    Returns (sets, truncated, searched_up_to). `truncated` is True whenever the
    search did not exhaust the possibilities -- the caller must say so.
    """
    accepted: list[tuple[str, ...]] = []
    tried = 0
    truncated = False
    searched = -1
    top = min(max_size, len(cands))
    for k in range(0, top + 1):
        for comb in itertools.combinations(cands, k):
            if tried >= MAX_TRIALS:
                truncated = True
                break
            s = set(comb)
            if any(set(a) <= s for a in accepted):
                continue  # a subset already works, so this one is not minimal
            tried += 1
            if _backdoor_ok(g, gbd, x, y, s):
                accepted.append(comb)
                if len(accepted) >= max_sets:
                    truncated = True
                    break
        if truncated:
            break
        searched = k
    if top < len(cands):
        truncated = True
    return [list(s) for s in accepted], truncated, max(searched, 0)


# ---------------------------------------------------------------------------
# Front door
# ---------------------------------------------------------------------------


def _frontdoor_ok(g: Graph, x: str, y: str, m: Iterable[str]) -> bool:
    ms = set(m)
    if not ms or ms & ({x, y} | g.latent):
        return False
    # (1) the set intercepts every directed path from treatment to outcome
    if g.directed_path_exists(x, y, blocked=ms):
        return False
    # (2) no unblocked back-door path from the treatment to the set
    if not _dsep(g.without_out_edges({x}), {x}, ms, set()):
        return False
    # (3) the treatment blocks every back-door path from the set to the outcome
    if not _dsep(g.without_out_edges(ms), ms, {y}, {x}):
        return False
    return True


def _frontdoor_sets(g: Graph, x: str, y: str, mediators: Sequence[str],
                    max_size: int = MAX_FRONTDOOR_SIZE) -> tuple[list[list[str]], bool]:
    accepted: list[tuple[str, ...]] = []
    pool = [m for m in mediators if m not in g.latent]
    truncated = min(max_size, len(pool)) < len(pool)
    for k in range(1, min(max_size, len(pool)) + 1):
        for comb in itertools.combinations(pool, k):
            if any(set(a) <= set(comb) for a in accepted):
                continue
            if _frontdoor_ok(g, x, y, comb):
                accepted.append(comb)
                if len(accepted) >= MAX_SETS:
                    return [list(s) for s in accepted], True
    return [list(s) for s in accepted], truncated


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------


def _instruments(g: Graph, x: str, y: str) -> list[dict[str, Any]]:
    """Graphical instruments, with the smallest set they need to be conditioned on.

    In the graph with the treatment's outgoing arrows cut, a valid instrument is
    still connected to the treatment (it is relevant) and no longer connected to
    the outcome (it moves the outcome only through the treatment).
    """
    gx = g.without_out_edges({x})
    de_x = g.descendants({x})
    pool = [n for n in g.observed if n not in de_x and n != y]
    out: list[dict[str, Any]] = []
    for z in pool:
        wpool = [w for w in g.observed if w not in de_x and w not in (z, y)]
        chosen: list[str] | None = None
        top = min(MAX_INSTRUMENT_CONDITION_SIZE, len(wpool))
        for k in range(0, top + 1):
            for comb in itertools.combinations(wpool, k):
                w = set(comb)
                if _dsep(gx, {z}, {x}, w):
                    continue  # not relevant: it does not move the treatment
                if not _dsep(gx, {z}, {y}, w):
                    continue  # it reaches the outcome by some other route
                chosen = list(comb)
                break
            if chosen is not None:
                break
        if chosen is None:
            continue
        if chosen:
            why = (f"'{g.label(z)}' moves '{g.label(x)}' and, once you hold "
                   f"{_join([g.label(w) for w in chosen])} fixed, reaches '{g.label(y)}' only through it.")
        else:
            why = (f"'{g.label(z)}' moves '{g.label(x)}' and reaches '{g.label(y)}' only through it, "
                   "if the arrows you drew are right.")
        out.append({
            "node": z, "variable": g.variable(z), "label": g.label(z),
            "conditional_on": chosen,
            "conditional_on_variables": [g.variable(w) for w in chosen],
            "why": why,
            "estimand": "LATE (the effect for the units this instrument moves), not the ATE.",
        })
    return out


# ---------------------------------------------------------------------------
# Bad controls and colliders
# ---------------------------------------------------------------------------


def _colliders(g: Graph, x: str | None = None, y: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    anc_x = g.ancestors({x}) if x else set()
    anc_y = g.ancestors({y}) if y else set()
    for n in g.order:
        if n in g.synthetic or n in (x, y):
            continue
        ps = g.sort(g.parents[n])
        if len(ps) < 2:
            continue
        between = False
        if x and y:
            between = any(a in anc_x and b in anc_y for a, b in itertools.permutations(ps, 2))
        out.append({
            "node": n, "variable": g.variable(n), "label": g.label(n),
            "parents": ps,
            "parent_labels": [g.label(p) for p in ps],
            "on_backdoor_path": bool(between),
            "why": (f"Two or more arrows point into '{g.label(n)}' "
                    f"(from {_join([g.label(p) for p in ps])}). It is closed while you leave it alone; "
                    "adjusting for it, or selecting your sample on it, opens a path through it."),
        })
    return out


def _bad_controls(g: Graph, gbd: Graph, x: str, y: str, minimal_sets: Sequence[Sequence[str]],
                  canonical: Sequence[str], canonical_ok: bool,
                  colliders: Sequence[Mapping[str, Any]],
                  frontdoor: Iterable[str] = ()) -> list[dict[str, Any]]:
    de_x = g.descendants({x}) - {x}
    de_y = g.descendants({y}) - {y}
    anc_y = g.ancestors({y})
    in_some_set = {n for s in minimal_sets for n in s}
    fd_nodes = set(frontdoor)
    collider_ids = {c["node"] for c in colliders}
    backdoor_collider_ids = {c["node"] for c in colliders if c.get("on_backdoor_path")}

    ref_sets: list[set[str]] = [set(s) for s in minimal_sets]
    if not ref_sets and canonical_ok:
        ref_sets = [set(canonical)]
    have_reference = bool(ref_sets)

    out: list[dict[str, Any]] = []
    for n in g.order:
        if n in (x, y) or n in g.latent or n in g.synthetic:
            continue
        kind: str | None = None
        why = ""
        if n in de_x and n in de_y:
            kind = "descendant_of_both"
            why = (f"'{g.label(n)}' happens after both the treatment and the outcome. "
                   "Adjusting for it, or filtering the sample on it, is selection bias: it makes "
                   "treated and untreated units look different for a reason that has nothing to do "
                   "with the effect.")
        elif n in de_x and n in anc_y:
            kind = "mediator"
            why = (f"'{g.label(n)}' is on the path from '{g.label(x)}' to '{g.label(y)}'. "
                   "Adjusting for it removes the part of the effect that runs through it, so what is "
                   "left is a direct effect, not the total effect you asked for.")
            if n in fd_nodes:
                why += (" The front-door route below uses it a different way -- it does not adjust "
                        "for it -- so listing it here is about adjustment only.")
        elif n in de_x:
            kind = "descendant_of_treatment"
            why = (f"'{g.label(n)}' happens after the treatment. Adjusting for something the treatment "
                   "caused can move the estimate in either direction, and nothing in the output warns you.")
        elif n not in in_some_set:
            opens = False
            if have_reference:
                for s in ref_sets:
                    if n in s:
                        continue
                    if _dsep(gbd, {x}, {y}, s) and not _dsep(gbd, {x}, {y}, s | {n}):
                        opens = True
                        break
            elif n in backdoor_collider_ids:
                opens = True
            if opens and n in collider_ids:
                kind = "collider"
                why = (f"'{g.label(n)}' is a collider: arrows from two different places point into it. "
                       f"Leaving it alone keeps that path closed. Adjusting for it opens a path between "
                       f"'{g.label(x)}' and '{g.label(y)}' that was not there before.")
            elif opens:
                kind = "opens_a_path"
                why = (f"Adjusting for '{g.label(n)}' opens a path between '{g.label(x)}' and "
                       f"'{g.label(y)}' that is closed while you leave it out.")
        if kind:
            out.append({
                "node": n, "variable": g.variable(n), "label": g.label(n),
                "kind": kind, "why": why,
            })
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve(g: Graph, key: Any, *, role: str | None = None, what: str = "variable") -> str:
    """Find a node by id, by column name, or (when key is None) by role."""
    if key is None or (isinstance(key, str) and not key.strip()):
        if role:
            hits = [n for n in g.order if str(g.node[n].get("role") or "") == role]
            if len(hits) == 1:
                return hits[0]
            if not hits:
                raise SpecError(
                    f"No node in this graph is marked as the {role}. "
                    f"Mark one, or tell me which variable is the {role}."
                )
            raise SpecError(
                f"{len(hits)} nodes are marked as the {role} ({_join([g.label(h) for h in hits])}). "
                f"Exactly one node has to be the {role}."
            )
        raise SpecError(f"No {what} was given, and the graph does not say which node it is.")
    k = str(key).strip()
    if k in g.node:
        return k
    for n in g.order:
        if g.node[n].get("variable") == k:
            return n
    lowered = k.casefold()
    for n in g.order:
        if str(g.node[n].get("label") or "").casefold() == lowered:
            return n
    known = _join([g.label(n) for n in g.observed[:12]]) or "nothing yet"
    raise SpecError(
        f"'{k}' is not in this graph. The nodes I can see are: {known}."
        + (" (and more)" if len(g.observed) > 12 else "")
    )


def validate(dag: Mapping[str, Any] | None) -> dict[str, Any]:
    """Structural check on a capy.dag.v1 object. Never raises, never crashes."""
    g, errors, warnings = _parse(dag)
    cycles = _cycles(g)
    cycle_rows = [_cycle_row(g, c) for c in cycles]
    for row in cycle_rows:
        errors.append(row["explain"])
    if not g.order and not errors:
        warnings.append("The graph is empty. Drag a variable onto the canvas to start.")
    isolated = [n for n in g.observed if not g.parents[n] and not g.children[n]]
    if isolated:
        warnings.append(
            f"{_join([g.label(n) for n in isolated])} "
            f"{'has' if len(isolated) == 1 else 'have'} no arrows. "
            "A node with no arrows is a claim that it causes nothing and is caused by nothing."
        )
    return {
        "caption": CAPTION,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "cycles": cycle_rows,
        "n_nodes": len([n for n in g.order if n not in g.synthetic]),
        "n_edges": sum(len(g.children[n]) for n in g.order if n not in g.synthetic) + len(g.bidirected),
        "n_unmeasured": len([n for n in g.latent if n not in g.synthetic]) + len(g.bidirected),
    }


def _cycle_row(g: Graph, cycle: Sequence[str]) -> dict[str, Any]:
    names = [g.label(n) for n in cycle]
    loop = " -> ".join(names + [names[0]])
    return {
        "nodes": list(cycle),
        "labels": names,
        "explain": (f"These arrows run in a circle: {loop}. A back-door argument needs a graph with no "
                    "loops, because 'what happened before what' has to be answerable. If two things "
                    "really do feed back on each other, that is a time story -- give them one node per "
                    "period, or use a design that models the feedback."),
    }


def bad_controls(dag: Mapping[str, Any] | None, treatment: Any = None,
                 outcome: Any = None) -> dict[str, Any]:
    """Variables the UI should glow red if they are dropped into 'adjust for'.

    Descendants of the treatment always qualify. Colliders qualify when
    conditioning on them opens a path that is otherwise closed, which needs an
    outcome; if none is given, the node marked as the outcome is used.
    """
    g, errors, warnings = _parse(dag)
    x = _resolve(g, treatment, role="treatment", what="treatment")
    y: str | None
    try:
        y = _resolve(g, outcome, role="outcome", what="outcome")
    except SpecError:
        y = None

    de_x = g.sort(g.descendants({x}) - {x} - g.synthetic)
    rows: list[dict[str, Any]] = []
    colliders: list[dict[str, Any]] = []
    if y is None:
        warnings.append(
            "No outcome was given, so this lists only what the treatment causes. "
            "Colliders need an outcome before I can tell you which paths they open."
        )
        anc_y: set[str] = set()
        for n in de_x:
            if n in g.latent:
                continue
            rows.append({
                "node": n, "variable": g.variable(n), "label": g.label(n),
                "kind": "descendant_of_treatment",
                "why": (f"'{g.label(n)}' happens after the treatment. Adjusting for something the "
                        "treatment caused blocks part of the effect you are trying to measure."),
            })
    else:
        gbd = g.without_out_edges({x})
        cands = _backdoor_candidates(g, gbd, x, y)
        canonical = list(cands)
        canonical_ok = _backdoor_ok(g, gbd, x, y, canonical)
        sets: list[list[str]] = []
        if canonical_ok:
            sets, _, _ = _minimal_backdoor_sets(g, gbd, x, y, cands, MAX_SET_SIZE, MAX_SETS)
        colliders = _colliders(g, x, y)
        rows = _bad_controls(g, gbd, x, y, sets, canonical, canonical_ok, colliders)
    return {
        "caption": CAPTION,
        "treatment": x,
        "outcome": y,
        "bad_controls": rows,
        "descendants_of_treatment": [n for n in de_x if n not in g.latent],
        "colliders": colliders,
        "errors": errors,
        "warnings": warnings,
    }


def identify(dag: Mapping[str, Any] | None, treatment: Any = None, outcome: Any = None,
             *, max_set_size: int = MAX_SET_SIZE, max_sets: int = MAX_SETS) -> dict[str, Any]:
    """Read a drawn graph back to the user: doors, sets, instruments, traps.

    `treatment` and `outcome` may be node ids, dataset column names, or None --
    in which case the nodes marked with those roles are used. Anything the user
    can fix in the editor raises SpecError with a sentence; anything about the
    picture itself (a loop, a dangling arrow) comes back inside the result, so
    the canvas can explain rather than blank out.
    """
    g, errors, warnings = _parse(dag)
    x = _resolve(g, treatment, role="treatment", what="treatment")
    y = _resolve(g, outcome, role="outcome", what="outcome")
    if x == y:
        raise SpecError("The treatment and the outcome are the same node. They have to be two things.")

    cycles = [_cycle_row(g, c) for c in _cycles(g)]
    notes: list[str] = []
    latents = []
    for n in g.order:
        if n not in g.latent:
            continue
        pair = _between(g, n)
        latents.append({
            "id": n, "label": g.label(n), "synthetic": n in g.synthetic,
            "between": list(pair) if pair[0] else None,
        })

    de_x = g.sort(g.descendants({x}) - {x} - g.synthetic)
    mediators = g.sort(((g.descendants({x}) - {x}) & (g.ancestors({y}) - {y})) - g.synthetic)
    colliders = _colliders(g, x, y)

    out: dict[str, Any] = {
        "schema": "capy.identification",
        "version": VERSION,
        "caption": CAPTION,
        "treatment": x,
        "treatment_label": g.label(x),
        "outcome": y,
        "outcome_label": g.label(y),
        "valid": not errors and not cycles,
        "errors": list(errors),
        "warnings": list(warnings),
        "cycles": cycles,
        "mediators": mediators,
        "descendants_of_treatment": de_x,
        "colliders": colliders,
        "latents": latents,
        "variables": {n: g.variable(n) for n in g.order if n not in g.synthetic},
        "labels": {n: g.label(n) for n in g.order if n not in g.synthetic},
        "notes": notes,
    }

    if cycles or errors:
        reason = (cycles[0]["explain"] if cycles else errors[0])
        out["backdoor"] = {"identified": False, "minimal_sets": [], "minimal_sets_variables": [],
                           "canonical_set": [], "reason": reason, "truncated": False,
                           "searched_up_to": 0, "n_candidates": 0, "candidates": []}
        out["frontdoor"] = {"identified": False, "sets": [], "sets_variables": [],
                            "reason": reason, "truncated": False}
        out["instruments"] = []
        out["bad_controls"] = []
        out["selected"] = None
        out["summary"] = ("I cannot read an adjustment set off this graph yet. " + reason)
        notes.append("Fix the picture and the identification panel fills itself in again. "
                     "Nothing was estimated and nothing was dropped.")
        return out

    gbd = g.without_out_edges({x})
    cands = _backdoor_candidates(g, gbd, x, y)
    canonical = list(cands)
    canonical_ok = _backdoor_ok(g, gbd, x, y, canonical)
    if canonical_ok:
        sets, truncated, searched = _minimal_backdoor_sets(g, gbd, x, y, cands, max_set_size, max_sets)
    else:
        # If adjusting for every candidate at once does not close the back doors,
        # no smaller set can either. Nothing to enumerate, and nothing truncated.
        sets, truncated, searched = [], False, 0
    identified = canonical_ok or bool(sets)

    if identified and sets:
        if sets == [[]]:
            reason = ("Every back-door path between them is already closed, so no adjustment is needed "
                      "-- as long as the arrows you drew are the ones that exist.")
        else:
            reason = ("Adjusting for any one of these sets closes every back-door path between "
                      f"'{g.label(x)}' and '{g.label(y)}', if the arrows you drew are the ones that exist.")
    elif identified:
        reason = ("A sufficient set exists, but it is larger than the sets I searched. Adjusting for "
                  "everything in 'canonical_set' closes the back doors.")
    else:
        blockers = _unmeasured_blockers(g, gbd, x, y, canonical)
        if blockers:
            reason = (f"At least one back-door path runs through something you have not measured "
                      f"({_join(blockers)}), so no set of measured variables closes it. "
                      "This design cannot answer the question by adjustment alone.")
        else:
            reason = ("No set of measured variables closes every back-door path in this graph. "
                      "Adjustment alone will not identify the effect here.")

    out["backdoor"] = {
        "identified": bool(identified),
        "minimal_sets": sets,
        "minimal_sets_variables": [[g.variable(n) for n in s] for s in sets],
        "minimal_sets_labels": [[g.label(n) for n in s] for s in sets],
        "canonical_set": canonical if canonical_ok else [],
        "canonical_set_variables": [g.variable(n) for n in canonical] if canonical_ok else [],
        "reason": reason,
        "truncated": bool(truncated),
        "searched_up_to": int(searched),
        "n_candidates": len(cands),
        "candidates": cands,
    }
    if truncated:
        looked = min(max_set_size, len(cands))
        if sets:
            note = (f"I listed the {len(sets)} smallest sufficient sets I found among {len(cands)} "
                    f"candidate variables, searching up to {looked} at a time. There may be other "
                    "minimal sets I did not list.")
        else:
            note = (f"No set of {looked} or fewer variables out of the {len(cands)} candidates closes "
                    "every back-door path, so none is listed here; that is the search hitting its "
                    "bound, not a verdict on the graph.")
        note += (" Whether *any* set works is answered exactly either way, by testing all "
                 f"{len(cands)} candidates together.")
        notes.append(note)
        out["backdoor"]["truncation_note"] = note

    fd_sets, fd_trunc = _frontdoor_sets(g, x, y, mediators)
    if fd_sets:
        fd_reason = ("Everything the treatment does to the outcome runs through this set, nothing "
                     "unmeasured pushes the treatment into it, and the treatment closes its own back "
                     "doors to the outcome. That is the front-door argument; it survives unmeasured "
                     "confounding between treatment and outcome.")
    elif y in g.children[x]:
        fd_reason = (f"'{g.label(x)}' has an arrow straight to '{g.label(y)}', so there is no set of "
                     "measured steps that everything must pass through. The front door is shut.")
    elif not mediators:
        fd_reason = ("Nothing sits between the treatment and the outcome, so there is no front-door "
                     "route.")
    elif any(m in g.latent for m in mediators):
        fd_reason = ("At least one step between the treatment and the outcome is unmeasured, so no "
                     "measured set catches everything the treatment does.")
    else:
        fd_reason = ("The measured steps between treatment and outcome do not meet the front-door "
                     "conditions: something else pushes both the treatment and those steps, or those "
                     "steps have their own back doors to the outcome.")
    out["frontdoor"] = {
        "identified": bool(fd_sets),
        "sets": fd_sets,
        "sets_variables": [[g.variable(n) for n in s] for s in fd_sets],
        "sets_labels": [[g.label(n) for n in s] for s in fd_sets],
        "reason": fd_reason,
        "truncated": bool(fd_trunc),
    }

    out["instruments"] = _instruments(g, x, y)
    out["bad_controls"] = _bad_controls(g, gbd, x, y, sets, canonical, canonical_ok, colliders,
                                        frontdoor={n for s in fd_sets for n in s})

    selected = dag.get("selected_adjustment_set") if isinstance(dag, Mapping) else None
    out["selected"] = _check_selected(g, gbd, x, y, selected)

    # Copy the UI leads with.
    if identified and sets == [[]]:
        summary = (f"With these arrows, nothing needs to be adjusted for: the paths between "
                   f"'{g.label(x)}' and '{g.label(y)}' that do not follow the arrow out of the "
                   "treatment are already closed.")
    elif identified and sets:
        first = _join([g.label(n) for n in sets[0]])
        more = f" ({len(sets)} sets in all)" if len(sets) > 1 else ""
        summary = (f"With these arrows, the effect of '{g.label(x)}' on '{g.label(y)}' is identified by "
                   f"adjusting for {first}{more}.")
    elif identified:
        summary = (f"With these arrows, the effect is identified, but only by adjusting for all "
                   f"{len(canonical)} candidate variables at once.")
    elif fd_sets:
        summary = (f"Adjustment cannot identify this effect, but the front door can: "
                   f"{_join([g.label(n) for n in fd_sets[0]])} carries everything the treatment does.")
    elif out["instruments"]:
        summary = (f"Adjustment cannot identify this effect. "
                   f"'{out['instruments'][0]['label']}' would work as an instrument, which answers a "
                   "different question: the effect for the units it moves.")
    else:
        summary = (f"With these arrows, no measured set identifies the effect of '{g.label(x)}' on "
                   f"'{g.label(y)}'. Either something is missing from the picture, or this question "
                   "needs a different design.")
    out["summary"] = summary

    notes.append("This is what the picture implies, not what the data says. No test in the world can "
                 "tell you an arrow is missing.")
    if latents:
        notes.append(f"{len(latents)} node(s) here are unmeasured, so they can never enter an "
                     "adjustment set. Dashed edges are read as an unmeasured common cause.")
    if out["bad_controls"]:
        notes.append(f"{len(out['bad_controls'])} variable(s) would do damage if you adjusted for them; "
                     "they are listed in 'bad_controls'.")
    return out


def _between(g: Graph, n: str) -> tuple[str, str] | tuple[None, None]:
    if n in g.synthetic:
        kids = g.sort(g.children[n])
        if len(kids) == 2:
            return kids[0], kids[1]
    return None, None


def _unmeasured_blockers(g: Graph, gbd: Graph, x: str, y: str, canonical: Sequence[str]) -> list[str]:
    """Which unmeasured nodes keep a back-door path open once you adjust for everything."""
    open_nodes = _reachable(gbd, {x}, set(canonical))
    out: list[str] = []
    for n in g.order:
        if n not in g.latent or n not in open_nodes:
            continue
        if n in gbd.ancestors({x}) and n in gbd.ancestors({y}):
            out.append(g.label(n))
    return out[:4]


def _check_selected(g: Graph, gbd: Graph, x: str, y: str, selected: Any) -> dict[str, Any] | None:
    if not selected:
        return None
    if isinstance(selected, str):
        selected = [selected]
    try:
        nodes = [_resolve(g, s) for s in selected]
    except SpecError as exc:
        return {"set": list(selected), "sufficient": False, "reason": str(exc)}
    ok = _backdoor_ok(g, gbd, x, y, nodes)
    if ok:
        reason = "The set you picked closes every back-door path, if the arrows you drew are right."
    else:
        offenders = [g.label(n) for n in nodes if n in g.descendants({x}) - {x}]
        if offenders:
            reason = (f"{_join(offenders)} happen after the treatment. Adjusting for them removes part "
                      "of the effect you are asking about.")
        elif [n for n in nodes if n in g.latent]:
            reason = "This set contains a node you have not measured, so it cannot be adjusted for."
        else:
            reason = ("The set you picked leaves at least one back-door path open, so the comparison "
                      "would still mix the effect with a difference between the groups.")
    return {
        "set": nodes,
        "variables": [g.variable(n) for n in nodes],
        "sufficient": bool(ok),
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# A starter graph from the roles the user already assigned
# ---------------------------------------------------------------------------


_ROLE_NOTES = {
    "confounder": "You said this moves both the treatment and the outcome. That is why it is drawn twice.",
    "instrument": "An instrument moves the treatment, and reaches the outcome only through it.",
    "mediator": "A mediator carries part of the effect. Adjusting for it answers a different question.",
    "running": "The running variable decides who is treated, and usually moves the outcome too.",
    "forbidden": "You marked this as post-treatment or a collider, so it is drawn after the treatment.",
}


def suggest_from_roles(roles: Mapping[str, Any] | None) -> dict[str, Any]:
    """A starter capy.dag.v1 drawn from the roles the user already assigned.

    It is a first draft of an argument, not a discovery. Every edge carries the
    note that says which role put it there, so the user can delete the ones they
    do not believe.
    """
    roles = roles or {}
    if isinstance(roles.get("roles"), Mapping):  # tolerate a whole spec
        spec = roles
        roles = spec["roles"]
    else:
        spec = {}
    design = str(spec.get("design") or roles.get("design") or "").strip()

    def one(key: str) -> str | None:
        v = roles.get(key)
        if isinstance(v, (list, tuple)):
            v = v[0] if v else None
        v = str(v).strip() if v not in (None, "") else None
        return v or None

    def many(key: str) -> list[str]:
        v = roles.get(key)
        if v in (None, ""):
            return []
        if isinstance(v, str):
            v = [v]
        out: list[str] = []
        for item in v:
            s = str(item).strip()
            if s and s not in out:
                out.append(s)
        return out

    treatment = one("treatment")
    outcome = one("outcome")
    notes: list[str] = []
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    used: set[str] = set()

    def node(var: str, role: str, x: float, y: float) -> str | None:
        if var in used:
            return None
        used.add(var)
        nodes.append({"id": var, "variable": var, "label": var, "role": role,
                      "latent": False, "x": float(x), "y": float(y)})
        return var

    def edge(a: str, b: str, note: str, kind: str = "directed") -> None:
        edges.append({"from": a, "to": b, "kind": kind, "note": note})

    if treatment:
        node(treatment, "treatment", 240, 320)
    if outcome:
        node(outcome, "outcome", 820, 320)
    if treatment and outcome:
        edge(treatment, outcome, "The effect you are asking about.")
    else:
        notes.append("Pick a treatment and an outcome and the arrow between them appears.")

    confounders = [c for c in many("confounders") if c not in (treatment, outcome)]
    for i, c in enumerate(confounders):
        added = node(c, "confounder", 380 + 140 * (i % 3), 80 + 70 * (i // 3))
        if added and treatment:
            edge(c, treatment, _ROLE_NOTES["confounder"])
        if added and outcome:
            edge(c, outcome, _ROLE_NOTES["confounder"])

    instruments = [z for z in many("instruments") if z not in (treatment, outcome)]
    for i, z in enumerate(instruments):
        added = node(z, "instrument", 40, 260 + 80 * i)
        if added and treatment:
            edge(z, treatment, _ROLE_NOTES["instrument"])

    mediators = [m for m in many("mediator") if m not in (treatment, outcome)]
    for i, m in enumerate(mediators):
        added = node(m, "mediator", 530, 320 + 90 * i)
        if added and treatment:
            edge(treatment, m, _ROLE_NOTES["mediator"])
        if added and outcome:
            edge(m, outcome, _ROLE_NOTES["mediator"])

    running = one("running")
    if running and running not in (treatment, outcome):
        node(running, "running", 120, 140)
        if treatment:
            edge(running, treatment, _ROLE_NOTES["running"])
        if outcome:
            edge(running, outcome, _ROLE_NOTES["running"])

    forbidden = [f for f in many("forbidden") if f not in (treatment, outcome)]
    for i, f in enumerate(forbidden):
        added = node(f, "other", 530 + 120 * i, 520)
        if added and treatment:
            edge(treatment, f, _ROLE_NOTES["forbidden"])
    if forbidden:
        notes.append("Variables you marked as forbidden are drawn after the treatment, which is why "
                     "the editor will glow them red if you try to adjust for them.")

    if instruments and treatment and outcome:
        edge(treatment, outcome,
             "An instrument is only worth using when you believe something unmeasured moves both the "
             "treatment and the outcome. That belief is this dashed edge; delete it if you do not hold it.",
             kind="bidirected")
        notes.append("A dashed edge between the treatment and the outcome was added because you named "
                     "an instrument: instruments exist to survive unmeasured confounding. While it is "
                     "there, no set of measured variables closes the back doors, so no adjustment set "
                     "is pre-selected.")
    elif treatment and outcome:
        notes.append("If you believe something you did not measure moves both the treatment and the "
                     "outcome, draw a dashed edge between them. No amount of data can tell you to.")

    skipped = [k for k in ("unit", "time", "cluster", "weight") if roles.get(k)]
    if skipped:
        names = _join([str(roles[k]) for k in skipped])
        tail = ("is a design fact -- who is a unit, when it happened, how the sample was drawn -- "
                "not a cause. It is left off the canvas on purpose; add it if you think it belongs "
                "in the story." if len(skipped) == 1 else
                "are design facts -- who is a unit, when it happened, how the sample was drawn -- "
                "not causes. They are left off the canvas on purpose; add them if you think they "
                "belong in the story.")
        notes.append(f"{names} {tail}")
    if design:
        notes.append(f"Drawn from a '{design}' design. It is a first draft of your argument, not a "
                     "discovery: delete the arrows you do not believe.")

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "id": "dag_from_roles",
        "name": (f"{outcome} <- {treatment}" if treatment and outcome else "Starter graph"),
        "nodes": nodes,
        "edges": edges,
        # Pre-selecting the confounders would contradict the dashed edge an
        # instrument implies, so it is only offered when the graph has no
        # unmeasured confounding drawn in it.
        "selected_adjustment_set": (list(confounders) if confounders and not instruments else None),
        "caption": CAPTION,
        "notes": notes,
        "generated_from": "roles",
    }


# ---------------------------------------------------------------------------


def _join(items: Sequence[str]) -> str:
    items = [str(i) for i in items if str(i)]
    if not items:
        return ""
    if len(items) == 1:
        return f"'{items[0]}'"
    quoted = [f"'{i}'" for i in items]
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]
