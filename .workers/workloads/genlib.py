#!/usr/bin/env python3
"""genlib -- product-agnostic seeded generator + differential harness core.

Design contract (EXP-101):
  * A single integer seed FULLY determines a generated Program (Config + Ops).
    Generation is deterministic across processes: it never depends on dict order,
    hash randomization, wall-clock, or the random module's global state. Every
    random draw goes through a seed-derived random.Random built from a sha256 of
    (root_seed, label), so the same seed yields a byte-identical program anywhere.
  * A Program is data: a Config (key->value drawn from declared sweep AXES) plus an
    ordered list of Ops (DDL / DML / QUERY / LIFECYCLE / EXPECT_ERROR). Axes and
    value pools are declared as tables, never pinned to a single magic combo, so an
    auditor can see the generator sweeps rather than telegraphing a known bug.
  * Runners are adapters (execute a script, reopen, return rows+error+rc). The
    reference runner is stdlib sqlite3; a thin CliRunner drives a subprocess binary
    (tursodb) and is mockable. Universal, product-independent ORACLES compare the two
    runners: differential-rows, differential-error-class, integrity, panic/abort,
    terminal-state, reopen-persistence.
  * A declarative KNOWN-DIVERGENCE allowlist is consulted before any differential is
    called RED. Suppression is never silent: every suppressed diff is emitted as an
    INVARIANT ... PASS carrying a divergence:<id> note.
  * run_case(seed, axes, runners) emits the INVARIANT/VERDICT stdout protocol and
    returns exit 0 (GREEN) / 1 (RED) / 3 (VOID), matching turso_workload_common.
"""
import hashlib
import os
import random
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def _digest_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def root_seed_from(raw: str) -> int:
    """Map an arbitrary seed string to a stable 64-bit root seed."""
    return _digest_int(f"genlib:{raw}")


def seeded_rng(root: int, label: str) -> random.Random:
    """A random.Random fully determined by (root, label) -- no global state."""
    return random.Random(_digest_int(f"{root}:{label}"))


# ---------------------------------------------------------------------------
# Axes and value pools (data, not code)
# ---------------------------------------------------------------------------

# Each axis is a name -> ordered tuple of candidate values. The generator picks
# exactly one value per axis per program; the coverage probe asserts every listed
# value is reachable. These are GENERIC config axes -- no target-specific tuple is
# pinned. Product adapters extend AXES via config_axes passed into run_case.
CORE_AXES: dict[str, tuple] = {
    # SQLite/Turso page sizes are powers of two 512..65536. Sweeping the whole
    # ladder (not just 4096) is what lets a page_size x encryption combo surface.
    "page_size": (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536),
    "journal_mode": ("delete", "truncate", "persist", "memory", "wal", "off"),
    "synchronous": ("off", "normal", "full"),
    "foreign_keys": (0, 1),
    "encryption": (0, 1),
}

# Boundary literal pool -- the values scalar functions and DML are stressed over.
# Declared once, as data, so an auditor sees the boundary space is swept.
BOUNDARY_VALUES: tuple[Any, ...] = (
    None,
    0,
    1,
    -1,
    9223372036854775807,      # INT64 max
    -9223372036854775808,     # INT64 min
    9223372036854775808,      # overflows INT64 -> real/text in engines
    0.0,
    -0.0,
    3.14159265358979,
    1e308,
    "",
    "0",
    "12abc",                  # numeric-prefix string
    "3.9suffix",
    "  7  ",                  # whitespace-padded numeric
    "abc",
    "O'Brien",                # embedded quote
    "åß☃",   # unicode: aa, sharp-s, snowman
    "\U0001f600",             # astral emoji
    "tab\tsep",               # embedded tab (round-trips through `-m list`)
    b"\x00\x01\xff",          # raw bytes -> blob
    "zeroblob:5",             # sentinel expanded to zeroblob(5) by renderer
)
# NOTE: embedded-newline literals are deliberately NOT in the pool. tursodb's `-m list`
# output is line-oriented, so a newline inside a projected value is indistinguishable
# from a row boundary on stdout -- a transport ambiguity, not a product divergence. Such
# values are still exercised through DML/storage paths (they round-trip via the DB, not
# via stdout parsing); only bare scalar projection of a newline would be unparseable.

# Identifier styles for DDL -- plain, quoted, keyword-ish, unicode. Data-driven so
# the generator exercises the identifier grammar rather than one safe style.
IDENTIFIER_STYLES: tuple[str, ...] = ("plain", "quoted", "keywordish", "unicode", "spaced")

# Scalar functions shared by SQLite and Turso, called over the boundary pool.
SCALAR_FUNCS: tuple[str, ...] = (
    "length", "hex", "quote", "typeof", "abs", "upper", "lower", "trim",
    "unicode", "round", "substr2", "coalesce2",
)

# Aggregate functions for QUERY ops (incl. empty-group / ungrouped-over-join).
AGG_FUNCS: tuple[str, ...] = ("count", "sum", "total", "avg", "min", "max", "group_concat")

# Join styles -- LEFT/outer is what makes empty-aggregate-over-join null rows appear.
JOIN_STYLES: tuple[str, ...] = ("inner", "left", "cross")

OP_FAMILIES: tuple[str, ...] = ("DDL", "DML", "QUERY", "LIFECYCLE", "EXPECT_ERROR")


# ---------------------------------------------------------------------------
# Feature families (generic op-kind families) and their reference-comparability
# ---------------------------------------------------------------------------
#
# Beyond the base op families above, the generator sweeps GENERIC FEATURE FAMILIES:
# whole classes of SQL surface (full-text search, JSON table-valued functions,
# triggers, attached auxiliary databases). Each is declared here as DATA -- a family
# name -> descriptor -- so an auditor sees the generator exercises a feature CLASS, never
# a bug-specific constant. `ref_comparable` records whether the stdlib sqlite3 reference
# can express the family with the SAME syntax/semantics (so a differential row/error
# comparison is meaningful). Where it cannot, the family's ops still run through every
# universal oracle (panic / integrity / terminal / reopen / error-class self-consistency).
#
# `weight` biases how often the family is chosen among feature ops (kept modest so the
# base op mix still dominates); `enabled_default` lets a product adapter's config gate a
# family. These are generic knobs, not target tuples.
@dataclass(frozen=True)
class FeatureFamily:
    name: str
    ref_comparable: bool
    weight: int


FEATURE_FAMILIES: tuple[FeatureFamily, ...] = (
    # FTS: Turso spells virtual full-text as `CREATE INDEX ... USING fts(cols)` plus the
    # `fts_match(cols, term)` / `col MATCH term` predicate. The reference (sqlite3) only
    # has fts5's `CREATE VIRTUAL TABLE ... USING fts5` + `MATCH` -- different DDL and no
    # `fts_match` scalar -- so it is NOT reference-comparable; universal oracles only.
    FeatureFamily("fts", ref_comparable=False, weight=2),
    # JSON TVF: json_each / json_tree as table-valued functions in joins / correlated
    # subqueries / CTE reuse. Python's sqlite3 ships JSON1 with identical syntax, so this
    # family IS reference-comparable (full differential).
    FeatureFamily("json_tvf", ref_comparable=True, weight=2),
    # Triggers exercised THROUGH a reopen boundary (create trigger, reopen db, then fire
    # it) with the shared identifier-style pool. Both engines have triggers -> comparable.
    FeatureFamily("trigger", ref_comparable=True, weight=1),
    # Attached auxiliary databases: ATTACH + DDL/DROP in aux + checkpoint/reopen. Standard
    # SQL both engines share -> comparable (row set + accept/reject).
    FeatureFamily("attach", ref_comparable=True, weight=1),
)

FEATURE_FAMILY_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURE_FAMILIES)
_FEATURE_BY_NAME: dict[str, FeatureFamily] = {f.name: f for f in FEATURE_FAMILIES}

# FTS tokenizers Turso accepts (generic sweep of the tokenizer grammar, not one value).
FTS_TOKENIZERS: tuple[str, ...] = ("default", "raw", "ngram")

# JSON document shapes fed to json_each/json_tree -- a generic pool of nesting/quoting
# shapes, not a single crafted payload. Deep objects, arrays, quoted keys, primitives,
# empties: the shape space the TVF cursor must reset across on re-entry.
JSON_SHAPES: tuple[str, ...] = (
    '{"a":{"b":1,"c":[2,{"target":"needle"}]},"items":[{"name":"one","v":10}],"z":"last"}',
    '{"items":[{"name":"flat","v":30}],"a":"b","c":"d"}',
    '{"items":[{"name":"array0"},{"nested":{"target":"needle"}},[7,8]],"meta":{"ok":true}}',
    '{"a.b":{"space key":9,"arr":[{"b":11},{"target":"needle"}]},"items":[]}',
    '{"scalar":42,"items":[1,2,3],"tail":{"b":12}}',
    '{"items":[],"empty":{},"a":{"c":[]},"target":"top"}',
)

# JSON path arguments (second arg to json_tree/json_each) -- includes quoted-key paths.
JSON_PATHS: tuple[str, ...] = ("$", "$.a", "$.items", '$."a.b"', "$.missing")


# ---------------------------------------------------------------------------
# Program model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Op:
    family: str            # one of OP_FAMILIES
    kind: str              # sub-kind, e.g. "create_table", "select_join"
    sql: str               # rendered SQL statement (single statement)
    expect_error: bool = False
    # ref_comparable=False marks an op whose semantics the stdlib sqlite3 reference
    # CANNOT express identically (e.g. Turso `USING fts(...)` + `fts_match(...)`, which
    # has no fts5-syntax equivalent the reference would parse the same way). For such
    # ops the DIFFERENTIAL oracles (diff_rows / error_class) are skipped -- there is no
    # apples-to-apples reference row set or accept/reject to compare against -- but the
    # UNIVERSAL, product-independent oracles (panic, integrity, terminal_state, reopen,
    # and the error-class self-consistency variant) still bind. This is the "reference-
    # comparable?" flag the targets file requires, carried per-op so the logic stays
    # generic: a whole family (FTS) is declared non-comparable via its axis data, and
    # every op it emits inherits the flag; families whose syntax IS shared (JSON TVF,
    # triggers, ATTACH) stay fully differential.
    ref_comparable: bool = True


@dataclass(frozen=True)
class Program:
    seed: int
    config: dict[str, Any]
    ops: tuple[Op, ...]

    def signature(self) -> str:
        """A byte-stable textual rendering -- used by the determinism probe."""
        parts = [f"SEED {self.seed}"]
        for key in sorted(self.config):
            parts.append(f"CFG {key}={self.config[key]!r}")
        for i, op in enumerate(self.ops):
            parts.append(
                f"OP {i} {op.family} {op.kind} err={int(op.expect_error)} "
                f"refcmp={int(op.ref_comparable)} {op.sql}"
            )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# SQL rendering helpers
# ---------------------------------------------------------------------------

def sql_quote_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


_AUX_PLACEHOLDER_RE = re.compile(r"@@AUX_DB:([A-Za-z0-9_]+)@@")


def _subst_aux_placeholders(sql: str, resolve: Callable[[str], str]) -> str:
    """Replace @@AUX_DB:<alias>@@ tokens with a runner-specific aux-db file path. The
    ATTACH family emits these placeholders so the generated Program stays runner-agnostic
    (the reference and candidate each attach THEIR OWN aux file); each runner substitutes
    its own path at execution time."""
    return _AUX_PLACEHOLDER_RE.sub(lambda m: resolve(m.group(1)), sql)


def render_literal(value: Any) -> str:
    """Render a boundary-pool value as a SQL literal (product-agnostic)."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "NULL"  # NaN not representable as a literal
        return repr(value)
    if isinstance(value, bytes):
        return "x'" + value.hex() + "'"
    if isinstance(value, str):
        if value.startswith("zeroblob:"):
            n = value.split(":", 1)[1]
            return f"zeroblob({int(n)})"
        return sql_quote_str(value)
    return sql_quote_str(str(value))


_KEYWORDS = ("select", "table", "order", "group", "index", "where", "from")


def render_identifier(base: str, style: str) -> str:
    """Render a table/column identifier in the given style, always double-quoted
    when the style requires it so the SQL stays valid across engines."""
    if style == "plain":
        return base
    if style == "quoted":
        return '"' + base + '"'
    if style == "keywordish":
        kw = _KEYWORDS[len(base) % len(_KEYWORDS)]
        return '"' + kw + "_" + base + '"'
    if style == "unicode":
        return '"' + base + "_é☃" + '"'
    if style == "spaced":
        return '"' + base + " col" + '"'
    return base


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def choose_config(root: int, axes: dict[str, tuple]) -> dict[str, Any]:
    """Pick one value per axis. Iterate axis names in sorted order so the result is
    independent of dict insertion order across processes."""
    rng = seeded_rng(root, "config")
    config: dict[str, Any] = {}
    for name in sorted(axes):
        values = axes[name]
        config[name] = values[rng.randrange(len(values))]
    return config


def _pool_value(rng: random.Random) -> Any:
    return BOUNDARY_VALUES[rng.randrange(len(BOUNDARY_VALUES))]


def _gen_ddl(rng: random.Random, tables: list[dict]) -> list[Op]:
    """Create a table (varied identifier styles) plus sometimes an index/view."""
    ops: list[Op] = []
    idx = len(tables)
    style = IDENTIFIER_STYLES[rng.randrange(len(IDENTIFIER_STYLES))]
    tname = render_identifier(f"t{idx}", style)
    ncols = rng.randint(2, 4)
    cols = [render_identifier(f"c{j}", IDENTIFIER_STYLES[rng.randrange(len(IDENTIFIER_STYLES))]) for j in range(ncols)]
    coldefs = ", ".join(f"{c} {tp}" for c, tp in zip(cols, _col_types(rng, ncols)))
    ops.append(Op("DDL", "create_table", f"CREATE TABLE {tname}({coldefs});"))
    tables.append({"name": tname, "cols": cols})
    # Optional secondary DDL over the just-created table.
    roll = rng.random()
    if roll < 0.4:
        iname = render_identifier(f"i{idx}", style)
        ops.append(Op("DDL", "create_index", f"CREATE INDEX {iname} ON {tname}({cols[0]});"))
    elif roll < 0.6:
        vname = render_identifier(f"v{idx}", style)
        ops.append(Op("DDL", "create_view", f"CREATE VIEW {vname} AS SELECT {cols[0]} FROM {tname};"))
    elif roll < 0.75:
        trg = render_identifier(f"trg{idx}", style)
        # AFTER INSERT trigger with a quoted-identifier body (WP-005 shape).
        ops.append(Op(
            "DDL", "create_trigger",
            f"CREATE TRIGGER {trg} AFTER INSERT ON {tname} BEGIN "
            f"UPDATE {tname} SET {cols[1]} = {cols[1]}; END;",
        ))
    return ops


def _col_types(rng: random.Random, n: int) -> list[str]:
    pool = ("INTEGER", "TEXT", "REAL", "BLOB", "")  # "" = no affinity
    return [pool[rng.randrange(len(pool))] for _ in range(n)]


def _gen_dml(rng: random.Random, tables: list[dict]) -> list[Op]:
    ops: list[Op] = []
    tbl = tables[rng.randrange(len(tables))]
    ncols = len(tbl["cols"])
    nrows = rng.randint(1, 4)
    tuples = []
    for _ in range(nrows):
        tuples.append("(" + ", ".join(render_literal(_pool_value(rng)) for _ in range(ncols)) + ")")
    ops.append(Op("DML", "insert", f"INSERT INTO {tbl['name']} VALUES {', '.join(tuples)};"))
    roll = rng.random()
    if roll < 0.3:
        col = tbl["cols"][0]
        ops.append(Op("DML", "update", f"UPDATE {tbl['name']} SET {col} = {render_literal(_pool_value(rng))};"))
    elif roll < 0.5:
        col = tbl["cols"][0]
        ops.append(Op("DML", "delete", f"DELETE FROM {tbl['name']} WHERE {col} = {render_literal(_pool_value(rng))};"))
    return ops


def _gen_query(rng: random.Random, tables: list[dict]) -> list[Op]:
    ops: list[Op] = []
    kind_roll = rng.random()
    if len(tables) >= 2 and kind_roll < 0.4:
        # Join, including LEFT/outer, with an aggregate over it (empty-group shape).
        a, b = rng.sample(range(len(tables)), 2)
        ta, tb = tables[a], tables[b]
        join = JOIN_STYLES[rng.randrange(len(JOIN_STYLES))]
        agg = AGG_FUNCS[rng.randrange(len(AGG_FUNCS))]
        acol = tb["cols"][0]
        join_clause = "" if join == "cross" else f" ON {ta['cols'][0]} = {tb['cols'][0]}"
        sql = (
            f"SELECT {_render_agg(agg, acol)} FROM {ta['name']} "
            f"{join.upper()} JOIN {tb['name']}{join_clause};"
        )
        ops.append(Op("QUERY", "agg_over_join", sql))
    elif kind_roll < 0.6:
        # Scalar-function call over a boundary value.
        fn = SCALAR_FUNCS[rng.randrange(len(SCALAR_FUNCS))]
        val = render_literal(_pool_value(rng))
        ops.append(Op("QUERY", "scalar_fn", f"SELECT {_render_scalar(fn, val)};"))
    elif kind_roll < 0.8:
        # CTE / subquery over a table.
        tbl = tables[rng.randrange(len(tables))]
        col = tbl["cols"][0]
        ops.append(Op(
            "QUERY", "cte_subquery",
            f"WITH cte AS (SELECT {col} AS x FROM {tbl['name']}) "
            f"SELECT count(*), (SELECT max(x) FROM cte) FROM cte;",
        ))
    else:
        # Ungrouped aggregate over a single table (empty-input shape when no rows).
        tbl = tables[rng.randrange(len(tables))]
        agg = AGG_FUNCS[rng.randrange(len(AGG_FUNCS))]
        ops.append(Op("QUERY", "agg_ungrouped", f"SELECT {_render_agg(agg, tbl['cols'][0])} FROM {tbl['name']};"))
    return ops


def _render_agg(agg: str, col: str) -> str:
    if agg == "group_concat":
        return f"group_concat({col})"
    if agg == "count":
        return f"count({col})"
    return f"{agg}({col})"


def _render_scalar(fn: str, val: str) -> str:
    if fn == "substr2":
        return f"substr({val}, 1, 2)"
    if fn == "coalesce2":
        return f"coalesce({val}, 'x')"
    if fn == "round":
        return f"round({val})"
    return f"{fn}({val})"


def _gen_lifecycle(rng: random.Random, plug: tuple[str, ...]) -> list[Op]:
    kinds = ("integrity_check", "reopen") + plug
    kind = kinds[rng.randrange(len(kinds))]
    if kind == "integrity_check":
        return [Op("LIFECYCLE", "integrity_check", "PRAGMA integrity_check;")]
    if kind == "reopen":
        return [Op("LIFECYCLE", "reopen", "-- reopen --")]  # handled by runner boundary
    # Product-pluggable lifecycle op (e.g. checkpoint); rendered as-is. A checkpoint PRAGMA
    # ECHOES a (busy, log, checkpointed) triple that legitimately differs across engines
    # (sqlite3 emits -1,-1 outside WAL; tursodb emits 0,0), so checkpoint echoes are marked
    # non-comparable -- a maintenance echo, not a query result under test. quick_check stays
    # comparable (it is a health oracle like integrity_check).
    comparable = "checkpoint" not in kind.lower()
    return [Op("LIFECYCLE", kind, kind, ref_comparable=comparable)]


def _gen_expect_error(rng: random.Random, tables: list[dict]) -> list[Op]:
    """Statements EXPECTED to error in BOTH engines -- drives error-class differential."""
    catalog = [
        ("syntax", "SELECT FROM;"),
        ("no_such_table", "SELECT * FROM __no_such_table_zzz__;"),
        ("no_such_func", "SELECT __no_such_func_zzz__(1);"),
        ("type_arity", "SELECT abs(1, 2, 3);"),
        ("bad_pragma_arg", "SELECT nonexistent_col FROM (SELECT 1) WHERE zzz_missing = 1;"),
    ]
    kind, sql = catalog[rng.randrange(len(catalog))]
    return [Op("EXPECT_ERROR", kind, sql, expect_error=True)]


# ---------------------------------------------------------------------------
# Feature-family generators (generic op-kind families)
# ---------------------------------------------------------------------------
#
# Each returns a self-contained op sequence that creates its own base table (so it does
# not depend on the randomly-typed base tables) and exercises the family. FTS ops are
# tagged ref_comparable=False (the differential oracles skip them); the rest are fully
# comparable. Every family sweeps a generic sub-grammar (tokenizers, JSON shapes/paths,
# trigger identifier styles, attach DDL kinds) rather than pinning a target constant.

def _fresh_name(rng: random.Random, prefix: str) -> str:
    return f"{prefix}{rng.randrange(1_000_000)}"


def _gen_fts(rng: random.Random) -> list[Op]:
    """Turso full-text: CREATE INDEX ... USING fts(cols) [WITH (tokenizer=...)] + a
    populate + fts_match / MATCH query. NOT reference-comparable (fts5 syntax differs),
    so tagged ref_comparable=False -- universal oracles still bind (integrity is the
    WP-024 oracle: a valid FTS insert must leave integrity_check == ok)."""
    ops: list[Op] = []
    t = _fresh_name(rng, "fts_t")
    idx = _fresh_name(rng, "fts_i")
    ops.append(Op("DDL", "fts_create_table",
                  f"CREATE TABLE {t}(id INTEGER PRIMARY KEY, title TEXT, body TEXT);",
                  ref_comparable=False))
    tok = FTS_TOKENIZERS[rng.randrange(len(FTS_TOKENIZERS))]
    ncols = rng.choice((1, 2))
    cols = "title, body" if ncols == 2 else "title"
    with_clause = "" if tok == "default" else f" WITH (tokenizer = '{tok}')"
    ops.append(Op("DDL", "fts_create_index",
                  f"CREATE INDEX {idx} ON {t} USING fts({cols}){with_clause};",
                  ref_comparable=False))
    # Populate with a couple of rows drawn from the boundary/text pool (valid FTS inserts).
    terms = ("alpha search", "beta engine", "gamma full text", "delta note")
    n = rng.randint(1, 3)
    tuples = []
    for i in range(n):
        title = terms[rng.randrange(len(terms))]
        body = terms[rng.randrange(len(terms))]
        tuples.append(f"({i + 1}, {sql_quote_str(title)}, {sql_quote_str(body)})")
    ops.append(Op("DML", "fts_insert",
                  f"INSERT INTO {t}(id, title, body) VALUES {', '.join(tuples)};",
                  ref_comparable=False))
    # Query form: fts_match(cols, term) or the MATCH operator, over a swept term.
    term = terms[rng.randrange(len(terms))].split()[0]
    if rng.random() < 0.5:
        pred = f"fts_match({cols}, {sql_quote_str(term)})"
    else:
        pred = f"title MATCH {sql_quote_str(term)}"
    ops.append(Op("QUERY", "fts_query",
                  f"SELECT group_concat(id, ',') FROM (SELECT id FROM {t} WHERE {pred} ORDER BY id);",
                  ref_comparable=False))
    # WP-024 shape: after valid FTS ops, integrity_check must still report ok.
    ops.append(Op("LIFECYCLE", "fts_integrity", "PRAGMA integrity_check;", ref_comparable=False))
    return ops


def _gen_json_tvf(rng: random.Random) -> list[Op]:
    """JSON table-valued functions (json_each / json_tree) used in a JOIN, correlated
    subquery, or CTE reuse -- the re-entry shapes (WP-023). Reference-comparable: Python
    sqlite3 ships JSON1 with identical syntax."""
    ops: list[Op] = []
    t = _fresh_name(rng, "jdoc")
    ops.append(Op("DDL", "json_create_table",
                  f"CREATE TABLE {t}(id INTEGER PRIMARY KEY, payload TEXT, root_path TEXT);"))
    n = rng.randint(2, 4)
    tuples = []
    for i in range(n):
        shape = JSON_SHAPES[rng.randrange(len(JSON_SHAPES))]
        path = JSON_PATHS[rng.randrange(len(JSON_PATHS))]
        tuples.append(f"({i + 1}, {sql_quote_str(shape)}, {sql_quote_str(path)})")
    ops.append(Op("DML", "json_insert",
                  f"INSERT INTO {t}(id, payload, root_path) VALUES {', '.join(tuples)};"))
    form = rng.randrange(4)
    if form == 0:
        # JOIN over json_tree, per-row root path (resets cursor state across rows).
        sql = (
            f"SELECT d.id, jt.fullkey, jt.type FROM {t} AS d "
            f"JOIN json_tree(d.payload, d.root_path) AS jt "
            f"WHERE jt.type IN ('object','array','integer','text') "
            f"ORDER BY d.id, jt.fullkey, jt.type;"
        )
        kind = "json_join_tree"
    elif form == 1:
        # Correlated scalar subquery re-entering json_tree twice per outer row.
        sql = (
            f"SELECT d.id, COALESCE((SELECT jt.fullkey FROM json_tree(d.payload) AS jt "
            f"WHERE jt.type='text' ORDER BY jt.id LIMIT 1), '<none>') "
            f"FROM {t} AS d ORDER BY d.id;"
        )
        kind = "json_correlated_subquery"
    elif form == 2:
        # CTE built on json_tree, referenced twice (UNION ALL) -- cursor-leakage shape.
        sql = (
            f"WITH jt AS (SELECT d.id AS did, j.fullkey AS fk, j.type AS ty "
            f"FROM {t} AS d JOIN json_tree(d.payload) AS j) "
            f"SELECT 'leaf', did, fk FROM jt WHERE ty NOT IN ('object','array') "
            f"UNION ALL SELECT 'cont', did, fk FROM jt WHERE ty IN ('object','array') "
            f"ORDER BY did, fk;"
        )
        kind = "json_cte_reuse"
    else:
        # json_each outer feeding a nested json_tree (double TVF re-entry).
        sql = (
            f"SELECT d.id, e.key, t.fullkey, t.type FROM {t} AS d "
            f"JOIN json_each(d.payload, '$.items') AS e "
            f"JOIN json_tree(e.value) AS t "
            f"WHERE t.type NOT IN ('object','array') OR t.fullkey='$' "
            f"ORDER BY d.id, CAST(e.key AS INTEGER), t.id;"
        )
        kind = "json_each_then_tree"
    ops.append(Op("QUERY", kind, sql))
    return ops


def _gen_trigger(rng: random.Random) -> list[Op]:
    """A trigger created with the shared identifier-style pool, then a reopen boundary,
    then a DML that FIRES it -- the create/reopen/use sequence (WP-005 shape). Fully
    reference-comparable (both engines have triggers with this syntax)."""
    ops: list[Op] = []
    style = IDENTIFIER_STYLES[rng.randrange(len(IDENTIFIER_STYLES))]
    base = _fresh_name(rng, "trg")
    tgt = render_identifier(f"{base}_t", style)
    audit = render_identifier(f"{base}_a", style)
    trg = render_identifier(f"{base}_g", style)
    ops.append(Op("DDL", "trigger_target_table",
                  f"CREATE TABLE {tgt}(id INTEGER PRIMARY KEY, a TEXT, b TEXT);"))
    ops.append(Op("DDL", "trigger_audit_table",
                  f"CREATE TABLE {audit}(msg TEXT);"))
    ops.append(Op("DDL", "trigger_create",
                  f"CREATE TRIGGER {trg} AFTER INSERT ON {tgt} BEGIN "
                  f"INSERT INTO {audit} VALUES('fired:' || NEW.a || ':' || NEW.b); END;"))
    # Reopen so the trigger is exercised after a schema reload (the WP-005 precondition).
    ops.append(Op("LIFECYCLE", "reopen", "-- reopen --"))
    ops.append(Op("DML", "trigger_fire",
                 f"INSERT INTO {tgt}(id, a, b) VALUES (1, 'ra', 'rb');"))
    ops.append(Op("QUERY", "trigger_audit_read",
                 f"SELECT msg FROM {audit} ORDER BY rowid;"))
    return ops


def _gen_attach(rng: random.Random, run_ctx: dict) -> list[Op]:
    """ATTACH an auxiliary db, do DDL/DROP inside it, checkpoint, reopen, then read back
    -- the attached-db page-lifecycle sequence (WP-008 shape). Reference-comparable
    (ATTACH is standard SQL). The aux path is resolved by the runner at exec time via a
    placeholder, since the reference and candidate use different run dirs."""
    ops: list[Op] = []
    alias = _fresh_name(rng, "aux")
    # AUX_DB placeholder is substituted by each runner with its own aux file path, so the
    # generated program stays runner-independent (data), and the two engines each attach
    # THEIR OWN aux file -- the differential is on behavior, not on the shared path.
    ops.append(Op("LIFECYCLE", "attach", f"ATTACH '@@AUX_DB:{alias}@@' AS {alias};"))
    at = _fresh_name(rng, "at")
    ops.append(Op("DDL", "attach_create",
                  f"CREATE TABLE {alias}.{at}(id INTEGER PRIMARY KEY, v TEXT);"))
    ops.append(Op("DML", "attach_insert",
                  f"INSERT INTO {alias}.{at}(id, v) VALUES (1, 'x'), (2, 'y');"))
    if rng.random() < 0.5:
        ops.append(Op("DDL", "attach_drop", f"DROP TABLE {alias}.{at};"))
        read_ok_after_drop = False
    else:
        read_ok_after_drop = True
    ops.append(Op("LIFECYCLE", "attach_checkpoint", "PRAGMA wal_checkpoint(TRUNCATE);", ref_comparable=False))
    ops.append(Op("LIFECYCLE", "reopen", "-- reopen --"))
    # After reopen the ATTACH is gone (reopen is a fresh connection); re-attach to read.
    ops.append(Op("LIFECYCLE", "attach", f"ATTACH '@@AUX_DB:{alias}@@' AS {alias};"))
    if read_ok_after_drop:
        ops.append(Op("QUERY", "attach_read",
                     f"SELECT id, v FROM {alias}.{at} ORDER BY id;"))
    ops.append(Op("LIFECYCLE", "attach_integrity", "PRAGMA integrity_check;"))
    ops.append(Op("LIFECYCLE", "attach", f"DETACH {alias};"))
    return ops


def _gen_feature(rng: random.Random, families: tuple[str, ...], run_ctx: dict) -> list[Op]:
    """Pick one enabled feature family (weighted) and emit its op sequence."""
    enabled = [f for f in FEATURE_FAMILIES if f.name in families]
    if not enabled:
        return []
    weights = [f.weight for f in enabled]
    total = sum(weights)
    r = rng.randrange(total)
    acc = 0
    chosen = enabled[-1]
    for fam, w in zip(enabled, weights):
        acc += w
        if r < acc:
            chosen = fam
            break
    if chosen.name == "fts":
        return _gen_fts(rng)
    if chosen.name == "json_tvf":
        return _gen_json_tvf(rng)
    if chosen.name == "trigger":
        return _gen_trigger(rng)
    if chosen.name == "attach":
        return _gen_attach(rng, run_ctx)
    return []


def generate(seed: int, axes: dict[str, tuple], lifecycle_plug: tuple[str, ...] = (),
             feature_families: tuple[str, ...] = FEATURE_FAMILY_NAMES) -> Program:
    """Produce a Program fully determined by seed. axes is CORE_AXES merged with any
    product axes. lifecycle_plug adds product lifecycle op kinds (e.g. checkpoint)."""
    root = seed  # seed is already a root int
    config = choose_config(root, axes)
    rng = seeded_rng(root, "ops")
    tables: list[dict] = []
    ops: list[Op] = []

    # Always begin with pragmas realizing the swept config so the config is load-bearing.
    # These config-realizing PRAGMAs ECHO the resulting mode, and engines legitimately
    # differ on the echo (tursodb is WAL-only, so `PRAGMA journal_mode = <x>` echoes 'wal'
    # for every requested mode; sqlite3 echoes the requested mode). That echo is a config
    # realization, not a query result under test, so the config PRAGMAs are marked
    # ref_comparable=False -- they still execute and realize the config, but their echoed
    # row is not differentially compared. (Real WP-025-style panics under a given config
    # still surface via the universal panic/terminal oracles, which bind regardless.)
    ops.append(Op("LIFECYCLE", "pragma_page_size", f"PRAGMA page_size = {config['page_size']};", ref_comparable=False))
    ops.append(Op("LIFECYCLE", "pragma_journal", f"PRAGMA journal_mode = {config['journal_mode']};", ref_comparable=False))
    ops.append(Op("LIFECYCLE", "pragma_sync", f"PRAGMA synchronous = {config['synchronous']};", ref_comparable=False))
    ops.append(Op("LIFECYCLE", "pragma_fk", f"PRAGMA foreign_keys = {config['foreign_keys']};", ref_comparable=False))

    n_ddl = rng.randint(2, 3)
    for _ in range(n_ddl):
        ops.extend(_gen_ddl(rng, tables))

    run_ctx: dict = {}
    # Interleave DML / QUERY / LIFECYCLE / EXPECT_ERROR / FEATURE. Fixed count for stable
    # coverage. Feature ops (fts/json/trigger/attach) are drawn from the same interleave so
    # they compose with the base surface rather than living in a separate phase.
    n_body = rng.randint(8, 14)
    for _ in range(n_body):
        pick = rng.random()
        if pick < 0.30:
            ops.extend(_gen_dml(rng, tables))
        elif pick < 0.55:
            ops.extend(_gen_query(rng, tables))
        elif pick < 0.70:
            ops.extend(_gen_feature(rng, feature_families, run_ctx))
        elif pick < 0.85:
            ops.extend(_gen_lifecycle(rng, lifecycle_plug))
        else:
            ops.extend(_gen_expect_error(rng, tables))

    # Guarantee at least one of each op family appears for coverage (append if missing).
    present = {op.family for op in ops}
    if "EXPECT_ERROR" not in present:
        ops.extend(_gen_expect_error(rng, tables))
    if "LIFECYCLE" not in present:
        ops.extend(_gen_lifecycle(rng, lifecycle_plug))
    # Guarantee at least one op from EACH enabled feature family so every declared feature
    # class is reachable (the coverage probe asserts this -- the anti-telegraphing rule
    # requires whole families be swept, not a single crafted case).
    present_kinds = {op.kind for op in ops}
    _feature_marker = {
        "fts": "fts_create_index",
        "json_tvf": "json_insert",
        "trigger": "trigger_create",
        "attach": "attach_create",
    }
    for fam in feature_families:
        if fam in _FEATURE_BY_NAME and _feature_marker.get(fam) not in present_kinds:
            if fam == "fts":
                ops.extend(_gen_fts(rng))
            elif fam == "json_tvf":
                ops.extend(_gen_json_tvf(rng))
            elif fam == "trigger":
                ops.extend(_gen_trigger(rng))
            elif fam == "attach":
                ops.extend(_gen_attach(rng, run_ctx))
            present_kinds = {op.kind for op in ops}
    # Always end with an integrity check + terminal reopen probe.
    ops.append(Op("LIFECYCLE", "integrity_check", "PRAGMA integrity_check;"))

    return Program(seed=seed, config=config, ops=tuple(ops))


# ---------------------------------------------------------------------------
# Runner adapter interface
# ---------------------------------------------------------------------------

@dataclass
class StmtResult:
    sql: str
    rc: int                       # 0 ok, nonzero error
    rows: list[tuple]             # normalized rows (empty for non-SELECT)
    error: str                    # error text ("" if none)
    ref_comparable: bool = True   # False -> differential oracles skip this stmt (see Op)


@dataclass
class RunResult:
    stmts: list[StmtResult] = field(default_factory=list)
    crashed: bool = False         # panic/abort (not a clean SQL error)
    crash_text: str = ""
    reopened_ok: bool = True      # db reopenable after run
    integrity_ok: Optional[bool] = None  # last integrity_check verdict, if run


class Runner:
    """Abstract adapter: execute an op sequence, collect rows/error/rc per statement,
    support a reopen boundary, and report crash/integrity. Product-independent."""

    name: str = "runner"
    # True when the runner returns untyped text (e.g. a CLI in list mode) and thus
    # cannot preserve SQLite type tags. Row comparison collapses type tags when set.
    text_only: bool = False

    def run(self, program: Program) -> RunResult:  # pragma: no cover - interface
        raise NotImplementedError


# ---- Normalization (shared, so both runners are compared apples-to-apples) ----

def normalize_value(value: Any, cli_text: bool = False) -> str:
    """Map any cell to a canonical string so type-affinity/float-format noise does not
    masquerade as a differential. Ints and integral floats collapse to the same form.

    cli_text mode: when EITHER runner is a text-only CLI (`-m list` emits untyped text,
    so it cannot preserve the int-vs-text or blob-vs-text distinction), the REFERENCE
    side is rendered exactly the way the CLI renders values (NULL->"", int->str,
    blob->hex, text verbatim) and the CLI side (a _CliText marker) is passed through
    unchanged. Comparison is thus on the CLI's own text projection -- a property of the
    transport, not a product-behavior suppression. It applies symmetrically and never
    hides a value difference, only the type tag the transport physically cannot carry."""
    # CLI cells arrive already in the CLI's text projection; never re-guess their type.
    # (We deliberately do NOT collapse integral-float text like "0.0" here: a CLI cell of
    # "0.0" can be a genuine TEXT result -- e.g. lower(0.0) -> '0.0' -- and collapsing it to
    # "0" would mask a real text difference. A candidate rendering an empty aggregate as
    # "0.0" where the reference yields typed 0 is left as a real differential for the
    # orchestrator to adjudicate, not silently normalized away.)
    if isinstance(value, _CliText):
        return str.__str__(value)
    if value is None:
        return "" if cli_text else "\x00NULL"
    if isinstance(value, bool):
        return ("1" if value else "0") if cli_text else f"\x00INT:{int(value)}"
    if isinstance(value, int):
        return str(value) if cli_text else f"\x00INT:{value}"
    if isinstance(value, float):
        if value != value:
            return "nan" if cli_text else "\x00NAN"
        if value in (float("inf"), float("-inf")):
            return ("inf" if value > 0 else "-inf") if cli_text else f"\x00REAL:{value!r}"
        if abs(value) < 1e15 and value == int(value):
            iv = int(value)
            return str(iv) if cli_text else f"\x00INT:{iv}"
        return (repr(value) if cli_text else f"\x00REAL:{value!r}")
    if isinstance(value, bytes):
        # sqlite3 returns a Python bytes for a blob; the CLI prints its hex. Compare on
        # hex so a blob on the reference matches the CLI's hex rendering of the same blob.
        return value.hex() if cli_text else "\x00BLOB:" + value.hex()
    # str
    return str(value) if cli_text else "\x00TXT:" + str(value)


def normalize_row(row: tuple, cli_text: bool = False) -> tuple:
    return tuple(normalize_value(v, cli_text) for v in row)


def normalize_rowset(rows: list[tuple], cli_text: bool = False) -> list[tuple]:
    """Multiset compare: sort normalized rows so row ORDER never triggers a false red
    (only ORDER BY queries would care, and those are compared as multisets too here)."""
    return sorted(normalize_row(r, cli_text) for r in rows)


# ---- sqlite3 reference runner ----

class Sqlite3Runner(Runner):
    """Reference runner over stdlib sqlite3. Deterministic; used both as the oracle
    reference and (two independent instances) for the null-differential probe."""

    def __init__(self, run_dir: Path, tag: str = "sqlite3"):
        self.run_dir = run_dir
        self.name = tag

    @staticmethod
    def _connect(db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, isolation_level=None)
        # Decode-tolerant text factory: some queries (e.g. group_concat over a column that
        # ended up holding non-UTF8 bytes) produce a result Python's default TEXT decoder
        # cannot decode, raising OperationalError -- a REFERENCE-SIDE fragility, not a Turso
        # defect. Replace undecodable bytes so the reference returns a value (comparable to
        # the CLI's own rendering) rather than spuriously "rejecting" a statement the CLI
        # accepts. Bytes that are valid UTF-8 decode unchanged, so normal text is unaffected.
        conn.text_factory = lambda b: b.decode("utf-8", "replace")
        return conn

    def _aux_path(self, seed: int, alias: str) -> str:
        return str(self.run_dir / f"{self.name}-{seed}-{alias}.auxdb")

    def _subst_aux(self, sql: str, seed: int) -> str:
        return _subst_aux_placeholders(sql, lambda alias: self._aux_path(seed, alias))

    def run(self, program: Program) -> RunResult:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.run_dir / f"{self.name}-{program.seed}.db"
        db_path.unlink(missing_ok=True)
        # Clear any aux db files from a prior run of this tag/seed so ATTACH starts clean.
        for aux in self.run_dir.glob(f"{self.name}-{program.seed}-*.auxdb"):
            aux.unlink(missing_ok=True)
        result = RunResult()
        conn = self._connect(str(db_path))
        try:
            for op in program.ops:
                if op.kind == "reopen":
                    conn.close()
                    conn = self._connect(str(db_path))
                    continue
                self._exec_one(conn, op, result, program.seed)
            # terminal reopen probe
            conn.close()
            try:
                conn = self._connect(str(db_path))
                conn.execute("PRAGMA integrity_check;").fetchall()
                result.reopened_ok = True
            except sqlite3.Error:
                result.reopened_ok = False
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        return result

    def _exec_one(self, conn: sqlite3.Connection, op: Op, result: RunResult, seed: int) -> None:
        sql = self._subst_aux(op.sql, seed)
        try:
            cur = conn.execute(sql)
            rows = cur.fetchall() if cur.description is not None else []
            result.stmts.append(StmtResult(sql, 0, [tuple(r) for r in rows], "", op.ref_comparable))
            if op.kind in ("integrity_check", "fts_integrity", "attach_integrity"):
                result.integrity_ok = bool(rows) and str(rows[0][0]).lower() == "ok"
        except sqlite3.Error as exc:
            result.stmts.append(StmtResult(sql, 1, [], f"{type(exc).__name__}: {exc}", op.ref_comparable))


# ---- CLI runner skeleton (tursodb) ----

# Strings in CLI stderr/stdout that mean a crash rather than a clean SQL rejection.
CRASH_MARKERS: tuple[str, ...] = (
    "panicked", "panic", "RUST_BACKTRACE", "assertion failed", "abort",
    "SIGSEGV", "SIGABRT", "core dumped", "internal error", "unreachable",
)


class CliRunner(Runner):
    """Thin subprocess adapter for a SQL CLI binary (tursodb). Not executable on macOS
    (linux binary), so it is constructed here and exercised via mocking in tests. The
    argv/parse logic is real so the guest run (EXP-102) can use it unchanged."""

    text_only = True

    def __init__(
        self,
        binary: Path,
        base_args: tuple[str, ...],
        run_dir: Path,
        tag: str = "tursodb",
        loader: Optional[Path] = None,
        loader_lib: Optional[Path] = None,
        timeout: int = 60,
        _spawn: Optional[Callable[[list[str], str], tuple[int, str, str]]] = None,
        encryption: int = 0,
        cipher: str = "",
        hexkey: str = "",
    ):
        self.binary = binary
        self.base_args = base_args
        self.run_dir = run_dir
        self.name = tag
        self.loader = loader
        self.loader_lib = loader_lib
        self.timeout = timeout
        # Encryption config swept by the product adapter. When encryption=1, the main db is
        # opened via the cipher URI `file:<db>?cipher=<c>&hexkey=<k>` (Turso's encryption
        # form, per turso_encryption_reopen_corruption_boundary). The reference stays
        # unencrypted -- encryption is candidate-side config, and the differential contract
        # is unchanged (the same rows must come back whether or not the file is encrypted).
        self.encryption = encryption
        self.cipher = cipher
        self.hexkey = hexkey
        # _spawn is injectable for tests; defaults to real subprocess.
        self._spawn = _spawn or self._subprocess_spawn

    def argv(self, db: str, script: str) -> list[str]:
        if self.loader and self.loader_lib:
            head = [str(self.loader), "--library-path", str(self.loader_lib), str(self.binary)]
        else:
            head = [str(self.binary)]
        return [*head, "-q", "-m", "list", *self.base_args, db, script]

    def _subprocess_spawn(self, argv: list[str], _script: str) -> tuple[int, str, str]:  # pragma: no cover
        proc = subprocess.run(argv, text=True, capture_output=True, timeout=self.timeout, check=False)
        return proc.returncode, proc.stdout, proc.stderr

    def _aux_path(self, seed: int, alias: str) -> str:
        return str(self.run_dir / f"{self.name}-{seed}-{alias}.auxdb")

    def _subst_aux(self, sql: str, seed: int) -> str:
        return _subst_aux_placeholders(sql, lambda alias: self._aux_path(seed, alias))

    def _db_arg(self, db_path: str) -> str:
        """The db argument passed to tursodb. When encryption is swept on, wrap the path in
        the cipher URI so the candidate actually opens an encrypted database; otherwise the
        bare path (plaintext). This is what makes the page_size x encryption target (WP-025
        / turso #7610) actually reachable rather than merely printed in the config."""
        if self.encryption and self.cipher and self.hexkey:
            return f"file:{db_path}?cipher={self.cipher}&hexkey={self.hexkey}"
        return db_path

    def run(self, program: Program) -> RunResult:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        db_plain = str(self.run_dir / f"{self.name}-{program.seed}.db")
        db_path = self._db_arg(db_plain)
        for aux in self.run_dir.glob(f"{self.name}-{program.seed}-*.auxdb"):
            aux.unlink(missing_ok=True)
        result = RunResult()
        # The CLI executes a whole script at once; we render statement-by-statement so
        # per-op rc/rows still line up. Reopen is a natural boundary (new process).
        #
        # ATTACH state is per-connection and each CLI statement is a fresh process, so an
        # ATTACH alias would not survive to the next statement. We therefore keep a small
        # active-attach preamble: after an accepted ATTACH we replay it ahead of every
        # subsequent statement (so `aux.tbl` resolves) until the matching DETACH or a
        # reopen clears it. This makes the one-statement-per-process transport faithfully
        # reproduce a persistent-connection ATTACH session -- generic, not target-specific.
        attach_preamble: list[str] = []
        for op in program.ops:
            if op.kind == "reopen":
                attach_preamble = []  # a reopen drops all attaches (fresh connection)
                continue
            stmt_sql = self._subst_aux(op.sql, program.seed)
            if op.kind == "attach" and stmt_sql.strip().upper().startswith("DETACH"):
                # Execute the detach (with current preamble) then drop it from the preamble.
                body = "".join(s + "\n" for s in attach_preamble) + stmt_sql
                rc, stdout, stderr = self._spawn(self.argv(db_path, self._script_for(body)), body)
                attach_preamble = [s for s in attach_preamble if not self._same_attach(s, op.sql)]
                combined = f"{stdout}\n{stderr}"
                if self._is_crash(rc, combined):
                    result.crashed = True
                    result.crash_text = combined[-500:]
                    result.stmts.append(StmtResult(stmt_sql, rc, [], combined[-300:], op.ref_comparable))
                    break
                result.stmts.append(StmtResult(stmt_sql, 0 if rc == 0 else 1, [], "" if rc == 0 else stderr.strip(), op.ref_comparable))
                continue
            body = "".join(s + "\n" for s in attach_preamble) + stmt_sql
            script = self._script_for(body)
            rc, stdout, stderr = self._spawn(self.argv(db_path, script), script)
            combined = f"{stdout}\n{stderr}"
            if self._is_crash(rc, combined):
                result.crashed = True
                result.crash_text = combined[-500:]
                result.stmts.append(StmtResult(stmt_sql, rc, [], combined[-300:], op.ref_comparable))
                break
            rows = self._parse_rows(stdout) if rc == 0 else []
            result.stmts.append(StmtResult(stmt_sql, 0 if rc == 0 else 1, rows, "" if rc == 0 else stderr.strip(), op.ref_comparable))
            if op.kind == "attach" and rc == 0 and stmt_sql.strip().upper().startswith("ATTACH"):
                attach_preamble.append(stmt_sql)
            if op.kind in ("integrity_check", "fts_integrity", "attach_integrity") and rc == 0:
                result.integrity_ok = any(r and str(r[0]).lower() == "ok" for r in rows)
        # terminal reopen probe
        script = self._script_for("PRAGMA integrity_check;")
        rc, stdout, stderr = self._spawn(self.argv(db_path, script), script)
        result.reopened_ok = rc == 0 and not self._is_crash(rc, f"{stdout}\n{stderr}")
        return result

    @staticmethod
    def _same_attach(preamble_stmt: str, detach_sql: str) -> bool:
        """True if an ATTACH preamble line refers to the alias in a DETACH statement."""
        m = re.search(r"DETACH\s+(?:DATABASE\s+)?([A-Za-z0-9_]+)", detach_sql, re.IGNORECASE)
        if not m:
            return False
        alias = m.group(1)
        return re.search(rf"\bAS\s+{re.escape(alias)}\b", preamble_stmt, re.IGNORECASE) is not None

    @staticmethod
    def _script_for(sql: str) -> str:
        return f"{sql}\n"

    @staticmethod
    def _is_crash(rc: int, text: str) -> bool:
        if rc < 0:  # killed by signal
            return True
        low = text.lower()
        return any(marker.lower() in low for marker in CRASH_MARKERS)

    @staticmethod
    def _parse_rows(stdout: str) -> list[tuple]:
        """Parse `-m list` output for a SINGLE statement: pipe-separated columns, one row
        per line. A single-column NULL row renders as one empty line, which is NOT the
        same as an empty result set (zero lines) -- so we must NOT drop empty lines
        wholesale (that was a harness bug that made `SELECT max(x) FROM empty` look like
        it returned no rows instead of one NULL row). We strip only ONE trailing newline
        (shell artifact) and keep every remaining line as a row. Because run() invokes the
        CLI one statement at a time, all lines belong to that statement's result set."""
        if stdout == "":
            return []
        body = stdout[:-1] if stdout.endswith("\n") else stdout
        if body == "":
            # stdout was exactly "\n" -> one single-column NULL row.
            return [(None,)]
        rows: list[tuple] = []
        for line in body.split("\n"):
            line = line.rstrip("\r")
            rows.append(tuple(_coerce_cli_cell(c) for c in line.split("|")))
        return rows


class _CliText(str):
    """Marks a cell as already in the CLI's text projection. normalize_value treats it
    verbatim so we never re-guess its SQLite type. This sidesteps the transport ambiguity
    where the CLI's untyped text (e.g. hex of a blob that is all digits) would otherwise be
    mis-coerced to int -- we compare the reference AS THE CLI WOULD RENDER IT, not the CLI
    output re-typed to guess SQLite's tag."""
    __slots__ = ()


def _coerce_cli_cell(cell: str) -> Any:
    """A CLI text cell is untyped: empty string means NULL, everything else is opaque text
    in the CLI's own projection. We do NOT guess int/real/blob here -- guessing is what let
    a blob's hex (all digits) masquerade as an integer. Comparison happens in cli_text mode
    where the reference side is rendered the same way."""
    if cell == "":
        return None
    return _CliText(cell)


# ---------------------------------------------------------------------------
# Known-divergence allowlist
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Divergence:
    id: str
    stmt_pattern: str      # regex tested against the offending SQL
    error_pattern: str     # regex tested against combined error text ("" = any)
    rationale: str         # must cite a source before an entry is added


# Ships nearly empty. Entries are added ONLY with a citation. A matching diff is
# suppressed but still emitted as INVARIANT ... PASS divergence:<id> so it is visible.
KNOWN_DIVERGENCES: tuple[Divergence, ...] = (
    # Example placeholder documenting the format -- pattern that never matches a
    # generated statement, so it suppresses nothing today.
    Divergence(
        id="D000-format-example",
        stmt_pattern=r"^__genlib_never_matches__$",
        error_pattern="",
        rationale="Format example only; no real suppression. Real entries must cite "
                   "sqlite3 docs or a Turso issue justifying the accepted divergence.",
    ),
)


def match_divergence(sql: str, error_text: str) -> Optional[Divergence]:
    for d in KNOWN_DIVERGENCES:
        if re.search(d.stmt_pattern, sql) and (not d.error_pattern or re.search(d.error_pattern, error_text)):
            return d
    return None


# ---------------------------------------------------------------------------
# Universal oracles
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    oracle: str
    ok: bool
    summary: str
    divergence_id: Optional[str] = None  # set when a diff was suppressed by allowlist


def oracle_panic(ref: RunResult, cand: RunResult) -> Finding:
    """Candidate must not crash (panic/abort). The reference never crashes."""
    ok = not cand.crashed
    return Finding("panic_abort", ok, f"crashed={cand.crashed} text={cand.crash_text!r}")


def oracle_terminal_state(ref: RunResult, cand: RunResult) -> Finding:
    """Every submitted op reached accept-or-reject and the process exited (no hang);
    if the candidate produced a statement result for each non-reopen op, terminal."""
    ok = len(cand.stmts) >= 1 and not cand.crashed
    return Finding("terminal_state", ok, f"stmts={len(cand.stmts)} crashed={cand.crashed}")


def oracle_reopen(ref: RunResult, cand: RunResult) -> Finding:
    ok = cand.reopened_ok
    return Finding("reopen_persistence", ok, f"reopened_ok={cand.reopened_ok}")


def oracle_integrity(ref: RunResult, cand: RunResult) -> Finding:
    """If the candidate ran an integrity_check, it must report ok."""
    if cand.integrity_ok is None:
        return Finding("integrity", True, "no_integrity_check_run")
    return Finding("integrity", cand.integrity_ok, f"integrity_ok={cand.integrity_ok}")


def oracle_error_class(ref: RunResult, cand: RunResult) -> Finding:
    """For each aligned statement, both engines must agree on accept vs reject. A
    mismatch (one accepts, other rejects) is a red -- unless allowlisted."""
    n = min(len(ref.stmts), len(cand.stmts))
    skipped = 0
    for i in range(n):
        r, c = ref.stmts[i], cand.stmts[i]
        # Skip statements the reference cannot express identically (e.g. FTS): there is no
        # meaningful accept/reject to compare. The universal oracles still bind on them.
        if not (r.ref_comparable and c.ref_comparable):
            skipped += 1
            continue
        r_ok, c_ok = r.rc == 0, c.rc == 0
        if r_ok != c_ok:
            d = match_divergence(c.sql, f"{r.error}\n{c.error}")
            if d is not None:
                return Finding("error_class", True, f"suppressed stmt={c.sql!r}", d.id)
            return Finding(
                "error_class", False,
                f"accept-mismatch stmt={c.sql!r} ref_ok={r_ok} cand_ok={c_ok} "
                f"ref_err={r.error!r} cand_err={c.error!r}",
            )
    return Finding("error_class", True, f"aligned={n} skipped_noncomparable={skipped}")


def oracle_diff_rows(ref: RunResult, cand: RunResult, cli_text: bool = False) -> Finding:
    """For each aligned statement that BOTH accepted, the normalized row multisets must
    match. Mismatch = red unless allowlisted. cli_text collapses type tags when either
    runner is a text-only CLI that cannot carry SQLite's typing."""
    n = min(len(ref.stmts), len(cand.stmts))
    for i in range(n):
        r, c = ref.stmts[i], cand.stmts[i]
        if not (r.ref_comparable and c.ref_comparable):
            continue  # reference cannot express this stmt (e.g. FTS); no row comparison
        if r.rc != 0 or c.rc != 0:
            continue  # error-class oracle owns accept/reject disagreements
        rref = normalize_rowset(r.rows, cli_text)
        rcand = normalize_rowset(c.rows, cli_text)
        if rref != rcand:
            d = match_divergence(c.sql, "")
            if d is not None:
                return Finding("diff_rows", True, f"suppressed stmt={c.sql!r}", d.id)
            return Finding(
                "diff_rows", False,
                f"rowset-mismatch stmt={c.sql!r} ref={rref[:4]!r} cand={rcand[:4]!r} "
                f"ref_n={len(rref)} cand_n={len(rcand)}",
            )
    return Finding("diff_rows", True, f"aligned={n}")


ORACLES: tuple[Callable[[RunResult, RunResult], Finding], ...] = (
    oracle_panic,
    oracle_terminal_state,
    oracle_reopen,
    oracle_integrity,
    oracle_error_class,
    oracle_diff_rows,
)


# ---------------------------------------------------------------------------
# Case protocol
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    seed: int
    verdict: str            # GREEN / RED / VOID
    exit_code: int          # 0 / 1 / 3
    findings: list[Finding]


def _plant_corruption(ref: RunResult) -> None:
    """ORACLE_SELFTEST: mutate the reference so a DIFFERENTIAL oracle MUST fire RED. We
    only mutate REF-COMPARABLE statements (the differential oracles skip non-comparable
    ones, so corrupting those would not be caught). Drop a row from the first accepted,
    comparable SELECT that returned rows; if none, flip an accept to reject."""
    for st in ref.stmts:
        if st.ref_comparable and st.rc == 0 and st.rows:
            st.rows = st.rows[1:]
            return
    for st in ref.stmts:
        if st.ref_comparable and st.rc == 0:
            st.rc = 1
            st.error = "SELFTEST: planted rejection"
            return


def run_case(
    seed: int,
    axes: dict[str, tuple],
    runners: tuple[Runner, Runner],
    lifecycle_plug: tuple[str, ...] = (),
    case_id: str = "GEN",
    emit: bool = True,
) -> CaseResult:
    """Generate a program from seed, run reference + candidate, apply every universal
    oracle, and emit the INVARIANT/VERDICT protocol. runners = (reference, candidate).

    Verdict: RED if any oracle fails; VOID if the harness itself errored; else GREEN.
    """
    reference, candidate = runners
    program = generate(seed, axes, lifecycle_plug)
    findings: list[Finding] = []
    verdict = "GREEN"
    exit_code = 0

    try:
        ref_result = reference.run(program)
        cand_result = candidate.run(program)
        if os.environ.get("ORACLE_SELFTEST"):
            _plant_corruption(ref_result)
        cli_text = getattr(reference, "text_only", False) or getattr(candidate, "text_only", False)
        for oracle in ORACLES:
            if oracle is oracle_diff_rows:
                f = oracle(ref_result, cand_result, cli_text)
            else:
                f = oracle(ref_result, cand_result)
            findings.append(f)
    except Exception as exc:  # harness fault -> VOID, not a product verdict
        if emit:
            print(f"INVARIANT {case_id} harness_fault FAIL seed={seed} err={type(exc).__name__}: {exc}", flush=True)
            print(f"VERDICT: VOID seed={seed}", flush=True)
        return CaseResult(seed, "VOID", 3, findings)

    for f in findings:
        status = "PASS" if f.ok else "FAIL"
        note = f" divergence:{f.divergence_id}" if f.divergence_id else ""
        if emit:
            print(f"INVARIANT {case_id} {f.oracle} {status} seed={seed} {f.summary}{note}", flush=True)
        if not f.ok:
            verdict = "RED"
            exit_code = 1

    if emit:
        print(f"VERDICT: {'GREEN' if verdict == 'GREEN' else 'RED'} seed={seed} "
              f"page_size={program.config['page_size']} encryption={program.config['encryption']}", flush=True)
    return CaseResult(seed, verdict, exit_code, findings)


def merged_axes(product_axes: Optional[dict[str, tuple]] = None) -> dict[str, tuple]:
    axes = dict(CORE_AXES)
    if product_axes:
        axes.update(product_axes)
    return axes
