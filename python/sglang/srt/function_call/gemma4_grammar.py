"""EBNF builder for Gemma4's native tool-call syntax.

Gemma4 emits ``<|tool_call>call:NAME{key:VALUE,key:VALUE}<tool_call|>`` where
strings are wrapped in ``<|"|>`` (a special token), keys are bare, objects use
``{k:v}``, arrays ``[v,v]``, booleans ``true``/``false``, numbers bare, and no
whitespace anywhere. Keys are emitted in sorted order, which is how the chat
template renders both the declarations and prior calls (``dictsort``).

The grammar is fed to xgrammar as ``{"type": "grammar"}`` inside a structural
tag; xgrammar matches the special tokens by their literal text. Constructs the
builder cannot express degrade to a generic value rule instead of raising, so
a constraint is always produced.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _lit(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


class _Builder:
    def __init__(self, string_delim: str):
        self.rules: List[Tuple[str, str]] = []
        self.names: set = {"root", "anyval", "anykey"}
        self.delim = _lit(string_delim)
        self.root_schema: Dict[str, Any] = {}
        self._any_added = False

    def rule(self, base: str, body: str) -> str:
        name = re.sub(r"[^A-Za-z0-9_]", "_", base)
        cand, i = name, 1
        while cand in self.names:
            i += 1
            cand = f"{name}_{i}"
        self.names.add(cand)
        self.rules.append((cand, body))
        return cand

    # -- primitives ---------------------------------------------------------
    def string(self) -> str:
        # any text except the delimiter's leading "<|" pair
        return f'{self.delim} ( [^<] | "<" [^|] )* {self.delim}'

    def integer(self) -> str:
        return '"-"? [0-9]+'

    def number(self) -> str:
        return '"-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?'

    def boolean(self) -> str:
        return '("true" | "false")'

    def anyval(self) -> str:
        if not self._any_added:
            self._any_added = True
            self.rules.append(
                (
                    "anyval",
                    f"{self.string()} | {self.number()} | {self.boolean()} | "
                    '"[" (anyval ("," anyval)*)? "]" | '
                    '"{" (anykey ":" anyval ("," anykey ":" anyval)*)? "}"',
                )
            )
            self.rules.append(("anykey", "[A-Za-z_] [A-Za-z0-9_.-]*"))
        return "anyval"

    # -- schema -------------------------------------------------------------
    def resolve(self, schema: Any) -> Any:
        if isinstance(schema, dict) and "$ref" in schema:
            ref = schema["$ref"]
            if isinstance(ref, str) and ref.startswith("#/"):
                node: Any = self.root_schema
                for part in ref[2:].split("/"):
                    node = node.get(part, {}) if isinstance(node, dict) else {}
                return node if isinstance(node, dict) else {}
            return {}
        return schema

    def value(self, schema: Any, hint: str, depth: int = 0) -> str:
        """Return an EBNF expression for a value of this schema."""
        schema = self.resolve(schema)
        if not isinstance(schema, dict) or depth > 12:
            return self.anyval()
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            alts = []
            for v in enum:
                if isinstance(v, bool):
                    alts.append("true" if v else "false")
                elif isinstance(v, (int, float)):
                    alts.append(_lit(str(v)))
                elif isinstance(v, str):
                    alts.append(f"{self.delim} {_lit(v)} {self.delim}")
                else:
                    return self.anyval()
            return "(" + " | ".join(alts) + ")"
        if "const" in schema:
            return self.value({"enum": [schema["const"]]}, hint, depth + 1)
        for key in ("anyOf", "oneOf"):
            alts = schema.get(key)
            if isinstance(alts, list) and alts:
                return (
                    "("
                    + " | ".join(self.value(x, hint, depth + 1) for x in alts)
                    + ")"
                )
        t = schema.get("type")
        if isinstance(t, list):
            ts = [x for x in t if x != "null"]
            if not ts:
                return self.anyval()
            if len(ts) == 1:
                t = ts[0]
            else:
                return (
                    "("
                    + " | ".join(
                        self.value({**schema, "type": x}, hint, depth + 1) for x in ts
                    )
                    + ")"
                )
        if t == "string":
            return self.string()
        if t == "integer":
            return self.integer()
        if t == "number":
            return self.number()
        if t == "boolean":
            return self.boolean()
        if t == "array":
            item = self.value(schema.get("items", {}), hint + "_item", depth + 1)
            item_rule = self.rule(hint + "_item", item)
            mx = schema.get("maxItems")
            mn = schema.get("minItems")
            mn = mn if isinstance(mn, int) and mn > 0 else 0
            if isinstance(mx, int) and mx >= 1:
                rep = f'{item_rule} ("," {item_rule}){{{max(0, mn - 1)},{mx - 1}}}'
                return f'"[" ({rep})? "]"' if mn == 0 else f'"[" {rep} "]"'
            if mn >= 1:
                return f'"[" {item_rule} ("," {item_rule}){{{mn - 1},}} "]"'
            return f'"[" ({item_rule} ("," {item_rule})*)? "]"'
        if t == "object" or "properties" in schema:
            return self.object(schema, hint, depth + 1)
        return self.anyval()

    def object(self, schema: Dict[str, Any], hint: str, depth: int = 0) -> str:
        props = schema.get("properties")
        props = props if isinstance(props, dict) else {}
        required = set(schema.get("required") or [])
        keys = sorted(props.keys())  # chat template renders with dictsort
        members: List[Tuple[str, str, bool]] = []
        for k in keys:
            v_expr = self.value(props[k], f"{hint}_{k}", depth + 1)
            v_rule = self.rule(f"{hint}_{k}", v_expr)
            members.append((k, v_rule, k in required))
        extra = schema.get("additionalProperties") is True
        if extra:
            anyv = self.anyval()
            extra_member = f'anykey ":" {anyv}'
        n = len(members)
        if n == 0:
            if extra:
                return f'"{{" ({extra_member} ("," {extra_member})*)? "}}"'
            return '"{" "}"'

        # E(i): members i.. with no leading comma (may be empty);
        # F(i): members i.. each preceded by "," (may be empty).
        e_next = self.rule(f"{hint}_e{n}", '""')
        f_next = self.rule(f"{hint}_f{n}", '""')
        for i in range(n - 1, -1, -1):
            k, vr, req = members[i]
            mem = f'{_lit(k)} ":" {vr}'
            if req:
                f_body = f'"," {mem} {f_next}'
                e_body = f"{mem} {f_next}"
            else:
                f_body = f'("," {mem})? {f_next}'
                e_body = f"({mem} {f_next} | {e_next})"
            f_next = self.rule(f"{hint}_f{i}", f_body)
            e_next = self.rule(f"{hint}_e{i}", e_body)
        if not extra:
            return f'"{{" {e_next} "}}"'
        tail = f'("," {extra_member})*'
        if any(m[2] for m in members):
            # at least one declared member is required, extras follow it
            return f'"{{" {e_next} {tail} "}}"'
        # all optional: declared members (maybe none) then extras, or extras alone
        return f'"{{" ({e_next} {tail} | {extra_member} {tail})? "}}"'


def build_gemma4_tool_call_ebnf(
    tools: List[Tuple[str, Dict[str, Any]]],
    parallel_tool_calls: bool = True,
    start_token: str = "<|tool_call>",
    end_token: str = "<tool_call|>",
    string_delim: str = '<|"|>',
) -> str:
    """Grammar for one or more (``parallel_tool_calls``) native Gemma4 calls
    drawn from ``tools`` = [(name, json_schema_parameters), ...]."""
    b = _Builder(string_delim)
    calls = []
    for name, params in tools:
        schema = params if isinstance(params, dict) else {}
        b.root_schema = schema
        if "properties" not in schema and schema.get("type") not in (None, "object"):
            schema = {"type": "object", "properties": {}}
        body = b.object(schema, f"t_{name}")
        call_rule = b.rule(
            f"call_{name}",
            f"{_lit(start_token)} {_lit('call:' + name)} {body} {_lit(end_token)}",
        )
        calls.append(call_rule)
    one = "(" + " | ".join(calls) + ")"
    root = f"{one}+" if parallel_tool_calls else one
    lines = [f"root ::= {root}"] + [f"{n} ::= {body}" for n, body in b.rules]
    return "\n".join(lines) + "\n"
