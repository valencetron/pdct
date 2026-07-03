"""Rule-based concept extraction for Dynamic Context Traversal.

Pure functions: (text) -> ordered deduplicated list of canonical slugs.
Three stackable rules: wikilinks, hashtags, Obsidian path references.
"""

from __future__ import annotations

import re


def to_slug(raw: str) -> str:
    s = raw.lower()
    if s.endswith(".md"):
        s = s[:-3]
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


_WIKILINK_RE = re.compile(r"\[\[([^\[\]|]+?)(?:\|[^\[\]]*)?\]\]")
_SCHEME_PREFIX_RE = re.compile(r"^[a-z][a-z0-9]{0,6}:")


def extract_wikilinks(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in _WIKILINK_RE.finditer(text):
        raw = match.group(1)
        if _SCHEME_PREFIX_RE.match(raw):
            continue
        slug = to_slug(raw)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        out.append(slug)
    return out


_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_HASHTAG_RE = re.compile(r"(?<![A-Za-z0-9_])#([A-Za-z][A-Za-z0-9-]{1,})")
_HEX_COLOR_RE = re.compile(r"^[0-9a-f]{3}$|^[0-9a-f]{6}$|^[0-9a-f]{8}$")
_HTML_ENTITY_RE = re.compile(r"^x[0-9a-f]{4,}$")


def _strip_code(text: str) -> str:
    stripped = _FENCED_CODE_RE.sub(" ", text)
    stripped = _INLINE_CODE_RE.sub(" ", stripped)
    return stripped


def _is_noise_hashtag_slug(slug: str) -> bool:
    return bool(_HEX_COLOR_RE.match(slug) or _HTML_ENTITY_RE.match(slug))


def extract_hashtags(text: str) -> list[str]:
    cleaned = _strip_code(text)
    seen: set[str] = set()
    out: list[str] = []
    for match in _HASHTAG_RE.finditer(cleaned):
        slug = to_slug(match.group(1))
        if not slug or slug in seen or _is_noise_hashtag_slug(slug):
            continue
        seen.add(slug)
        out.append(slug)
    return out


_PATH_REF_RE = re.compile(
    r"""
    (?:^|[\s`"'(])
    ([^\s`"'(]*[Oo]rion[ ][Aa]perture/.*?\.md)
    """,
    re.VERBOSE,
)


def extract_path_refs(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in _PATH_REF_RE.finditer(text):
        path = match.group(1)
        filename = path.rsplit("/", 1)[-1]
        slug = to_slug(filename)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        out.append(slug)
    return out


_DENYLIST: frozenset[str] = frozenset({
    # generic directory names
    "src", "app", "tools", "scripts", "bin", "lib", "components", "pages",
    "public", "dist", "build", "node-modules", "test", "tests", "docs",
    "utils", "models", "views", "api", "hooks", "styles", "assets", "static",
    "config", "common", "shared", "types", "interfaces", "helpers",
    "pycache", "venv",
    # user / host / OS fragments
    "users", "user", "home", "usr", "var", "opt", "etc", "tmp",
    # project-root names (observed in real data)
    "example-stack", "claude",
    # generic filename stems
    "index", "main", "init",
    # MC codebase triad — present on every mission-control path, not distinguishing
    "dashboard", "mission-control-app", "app-shell",
    # superpowers skill-invocation slugs (structural, not content topics)
    "using-superpowers", "superpowers-writing-plans",
    "superpowers-subagent-driven-development", "superpowers-brainstorming",
    # meta-tooling names (the DCT extractor talks about itself)
    "wikilinks", "wikilink",
    # MC card status vocabulary — closed set, extracted from `status` field but
    # not a content topic. "active"/"done"/"abandoned" are state markers.
    "done", "active", "abandoned", "now", "next", "later",
    "pending", "in-progress", "completed", "deleted",
})

_MIN_SLUG_LEN = 3
_MAX_SLUG_LEN = 80

_ALIASES: dict[str, str] = {}


def _filter_slugs(raw_slugs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for slug in raw_slugs:
        canonical = _ALIASES.get(slug, slug)
        if not canonical:
            continue
        if len(canonical) < _MIN_SLUG_LEN:
            continue
        if len(canonical) > _MAX_SLUG_LEN:
            continue
        if canonical in _DENYLIST:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


_HOME_PREFIX_RE = re.compile(r"^(?:~|/Users/[^/]+)/")


def extract_from_paths(paths: list) -> list[str]:
    raw: list[str] = []
    for path in paths:
        if not isinstance(path, str) or not path:
            continue
        trimmed = _HOME_PREFIX_RE.sub("", path)
        parts = [p for p in trimmed.split("/") if p]
        if not parts:
            continue
        for part in parts[:-1]:
            slug = to_slug(part)
            if slug:
                raw.append(slug)
        last = parts[-1]
        stem = last.rsplit(".", 1)[0] if "." in last else last
        slug = to_slug(stem)
        if slug:
            raw.append(slug)
    return _filter_slugs(raw)


_STRUCTURED_FIELD_KEYS = ("slug", "title", "status", "tags", "skill")


def extract_from_structured_fields(fields: dict) -> list[str]:
    raw: list[str] = []
    for key in _STRUCTURED_FIELD_KEYS:
        val = fields.get(key)
        if not isinstance(val, str) or not val:
            continue
        if key == "tags":
            for part in re.split(r"[,\s]+", val):
                if part:
                    raw.append(to_slug(part))
        else:
            raw.append(to_slug(val))
    return _filter_slugs(raw)


def extract(text: str) -> list[str]:
    raw: list[str] = []
    for rule in (extract_wikilinks, extract_hashtags, extract_path_refs):
        raw.extend(rule(text))
    return _filter_slugs(raw)
