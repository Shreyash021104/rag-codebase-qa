"""AST-aware code chunking.

The naive approach — split every file into fixed-size line windows — routinely
cuts functions in half, which poisons retrieval: an embedding of "the bottom
half of one function plus the top half of the next" doesn't represent either
one, so queries about either function match it poorly. Instead, we parse each
file with tree-sitter and cut chunks along the boundaries the code itself
declares: one chunk per top-level function/class, with small loose statements
(imports, constants) grouped together.

Anything tree-sitter can't parse (markdown, configs, unknown languages) falls
back to fixed line windows — chunking imperfectly beats not indexing at all.
"""

from dataclasses import dataclass

from tree_sitter_language_pack import get_parser

from .config import CHUNK_MAX_LINES, CHUNK_TARGET_LINES


@dataclass
class Chunk:
    file_path: str
    start_line: int  # 1-indexed, inclusive
    end_line: int
    symbol: str | None
    language: str | None
    content: str


EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".lua": "lua",
    ".sh": "bash",
    ".sql": "sql",
}

# Extensions worth indexing as plain text even though they aren't parseable
# code — READMEs and configs answer a lot of "how do I run this" questions.
TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".yaml", ".yml", ".toml", ".json", ".env.example"}


def language_for_path(path: str) -> str | None:
    for ext, lang in EXTENSION_LANGUAGES.items():
        if path.endswith(ext):
            return lang
    return None


def is_indexable(path: str) -> bool:
    return language_for_path(path) is not None or any(
        path.endswith(ext) for ext in TEXT_EXTENSIONS
    )


def _node_symbol(node) -> str | None:
    """Best-effort name extraction for a definition node.

    Most tree-sitter grammars expose the defined name as a `name` field;
    for wrapper nodes (decorated definitions, export statements) the named
    definition is nested one level down, so we look through children too.
    """
    name = node.child_by_field_name("name")
    if name is not None:
        return name.text.decode("utf-8", errors="replace")
    for child in node.named_children:
        found = _node_symbol(child)
        if found:
            return found
    return None


def _is_definition(node) -> bool:
    # Grammar node-type names differ across languages, but definitions
    # consistently contain one of these substrings (function_definition,
    # class_declaration, method_declaration, decorated_definition, ...).
    # A substring heuristic keeps this working across every language in
    # the pack without maintaining a per-grammar allowlist.
    t = node.type
    if any(k in t for k in ("function", "class", "method", "definition", "declaration")):
        return True
    # Wrapper nodes (JS/TS `export_statement`, etc.) hide the definition one
    # level down — without this unwrap, every exported class/function in a
    # TS file got lumped into anonymous "loose statement" chunks.
    return any(
        any(k in child.type for k in ("function", "class", "definition", "declaration"))
        for child in node.named_children
    )


def _line_windows(
    file_path: str, lines: list[str], language: str | None, window: int, overlap: int = 5
) -> list[Chunk]:
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + window, len(lines))
        content = "\n".join(lines[start:end]).strip()
        if content:
            chunks.append(
                Chunk(
                    file_path=file_path,
                    start_line=start + 1,
                    end_line=end,
                    symbol=None,
                    language=language,
                    content=content,
                )
            )
        if end == len(lines):
            break
        start = end - overlap
    return chunks


def chunk_file(file_path: str, source: str) -> list[Chunk]:
    lines = source.splitlines()
    if not lines:
        return []

    language = language_for_path(file_path)
    if language is None:
        # Plain-text fallback (docs/configs): fixed windows.
        return _line_windows(file_path, lines, None, CHUNK_TARGET_LINES * 2)

    try:
        parser = get_parser(language)
        tree = parser.parse(source.encode("utf-8"))
    except Exception:
        return _line_windows(file_path, lines, language, CHUNK_TARGET_LINES * 2)

    chunks: list[Chunk] = []
    pending_start: int | None = None  # start of the current "loose statements" group

    def flush_pending(up_to_line: int) -> None:
        nonlocal pending_start
        if pending_start is None:
            return
        content = "\n".join(lines[pending_start : up_to_line + 1]).strip()
        if content:
            chunks.append(
                Chunk(
                    file_path=file_path,
                    start_line=pending_start + 1,
                    end_line=up_to_line + 1,
                    symbol=None,
                    language=language,
                    content=content,
                )
            )
        pending_start = None

    prev_end = -1
    for node in tree.root_node.named_children:
        node_start = node.start_point[0]  # 0-indexed row
        node_end = node.end_point[0]
        node_lines = node_end - node_start + 1

        if _is_definition(node) and node_lines > 3:
            flush_pending(prev_end)
            symbol = _node_symbol(node)
            if node_lines <= CHUNK_MAX_LINES:
                content = "\n".join(lines[node_start : node_end + 1])
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        start_line=node_start + 1,
                        end_line=node_end + 1,
                        symbol=symbol,
                        language=language,
                        content=content,
                    )
                )
            else:
                # A single oversized definition (usually a big class): split
                # it into windows but keep the symbol on every piece, so a
                # match on any window still cites the right class.
                for piece in _line_windows(
                    file_path,
                    lines[node_start : node_end + 1],
                    language,
                    CHUNK_MAX_LINES,
                ):
                    chunks.append(
                        Chunk(
                            file_path=file_path,
                            start_line=piece.start_line + node_start,
                            end_line=piece.end_line + node_start,
                            symbol=symbol,
                            language=language,
                            content=piece.content,
                        )
                    )
        else:
            # Loose top-level statement (import, constant, expression):
            # accumulate into a group, flushing when the group gets big.
            if pending_start is None:
                pending_start = node_start
            elif node_end - pending_start + 1 > CHUNK_TARGET_LINES:
                flush_pending(prev_end)
                pending_start = node_start

        prev_end = node_end

    flush_pending(prev_end)
    return chunks
