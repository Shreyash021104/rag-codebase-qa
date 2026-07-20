"""Chunker tests — the most important module, and the one with a real bug
history (see the README's TypeScript export-wrapper story). No DB, no
embeddings, no API keys: pure parse-and-chunk correctness.
"""

from app.chunker import chunk_file, is_indexable

PY_SOURCE = '''\
import os


def load_config(path):
    with open(path) as f:
        raw = f.read()
    parsed = {}
    for line in raw.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            parsed[k.strip()] = v.strip()
    return parsed


class Server:
    def start(self):
        self.running = True
        return "listening"

    def stop(self):
        self.running = False
        return "stopped"
'''

TS_SOURCE = '''\
import { Redis } from "ioredis";

export class Room {
  private connections = new Set();

  broadcast(msg: string) {
    for (const c of this.connections) {
      c.send(msg);
    }
  }
}

export function makeRoom(name: string): Room {
  const room = new Room();
  registry.set(name, room);
  return room;
}
'''


def _symbols(chunks):
    return {c.symbol for c in chunks if c.symbol}


def test_python_functions_and_classes_become_named_chunks():
    chunks = chunk_file("app/server.py", PY_SOURCE)
    symbols = _symbols(chunks)
    assert "load_config" in symbols
    assert "Server" in symbols


def test_typescript_exported_symbols_are_named():
    # Regression test for the export-wrapper bug: `export class Room` parses
    # as an export_statement wrapping the class, and without unwrapping it
    # the whole thing came out as an anonymous blob chunk.
    chunks = chunk_file("src/room.ts", TS_SOURCE)
    symbols = _symbols(chunks)
    assert "Room" in symbols, f"expected Room, got {symbols}"
    assert "makeRoom" in symbols, f"expected makeRoom, got {symbols}"


def test_every_chunk_records_its_line_range():
    chunks = chunk_file("app/server.py", PY_SOURCE)
    assert chunks
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line


def test_indexable_extensions_include_code_docs_and_web():
    assert is_indexable("main.py")
    assert is_indexable("index.ts")
    assert is_indexable("README.md")
    assert is_indexable("index.html")  # regression: web files were skipped
    assert is_indexable("style.css")
    assert not is_indexable("logo.png")


def test_unparseable_file_falls_back_to_windows_not_crash():
    # A file with an extension we treat as text but isn't real code should
    # still produce chunks via the window fallback rather than erroring.
    chunks = chunk_file("notes.md", "# Title\n\nSome prose.\n" * 50)
    assert chunks
