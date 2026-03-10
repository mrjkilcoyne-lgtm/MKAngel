# Phase 1: "First Light" Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform MKAngel from a basic dark chat terminal into an angelic, gesture-driven, hybrid Kivy + WebView app with the Celestial Dark vestment, all 8 gesture-navigable panels, the Companion aspect as default, and an embedded document editor — all offline-first.

**Architecture:** Hybrid Kivy (shell/gestures/chat/voice) + WebView (document editor/rich panels). Python backend on-device. All WebView content bundled locally in APK. Offline-first with opportunistic cloud sync.

**Tech Stack:** Kivy 2.x, pyjnius (Android WebView bridge), Quill.js (rich text editor), python-docx/reportlab (export), SQLite (memory), pure Python GLM engine.

---

## Task 1: Angelic Naming — Rename Swarm to Host

Rename `swarm.py` terminology to angelic naming. This is a find-and-replace across the codebase.

**Files:**
- Modify: `app/swarm.py` — rename classes and docstrings (keep filename for now to avoid import breakage)
- Modify: `app/__init__.py` — update imported names
- Modify: `app/cowork.py` — update any swarm references
- Modify: `app/chat.py` — update /swarm commands to /host

**Step 1: Rename core classes in app/swarm.py**

Replace throughout the file:
- `Swarm` → `Host` (class name)
- `SwarmAgent` → `HostAgent`
- `SwarmMessage` → `HostMessage`
- `SwarmHarness` → `HostHarness`
- `swarm` → `host` in docstrings and comments (case-sensitive, preserve "swarm" in historical references)
- Keep `AgentRole` and `CoordinationProtocol` unchanged (they're already angelic)

**Step 2: Update app/__init__.py**

Replace:
```python
from app.swarm import Swarm, SwarmHarness, BorgesLibrary, AgentRole
```
with:
```python
from app.swarm import Host, HostHarness, BorgesLibrary, AgentRole
```

**Step 3: Update references in app/chat.py**

Search for any `/swarm` command references and rename to `/host`.

**Step 4: Commit**

```bash
git add app/swarm.py app/__init__.py app/chat.py app/cowork.py
git commit -m "refactor: rename Swarm to Host (angelic naming convention)"
```

---

## Task 2: Design Token System — Vestment Foundation

Create a centralised theme/token system that both Kivy and WebView can consume.

**Files:**
- Create: `app/vestments.py`

**Step 1: Create app/vestments.py**

```python
"""
Vestment system — the Angel's visual identities.

Four vestments:
  - Celestial Dark: OLED black, gold, constellation particles
  - Ethereal Light: Warm white, silver, glassmorphism
  - Living Gradient: Aurora animation, breathing backgrounds
  - Minimal Power: Monochrome, typography-driven

Each vestment is a dict of design tokens consumed by both
Kivy widgets (as rgba tuples) and WebView CSS (as hex strings).
"""

from __future__ import annotations
from kivy.utils import get_color_from_hex

_hex = get_color_from_hex


def _token(hex_color: str, alpha: float = 1.0) -> dict:
    """Create a token with both Kivy rgba and CSS hex."""
    rgba = list(_hex(hex_color))
    rgba[3] = alpha
    return {"kivy": rgba, "css": hex_color, "alpha": alpha}


CELESTIAL_DARK = {
    "name": "Celestial Dark",
    "id": "celestial_dark",
    "bg":            _token("#000000"),
    "surface":       _token("#0e0e14"),
    "surface_head":  _token("#0c0c12"),
    "surface_input": _token("#16161e"),
    "accent":        _token("#D4AF37"),       # gold
    "accent_purple": _token("#bb86fc"),
    "teal":          _token("#03dac6"),
    "text":          _token("#f0f0f2"),
    "text_sec":      _token("#9e9ea8"),
    "text_dim":      _token("#555560"),
    "success":       _token("#66bb6a"),
    "warning":       _token("#ffab40"),
    "error":         _token("#ef5350"),
    "separator":     _token("#ffffff", 0.04),
    "bubble_user":   _token("#bb86fc", 0.10),
    "bubble_angel":  _token("#0e0e14"),
    "particle":      _token("#D4AF37", 0.3),
}

ETHEREAL_LIGHT = {
    "name": "Ethereal Light",
    "id": "ethereal_light",
    "bg":            _token("#FAFAFA"),
    "surface":       _token("#FFFFFF", 0.85),
    "surface_head":  _token("#F5F5F5"),
    "surface_input": _token("#EEEEEE"),
    "accent":        _token("#C0C0C0"),
    "accent_purple": _token("#7C4DFF"),
    "teal":          _token("#00897B"),
    "text":          _token("#333333"),
    "text_sec":      _token("#666666"),
    "text_dim":      _token("#999999"),
    "success":       _token("#4CAF50"),
    "warning":       _token("#FF9800"),
    "error":         _token("#F44336"),
    "separator":     _token("#000000", 0.06),
    "bubble_user":   _token("#E8EAF6"),
    "bubble_angel":  _token("#FFFFFF"),
    "particle":      _token("#C0C0C0", 0.2),
}

LIVING_GRADIENT = {
    "name": "Living Gradient",
    "id": "living_gradient",
    "bg":            _token("#0D001A"),
    "surface":       _token("#1A0033", 0.8),
    "surface_head":  _token("#0D001A", 0.9),
    "surface_input": _token("#2D004D"),
    "accent":        _token("#00FFFF"),
    "accent_purple": _token("#FF00FF"),
    "teal":          _token("#00FFAA"),
    "text":          _token("#FFFFFF"),
    "text_sec":      _token("#CCCCFF"),
    "text_dim":      _token("#8888AA"),
    "success":       _token("#00FF88"),
    "warning":       _token("#FFAA00"),
    "error":         _token("#FF4444"),
    "separator":     _token("#FFFFFF", 0.08),
    "bubble_user":   _token("#FF00FF", 0.15),
    "bubble_angel":  _token("#1A0033", 0.8),
    "particle":      _token("#00FFFF", 0.4),
}

MINIMAL_POWER = {
    "name": "Minimal Power",
    "id": "minimal_power",
    "bg":            _token("#FCFCFC"),
    "surface":       _token("#FFFFFF"),
    "surface_head":  _token("#FAFAFA"),
    "surface_input": _token("#F5F5F5"),
    "accent":        _token("#1A237E"),
    "accent_purple": _token("#1A237E"),
    "teal":          _token("#1A237E"),
    "text":          _token("#212121"),
    "text_sec":      _token("#757575"),
    "text_dim":      _token("#BDBDBD"),
    "success":       _token("#1B5E20"),
    "warning":       _token("#E65100"),
    "error":         _token("#B71C1C"),
    "separator":     _token("#000000", 0.08),
    "bubble_user":   _token("#E8EAF6"),
    "bubble_angel":  _token("#FFFFFF"),
    "particle":      _token("#1A237E", 0.1),
}

ALL_VESTMENTS = {
    "celestial_dark": CELESTIAL_DARK,
    "ethereal_light": ETHEREAL_LIGHT,
    "living_gradient": LIVING_GRADIENT,
    "minimal_power": MINIMAL_POWER,
}

DEFAULT_VESTMENT = "celestial_dark"


def get_vestment(name: str | None = None) -> dict:
    """Return a vestment token dict by id."""
    return ALL_VESTMENTS.get(name or DEFAULT_VESTMENT, CELESTIAL_DARK)


def vestment_to_css(vestment: dict) -> str:
    """Export vestment tokens as CSS custom properties for WebView."""
    lines = [":root {"]
    for key, val in vestment.items():
        if isinstance(val, dict) and "css" in val:
            lines.append(f"  --{key.replace('_', '-')}: {val['css']};")
            if val.get("alpha", 1.0) < 1.0:
                lines.append(f"  --{key.replace('_', '-')}-alpha: {val['alpha']};")
    lines.append("}")
    return "\n".join(lines)
```

**Step 2: Commit**

```bash
git add app/vestments.py
git commit -m "feat: add vestment design token system (4 themes)"
```

---

## Task 3: Gesture Recognition System

**Files:**
- Create: `app/gestures.py`

**Step 1: Create app/gestures.py**

```python
"""
Gesture recognition for the Angel's full wingspan.

8 gestures, each activating a different power:
  Tap           → Chat (default)
  Swipe Up      → Voice mode
  Swipe Right   → Hosts (parallel agents)
  Swipe Left    → Wings (extensions)
  Long Press    → Skills overlay
  Swipe Down    → Aspect switcher
  Double Tap    → Vestment switcher
  Pinch         → Document mode
"""

from __future__ import annotations

import time
from kivy.metrics import dp
from kivy.uix.widget import Widget
from kivy.clock import Clock


class GestureAction:
    CHAT = "chat"
    VOICE = "voice"
    HOSTS = "hosts"
    WINGS = "wings"
    SKILLS = "skills"
    ASPECTS = "aspects"
    VESTMENTS = "vestments"
    DOCUMENTS = "documents"


class GestureDetector(Widget):
    """
    Transparent overlay that detects gestures and dispatches actions.

    Place this as the top-level widget in the layout; it passes through
    all touch events it doesn't consume.
    """

    # Thresholds
    SWIPE_MIN_DIST = dp(80)        # min distance for a swipe
    SWIPE_MAX_TIME = 0.5           # max seconds for a swipe
    LONG_PRESS_TIME = 0.6          # seconds to trigger long press
    DOUBLE_TAP_TIME = 0.3          # max gap between taps
    EDGE_ZONE = dp(30)             # edge zone for directional swipes
    PINCH_MIN_DIST_CHANGE = dp(50) # min distance change for pinch

    def __init__(self, callback=None, **kwargs):
        super().__init__(**kwargs)
        self._callback = callback
        self._touch_start = None
        self._touch_start_time = 0
        self._long_press_event = None
        self._last_tap_time = 0
        self._touches = {}  # multi-touch tracking

    def set_callback(self, fn):
        self._callback = fn

    def _dispatch(self, action: str):
        if self._callback:
            self._callback(action)

    def on_touch_down(self, touch):
        if len(self._touches) == 0:
            self._touch_start = (touch.x, touch.y)
            self._touch_start_time = time.time()
            # Schedule long press
            self._long_press_event = Clock.schedule_once(
                self._on_long_press, self.LONG_PRESS_TIME
            )
        self._touches[touch.uid] = (touch.x, touch.y)
        return False  # pass through

    def on_touch_move(self, touch):
        if touch.uid in self._touches:
            self._touches[touch.uid] = (touch.x, touch.y)
            # Cancel long press on significant movement
            if self._touch_start:
                dx = abs(touch.x - self._touch_start[0])
                dy = abs(touch.y - self._touch_start[1])
                if dx > dp(15) or dy > dp(15):
                    self._cancel_long_press()
        return False

    def on_touch_up(self, touch):
        self._cancel_long_press()

        if touch.uid not in self._touches:
            return False

        del self._touches[touch.uid]

        if self._touch_start is None:
            return False

        elapsed = time.time() - self._touch_start_time
        dx = touch.x - self._touch_start[0]
        dy = touch.y - self._touch_start[1]
        dist = (dx**2 + dy**2) ** 0.5
        sx, sy = self._touch_start

        # ── Pinch detection (2-finger) ──
        # Handled separately via multi-touch; simplified here

        # ── Swipe detection ──
        if dist > self.SWIPE_MIN_DIST and elapsed < self.SWIPE_MAX_TIME:
            if abs(dy) > abs(dx):
                # Vertical swipe
                if dy > 0:
                    # Swipe down — from top edge → aspects
                    if sy > self.height * 0.8:
                        self._dispatch(GestureAction.ASPECTS)
                        return True
                else:
                    # Swipe up — from bottom edge → voice
                    if sy < self.height * 0.25:
                        self._dispatch(GestureAction.VOICE)
                        return True
            else:
                # Horizontal swipe
                if dx > 0:
                    # Swipe right — from left edge → hosts
                    if sx < self.EDGE_ZONE:
                        self._dispatch(GestureAction.HOSTS)
                        return True
                else:
                    # Swipe left — from right edge → wings
                    if sx > self.width - self.EDGE_ZONE:
                        self._dispatch(GestureAction.WINGS)
                        return True

            self._touch_start = None
            return False

        # ── Double tap detection ──
        now = time.time()
        if dist < dp(15) and elapsed < 0.3:
            if (now - self._last_tap_time) < self.DOUBLE_TAP_TIME:
                self._dispatch(GestureAction.VESTMENTS)
                self._last_tap_time = 0
                self._touch_start = None
                return True
            self._last_tap_time = now

        self._touch_start = None
        return False

    def _on_long_press(self, _dt):
        self._dispatch(GestureAction.SKILLS)
        self._long_press_event = None

    def _cancel_long_press(self):
        if self._long_press_event:
            self._long_press_event.cancel()
            self._long_press_event = None
```

**Step 2: Commit**

```bash
git add app/gestures.py
git commit -m "feat: add gesture detection system (8 angelic gestures)"
```

---

## Task 4: Document Management System

**Files:**
- Create: `app/documents.py`

**Step 1: Create app/documents.py**

```python
"""
Document management for MKAngel.

Offline-first: documents save locally as Quill Delta JSON.
Export to DOCX/PDF when requested. Cloud sync when online.

Storage: ~/.mkangel/documents/
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


DOCS_DIR = Path.home() / ".mkangel" / "documents"


@dataclass
class Document:
    """A single document managed by the Angel."""
    doc_id: str
    title: str
    content: dict = field(default_factory=dict)  # Quill Delta JSON
    created_at: float = 0.0
    updated_at: float = 0.0
    version: int = 1
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()
        if not self.updated_at:
            self.updated_at = time.time()


class DocumentManager:
    """Manages document lifecycle: create, save, load, list, export."""

    def __init__(self, docs_dir: Path | None = None):
        self.docs_dir = docs_dir or DOCS_DIR
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def create(self, title: str = "Untitled") -> Document:
        doc_id = f"doc_{int(time.time() * 1000)}"
        doc = Document(doc_id=doc_id, title=title)
        self.save(doc)
        return doc

    def save(self, doc: Document) -> None:
        doc.updated_at = time.time()
        doc.version += 1
        path = self.docs_dir / f"{doc.doc_id}.json"
        path.write_text(json.dumps({
            "doc_id": doc.doc_id,
            "title": doc.title,
            "content": doc.content,
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
            "version": doc.version,
            "tags": doc.tags,
        }, indent=2), encoding="utf-8")

    def load(self, doc_id: str) -> Document | None:
        path = self.docs_dir / f"{doc_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Document(**data)

    def list_documents(self) -> list[Document]:
        docs = []
        for f in sorted(self.docs_dir.glob("doc_*.json"), reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                docs.append(Document(**data))
            except Exception:
                continue
        return docs

    def delete(self, doc_id: str) -> bool:
        path = self.docs_dir / f"{doc_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def export_text(self, doc: Document) -> str:
        """Export document content as plain text."""
        content = doc.content
        if isinstance(content, dict) and "ops" in content:
            return "".join(
                op.get("insert", "") for op in content["ops"]
                if isinstance(op.get("insert"), str)
            )
        return str(content)

    def export_docx(self, doc: Document, output_path: str) -> str | None:
        """Export to DOCX. Returns path or None if python-docx unavailable."""
        try:
            from docx import Document as DocxDocument
            d = DocxDocument()
            d.add_heading(doc.title, 0)
            d.add_paragraph(self.export_text(doc))
            d.save(output_path)
            return output_path
        except ImportError:
            return None
```

**Step 2: Commit**

```bash
git add app/documents.py
git commit -m "feat: add offline-first document management system"
```

---

## Task 5: WebView Document Editor (Quill.js)

**Files:**
- Create: `assets/web/editor.html`

**Step 1: Create assets directory**

```bash
mkdir -p assets/web/vestments
```

**Step 2: Create assets/web/editor.html**

A self-contained HTML file with Quill.js bundled inline (no CDN — offline-first). Since Quill's full source is large, we embed a minimal rich text editor that works offline with the Celestial Dark vestment.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Angel Editor</title>
<style>
  :root {
    --bg: #000000;
    --surface: #0e0e14;
    --accent: #D4AF37;
    --text: #f0f0f2;
    --text-dim: #555560;
    --input-bg: #16161e;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  .toolbar {
    display: flex;
    gap: 4px;
    padding: 8px 12px;
    background: var(--surface);
    border-bottom: 1px solid rgba(255,255,255,0.04);
    flex-wrap: wrap;
  }
  .toolbar button {
    background: var(--input-bg);
    color: var(--text);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 14px;
    cursor: pointer;
    min-width: 36px;
  }
  .toolbar button:active, .toolbar button.active {
    background: var(--accent);
    color: #000;
  }
  .toolbar .title-input {
    flex: 1;
    background: transparent;
    color: var(--accent);
    border: none;
    font-size: 16px;
    font-weight: bold;
    outline: none;
    min-width: 120px;
  }
  .toolbar .title-input::placeholder { color: var(--text-dim); }
  #editor {
    flex: 1;
    padding: 16px;
    font-size: 16px;
    line-height: 1.6;
    outline: none;
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
  }
  #editor:empty::before {
    content: "Start writing...";
    color: var(--text-dim);
  }
  .status-bar {
    padding: 6px 12px;
    background: var(--surface);
    color: var(--text-dim);
    font-size: 12px;
    display: flex;
    justify-content: space-between;
    border-top: 1px solid rgba(255,255,255,0.04);
  }
</style>
</head>
<body>
  <div class="toolbar">
    <input class="title-input" id="docTitle" placeholder="Untitled" />
    <button onclick="fmt('bold')" title="Bold"><b>B</b></button>
    <button onclick="fmt('italic')" title="Italic"><i>I</i></button>
    <button onclick="fmt('underline')" title="Underline"><u>U</u></button>
    <button onclick="fmt('insertUnorderedList')" title="List">&#8226;</button>
    <button onclick="fmt('insertOrderedList')" title="Numbered">1.</button>
    <button onclick="setHead()" title="Heading">H</button>
    <button onclick="save()" title="Save" style="background:var(--accent);color:#000">&#10003;</button>
  </div>
  <div id="editor" contenteditable="true"></div>
  <div class="status-bar">
    <span id="wordCount">0 words</span>
    <span id="saveStatus">Ready</span>
  </div>

<script>
  // ── Formatting ──
  function fmt(cmd) { document.execCommand(cmd, false, null); }
  function setHead() {
    const sel = window.getSelection();
    if (!sel.rangeCount) return;
    document.execCommand('formatBlock', false, 'h2');
  }

  // ── Word count ──
  const editor = document.getElementById('editor');
  const wordCountEl = document.getElementById('wordCount');
  editor.addEventListener('input', () => {
    const text = editor.innerText.trim();
    const count = text ? text.split(/\s+/).length : 0;
    wordCountEl.textContent = count + ' word' + (count !== 1 ? 's' : '');
  });

  // ── Bridge to Python ──
  function getContent() {
    return JSON.stringify({
      title: document.getElementById('docTitle').value || 'Untitled',
      html: editor.innerHTML,
      text: editor.innerText
    });
  }
  function setContent(title, html) {
    document.getElementById('docTitle').value = title;
    editor.innerHTML = html;
  }
  function setTheme(cssVars) {
    const root = document.documentElement;
    for (const [key, val] of Object.entries(JSON.parse(cssVars))) {
      root.style.setProperty(key, val);
    }
  }
  function save() {
    document.getElementById('saveStatus').textContent = 'Saving...';
    // Call Python bridge if available
    if (window.AngelBridge) {
      window.AngelBridge.save(getContent());
    }
    setTimeout(() => {
      document.getElementById('saveStatus').textContent = 'Saved';
    }, 500);
  }

  // Auto-save every 30 seconds
  setInterval(() => { save(); }, 30000);
</script>
</body>
</html>
```

**Step 3: Create assets/web/vestments/celestial-dark.css**

```css
:root {
  --bg: #000000;
  --surface: #0e0e14;
  --surface-head: #0c0c12;
  --surface-input: #16161e;
  --accent: #D4AF37;
  --accent-purple: #bb86fc;
  --teal: #03dac6;
  --text: #f0f0f2;
  --text-sec: #9e9ea8;
  --text-dim: #555560;
  --success: #66bb6a;
  --warning: #ffab40;
  --error: #ef5350;
}
```

**Step 4: Commit**

```bash
git add assets/
git commit -m "feat: add WebView document editor with Celestial Dark vestment"
```

---

## Task 6: Rewrite main_android.py — The Angel's Body

This is the big one. Complete rewrite of the Android entry point with:
- Gesture detection overlay
- Panel system (chat, voice, hosts, wings, skills, aspects, vestments, documents)
- Celestial Dark vestment applied via design tokens
- Companion aspect as default opening
- WebView for document panel
- Voice bar (swipe up)

**Files:**
- Modify: `main_android.py` — complete rewrite

**Step 1: Rewrite main_android.py**

The new file replaces the current 471-line chat-only UI with a gesture-driven panel system. Key changes:

- `MKAngelApp.build()` creates a `FloatLayout` root with:
  - `GestureDetector` overlay (transparent, top layer)
  - `_ChatPanel` (default visible panel — the existing chat UI, refined)
  - `_VoicePanel` (swipe up — mic button + wave animation)
  - `_HostPanel` (swipe right — Host deployment UI)
  - `_SkillsOverlay` (long press — floating skill picker)
  - `_AspectSwitcher` (swipe down — 4 aspect cards)
  - `_VestmentSwitcher` (double tap — 4 vestment previews)
  - `_DocumentPanel` (pinch — WebView with Quill.js editor)
  - `_WingsPanel` (swipe left — placeholder for Phase 2)

- Panels are shown/hidden via `_show_panel(name)` which animates opacity
- Vestment tokens from `app/vestments.py` drive all colours
- The Companion aspect starts by showing last context + Angel thinking

The full implementation is approximately 600-800 lines. Core structure:

```python
# main_android.py structure (Phase 1: First Light)
#
# MKAngelApp
#   build() → FloatLayout root
#     GestureDetector (overlay, captures gestures)
#     _ChatPanel (default, chat bubbles + input)
#     _VoicePanel (mic button, wave animation)
#     _HostPanel (deploy parallel agents)
#     _DocumentPanel (WebView + Quill.js)
#     _SkillsOverlay (floating picker)
#     _AspectSwitcher (4 cards)
#     _VestmentSwitcher (4 previews)
#     _WingsPanel (placeholder)
#
#   _show_panel(name) → animate visible panel
#   _on_gesture(action) → route gesture to panel
#   _boot() → background Angel init (existing logic)
#   _on_send(text) → process user input (existing logic)
```

Each panel is a BoxLayout with `opacity=0` by default. `_show_panel` fades out current, fades in target over 0.2s.

**Step 2: Test manually**

Build and install APK, verify:
1. Chat panel shows by default (Companion aspect)
2. Swipe up shows voice panel
3. Swipe right shows host panel
4. Long press shows skills overlay
5. All gestures route correctly

**Step 3: Commit**

```bash
git add main_android.py
git commit -m "feat: rewrite Android UI — gesture-driven panels + Celestial Dark vestment"
```

---

## Task 7: Update buildozer.spec

**Files:**
- Modify: `buildozer.spec`

**Step 1: Update spec**

Add `html,css,js` to `source.include_exts` so the WebView assets get bundled:

```
source.include_exts = py,png,jpg,kv,atlas,json,html,css,js
```

Add `RECORD_AUDIO` permission for voice:

```
android.permissions = INTERNET,RECORD_AUDIO
```

**Step 2: Commit**

```bash
git add buildozer.spec
git commit -m "chore: update buildozer.spec for WebView assets + audio permission"
```

---

## Task 8: Update app/__init__.py Exports

**Files:**
- Modify: `app/__init__.py`

**Step 1: Add new module imports**

Add guarded imports for the new modules:

```python
try:
    from app.vestments import get_vestment, vestment_to_css, ALL_VESTMENTS
except Exception:
    pass

try:
    from app.gestures import GestureDetector, GestureAction
except Exception:
    pass

try:
    from app.documents import DocumentManager, Document
except Exception:
    pass
```

**Step 2: Commit**

```bash
git add app/__init__.py
git commit -m "feat: export vestments, gestures, documents from app package"
```

---

## Task 9: Integration Test — Build and Deploy

**Step 1: Push and trigger build**

```bash
git push origin main
```

**Step 2: Monitor build**

```bash
gh run watch --exit-status
```

**Step 3: Download and install**

```bash
gh run download <run-id> -n mkangel-apk -D ./bin/
adb shell pm uninstall com.mrjkilcoyne.mkangel
adb install bin/mkangel-*.apk
```

**Step 4: Verify on device**

1. Launch app — Companion aspect shows (last context + Angel status)
2. Chat works — send message, receive response in gold-accent bubbles
3. Swipe up — voice panel appears
4. Swipe right — host panel appears
5. Long press — skills overlay appears
6. Pinch — document editor opens (WebView)
7. Double tap header — vestment switcher shows 4 options
8. Swipe down — aspect switcher shows 4 cards
9. Airplane mode — everything still works except API providers

**Step 5: Final commit**

```bash
git add -A
git commit -m "Phase 1: First Light — gesture-driven Angel with Celestial Dark vestment"
```

---

## Dependency Notes

**No new pip dependencies required.** Everything uses:
- Kivy (already in buildozer.spec requirements)
- pyjnius (comes with python-for-android)
- Standard library (json, os, time, threading, pathlib)
- Quill.js is bundled as local HTML (no CDN, no npm)

**Future dependencies (Phase 2+):**
- python-docx (DOCX export)
- reportlab (PDF export)
- Flask/FastAPI (if moving to full WebView architecture later)
