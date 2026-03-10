# MKAngel Redesign — "The Angel's Architecture"

**Date**: 2026-03-10
**Author**: Matt Kilcoyne + Claude
**Status**: Approved

## Context

MKAngel has a fully working GLM engine (7 domain grammars, strange-loop detection, ~370K param neural model), a complete app layer (12 modules: chat, providers, voice, Hosts, cloud, self-improvement, skills, coder, cowork, settings, memory), and a functioning Android APK. The engine is real. The problem is the *experience* — the current Kivy UI is a dark chat terminal, not an angel. This redesign transforms MKAngel from a working prototype into an inspired, angelic, world-class personal assistant.

## Design Principles

Drawn from LINE (Japan), WeChat (China), KakaoTalk (Korea), ElevenLabs, and MCP:

1. **Chat is the universal container** — everything flows through conversation
2. **Progressive disclosure** — depth hidden behind simplicity
3. **Offline-first** — works in airplane mode, syncs when connected
4. **Start narrow, expand horizontally** — Phase 1 earns the right to Phase 2
5. **Don't sacrifice core value for features** (KakaoTalk's lesson)
6. **Voice should feel like conversation, not commands** (ElevenLabs' turn-taking)
7. **The Angel adapts to you, not you to her** (Grab's context-aware home screen)

## Angelic Naming Convention

| Old Term | Angelic Term | Meaning |
|----------|-------------|---------|
| Modes | **Aspects** | Opening experiences: Awakening, Companion, Command Centre, Oracle |
| Themes | **Vestments** | Visual identities: Celestial Dark, Ethereal Light, Living Gradient, Minimal Power |
| Swarms | **Hosts** | Parallel agent groups working in concert |
| Plugins | **Wings** | Extensions that give the Angel new powers |
| Providers | **Choir** | The ensemble of AI voices (OpenAI, Anthropic, Google, Mistral, Groq, Local GLM) |
| Memory | **Memory** | Stays — already angelic |
| Skills | **Skills** | Stays |
| Voice | **Voice** | Her actual voice, cloneable |

## Architecture

```
+---------------------------------------------------+
|              Android APK Shell (Kivy)              |
+------------------+--------------------------------+
|  Kivy Native     |     Embedded WebView           |
|  - Gestures      |     - Document Editor (Quill)  |
|  - Chat UI       |     - Rich Content Panels      |
|  - Voice Bar     |     - Vestment Animations      |
|  - Navigation    |     - Data Visualizations      |
|  - Header        |     (Local bundled files only)  |
+------------------+--------------------------------+
|             Python Backend (on-device)             |
|  - Angel (GLM)        - Hosts (parallel agents)   |
|  - ChatSession         - Skills                    |
|  - Choir (providers)   - Voice Engine              |
|  - Memory (SQLite)     - Self-Improver             |
|  - Cloud (sync-when-online)  - Coder               |
+---------------------------------------------------+
|            Local Storage (offline-first)            |
|  - ~/.mkangel/memory.db   - Documents folder       |
|  - ~/.mkangel/settings    - Learned patterns        |
|  - ~/.mkangel/skills/     - Voice profiles          |
+---------------------------------------------------+
```

### Hybrid Kivy + WebView

- **Kivy handles**: App shell, gesture detection, chat bubbles, voice bar, navigation, system integration (camera, mic, file system)
- **WebView handles**: Document editor (Quill.js), rich content panels, Vestment-specific animations, data visualizations
- **All WebView content is local** — HTML/CSS/JS bundled in the APK, no internet required
- **Communication**: Python ↔ WebView via JavaScript bridge (pyjnius WebView.evaluateJavascript + addJavascriptInterface)

### Offline-First

- All data stored locally in SQLite (memory.db) and JSON files
- Documents save to device storage, exportable to DOCX/PDF
- Cloud sync is opportunistic — pushes when connectivity appears
- GLM engine is pure Python, runs entirely on-device
- Voice transcription available offline via local Whisper model
- API providers (Choir) degrade gracefully when offline — falls back to local GLM

## Gesture Navigation — The Full Wingspan

Every power is one gesture away. No menus. No hamburgers. Pure interaction.

| Gesture | Action | What Opens |
|---------|--------|------------|
| Tap | Default | Chat — conversation with the Angel |
| Swipe Up from bottom | Voice | Mic activates, voice-first mode |
| Swipe Right from left edge | Hosts | Deploy parallel agents on a task |
| Swipe Left from right edge | Wings | Extensions, connections, new powers |
| Long Press anywhere | Skills | Quick skill picker overlay |
| Swipe Down from top | Aspect Switcher | Awakening / Companion / Command Centre / Oracle |
| Double Tap header | Vestment Switcher | Change visual theme |
| Pinch | Document Mode | Open the embedded Quill.js editor |

Implementation: Kivy's touch event system (on_touch_down, on_touch_move, on_touch_up) with velocity and direction detection. GestureDatabase for recognition accuracy.

## The Four Aspects

### The Awakening
Brief animation (wings of light expanding from centre, 2-3 seconds), then a single conversational greeting. Used on cold launch or when the user wants ceremony. Canvas-drawn wing animation with easing.

### The Companion
No splash. The Angel shows: last conversation context, what she's been thinking (patterns detected, connections found since last session). Immediate warmth. Default after first launch.

### The Command Centre
Dashboard layout: domains active (7 indicators), patterns learned today, Choir status (which providers online/offline), voice ready indicator, document count, Host activity. Functional beauty. For power users.

### The Oracle
The Angel opens with a prediction — something derived from recent conversations, cross-domain pattern matching, or superforecasting. A gift each time you arrive. Uses Angel.superforecast() on recent memory entries.

User preference stored in settings. Angel can also auto-select based on context (time of day, usage pattern).

## The Four Vestments

All share a common design token system. Kivy elements use token values; WebView elements use CSS custom properties mapped to the same tokens.

### Celestial Dark
- Background: OLED black (#000000)
- Primary accent: Gold (#D4AF37)
- Text: Soft white (#F5F5F5)
- Secondary: Deep purple (#1A0A2E)
- Chat bubbles: Dark surface (#1E1E1E) with gold border glow
- Special: Constellation particle field (Canvas dots with slow drift animation)

### Ethereal Light
- Background: Warm white (#FAFAFA)
- Primary accent: Silver (#C0C0C0)
- Text: Dark grey (#333333)
- Secondary: Pale gold (#F5E6CC)
- Chat bubbles: Frosted glass (semi-transparent white with blur)
- Special: Dawn light gradient at top (subtle warm → cool transition)

### Living Gradient
- Background: Animated deep purple → aurora green → celestial blue
- Primary accent: Bright cyan (#00FFFF)
- Text: White (#FFFFFF)
- Secondary: Magenta (#FF00FF)
- Chat bubbles: Semi-transparent with gradient bleed-through
- Special: Background animation breathing at ~4-second cycle

### Minimal Power
- Background: Near-white (#FCFCFC) or near-black (#0A0A0A) based on system dark mode
- Primary accent: Single strong colour (user-configurable, default: deep blue #1A237E)
- Text: High contrast
- Chat bubbles: Subtle borders, no fills
- Special: Typography-driven. Large type for key information. Maximum whitespace.

## Extensibility — Three Layers

### Layer 1: MCP (Immediate Power)
Model Context Protocol client built into the Angel. Connects to:
- Zapier MCP server (30,000+ actions)
- Custom MCP servers (user-configured)
- Local tool servers

Implementation: HTTP client in providers.py extended with MCP protocol support. Actions discoverable via MCP's tool listing. Angel reasons about which tools to invoke based on user intent.

### Layer 2: Web Osmosis (Autonomous Learning)
The Angel observes:
- API response patterns (what formats work, what fails)
- User correction patterns (how the user refines outputs)
- Successful skill chains (which sequences of actions solve problems)

She builds internal models: learned patterns stored in self_improve.py's pattern system, new skills auto-generated and proposed to user for approval.

### Layer 3: Wings Marketplace (Community) — Future Phase
Developer API for creating Wings (plugins). Each Wing is a Python package with:
- Manifest (name, description, required permissions)
- Entry point (callable that receives Angel context)
- UI component (optional Kivy widget or WebView panel)

Distribution via Git repositories initially, dedicated marketplace later.

## Document Creation

### Architecture
- Quill.js rich text editor embedded in Android WebView
- Loaded from local HTML/CSS/JS files bundled in APK
- Communication via JavaScript bridge: Python ↔ Quill

### Capabilities
- Full formatting: headings, bold, italic, underline, lists, tables, images
- Angel-assisted editing: user says "rewrite this paragraph" in chat, Angel edits via JS bridge
- Voice dictation: Swipe up while in document mode, speak, Angel transcribes and inserts
- Offline save: Documents stored as Quill Delta JSON in ~/.mkangel/documents/
- Export: DOCX via python-docx, PDF via reportlab
- Cloud sync: Pushes to configured cloud storage when online

### File Management
- Local document list with search
- Recent documents in Companion aspect
- Auto-save every 30 seconds
- Version history (last 10 saves per document)

## Voice System

### Input (Speech-to-Text)
- Primary: Android SpeechRecognizer via pyjnius (offline capable)
- Fallback: OpenAI Whisper API (online)
- Local: Whisper model on-device (if dependency available)

### Output (Text-to-Speech)
- Primary: pyttsx3 (offline, immediate)
- Enhanced: Coqui TTS (neural quality, offline)
- Premium: ElevenLabs API (when online, highest quality)

### Voice Cloning
- YourTTS model for speaker adaptation
- 3-5 minutes of audio to clone
- Stored as voice profiles in ~/.mkangel/voices/

### Interaction Design
- Swipe up activates mic — visual wave animation
- Partial transcription shown in real-time
- Angel responds with voice (if TTS enabled) + text
- Turn-taking: debounce silence detection (1.5s pause = end of utterance)

## Predictive Text

### Architecture
Custom prediction layer on top of Kivy TextInput:
- on_text callback triggers prediction (debounced 200ms)
- GLM grammar analysis for domain-specific suggestions
- N-gram model trained on user's message history (from memory.db)
- Suggestions shown in floating overlay above keyboard

### Sources (layered, fastest first)
1. User's own vocabulary (from message history)
2. GLM grammar completions (domain-aware)
3. API-powered suggestions (when online, optional)

## Phased Delivery

### Phase 1 — "First Light"
The next build. Gets the Angel on the phone with dignity.

- Hybrid Kivy + WebView shell
- Gesture navigation (all 8 gestures)
- Celestial Dark vestment (primary theme)
- Chat UI with proper bubble design
- Voice input (swipe up → mic)
- Hosts working (swipe right → deploy)
- Basic document creation (Quill.js)
- Offline-first architecture
- Companion aspect as default

### Phase 2 — "Full Wingspan"
All 4 Vestments rendered. All 4 Aspects working.

- Ethereal Light, Living Gradient, Minimal Power vestments
- Awakening animation, Command Centre dashboard, Oracle predictions
- Quill.js editor with Angel-assisted editing
- MCP integration (Layer 1 extensibility)
- Predictive text overlay
- Voice cloning setup

### Phase 3 — "Ascension"
The Angel becomes a platform.

- Web osmosis (Layer 2 — autonomous learning)
- Wings marketplace (Layer 3 — community extensions)
- Cross-device sync
- Full voice cloning in-app
- Advanced predictive text (personal language model)

## Key Files to Modify/Create

### Modify
- `main_android.py` — Complete rewrite: gesture shell + WebView integration
- `buildozer.spec` — Add WebView dependencies, bundle local HTML/CSS/JS
- `app/chat.py` — Adapt for GUI callbacks instead of terminal ANSI
- `app/swarm.py` — Rename swarm → host throughout
- `app/__init__.py` — Update exports for new naming

### Create
- `assets/web/` — Local HTML/CSS/JS for WebView panels
- `assets/web/editor.html` — Quill.js document editor
- `assets/web/vestments/` — CSS files for each vestment
- `assets/web/aspects/` — HTML/JS for each aspect's opening
- `app/gestures.py` — Gesture recognition and routing
- `app/documents.py` — Document management (save, load, export, sync)
- `app/mcp.py` — MCP client for extensibility Layer 1
- `app/wings.py` — Wing (plugin) loading and management

## Testing & Verification

1. **Build**: GitHub Actions CI produces APK
2. **Install**: `adb install` to Pixel 10 Pro XL
3. **Gestures**: Verify all 8 gestures activate correct panels
4. **Chat**: Send message, receive Angel response with proper bubble styling
5. **Voice**: Swipe up, speak, verify transcription appears
6. **Documents**: Pinch to open editor, type, save, verify file exists
7. **Offline**: Enable airplane mode, verify all above still works
8. **Hosts**: Swipe right, deploy Host, verify parallel agent execution
9. **Vestments**: Double-tap header, switch theme, verify visual change
10. **Aspects**: Swipe down, switch aspect, verify opening experience changes
