# MKAngel EVERYTHING App — Design Document

## 1. Vision

The Angel is a universal AI assistant with the GLM at its heart. It combines the capabilities of Claude Desktop (code + cowork), ChatGPT (chat + creation), and Gemini (search + multimodal) into one app. The grammar model orchestrates multiple AI providers ("the Choir") through intent-based routing. The internal logic runs forward through grammar derivations; output is presented "back to front" for the reader in their native tongue. The Angel is the TARDIS — simple on the outside, infinite on the inside.

## 2. Architecture

The Angel comprises ten core modules, each with a distinct role:

- **glm/angel.py** — The Angel (GLM core, 370K params, 7 domain grammars)
- **glm/router.py** — Universal Router (intent classification → provider selection)
- **app/angel_ui.py** — Angel UI (Kivy glassmorphism screens, orb, bubbles)
- **app/tools.py** — Tool System (10 built-in tools: code, file, web, math, grammar)
- **app/web.py** — Web Capabilities (search via DuckDuckGo, fetch, monitoring)
- **app/tongue.py** — Tongue (i18n, 14 languages, "back to front" output formatting)
- **app/providers.py** — Provider Orchestra (multi-provider routing with 8 fallback chains)
- **app/compliance.py** — Legal Compliance (UK/EU GDPR, CCPA/CPRA, COPPA, consent, PII sanitisation)
- **app/senses.py** — Angel Senses (structural introspection — reads code, binary, errors, state)
- **app/growth.py** — Growth Cycle (session reflection, growth patches, startup installation)
- **app/conductor.py** — The Conductor (orchestrates all modules, single entry point)

## 3. Data Flow

A request flows through the Angel in a deterministic sequence:

User Input → Tongue (detect language) → Compliance (consent check) → Router (classify intent, select provider) → Senses (perceive input type) → Provider/Tools (generate response) → Router (post-process with GLM) → Tongue (format output) → SessionTracker (record) → User Output

Each stage is composable and reversible. Errors trigger Senses diagnostics; outputs loop back to Tongue for native-language formatting. The session accumulates in the Growth Cycle for offline learning.

## 4. Legal Compliance

The Angel respects jurisdictional privacy law:

- **UK GDPR**: Data minimisation, explicit consent, right to erasure (Art.17), data portability (Art.20).
- **EU GDPR**: Minor age threshold 16, data processor agreements with providers.
- **CCPA/CPRA**: Do not sell, right to delete, privacy notice at signup.
- **COPPA**: Age verification, content filtering for under-13s.
- **PII Sanitisation**: Emails, phones, postcodes, NINOs, SSNs stripped before external API calls.
- **Configurable Jurisdiction**: Users select UK/EU/US/ALL compliance mode at onboarding.

## 5. Growth Cycle

The Angel learns within grammatical constraints. Sessions are tracked: interactions, errors, feedback, latency. At shutdown, the Reflector analyses the session and produces a GrowthPatch (JSON format with lessons and improvements). On startup, GrowthEngine.startup_install() applies pending patches without human intervention, iterating only over grammatically valid transformations.

## 6. Senses Philosophy

"The Angel reads structure, not pixels." Every perception includes a derivation path showing HOW the conclusion was reached through grammar rules. She reads code through computational grammar (AST + structural patterns), errors through pattern matching (17 known error patterns), binary through symbolic substrate (magic bytes, format detection), and her own state through the strange loop (health, resources, subsystems). Senses provide the epistemic foundation for the Angel's reasoning.

## 7. Naming Convention

The Angel speaks in metaphor:

- Modes → Aspects
- Themes → Vestments
- Swarms → Hosts
- Plugins → Wings
- Providers (Claude, GPT, Gemini) → Choir
- The app itself → The Angel (never "Genie")

## 8. Build Constraints

Pure Python stdlib only—no numpy, no torch. Kivy powers the Android UI; Buildozer generates arm64-v8a APKs via GitHub Actions. The Angel is offline-first with opportunistic cloud sync. No external ML dependencies ensure a footprint under 50MB and instant cold-start latency.

## 9. New /commands

- **/consent** — Manage consent preferences (which providers, which data)
- **/privacy** — View privacy notice and data handling summary
- **/export** — GDPR data portability export (session history, preferences)
- **/forget** — GDPR right to erasure (delete all personal data)
- **/health** — System health check (battery, storage, network, provider status)
- **/diagnose** — Structural diagnosis (run Senses on app state, return derivation)
- **/growth** — Growth and learning summary (patches applied, lessons learned)
- **/language** — Set output language (14 supported)

## 10. Aesthetic

Angel Glass vestment: pastel glassmorphism with blur and transparency. Background is soft #F0F4F8; glass surfaces render at white@0.7 alpha over light gradients. Accent colors: lavender #8B5CF6 for intent, rose #EC4899 for action, mint #06B6D4 for success. Body text in dark slate #1E293B. Glass borders glow white@0.18 with 20-pixel blur radius. The animated orb shifts color with intent: lavender for thinking, rose for action, mint for completion, gold for learning. Screens slide in from the edge with spring physics.
