# MKAngel -- Project Memory

## Architecture
- **glm/** -- Grammar Language Model core (pure Python, no heavy deps)
  - core/ -- Grammar, Rule, Production, StrangeLoop, Substrate, Lexicon, DerivationEngine
  - grammars/ -- Domain rule sets: linguistic, etymological, chemical, biological, computational, mathematical, physics
  - substrates/ -- Phonological, morphological, molecular, symbolic, mathematical
  - model/ -- Neural GLM with FugueAttention, StrangeLoopAttention, TemporalAttention (370K params, pure Python)
  - mnemo/ -- Mnemonic language system
  - angel.py -- The Angel: unifies all layers
- **app/** -- Application layer: Settings, Memory (sqlite3), Providers, ChatSession, Skills, Coder, Voice, Swarm, Cloud, SelfImprove
- **main_android.py** -- Kivy Android entry point (must be copied to main.py at build time)
- **buildozer.spec** -- Android build config

## Build Notes
- Buildozer's `entrypoint` setting does NOT work -- always `cp main_android.py main.py` before building
- `yes | buildozer -v android debug` -- pipe yes to handle SDK license prompts and root warnings
- Requirements: `python3,kivy,pysqlite3` (pure Python stack, no numpy/torch)
- GitHub Actions workflow at `.github/workflows/build-apk.yml` -- must be on `main` branch to appear in Actions tab
- Target: arm64-v8a, API 33, minAPI 24

## User Context
- User works from Android phone / GitHub mobile app -- keep instructions concise and phone-friendly
- Repo: mrjkilcoyne-lgtm/MKAngel

## Gotchas
- Workflow files on feature branches won't show in the GitHub Actions tab until merged to main
- `apt-get update` required before installing packages in CI/Colab (404 errors otherwise)
- sqlite3 is used by app/memory.py -- needs pysqlite3 in buildozer requirements
