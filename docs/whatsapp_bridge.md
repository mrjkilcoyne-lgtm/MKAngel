# MKAngel WhatsApp Bridge

Message a WhatsApp number, have a Claude agent work on the MKAngel repo,
get the result back as a WhatsApp message. Runs on your Android phone
via Termux, or on any Linux VPS.

## How it works

```
WhatsApp  <->  Baileys (Node)  <->  bridge.py  <->  Claude Agent SDK  <->  MKAngel repo
```

- **Baileys** is a free, community library that talks to WhatsApp's
  multi-device protocol by pretending to be a linked device. You pair
  it once by scanning a QR code from WhatsApp > Linked Devices.
- **bridge.py** is a small asyncio loop that spawns the Baileys
  subprocess, enforces the allow-list, and runs one Claude Agent SDK
  session per inbound message.
- **claude-agent-sdk** gives the agent Read, Edit, Write, Bash, Glob
  and Grep tools with its working directory pinned to the MKAngel
  checkout, so it can commit and push just like this Claude Code
  session can.

## Prerequisites

- A spare WhatsApp account for the bot (see "Getting a bot number"
  below).
- Your own WhatsApp number to message the bot from.
- An Anthropic API key (https://console.anthropic.com).
- Node.js 18+ and Python 3.10+.
- SSH or `gh` auth already set up so the agent can `git push`.

## Getting a bot number (pick one)

1. **Second SIM / eSIM.** Cheapest eSIM from Giffgaff/Smarty/etc.,
   register WhatsApp on it, done. Best option -- the bot has its own
   identity.
2. **WhatsApp Business app.** Install WhatsApp Business on the same
   phone alongside regular WhatsApp, register it to a second number
   you already own (landline works; it calls you to read out the code).
3. **Twilio WhatsApp Sandbox.** Free tier, no SIM needed, but you join
   via a sandbox keyword every 72 hours. Code would need to be swapped
   from Baileys to Twilio -- not wired up in v1.

## Install on Termux (Android)

```sh
# One-off Termux setup
pkg update
pkg install -y nodejs python git openssh
termux-setup-storage

# Clone MKAngel (if not already)
cd ~
git clone https://github.com/mrjkilcoyne-lgtm/MKAngel.git
cd MKAngel

# Configure the bridge
cp app/whatsapp/.env.example app/whatsapp/.env
nano app/whatsapp/.env
# -> set ANTHROPIC_API_KEY
# -> set WHATSAPP_ALLOWLIST to your own WhatsApp number (digits, no +)

# Run it (first launch installs dependencies automatically)
./scripts/run_whatsapp_bridge.sh
```

On the first run you'll see an ASCII QR code in the terminal. On your
phone: **WhatsApp > Settings > Linked Devices > Link a device** and
point the camera at the terminal. Pairing state is saved in
`app/whatsapp/baileys/auth_info/` so future launches don't need the QR.

Keep Termux alive with `termux-wake-lock` if you want the bridge to
survive screen-off.

## Install on a Linux VPS

```sh
# Debian/Ubuntu
sudo apt update
sudo apt install -y nodejs npm python3 python3-pip git

git clone https://github.com/mrjkilcoyne-lgtm/MKAngel.git
cd MKAngel
cp app/whatsapp/.env.example app/whatsapp/.env
${EDITOR:-nano} app/whatsapp/.env

./scripts/run_whatsapp_bridge.sh
```

For "always on" use a systemd unit (example snippet):

```ini
[Unit]
Description=MKAngel WhatsApp Bridge
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/matt/MKAngel
ExecStart=/home/matt/MKAngel/scripts/run_whatsapp_bridge.sh
Restart=on-failure
User=matt

[Install]
WantedBy=multi-user.target
```

## Using it

Once paired, just WhatsApp the bot from your allow-listed number. A few
warm-up prompts:

- `what branch am i on and what's dirty?`
- `read docs/freebies_canzuk.md and summarise`
- `in glm/angel.py what does StrangeLoop do?`
- `add a TODO to CLAUDE.md under Gotchas about X, commit and push`

The agent replies in-thread. If a message comes in while one is in
flight, you'll get a `working on your previous one, queued this`
acknowledgement.

## Safety notes

- **Allow-list** is the only authentication. Anyone messaging the bot
  number from a number not in `WHATSAPP_ALLOWLIST` is silently ignored.
  Do not publish the bot's number.
- The agent runs with `permission_mode="acceptEdits"` -- it will edit
  and commit without asking. Treat it as a real developer with push
  access to this branch. Do not run it pointing at `main`; keep the
  `WHATSAPP_REPO_ROOT` on a feature branch.
- All inbound and outbound messages are logged to
  `app/whatsapp/whatsapp.log` for audit.
- The `.env` file and Baileys `auth_info/` directory are both
  git-ignored -- never commit either.

## Roadmap

- [ ] Conversational memory across messages (use `app.memory`).
- [ ] Voice note transcription on the Baileys side (Whisper via
      `app.voice`).
- [ ] Multi-user allow-list with per-user repo checkouts.
- [ ] Twilio transport as an alternative to Baileys.
- [ ] Slash-commands (`/branch`, `/status`, `/freebies`) for common
      operations without spinning up a full agent turn.
