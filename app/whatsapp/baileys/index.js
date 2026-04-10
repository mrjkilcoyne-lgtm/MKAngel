/*
 * Baileys transport for the MKAngel WhatsApp bridge.
 *
 * Reads send commands as newline-delimited JSON from stdin and writes
 * events as newline-delimited JSON to stdout. Logs (including the pairing
 * QR code) go to stderr so they don't pollute the JSON channel.
 *
 * Events out (stdout):
 *     {"type":"ready"}
 *     {"type":"qr","qr":"<ascii-art qr code>"}
 *     {"type":"msg","from":"447700900123@s.whatsapp.net","text":"..."}
 *     {"type":"audio","from":"447700900123@s.whatsapp.net","path":"/abs/path.ogg","mimetype":"audio/ogg"}
 *     {"type":"error","error":"..."}
 *
 * Commands in (stdin):
 *     {"type":"send","to":"447700900123@s.whatsapp.net","text":"..."}
 *
 * Auth state is persisted in ./auth_info/ so you only scan the QR once.
 */

const fs = require("fs");
const path = require("path");
const readline = require("readline");
const pino = require("pino");
const qrcode = require("qrcode-terminal");
const {
  default: makeWASocket,
  useMultiFileAuthState,
  DisconnectReason,
  fetchLatestBaileysVersion,
  downloadMediaMessage,
} = require("@whiskeysockets/baileys");

const AUTH_DIR = path.join(__dirname, "auth_info");
const TMP_DIR = path.join(__dirname, "tmp");

// Shared pino logger so downloadMediaMessage has something to call into.
const logger = pino({ level: "warn" });

function emit(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

function logErr(...args) {
  // Anything on stderr gets forwarded as [baileys] log lines by the Python side.
  console.error(...args);
}

async function main() {
  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version } = await fetchLatestBaileysVersion();
  logErr(`using WA version ${version.join(".")}`);

  const sock = makeWASocket({
    version,
    auth: state,
    printQRInTerminal: false,
    logger,
    browser: ["MKAngel", "Chrome", "1.0.0"],
  });

  sock.ev.on("creds.update", saveCreds);

  sock.ev.on("connection.update", (update) => {
    const { connection, lastDisconnect, qr } = update;
    if (qr) {
      // Render ASCII QR to stderr for easy scanning in a terminal.
      qrcode.generate(qr, { small: true }, (ascii) => {
        logErr("\n" + ascii);
      });
      emit({ type: "qr", qr });
    }
    if (connection === "open") {
      logErr("baileys: connection open");
      emit({ type: "ready" });
    }
    if (connection === "close") {
      const statusCode =
        lastDisconnect && lastDisconnect.error
          ? lastDisconnect.error.output && lastDisconnect.error.output.statusCode
          : undefined;
      const shouldReconnect = statusCode !== DisconnectReason.loggedOut;
      logErr(`baileys: connection closed (status ${statusCode}); reconnect=${shouldReconnect}`);
      if (shouldReconnect) {
        setTimeout(() => main().catch((e) => emit({ type: "error", error: String(e) })), 2000);
      } else {
        emit({ type: "error", error: "logged out; delete auth_info/ and rescan" });
        process.exit(1);
      }
    }
  });

  sock.ev.on("messages.upsert", async (payload) => {
    if (payload.type !== "notify") return;
    for (const m of payload.messages) {
      if (!m.message || m.key.fromMe) continue;
      const from = m.key.remoteJid || "";
      // Only handle 1:1 personal chats (not groups, not newsletters).
      if (!from.endsWith("@s.whatsapp.net")) continue;

      // Voice notes: audioMessage covers both normal audio and ptt
      // (push-to-talk). WhatsApp voice notes are opus in an ogg container.
      const audioMsg =
        m.message.audioMessage || m.message.pttMessage || null;
      if (audioMsg) {
        try {
          if (!fs.existsSync(TMP_DIR)) {
            fs.mkdirSync(TMP_DIR, { recursive: true });
          }
          const buffer = await downloadMediaMessage(
            m,
            "buffer",
            {},
            { logger, reuploadRequest: sock.updateMediaMessage },
          );
          const mimetype = audioMsg.mimetype || "audio/ogg";
          const ext = mimetype.includes("ogg")
            ? "ogg"
            : mimetype.includes("mp4")
              ? "m4a"
              : mimetype.includes("mpeg")
                ? "mp3"
                : "bin";
          const rand = Math.random().toString(36).slice(2, 10);
          const filename = `audio-${Date.now()}-${rand}.${ext}`;
          const absPath = path.join(TMP_DIR, filename);
          fs.writeFileSync(absPath, buffer);
          emit({ type: "audio", from, path: absPath, mimetype });
        } catch (e) {
          emit({ type: "error", error: `audio download failed: ${e.message}` });
        }
        continue;
      }

      const text =
        (m.message.conversation) ||
        (m.message.extendedTextMessage && m.message.extendedTextMessage.text) ||
        (m.message.imageMessage && m.message.imageMessage.caption) ||
        "";
      if (!text) continue;
      emit({ type: "msg", from, text });
    }
  });

  // Read outbound commands from stdin, line by line.
  const rl = readline.createInterface({ input: process.stdin });
  rl.on("line", async (line) => {
    line = line.trim();
    if (!line) return;
    let cmd;
    try {
      cmd = JSON.parse(line);
    } catch (e) {
      emit({ type: "error", error: `bad json from python: ${e.message}` });
      return;
    }
    if (cmd.type === "send" && cmd.to && typeof cmd.text === "string") {
      try {
        await sock.sendMessage(cmd.to, { text: cmd.text });
      } catch (e) {
        emit({ type: "error", error: `send failed: ${e.message}` });
      }
    }
  });

  rl.on("close", () => {
    logErr("stdin closed; shutting down");
    process.exit(0);
  });
}

main().catch((e) => {
  emit({ type: "error", error: String(e) });
  process.exit(1);
});
