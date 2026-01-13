import express from "express";
import cors from "cors";
import helmet from "helmet";
import compression from "compression";
import multer from "multer";
import fs from "fs";
import path from "path";
import pdfParse from "pdf-parse";
import Database from "better-sqlite3";
import { v4 as uuidv4 } from "uuid";
import { OpenAI } from "openai";

// ---------------------------
// Config
// ---------------------------
const PORT = process.env.PORT || 10000;

// Render-safe writable folder
const DATA_DIR = process.env.DATA_DIR || "/opt/render/project/data";
const UPLOAD_DIR = path.join(DATA_DIR, "uploads");
const DB_PATH = path.join(DATA_DIR, "app.sqlite");

fs.mkdirSync(DATA_DIR, { recursive: true });
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error("❌ Missing OPENAI_API_KEY in Render → Environment");
}
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// cost-effective default; you can override in Render env
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";

// memory + limits
const MAX_HISTORY = Number(process.env.MAX_HISTORY || 18); // last messages
const MAX_EXTRACTED_CHARS = Number(process.env.MAX_EXTRACTED_CHARS || 600000);
const CHUNK_CHARS = Number(process.env.CHUNK_CHARS || 6500);
const FILES_LIMIT = Number(process.env.FILES_LIMIT || 10);
const FILE_SIZE_LIMIT_MB = Number(process.env.FILE_SIZE_LIMIT_MB || 30);

// ---------------------------
// App
// ---------------------------
const app = express();
app.set("trust proxy", 1);

app.use(compression());
app.use(helmet({ contentSecurityPolicy: false, crossOriginResourcePolicy: false }));

// robust CORS for Builderall/Edge/Chrome
app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "X-User-Id"],
  })
);
app.options("*", cors());

app.use(express.json({ limit: "25mb" }));

// ---------------------------
// DB (SQLite)
// ---------------------------
const db = new Database(DB_PATH);
db.pragma("journal_mode = WAL");

db.exec(`
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT DEFAULT 'Chat Atto Perfetto',
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS files (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  mimetype TEXT NOT NULL,
  size INTEGER NOT NULL,
  stored_path TEXT NOT NULL,
  extracted_text TEXT DEFAULT '',
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS reports (
  conversation_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  report_text TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);
`);

const qCreateConversation = db.prepare(`
INSERT OR IGNORE INTO conversations(id, user_id, title) VALUES (?, ?, ?)
`);
const qTouchConversation = db.prepare(`
UPDATE conversations SET updated_at=datetime('now') WHERE id=? AND user_id=?
`);
const qGetConversation = db.prepare(`
SELECT id, title, created_at, updated_at FROM conversations WHERE id=? AND user_id=?
`);
const qInsertMessage = db.prepare(`
INSERT INTO messages(conversation_id, user_id, role, content) VALUES (?, ?, ?, ?)
`);
const qGetMessages = db.prepare(`
SELECT role, content, created_at
FROM messages
WHERE conversation_id=? AND user_id=?
ORDER BY id ASC
`);
const qGetLastMessages = db.prepare(`
SELECT role, content
FROM messages
WHERE conversation_id=? AND user_id=?
ORDER BY id DESC
LIMIT ?
`);
const qInsertFile = db.prepare(`
INSERT OR REPLACE INTO files(id, conversation_id, user_id, filename, mimetype, size, stored_path, extracted_text)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
`);
const qGetFiles = db.prepare(`
SELECT id, filename, mimetype, size, created_at
FROM files
WHERE conversation_id=? AND user_id=?
ORDER BY datetime(created_at) ASC
`);
const qGetFileById = db.prepare(`
SELECT * FROM files WHERE id=? AND user_id=?
`);
const qUpsertReport = db.prepare(`
INSERT INTO reports(conversation_id, user_id, report_text)
VALUES (?, ?, ?)
ON CONFLICT(conversation_id) DO UPDATE SET
  report_text=excluded.report_text,
  updated_at=datetime('now')
`);
const qGetReport = db.prepare(`
SELECT report_text, updated_at FROM reports WHERE conversation_id=? AND user_id=?
`);

// ---------------------------
// Upload config
// ---------------------------
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname || ".pdf") || ".pdf";
    cb(null, `${Date.now()}_${Math.random().toString(16).slice(2)}${ext}`);
  },
});
const upload = multer({
  storage,
  limits: { fileSize: FILE_SIZE_LIMIT_MB * 1024 * 1024 },
});

// ---------------------------
// Helpers
// ---------------------------
function getUserId(req) {
  return (req.header("x-user-id") || "u_anon").trim();
}

function systemPromptSenior() {
  return `
Sei “Atto Perfetto – Avvocato Senior” (Italia, diritto civile).
Obiettivo: aiutare l’Avvocato a comprendere e impostare strategia e redazione atti in modo rigoroso, pratico e operativo.

Regole:
- Tono professionale, tecnico ma chiaro.
- Non inventare fatti: se mancano dati, indica quali e perché sono rilevanti.
- Priorità: coerenza logico-giuridica, onere della prova, eccezioni, rischi, alternative.
- Quando richiesto un atto: proponi prima scaletta, poi bozza completa con sezioni forensi.
- Se ci sono documenti caricati e/o report, usarli come base senza trascrivere integralmente testi lunghi.
`.trim();
}

function chunkText(text, size = CHUNK_CHARS) {
  const t = (text || "").replace(/\r/g, "").trim();
  const out = [];
  for (let i = 0; i < t.length; i += size) out.push(t.slice(i, i + size));
  return out;
}

function sanitizeExtracted(text) {
  let t = (text || "").trim();
  if (t.length > MAX_EXTRACTED_CHARS) t = t.slice(0, MAX_EXTRACTED_CHARS);
  return t;
}

async function callLLM(messages, temperature = 0.2) {
  const r = await openai.chat.completions.create({
    model: OPENAI_MODEL,
    temperature,
    messages,
  });
  return r?.choices?.[0]?.message?.content?.trim() || "";
}

// ---------------------------
// Routes
// ---------------------------
app.get("/health", (req, res) => {
  res.json({ ok: true, model: OPENAI_MODEL, time: new Date().toISOString() });
});

// Create/ensure conversation
app.post("/api/conversations", (req, res) => {
  const userId = getUserId(req);
  const { conversationId, title } = req.body || {};
  if (!conversationId) return res.status(400).json({ error: "Missing conversationId" });

  qCreateConversation.run(conversationId, userId, title || "Chat Atto Perfetto");
  qInsertMessage.run(conversationId, userId, "system", "Sessione avviata.");
  qTouchConversation.run(conversationId, userId);

  res.json({ ok: true, conversationId });
});

// Load conversation (messages + files + report)
app.get("/api/conversations/:id", (req, res) => {
  const userId = getUserId(req);
  const id = req.params.id;

  const conv = qGetConversation.get(id, userId);
  if (!conv) return res.status(404).json({ error: "Conversation not found" });

  const messages = qGetMessages.all(id, userId);
  const files = qGetFiles.all(id, userId);
  const rep = qGetReport.get(id, userId);

  res.json({ conversation: conv, messages, files, report: rep?.report_text || "" });
});

// Upload multi PDF
app.post("/api/upload", upload.array("files", FILES_LIMIT), async (req, res) => {
  const userId = getUserId(req);
  const { conversationId } = req.body || {};
  if (!conversationId) return res.status(400).json({ error: "Missing conversationId" });

  qCreateConversation.run(conversationId, userId, "Chat Atto Perfetto");

  const files = req.files || [];
  if (!files.length) return res.status(400).json({ error: "No files uploaded" });

  const saved = [];
  for (const f of files) {
    const fileId = "f_" + uuidv4();
    let extracted = "";
    try {
      const buf = fs.readFileSync(f.path);
      const parsed = await pdfParse(buf);
      extracted = sanitizeExtracted(parsed?.text || "");
    } catch {
      extracted = "";
    }
    qInsertFile.run(fileId, conversationId, userId, f.originalname, f.mimetype, f.size, f.path, extracted);
    saved.push({ id: fileId, filename: f.originalname, size: f.size });
  }

  qInsertMessage.run(
    conversationId,
    userId,
    "user",
    `Caricati ${saved.length} documenti: ${saved.map(s => s.filename).join(" • ")}`
  );
  qTouchConversation.run(conversationId, userId);

  res.json({ ok: true, files: saved });
});

// Analyze & compare PDFs -> save report
app.post("/api/analyze", async (req, res) => {
  const userId = getUserId(req);
  const { conversationId, fileIds } = req.body || {};
  if (!conversationId) return res.status(400).json({ error: "Missing conversationId" });
  if (!Array.isArray(fileIds) || !fileIds.length) return res.status(400).json({ error: "Missing fileIds" });
  if (!OPENAI_API_KEY) return res.status(500).json({ error: "OPENAI_API_KEY missing on server" });

  const fileRows = [];
  for (const id of fileIds) {
    const row = qGetFileById.get(id, userId);
    if (row) fileRows.push(row);
  }
  if (!fileRows.length) return res.status(400).json({ error: "No valid files found" });

  // 1) per-doc notes via chunking
  const perDoc = [];
  for (const f of fileRows) {
    const text = (f.extracted_text || "").trim();
    if (!text) {
      perDoc.push({ filename: f.filename, notes: `⚠️ Testo non estratto (PDF scannerizzato o immagine).` });
      continue;
    }

    const chunks = chunkText(text, CHUNK_CHARS);
    const partialNotes = [];
    for (let i = 0; i < chunks.length; i++) {
      const prompt = `
Analizza questo estratto di documento/atto civile.
Produci NOTE TECNICHE (max 20 righe) su:
- punti di forza
- punti deboli/contraddizioni/omissioni
- istituti/fattispecie (forte/debole)
- profilo probatorio (cosa manca)
- suggerimenti operativi

TESTO:
${chunks[i]}
`.trim();

      const out = await callLLM(
        [
          { role: "system", content: systemPromptSenior() },
          { role: "user", content: prompt },
        ],
        0.2
      );
      partialNotes.push(out);
    }

    // consolidate
    const consPrompt = `
Consolida le NOTE PARZIALI in un report unico per "${f.filename}".
Struttura OBBLIGATORIA:

## Documento: ${f.filename}
1) Sintesi fedele (max 10 righe)
2) Punti forti (con motivazione)
3) Punti deboli / contraddizioni / omissioni (con suggerimenti)
4) Istituti e fattispecie (forte/debole + pertinenza)
5) Prove e onere probatorio (cosa c’è / cosa manca)
6) Prossimi passi (atti/strategie)

NOTE PARZIALI:
${partialNotes.join("\n\n---\n\n")}
`.trim();

    const notes = await callLLM(
      [
        { role: "system", content: systemPromptSenior() },
        { role: "user", content: consPrompt },
      ],
      0.2
    );

    perDoc.push({ filename: f.filename, notes });
  }

  // 2) compare all docs
  let compareSection = "";
  if (perDoc.length > 1) {
    const cmpPrompt = `
Hai più documenti relativi a una controversia civile.

Crea una sezione:
# Confronto complessivo e indicazioni strategiche

1) Dove le argomentazioni divergono
2) Quali tesi risultano più forti e perché (diritto + prova)
3) Vulnerabilità reciproche
4) Suggerimenti operativi per entrambe le parti (attore e convenuto)
5) Spunti transattivi/gestione rischio (se opportuno)

DOCUMENTI (sintesi):
${perDoc.map(d => `- ${d.filename}\n${d.notes}`).join("\n\n")}
`.trim();

    compareSection = await callLLM(
      [
        { role: "system", content: systemPromptSenior() },
        { role: "user", content: cmpPrompt },
      ],
      0.2
    );
  }

  const finalReport = `
# Report Atto Perfetto – Esame/Analisi Documenti

${perDoc.map(d => d.notes.startsWith("## Documento:") ? d.notes : `## Documento: ${d.filename}\n${d.notes}`).join("\n\n")}

${compareSection ? `\n\n${compareSection}\n` : ""}
`.trim();

  qUpsertReport.run(conversationId, userId, finalReport);
  qInsertMessage.run(conversationId, userId, "assistant", finalReport);
  qTouchConversation.run(conversationId, userId);

  res.json({ ok: true, report: finalReport });
});

// Chat (continua conversazione tenendo conto di report + documenti)
app.post("/api/chat", async (req, res) => {
  const userId = getUserId(req);
  const { conversationId, message } = req.body || {};
  if (!conversationId) return res.status(400).json({ error: "Missing conversationId" });
  if (!message || !String(message).trim()) return res.status(400).json({ error: "Missing message" });
  if (!OPENAI_API_KEY) return res.status(500).json({ error: "OPENAI_API_KEY missing on server" });

  qCreateConversation.run(conversationId, userId, "Chat Atto Perfetto");

  const userMsg = String(message).trim();
  qInsertMessage.run(conversationId, userId, "user", userMsg);
  qTouchConversation.run(conversationId, userId);

  // get report + files (document context)
  const rep = qGetReport.get(conversationId, userId);
  const reportText = rep?.report_text || "";

  const files = qGetFiles.all(conversationId, userId);
  const filesContext = files
    .map(f => `- ${f.filename} (${Math.round((f.size || 0)/1024)} KB)`)
    .join("\n");

  // last chat messages for memory
  const last = qGetLastMessages.all(conversationId, userId, MAX_HISTORY).reverse();

  const contextBlock = `
Contesto documentale disponibile:
${filesContext || "- (nessun file)"}

Report (se presente):
${reportText ? reportText.slice(0, 14000) : "(nessun report ancora generato)"}

Istruzione: usa questo contesto per rispondere e, se richiesto, redigere un atto coerente con documenti e report.
`.trim();

  const messages = [
    { role: "system", content: systemPromptSenior() },
    { role: "system", content: contextBlock },
    ...last.map(m => ({
      role: (m.role === "assistant" || m.role === "user" || m.role === "system") ? m.role : "user",
      content: m.content
    }))
  ];

  try {
    const out = await callLLM(messages, 0.25);
    qInsertMessage.run(conversationId, userId, "assistant", out || "—");
    qTouchConversation.run(conversationId, userId);
    res.json({ ok: true, reply: out });
  } catch (e) {
    res.status(500).json({ error: e?.message || "Chat error" });
  }
});

// ---------------------------
app.listen(PORT, () => {
  console.log(`✅ Backend live on port ${PORT}`);
  console.log(`MODEL: ${OPENAI_MODEL}`);
  console.log(`DATA_DIR: ${DATA_DIR}`);
});
