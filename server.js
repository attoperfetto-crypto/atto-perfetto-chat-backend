import express from "express";
import cors from "cors";
import helmet from "helmet";
import compression from "compression";
import multer from "multer";
import fs from "fs";
import path from "path";
import pdfParse from "pdf-parse";
import { OpenAI } from "openai";
import Database from "better-sqlite3";
import { v4 as uuidv4 } from "uuid";

// -------------------------
// Config (Render-safe paths)
// -------------------------
const PORT = process.env.PORT || 10000;
const DATA_DIR = process.env.DATA_DIR || "/opt/render/project/data"; // writable on Render
const UPLOAD_DIR = `${DATA_DIR}/uploads`;
const DB_PATH = `${DATA_DIR}/atto_perfetto_chat.sqlite`;

fs.mkdirSync(DATA_DIR, { recursive: true });
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// -------------------------
// OpenAI
// -------------------------
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) console.error("Missing OPENAI_API_KEY in Render → Environment");

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// default economical model; override in Render env OPENAI_MODEL if needed
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";

// Safety knobs
const MAX_HISTORY_MESSAGES = Number(process.env.MAX_HISTORY_MESSAGES || 20); // memory per convo
const MAX_USER_MESSAGE_CHARS = Number(process.env.MAX_USER_MESSAGE_CHARS || 12000);
const MAX_EXTRACTED_TEXT_CHARS = Number(process.env.MAX_EXTRACTED_TEXT_CHARS || 700000);

// -------------------------
// App
// -------------------------
const app = express();
app.set("trust proxy", 1);

// CORS molto permissivo (ideale in fase di test Builderall)
app.use(cors({
  origin: "*",
  methods: ["GET","POST","PUT","DELETE","OPTIONS"],
  allowedHeaders: ["Content-Type","X-User-Id"],
}));

// Rispondi sempre ai preflight
app.options("*", cors());

app.use(helmet({
  crossOriginResourcePolicy: false,
  contentSecurityPolicy: false
}));
app.use(compression());
app.use(express.json({ limit: "20mb" }));

// -------------------------
// DB
// -------------------------
const db = new Database(DB_PATH);
db.pragma("journal_mode = WAL");

db.exec(`
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT DEFAULT 'Chat',
  folder TEXT DEFAULT 'Generale',
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  role TEXT NOT NULL, -- 'user' | 'assistant' | 'system'
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

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
USING fts5(
  conversation_id,
  user_id,
  role,
  content,
  content='messages',
  content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, conversation_id, user_id, role, content)
  VALUES (new.id, new.conversation_id, new.user_id, new.role, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, conversation_id, user_id, role, content)
  VALUES ('delete', old.id, old.conversation_id, old.user_id, old.role, old.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, conversation_id, user_id, role, content)
  VALUES ('delete', old.id, old.conversation_id, old.user_id, old.role, old.content);
  INSERT INTO messages_fts(rowid, conversation_id, user_id, role, content)
  VALUES (new.id, new.conversation_id, new.user_id, new.role, new.content);
END;
`);

const stmtEnsureUser = db.prepare(`INSERT OR IGNORE INTO users(id) VALUES (?)`);

const stmtUpsertConversation = db.prepare(`
  INSERT OR REPLACE INTO conversations(id, user_id, title, folder, created_at, updated_at)
  VALUES (
    @id, @user_id, @title, @folder,
    COALESCE((SELECT created_at FROM conversations WHERE id=@id AND user_id=@user_id), datetime('now')),
    datetime('now')
  )
`);

const stmtGetConversation = db.prepare(`
  SELECT id, title, folder, created_at, updated_at
  FROM conversations
  WHERE id=? AND user_id=?
`);

const stmtListConversations = db.prepare(`
  SELECT id, title, folder, updated_at
  FROM conversations
  WHERE user_id=?
  ORDER BY datetime(updated_at) DESC
`);

const stmtUpdateConversation = db.prepare(`
  UPDATE conversations
  SET title=@title, folder=@folder, updated_at=datetime('now')
  WHERE id=@id AND user_id=@user_id
`);

const stmtTouchConversation = db.prepare(`
  UPDATE conversations SET updated_at=datetime('now')
  WHERE id=? AND user_id=?
`);

const stmtInsertMessage = db.prepare(`
  INSERT INTO messages(conversation_id, user_id, role, content)
  VALUES (?, ?, ?, ?)
`);

const stmtGetMessages = db.prepare(`
  SELECT role, content, created_at
  FROM messages
  WHERE conversation_id=? AND user_id=?
  ORDER BY id ASC
`);

const stmtGetLastMessages = db.prepare(`
  SELECT role, content
  FROM messages
  WHERE conversation_id=? AND user_id=?
  ORDER BY id DESC
  LIMIT ?
`);

const stmtInsertFile = db.prepare(`
  INSERT OR REPLACE INTO files(id, conversation_id, user_id, filename, mimetype, size, stored_path, extracted_text)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?)
`);

const stmtGetFilesByConversation = db.prepare(`
  SELECT id, filename, mimetype, size, created_at
  FROM files
  WHERE conversation_id=? AND user_id=?
  ORDER BY datetime(created_at) ASC
`);

const stmtGetFileById = db.prepare(`
  SELECT * FROM files
  WHERE id=? AND user_id=?
`);

// -------------------------
// User middleware
// -------------------------
app.use((req, res, next) => {
  let userId = req.header("x-user-id")?.trim();
  if (!userId) userId = "u_anon";
  stmtEnsureUser.run(userId);
  req.userId = userId;
  next();
});

// -------------------------
// Upload (multer)
// -------------------------
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    const ts = Date.now();
    const rnd = Math.random().toString(16).slice(2);
    const ext = path.extname(file.originalname || ".pdf") || ".pdf";
    cb(null, `${ts}_${rnd}${ext}`);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 30 * 1024 * 1024 } // 30MB each
});

// -------------------------
// SSE helpers
// -------------------------
function sseHeaders(res) {
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();
}
function sseEvent(res, event, data) {
  res.write(`event: ${event}\n`);
  const payload = String(data ?? "").replace(/\n/g, "\\n");
  res.write(`data: ${payload}\n\n`);
}

// -------------------------
// System prompt (senior)
// -------------------------
function systemPromptSenior() {
  return `
Sei “Atto Perfetto – Avvocato Senior” (Italia, diritto civile). 
Agisci come un legale esperto: impostazione rigorosa, concreta, orientata a strategia e redazione.

Regole:
- Tono professionale, tecnico ma chiaro.
- Non inventare dati/fatti: se manca qualcosa, indica cosa manca e perché è importante.
- Evidenzia: struttura, domanda, eccezioni, onere probatorio, criticità, contraddizioni e linee difensive/offensive.
- Se l’utente chiede un atto, proponi una scaletta e poi una bozza pulita, con sezioni e stile forense.
- Se l’utente chiede analisi, produci un’analisi ordinata per punti e suggerisci come rafforzare.

Se sono presenti documenti caricati, puoi usarli come base ma senza trascrivere integralmente testi lunghi.
  `.trim();
}

// -------------------------
// Chunking
// -------------------------
function chunkText(text, maxChars = 6500) {
  const clean = (text || "").replace(/\r/g, "").trim();
  if (!clean) return [];
  const chunks = [];
  let i = 0;
  while (i < clean.length) {
    chunks.push(clean.slice(i, i + maxChars));
    i += maxChars;
  }
  return chunks;
}

// -------------------------
// Routes
// -------------------------
app.get("/health", (req, res) => {
  res.json({ ok: true, model: OPENAI_MODEL, time: new Date().toISOString() });
});

// Create conversation
app.post("/api/conversations", (req, res) => {
  const userId = req.userId;
  const { id, title, folder } = req.body || {};
  if (!id) return res.status(400).json({ error: "Missing id" });

  stmtUpsertConversation.run({
    id,
    user_id: userId,
    title: title || "Chat Atto Perfetto",
    folder: folder || "Generale"
  });

  // First system msg (optional)
  stmtInsertMessage.run(id, userId, "system", "Sessione avviata.");
  stmtTouchConversation.run(id, userId);

  res.json({ ok: true, id });
});

// List conversations
app.get("/api/conversations", (req, res) => {
  const userId = req.userId;
  res.json(stmtListConversations.all(userId));
});

// Get conversation detail
app.get("/api/conversations/:id", (req, res) => {
  const userId = req.userId;
  const id = req.params.id;

  const conv = stmtGetConversation.get(id, userId);
  if (!conv) return res.status(404).json({ error: "Conversation not found" });

  const messages = stmtGetMessages.all(id, userId);
  const files = stmtGetFilesByConversation.all(id, userId);

  res.json({ conversation: conv, messages, files });
});

// Update conversation (title/folder)
app.put("/api/conversations/:id", (req, res) => {
  const userId = req.userId;
  const id = req.params.id;
  const { title, folder } = req.body || {};

  const conv = stmtGetConversation.get(id, userId);
  if (!conv) return res.status(404).json({ error: "Conversation not found" });

  stmtUpdateConversation.run({
    id,
    user_id: userId,
    title: title || conv.title,
    folder: folder || conv.folder
  });

  res.json({ ok: true });
});

// Global search (FTS)
app.get("/api/search", (req, res) => {
  const userId = req.userId;
  const q = (req.query.q || "").toString().trim();
  if (!q) return res.json([]);

  const safe = q.replace(/"/g, '""');
  const sql = `
    SELECT conversation_id,
           snippet(messages_fts, 3, '⟦', '⟧', '…', 14) AS snippet
    FROM messages_fts
    WHERE messages_fts MATCH ?
      AND user_id = ?
    ORDER BY rank
    LIMIT 50
  `;

  try {
    const stmt = db.prepare(sql);
    const rows = stmt.all(`"${safe}"`, userId);
    res.json(rows);
  } catch {
    try {
      const stmt = db.prepare(sql);
      const rows = stmt.all(safe, userId);
      res.json(rows);
    } catch {
      res.status(500).json({ error: "Search error" });
    }
  }
});

// Upload multi PDFs
app.post("/api/upload", upload.array("files", 10), async (req, res) => {
  const userId = req.userId;
  const conversationId = req.body.conversationId;

  if (!conversationId) return res.status(400).json({ error: "Missing conversationId" });

  // ensure conversation exists
  const existing = stmtGetConversation.get(conversationId, userId);
  if (!existing) {
    stmtUpsertConversation.run({
      id: conversationId,
      user_id: userId,
      title: "Chat Atto Perfetto",
      folder: "Generale"
    });
  }

  const files = req.files || [];
  if (!files.length) return res.status(400).json({ error: "No files uploaded" });

  const out = [];
  for (const f of files) {
    const fileId = "f_" + uuidv4();

    let extracted = "";
    try {
      const buf = fs.readFileSync(f.path);
      const parsed = await pdfParse(buf);
      extracted = (parsed?.text || "").trim();
      if (extracted.length > MAX_EXTRACTED_TEXT_CHARS) {
        extracted = extracted.slice(0, MAX_EXTRACTED_TEXT_CHARS);
      }
    } catch {
      extracted = "";
    }

    stmtInsertFile.run(
      fileId,
      conversationId,
      userId,
      f.originalname,
      f.mimetype,
      f.size,
      f.path,
      extracted
    );

    out.push({ id: fileId, name: f.originalname, size: f.size });
  }

  stmtTouchConversation.run(conversationId, userId);
  stmtInsertMessage.run(
    conversationId,
    userId,
    "user",
    `Caricati ${out.length} PDF: ${out.map(x => x.name).join(" • ")}`
  );
  stmtTouchConversation.run(conversationId, userId);

  res.json({ ok: true, files: out });
});

// -------------------------
// CHAT LIBERA (SSE streaming)
// -------------------------
app.post("/api/chat/stream", async (req, res) => {
  const userId = req.userId;
  const { conversationId, message } = req.body || {};

  if (!conversationId) return res.status(400).json({ error: "Missing conversationId" });
  if (!message || !String(message).trim()) return res.status(400).json({ error: "Missing message" });
  if (!OPENAI_API_KEY) return res.status(500).json({ error: "OPENAI_API_KEY missing on server" });

  const msg = String(message).slice(0, MAX_USER_MESSAGE_CHARS);

  // ensure conversation exists
  const conv = stmtGetConversation.get(conversationId, userId);
  if (!conv) {
    stmtUpsertConversation.run({
      id: conversationId,
      user_id: userId,
      title: "Chat Atto Perfetto",
      folder: "Generale"
    });
    stmtInsertMessage.run(conversationId, userId, "system", "Sessione avviata.");
  }

  // save user msg
  stmtInsertMessage.run(conversationId, userId, "user", msg);
  stmtTouchConversation.run(conversationId, userId);

  // build memory (last N)
  const last = stmtGetLastMessages.all(conversationId, userId, MAX_HISTORY_MESSAGES).reverse();

  const messages = [
    { role: "system", content: systemPromptSenior() },
    ...last.map(m => ({ role: m.role === "system" ? "system" : m.role, content: m.content }))
  ];

  // SSE stream
  sseHeaders(res);
  sseEvent(res, "meta", JSON.stringify({ model: OPENAI_MODEL }));

  try {
    const stream = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      temperature: 0.25,
      stream: true,
      messages
    });

    let full = "";
    for await (const part of stream) {
      const delta = part?.choices?.[0]?.delta?.content || "";
      if (delta) {
        full += delta;
        sseEvent(res, "delta", delta);
      }
    }

    const finalText = full.trim() || "—";
    stmtInsertMessage.run(conversationId, userId, "assistant", finalText);
    stmtTouchConversation.run(conversationId, userId);

    sseEvent(res, "done", "");
    res.end();
  } catch (e) {
    sseEvent(res, "error", e?.message || "Errore chat");
    sseEvent(res, "done", "");
    res.end();
  }
});

// -------------------------
// PDF ANALYZE -> REPORT (SSE)
// -------------------------
app.post("/api/analyze/stream", async (req, res) => {
  const userId = req.userId;
  const { conversationId, fileIds } = req.body || {};

  if (!conversationId) return res.status(400).json({ error: "Missing conversationId" });
  if (!Array.isArray(fileIds) || fileIds.length === 0) {
    return res.status(400).json({ error: "Missing fileIds" });
  }
  if (!OPENAI_API_KEY) return res.status(500).json({ error: "OPENAI_API_KEY missing on server" });

  // ensure conversation exists
  const conv = stmtGetConversation.get(conversationId, userId);
  if (!conv) {
    stmtUpsertConversation.run({
      id: conversationId,
      user_id: userId,
      title: "Chat Atto Perfetto",
      folder: "Generale"
    });
    stmtInsertMessage.run(conversationId, userId, "system", "Sessione avviata.");
  }

  sseHeaders(res);
  sseEvent(res, "meta", JSON.stringify({ model: OPENAI_MODEL, files: fileIds.length }));

  // Load files
  const fileRows = [];
  for (const id of fileIds) {
    const row = stmtGetFileById.get(id, userId);
    if (row) fileRows.push(row);
  }
  if (!fileRows.length) {
    sseEvent(res, "error", "Nessun file valido trovato.");
    sseEvent(res, "done", "");
    return res.end();
  }

  const perDocReports = [];
  let globalReportText = "";

  try {
    for (let idx = 0; idx < fileRows.length; idx++) {
      const f = fileRows[idx];
      const text = (f.extracted_text || "").trim();
      sseEvent(res, "status", `Documento ${idx + 1}/${fileRows.length}: ${f.filename}`);

      if (!text) {
        const warn = `⚠️ "${f.filename}": testo non estratto (PDF scannerizzato/immagine).`;
        const docRep = `# REPORT DOCUMENTO: ${f.filename}\n\n${warn}\n`;
        perDocReports.push({ filename: f.filename, report: docRep });
        globalReportText += `\n\n${docRep}`;
        sseEvent(res, "delta", `\n\n${docRep}\n`);
        continue;
      }

      const chunks = chunkText(text, 6500);
      const partial = [];

      for (let c = 0; c < chunks.length; c++) {
        sseEvent(res, "progress", JSON.stringify({ file: f.filename, chunk: c + 1, totalChunks: chunks.length }));

        const chunkPrompt = `
Analizza questo estratto di atto/documento (diritto civile italiano).

Produci note tecniche (max 20 righe) su:
- punti di forza
- debolezze/contraddizioni/omissioni
- istituti e fattispecie (forte/debole)
- profilo probatorio (cosa manca)
- suggerimenti operativi

TESTO:
${chunks[c]}
        `.trim();

        const r = await openai.chat.completions.create({
          model: OPENAI_MODEL,
          temperature: 0.2,
          messages: [
            { role: "system", content: systemPromptSenior() },
            { role: "user", content: chunkPrompt }
          ]
        });

        partial.push(r?.choices?.[0]?.message?.content?.trim() || "");
      }

      const consolidatePrompt = `
Consolida le note parziali in un report unico e completo per "${f.filename}".
Usa questa struttura OBBLIGATORIA:

# REPORT DOCUMENTO: ${f.filename}

## 1) Sintesi fedele (max 12 righe)
## 2) Punti di forza
## 3) Debolezze / Contraddizioni / Omissioni (con suggerimenti operativi)
## 4) Istituti e fattispecie (forte/debole + pertinenza)
## 5) Prove e onere probatorio (cosa c’è / cosa manca)
## 6) Strategie operative (prossimi atti possibili)

NOTE PARZIALI:
${partial.join("\n\n---\n\n")}
      `.trim();

      const r2 = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0.2,
        messages: [
          { role: "system", content: systemPromptSenior() },
          { role: "user", content: consolidatePrompt }
        ]
      });

      const docReport = r2?.choices?.[0]?.message?.content?.trim() || "";
      perDocReports.push({ filename: f.filename, report: docReport });
      globalReportText += `\n\n${docReport}`;

      sseEvent(res, "delta", `\n\n${docReport}\n`);
      sseEvent(res, "separator", "\n\n---\n\n");
    }

    if (perDocReports.length > 1) {
      sseEvent(res, "status", "Confronto complessivo tra documenti…");

      const comparePrompt = `
Sono stati analizzati più documenti relativi ad una controversia civile.

Genera una sezione finale con:
# CONFRONTO COMPLESSIVO E INDICAZIONI STRATEGICHE

1) Confronto tra tesi e argomentazioni (coerenza, diritto, prova)
2) Dove uno tende a prevalere sull’altro e perché
3) Punti discordanti e vulnerabilità reciproche
4) Suggerimenti per ENTRAMBE le parti (attore e convenuto) sui prossimi atti
5) Spunti per gestione del rischio / transazione (se sensato)

REPORT DOCUMENTI:
${perDocReports.map(d => `DOCUMENTO: ${d.filename}\n${d.report}`).join("\n\n=====\n\n")}
      `.trim();

      const r3 = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0.2,
        messages: [
          { role: "system", content: systemPromptSenior() },
          { role: "user", content: comparePrompt }
        ]
      });

      const compareOut = r3?.choices?.[0]?.message?.content?.trim() || "";
      globalReportText += `\n\n${compareOut}`;
      sseEvent(res, "delta", `\n\n${compareOut}\n`);
    }

    // Save report as assistant msg (so searchable globally)
    stmtInsertMessage.run(conversationId, userId, "assistant", globalReportText.trim() || "Report generato.");
    stmtTouchConversation.run(conversationId, userId);

    sseEvent(res, "done", "");
    return res.end();
  } catch (e) {
    sseEvent(res, "error", e?.message || "Errore durante l’analisi");
    sseEvent(res, "done", "");
    return res.end();
  }
});

// -------------------------
// Start
// -------------------------
app.listen(PORT, () => {
  console.log(`✅ Atto Perfetto Chat backend on port ${PORT}`);
  console.log(`MODEL: ${OPENAI_MODEL}`);
  console.log(`DATA_DIR: ${DATA_DIR}`);
});
