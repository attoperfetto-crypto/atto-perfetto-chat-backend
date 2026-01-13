import express from "express";
import cors from "cors";
import helmet from "helmet";
import compression from "compression";
import cookieParser from "cookie-parser";
import multer from "multer";
import fs from "fs";
import path from "path";
import pdfParse from "pdf-parse";
import { OpenAI } from "openai";
import Database from "better-sqlite3";

// -------------------------
// Render-safe storage paths
// -------------------------
const PORT = process.env.PORT || 10000;

// IMPORTANT: Render-safe writable dir (works without Disk)
const DATA_DIR = process.env.DATA_DIR || "/opt/render/project/data";
const UPLOAD_DIR = `${DATA_DIR}/uploads`;
const DB_PATH = `${DATA_DIR}/atto_perfetto.sqlite`;

fs.mkdirSync(DATA_DIR, { recursive: true });
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// -------------------------
// OpenAI
// -------------------------
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error("Missing OPENAI_API_KEY in Render → Environment");
}
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// Default model (you can change in Render env: OPENAI_MODEL)
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";

// -------------------------
// App
// -------------------------
const app = express();

// CORS (set to your domain in production if you want strict)
app.use(cors({ origin: true, credentials: true }));

app.use(helmet({
  crossOriginResourcePolicy: false,
}));
app.use(compression());
app.use(express.json({ limit: "10mb" }));
app.use(cookieParser());

// -------------------------
// DB (SQLite)
// -------------------------
const db = new Database(DB_PATH);
db.pragma("journal_mode = WAL");

// Tables
db.exec(`
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT DEFAULT 'Conversazione',
  folder TEXT DEFAULT 'Generale',
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  role TEXT NOT NULL, -- 'user' | 'assistant'
  content TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(conversation_id) REFERENCES conversations(id)
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
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(conversation_id) REFERENCES conversations(id)
);

-- FTS for global search over messages
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
USING fts5(
  conversation_id,
  user_id,
  role,
  content,
  content='messages',
  content_rowid='id'
);

-- triggers to keep FTS in sync
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

// Prepared statements
const stmtEnsureUser = db.prepare(`INSERT OR IGNORE INTO users(id) VALUES (?)`);

const stmtCreateConversation = db.prepare(`
  INSERT OR REPLACE INTO conversations(id, user_id, title, folder, created_at, updated_at)
  VALUES (@id, @user_id, @title, @folder,
    COALESCE((SELECT created_at FROM conversations WHERE id=@id AND user_id=@user_id), datetime('now')),
    datetime('now')
  )
`);

const stmtListConversations = db.prepare(`
  SELECT id, title, folder, updated_at
  FROM conversations
  WHERE user_id=?
  ORDER BY datetime(updated_at) DESC
`);

const stmtGetConversation = db.prepare(`
  SELECT id, title, folder, created_at, updated_at
  FROM conversations
  WHERE id=? AND user_id=?
`);

const stmtUpdateConversation = db.prepare(`
  UPDATE conversations
  SET title=@title, folder=@folder, updated_at=datetime('now')
  WHERE id=@id AND user_id=@user_id
`);

const stmtInsertMessage = db.prepare(`
  INSERT INTO messages(conversation_id, user_id, role, content)
  VALUES (?, ?, ?, ?)
`);

const stmtUpdateConversationTouched = db.prepare(`
  UPDATE conversations SET updated_at=datetime('now')
  WHERE id=? AND user_id=?
`);

const stmtGetMessages = db.prepare(`
  SELECT role, content, created_at
  FROM messages
  WHERE conversation_id=? AND user_id=?
  ORDER BY id ASC
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
  SELECT *
  FROM files
  WHERE id=? AND user_id=?
`);

// -------------------------
// User identification middleware
// -------------------------
app.use((req, res, next) => {
  // prefer header X-User-Id from your Builderall HTML
  let userId = req.header("x-user-id")?.trim();

  // Optional cookies (if you ever want)
  if (!userId) userId = req.cookies?.atto_user_id;

  // Fallback
  if (!userId) userId = "u_anon";

  // Persist user
  stmtEnsureUser.run(userId);

  req.userId = userId;
  next();
});

// -------------------------
// Multer (upload)
// -------------------------
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    // safe unique name
    const ts = Date.now();
    const rand = Math.random().toString(16).slice(2);
    const ext = path.extname(file.originalname || ".pdf") || ".pdf";
    cb(null, `${ts}_${rand}${ext}`);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 25 * 1024 * 1024 } // 25MB per file (adjust)
});

// -------------------------
// Health
// -------------------------
app.get("/health", (req, res) => {
  res.json({ ok: true, time: new Date().toISOString() });
});

// -------------------------
// Conversations
// -------------------------
app.post("/api/conversations", (req, res) => {
  const userId = req.userId;
  const { id, title, folder } = req.body || {};

  if (!id) return res.status(400).json({ error: "Missing id" });

  stmtCreateConversation.run({
    id,
    user_id: userId,
    title: title || "Nuova conversazione",
    folder: folder || "Generale",
  });

  res.json({ ok: true, id });
});

app.get("/api/conversations", (req, res) => {
  const userId = req.userId;
  const rows = stmtListConversations.all(userId);
  res.json(rows);
});

app.get("/api/conversations/:id", (req, res) => {
  const userId = req.userId;
  const id = req.params.id;

  const conv = stmtGetConversation.get(id, userId);
  if (!conv) return res.status(404).json({ error: "Conversation not found" });

  const messages = stmtGetMessages.all(id, userId);
  const files = stmtGetFilesByConversation.all(id, userId);

  res.json({ conversation: conv, messages, files });
});

app.put("/api/conversations/:id", (req, res) => {
  const userId = req.userId;
  const id = req.params.id;
  const { title, folder } = req.body || {};

  const conv = stmtGetConversation.get(id, userId);
  if (!conv) return res.status(404).json({ error: "Conversation not found" });

  stmtUpdateConversation.run({
    id,
    user_id: userId,
    title: title || conv.title || "Conversazione",
    folder: folder || conv.folder || "Generale",
  });

  res.json({ ok: true });
});

// -------------------------
// Global search (FTS)
// -------------------------
app.get("/api/search", (req, res) => {
  const userId = req.userId;
  const q = (req.query.q || "").toString().trim();
  if (!q) return res.json([]);

  // Use FTS match query (basic)
  // Escape quotes
  const safe = q.replace(/"/g, '""');
  const sql = `
    SELECT
      conversation_id,
      snippet(messages_fts, 3, '⟦', '⟧', '…', 12) AS snippet
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
  } catch (e) {
    // fallback: no quotes (some FTS cases)
    try {
      const stmt = db.prepare(sql);
      const rows = stmt.all(safe, userId);
      res.json(rows);
    } catch (e2) {
      res.status(500).json({ error: "Search error" });
    }
  }
});

// -------------------------
// Upload PDF (multi-file)
// -------------------------
app.post("/api/upload", upload.array("files", 10), async (req, res) => {
  const userId = req.userId;
  const conversationId = req.body.conversationId;

  if (!conversationId) {
    return res.status(400).json({ error: "Missing conversationId" });
  }

  // Ensure conversation exists
  const conv = stmtGetConversation.get(conversationId, userId);
  if (!conv) {
    stmtCreateConversation.run({
      id: conversationId,
      user_id: userId,
      title: "Conversazione",
      folder: "Generale"
    });
  }

  const files = req.files || [];
  if (!files.length) return res.status(400).json({ error: "No files uploaded" });

  const out = [];

  for (const f of files) {
    const fileId = `f_${Date.now()}_${Math.random().toString(16).slice(2)}`;

    let extracted = "";
    try {
      const buf = fs.readFileSync(f.path);
      const parsed = await pdfParse(buf);
      extracted = (parsed?.text || "").trim();
      if (extracted.length > 500000) extracted = extracted.slice(0, 500000); // safety cap
    } catch (e) {
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

  stmtUpdateConversationTouched.run(conversationId, userId);

  res.json({ ok: true, files: out });
});

// -------------------------
// SSE utilities
// -------------------------
function sseHeaders(res) {
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();
}

function sseEvent(res, event, data) {
  res.write(`event: ${event}\n`);
  // encode newlines for safe client parsing
  const payload = String(data ?? "").replace(/\n/g, "\\n");
  res.write(`data: ${payload}\n\n`);
}

// -------------------------
// System prompt (Avvocato senior)
// -------------------------
function seniorLawyerSystemPrompt() {
  return `
Sei "Atto Perfetto – Avvocato Senior", un assistente IA per Avvocati e Studi Legali italiani.
Obiettivo: produrre analisi e testi giuridici civili di livello eccellente, chiari, tecnici e operativi.

REGOLE:
- Mantieni tono professionale, sobrio e determinato.
- Non inventare fatti: se mancano dati essenziali, chiedi SOLO ciò che serve davvero.
- Quando redigi o analizzi, struttura sempre per sezioni, con titoli chiari.
- Evidenzia punti di forza/debolezza, eccezioni, onere probatorio, rischi e strategie.
- Suggerisci in modo pragmatico come migliorare un atto o impostare una difesa/azione.
- Non sostituirti al giudice: parla in termini di probabilità/valutazioni ragionevoli.

Se l’utente chiede una bozza di atto, produci:
1) Inquadramento (procedura, competenza, rito)
2) Fatti (chiari e cronologici)
3) Diritto (normativa + principi + giurisprudenza se utile)
4) Domande/Conclusioni
5) Mezzi di prova (documenti, testi, CTU se pertinente)
6) Note strategiche finali

Se l’utente carica atti (PDF), analizza con precisione: struttura, tesi, prove, contraddizioni, istituti e fattispecie.
  `.trim();
}

// -------------------------
// Build conversation memory
// -------------------------
function buildMessagesForModel(userId, conversationId, userMessage) {
  const history = stmtGetMessages.all(conversationId, userId);

  const msgs = [
    { role: "system", content: seniorLawyerSystemPrompt() },
  ];

  // Cap history (but keep a lot, user wants strong memory)
  // You can adjust limit here
  const MAX_HISTORY = 40;
  const slice = history.slice(Math.max(0, history.length - MAX_HISTORY));

  for (const m of slice) {
    const role = m.role === "user" ? "user" : "assistant";
    msgs.push({ role, content: m.content });
  }

  msgs.push({ role: "user", content: userMessage });
  return msgs;
}

// -------------------------
// Chat SSE endpoint
// -------------------------
app.post("/api/chat/stream", async (req, res) => {
  const userId = req.userId;
  const { conversationId, message } = req.body || {};

  if (!conversationId) return res.status(400).json({ error: "Missing conversationId" });
  if (!message || !message.trim()) return res.status(400).json({ error: "Missing message" });

  // Ensure conversation exists
  const conv = stmtGetConversation.get(conversationId, userId);
  if (!conv) {
    stmtCreateConversation.run({
      id: conversationId,
      user_id: userId,
      title: "Conversazione",
      folder: "Generale",
    });
  }

  // Save user message
  stmtInsertMessage.run(conversationId, userId, "user", message.trim());
  stmtUpdateConversationTouched.run(conversationId, userId);

  // SSE
  sseHeaders(res);
  sseEvent(res, "meta", JSON.stringify({ model: OPENAI_MODEL }));

  if (!OPENAI_API_KEY) {
    sseEvent(res, "error", "OPENAI_API_KEY mancante sul server.");
    sseEvent(res, "done", "");
    return res.end();
  }

  const msgs = buildMessagesForModel(userId, conversationId, message.trim());

  try {
    const stream = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      messages: msgs,
      temperature: 0.2,
      stream: true,
    });

    let full = "";

    for await (const part of stream) {
      const delta = part?.choices?.[0]?.delta?.content || "";
      if (delta) {
        full += delta;
        sseEvent(res, "delta", delta);
      }
    }

    // Save assistant reply
    const final = full.trim() || " ";
    stmtInsertMessage.run(conversationId, userId, "assistant", final);
    stmtUpdateConversationTouched.run(conversationId, userId);

    sseEvent(res, "done", "");
    res.end();
  } catch (e) {
    sseEvent(res, "error", e?.message || "Errore OpenAI");
    sseEvent(res, "done", "");
    res.end();
  }
});

// -------------------------
// Chunking utilities
// -------------------------
function chunkText(text, maxChars = 6000) {
  const clean = (text || "").replace(/\r/g, "").trim();
  if (!clean) return [];
  const chunks = [];
  let start = 0;
  while (start < clean.length) {
    let end = Math.min(start + maxChars, clean.length);
    chunks.push(clean.slice(start, end));
    start = end;
  }
  return chunks;
}

// -------------------------
// Analyze PDF(s) SSE endpoint
// -------------------------
app.post("/api/analyze/stream", async (req, res) => {
  const userId = req.userId;
  const { conversationId, fileIds } = req.body || {};

  if (!conversationId) return res.status(400).json({ error: "Missing conversationId" });
  if (!Array.isArray(fileIds) || fileIds.length === 0) {
    return res.status(400).json({ error: "Missing fileIds" });
  }

  // Ensure conversation exists
  const conv = stmtGetConversation.get(conversationId, userId);
  if (!conv) {
    stmtCreateConversation.run({
      id: conversationId,
      user_id: userId,
      title: "Conversazione",
      folder: "Generale",
    });
  }

  // SSE
  sseHeaders(res);
  sseEvent(res, "meta", JSON.stringify({ model: OPENAI_MODEL, files: fileIds.length }));

  if (!OPENAI_API_KEY) {
    sseEvent(res, "error", "OPENAI_API_KEY mancante sul server.");
    sseEvent(res, "done", "");
    return res.end();
  }

  // Load files
  const fileRows = [];
  for (const id of fileIds) {
    const row = stmtGetFileById.get(id, userId);
    if (row) fileRows.push(row);
  }
  if (!fileRows.length) {
    sseEvent(res, "error", "Nessun file valido trovato per questo utente.");
    sseEvent(res, "done", "");
    return res.end();
  }

  // Build per-file analysis (chunked) then compare
  const perFileSummaries = [];
  let fileIndex = 0;

  for (const f of fileRows) {
    fileIndex++;
    const text = (f.extracted_text || "").trim();
    const chunks = chunkText(text, 6000);

    sseEvent(res, "status", `Analisi documento ${fileIndex}/${fileRows.length}: ${f.filename}`);

    // If no text extracted, warn
    if (!chunks.length) {
      const msg = `Documento "${f.filename}": testo non estratto (PDF scannerizzato o vuoto).`;
      perFileSummaries.push({ filename: f.filename, summary: msg });
      sseEvent(res, "delta", `\n\n## Documento: ${f.filename}\n`);
      sseEvent(res, "delta", `⚠️ ${msg}\n`);
      continue;
    }

    // Analyze each chunk and then consolidate
    const partialNotes = [];
    let c = 0;

    for (const chunk of chunks) {
      c++;
      sseEvent(res, "progress", JSON.stringify({ file: f.filename, chunk: c, totalChunks: chunks.length }));

      const prompt = `
Analizza in modo tecnico e operativo questo estratto di atto giudiziario civile italiano.

OBIETTIVI:
- Evidenzia punti di forza e punti deboli/contraddizioni del testo.
- Elenca istituti giuridici e fattispecie trattate, con valutazione (forte/debole).
- Segnala eventuali omissioni rilevanti, passaggi illogici, carenze probatorie.
- Mantieni un taglio pratico “da avvocato senior”.

TESTO (estratto):
${chunk}
      `.trim();

      try {
        const r = await openai.chat.completions.create({
          model: OPENAI_MODEL,
          messages: [
            { role: "system", content: seniorLawyerSystemPrompt() },
            { role: "user", content: prompt }
          ],
          temperature: 0.2,
        });
        const out = r?.choices?.[0]?.message?.content?.trim() || "";
        partialNotes.push(out);
      } catch (e) {
        partialNotes.push(`Errore analisi chunk: ${e?.message || "OpenAI error"}`);
      }
    }

    // Consolidate chunk analyses into single per-file summary
    const consolidatePrompt = `
Unisci e consolida le analisi parziali in un report unico e coerente per il documento "${f.filename}".

Struttura OBBLIGATORIA:
1) Punti di forza
2) Debolezze e contraddizioni
3) Istituti giuridici e fattispecie (con valutazione forte/debole)
4) Strategie operative (come agire nei prossimi atti / difese)
5) Check-list prove e onere probatorio

ANALISI PARZIALI:
${partialNotes.join("\n\n---\n\n")}
    `.trim();

    let consolidated = "";
    try {
      const r2 = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        messages: [
          { role: "system", content: seniorLawyerSystemPrompt() },
          { role: "user", content: consolidatePrompt }
        ],
        temperature: 0.2,
      });
      consolidated = r2?.choices?.[0]?.message?.content?.trim() || "";
    } catch (e) {
      consolidated = `Errore consolidamento: ${e?.message || "OpenAI error"}`;
    }

    perFileSummaries.push({ filename: f.filename, summary: consolidated });

    // Stream this doc report
    sseEvent(res, "delta", `\n\n# REPORT DOCUMENTO: ${f.filename}\n\n`);
    sseEvent(res, "delta", consolidated + "\n");
    sseEvent(res, "separator", "\n\n---\n\n");
  }

  // If multiple documents: compare
  if (perFileSummaries.length > 1) {
    sseEvent(res, "status", "Confronto tra documenti e valutazione strategica…");

    const comparePrompt = `
Sono stati analizzati più atti/documenti relativi ad una controversia civile.

1) Metti a confronto le argomentazioni e l’impianto logico-giuridico dei documenti.
2) Valuta se uno tende a prevalere sull’altro e perché (coerenza, prova, diritto, strategia).
3) Evidenzia argomentazioni discordanti e punti di frizione.
4) Suggerisci a ENTRAMBE le parti (attore e convenuto) come impostare i prossimi atti,
   quali eccezioni/punti enfatizzare, quali rischi mitigare, quali prove integrare.
5) Se opportuno, indica spunti transattivi o di gestione del rischio.

REPORT SINGOLI:
${perFileSummaries.map(x => `DOCUMENTO: ${x.filename}\n${x.summary}`).join("\n\n===== \n\n")}
    `.trim();

    let compareOut = "";
    try {
      const r3 = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        messages: [
          { role: "system", content: seniorLawyerSystemPrompt() },
          { role: "user", content: comparePrompt }
        ],
        temperature: 0.2,
      });
      compareOut = r3?.choices?.[0]?.message?.content?.trim() || "";
    } catch (e) {
      compareOut = `Errore confronto: ${e?.message || "OpenAI error"}`;
    }

    sseEvent(res, "delta", `\n\n# CONFRONTO E STRATEGIA COMPLESSIVA\n\n`);
    sseEvent(res, "delta", compareOut + "\n");
  }

  // Save final report in conversation as assistant message (one big message)
  // NOTE: we cannot easily reconstruct full text from streamed chunks,
  // so we save a short marker; the front-end already stores the displayed report for export.
  stmtInsertMessage.run(conversationId, userId, "assistant", "Report ACTA SCAN generato (vedi output in chat).");
  stmtUpdateConversationTouched.run(conversationId, userId);

  sseEvent(res, "done", "");
  res.end();
});

// -------------------------
// Start
// -------------------------
app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
  console.log(`DATA_DIR: ${DATA_DIR}`);
});
