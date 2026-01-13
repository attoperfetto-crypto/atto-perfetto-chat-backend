/**
 * server.js — Atto Perfetto Chat Backend (Render)
 *
 * Features:
 * - SSE streaming chat (memoria storica per conversazione)
 * - Fascicoli/Clienti (folders) persistenti server-side
 * - Ricerca globale su TUTTE le chat dell'utente (SQLite FTS5)
 * - Upload PDF multi-file + estrazione testo + chunking
 * - Analisi atti singola + confronto (se più file)
 * - Salvataggio conversazioni e report per singolo utente
 *
 * ENV:
 * - OPENAI_API_KEY (obbligatoria)
 * - OPENAI_MODEL (opzionale) es: gpt-5.2
 * - DATA_DIR (opzionale) es: /var/data (Render Disk)
 * - PORT (opzionale)
 */

import fs from "fs";
import path from "path";
import express from "express";
import cors from "cors";
import multer from "multer";
import cookieParser from "cookie-parser";
import helmet from "helmet";
import compression from "compression";
import { v4 as uuidv4 } from "uuid";
import pdfParse from "pdf-parse";
import OpenAI from "openai";
import Database from "better-sqlite3";

// -------------------------
// Config
// -------------------------
const PORT = process.env.PORT || 10000;
const DATA_DIR = process.env.DATA_DIR || "/var/data";
const UPLOAD_DIR = path.join(DATA_DIR, "uploads");
const DB_PATH = path.join(DATA_DIR, "atto_perfetto.sqlite");

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error("❌ Missing OPENAI_API_KEY in environment.");
  process.exit(1);
}
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-5.2"; // puoi mettere gpt-4.1-mini per risparmiare

// Ensure directories
fs.mkdirSync(DATA_DIR, { recursive: true });
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// -------------------------
// OpenAI client
// -------------------------
const client = new OpenAI({ apiKey: OPENAI_API_KEY });

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
  title TEXT,
  folder TEXT DEFAULT 'Generale',
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL,
  role TEXT NOT NULL, -- user | assistant | system
  content TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS files (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  original_name TEXT,
  mime TEXT,
  size INTEGER,
  disk_path TEXT,
  extracted_text TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(user_id) REFERENCES users(id),
  FOREIGN KEY(conversation_id) REFERENCES conversations(id)
);

-- Full-text search over messages (FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
USING fts5(content, conversation_id, user_id, role, tokenize='unicode61');

-- Keep FTS in sync
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(content, conversation_id, user_id, role)
  VALUES (new.content, new.conversation_id,
          (SELECT user_id FROM conversations WHERE id=new.conversation_id),
          new.role);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  DELETE FROM messages_fts WHERE rowid=old.id;
END;
`);

// Prepared statements
const stmtEnsureUser = db.prepare(`INSERT OR IGNORE INTO users(id) VALUES (?)`);
const stmtCreateConv = db.prepare(`
  INSERT INTO conversations(id, user_id, title, folder)
  VALUES (@id, @user_id, @title, @folder)
`);
const stmtListConvs = db.prepare(`
  SELECT id, title, folder, created_at, updated_at
  FROM conversations
  WHERE user_id = @user_id
  AND (@folder IS NULL OR folder = @folder)
  ORDER BY datetime(updated_at) DESC
`);
const stmtGetConv = db.prepare(`
  SELECT id, title, folder, created_at, updated_at
  FROM conversations
  WHERE id = @id AND user_id = @user_id
`);
const stmtUpdateConv = db.prepare(`
  UPDATE conversations SET title=@title, folder=@folder, updated_at=datetime('now')
  WHERE id=@id AND user_id=@user_id
`);
const stmtTouchConv = db.prepare(`
  UPDATE conversations SET updated_at=datetime('now')
  WHERE id=@id AND user_id=@user_id
`);
const stmtInsertMsg = db.prepare(`
  INSERT INTO messages(conversation_id, role, content)
  VALUES (@conversation_id, @role, @content)
`);
const stmtGetMsgs = db.prepare(`
  SELECT id, role, content, created_at
  FROM messages
  WHERE conversation_id=@conversation_id
  ORDER BY id ASC
`);
const stmtInsertFile = db.prepare(`
  INSERT INTO files(id, user_id, conversation_id, original_name, mime, size, disk_path, extracted_text)
  VALUES (@id, @user_id, @conversation_id, @original_name, @mime, @size, @disk_path, @extracted_text)
`);
const stmtListFilesForConv = db.prepare(`
  SELECT id, original_name, mime, size, created_at
  FROM files
  WHERE user_id=@user_id AND conversation_id=@conversation_id
  ORDER BY datetime(created_at) ASC
`);
const stmtGetFileById = db.prepare(`
  SELECT * FROM files
  WHERE id=@id AND user_id=@user_id
`);
const stmtSearch = db.prepare(`
  SELECT
    messages_fts.rowid as msg_id,
    conversation_id,
    role,
    snippet(messages_fts, 0, '⟦', '⟧', '…', 12) AS snippet
  FROM messages_fts
  WHERE messages_fts MATCH @q
    AND user_id = @user_id
  ORDER BY rank
  LIMIT 50
`);

// -------------------------
// App
// -------------------------
const app = express();
app.use(helmet({ contentSecurityPolicy: false }));
app.use(compression());
app.use(cors({ origin: true, credentials: true }));
app.use(express.json({ limit: "25mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());

// -------------------------
// User identification (per-user isolation)
// -------------------------
// Builderall: se hai un tuo userId reale, passa header "x-user-id".
// Altrimenti: cookie anonimo. Così nessuno vede chat/file altrui.
app.use((req, res, next) => {
  let userId = req.header("x-user-id")?.trim();
  if (!userId) {
    userId = req.cookies?.ap_uid;
    if (!userId) {
      userId = "u_" + uuidv4();
      res.cookie("ap_uid", userId, {
        httpOnly: true,
        sameSite: "lax",
        secure: true,
        maxAge: 1000 * 60 * 60 * 24 * 365,
      });
    }
  }

  // Ensure user exists
  stmtEnsureUser.run(userId);
  req.userId = userId;
  next();
});

// -------------------------
// Multer upload
// -------------------------
const storage = multer.diskStorage({
  destination: (_, __, cb) => cb(null, UPLOAD_DIR),
  filename: (_, file, cb) => cb(null, `${Date.now()}_${uuidv4()}_${file.originalname}`),
});
const upload = multer({
  storage,
  limits: { fileSize: 25 * 1024 * 1024 }, // 25MB per file (modifica se vuoi)
});

// -------------------------
// Helpers
// -------------------------
function sseInit(res) {
  res.status(200);
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();
}

function sseSend(res, event, data) {
  // data must be string
  res.write(`event: ${event}\n`);
  res.write(`data: ${data.replace(/\n/g, "\\n")}\n\n`);
}

function safeJson(obj) {
  return JSON.stringify(obj ?? {});
}

function chunkText(text, chunkSize = 14000, overlap = 1200) {
  const out = [];
  const clean = (text || "").trim();
  if (!clean) return out;

  let i = 0;
  while (i < clean.length) {
    const end = Math.min(i + chunkSize, clean.length);
    const slice = clean.slice(i, end);
    out.push(slice);
    if (end === clean.length) break;
    i = Math.max(0, end - overlap);
  }
  return out;
}

function buildSeniorLawyerInstructions(extra = "") {
  return `
Sei “Atto Perfetto – Avvocato Senior”.
Agisci come un avvocato civilista senior italiano (taglio strategico e processuale), con stile:
- tecnico ma chiarissimo,
- rigoroso, puntuale, verificabile,
- operativo (indicazioni pratiche, step, rischi, alternative),
- imparziale (pro/contro, punti forti e vulnerabilità),
- orientato alla redazione di atti (coerenza, struttura, argomentazione, prova).

Regole:
1) Non inventare dati: se manca un elemento, segnala “Dato non presente nel testo” e indica cosa servirebbe.
2) Evidenzia contraddizioni interne e omissioni rilevanti.
3) Organizza sempre l’output in sezioni con titoli e bullet point quando utile.
4) Se analizzi atti di parte, ragiona anche in ottica “controparte” (dove potrebbe attaccare).
5) Se ricevi più documenti, analizzali separatamente e poi fai un confronto strategico complessivo.

${extra}
`.trim();
}

async function openaiStreamToSSE({ res, requestBody, onTextDelta, onDone }) {
  // Streaming via OpenAI SDK (Responses API)
  // https://platform.openai.com/docs/guides/streaming-responses :contentReference[oaicite:1]{index=1}
  const stream = await client.responses.create({ ...requestBody, stream: true });

  let fullText = "";
  for await (const event of stream) {
    if (event.type === "response.output_text.delta") {
      const delta = event.delta || "";
      fullText += delta;
      onTextDelta?.(delta);
    }
    if (event.type === "response.completed") {
      onDone?.(fullText, event);
      break;
    }
    if (event.type === "error") {
      throw new Error(event?.error?.message || "OpenAI stream error");
    }
  }
  return fullText;
}

function getConversationOrThrow(userId, conversationId) {
  const conv = stmtGetConv.get({ id: conversationId, user_id: userId });
  if (!conv) {
    const err = new Error("Conversation not found");
    err.status = 404;
    throw err;
  }
  return conv;
}

// -------------------------
// Routes
// -------------------------
app.get("/health", (req, res) => res.json({ ok: true }));

// Create conversation
app.post("/api/conversations", (req, res) => {
  const userId = req.userId;
  const id = req.body?.id || "c_" + uuidv4();
  const title = (req.body?.title || "Nuova conversazione").slice(0, 120);
  const folder = (req.body?.folder || "Generale").slice(0, 80);

  stmtCreateConv.run({ id, user_id: userId, title, folder });
  res.json({ id, title, folder });
});

// List conversations (optional folder)
app.get("/api/conversations", (req, res) => {
  const userId = req.userId;
  const folder = req.query?.folder ? String(req.query.folder) : null;
  const rows = stmtListConvs.all({ user_id: userId, folder });
  res.json(rows);
});

// Get single conversation + messages + files
app.get("/api/conversations/:id", (req, res) => {
  const userId = req.userId;
  const id = req.params.id;
  const conv = stmtGetConv.get({ id, user_id: userId });
  if (!conv) return res.status(404).json({ error: "Not found" });

  const messages = stmtGetMsgs.all({ conversation_id: id });
  const files = stmtListFilesForConv.all({ user_id: userId, conversation_id: id });
  res.json({ conversation: conv, messages, files });
});

// Update folder/title
app.put("/api/conversations/:id", (req, res) => {
  const userId = req.userId;
  const id = req.params.id;
  const conv = getConversationOrThrow(userId, id);

  const title = (req.body?.title ?? conv.title ?? "Conversazione").slice(0, 120);
  const folder = (req.body?.folder ?? conv.folder ?? "Generale").slice(0, 80);

  stmtUpdateConv.run({ id, user_id: userId, title, folder });
  res.json({ ok: true });
});

// Global search across all chats (FTS)
app.get("/api/search", (req, res) => {
  const userId = req.userId;
  const q = String(req.query?.q || "").trim();
  if (!q) return res.json([]);

  // FTS syntax: words, AND/OR, quotes. Escape minimal.
  const safeQ = q.replace(/["']/g, " ").trim();
  const rows = stmtSearch.all({ q: safeQ, user_id: userId });
  res.json(rows);
});

// Upload PDFs (multi-file)
app.post("/api/upload", upload.array("files", 10), async (req, res) => {
  try {
    const userId = req.userId;
    const conversationId = String(req.body?.conversationId || "").trim();
    if (!conversationId) return res.status(400).json({ error: "conversationId required" });
    getConversationOrThrow(userId, conversationId);

    const files = req.files || [];
    if (!files.length) return res.status(400).json({ error: "No files uploaded" });

    const saved = [];
    for (const f of files) {
      const fileId = "f_" + uuidv4();
      let extracted = "";

      // Extract text if PDF
      if ((f.mimetype || "").includes("pdf") || f.originalname?.toLowerCase().endsWith(".pdf")) {
        const buf = fs.readFileSync(f.path);
        const parsed = await pdfParse(buf);
        extracted = (parsed?.text || "").trim();
      } else {
        // For non-PDF: store as is (optional)
        extracted = "";
      }

      stmtInsertFile.run({
        id: fileId,
        user_id: userId,
        conversation_id: conversationId,
        original_name: f.originalname,
        mime: f.mimetype,
        size: f.size,
        disk_path: f.path,
        extracted_text: extracted,
      });

      saved.push({
        id: fileId,
        name: f.originalname,
        size: f.size,
        mime: f.mimetype,
        extractedChars: extracted.length,
      });
    }

    stmtTouchConv.run({ id: conversationId, user_id: userId });
    res.json({ ok: true, files: saved });
  } catch (e) {
    console.error(e);
    res.status(e.status || 500).json({ error: e.message || "Upload error" });
  }
});

// CHAT stream (SSE) — with memory
app.post("/api/chat/stream", async (req, res) => {
  const userId = req.userId;

  try {
    const conversationId = String(req.body?.conversationId || "").trim();
    const message = String(req.body?.message || "").trim();
    const systemExtra = String(req.body?.systemExtra || "").trim();

    if (!conversationId) return res.status(400).json({ error: "conversationId required" });
    if (!message) return res.status(400).json({ error: "message required" });

    const conv = getConversationOrThrow(userId, conversationId);

    sseInit(res);

    // Save user message
    stmtInsertMsg.run({
      conversation_id: conversationId,
      role: "user",
      content: message,
    });
    stmtTouchConv.run({ id: conversationId, user_id: userId });

    // Build memory: entire conversation messages
    const history = stmtGetMsgs.all({ conversation_id: conversationId });

    const instructions = buildSeniorLawyerInstructions(systemExtra);

    // Convert to Responses "input" format
    // We keep the whole history to preserve memory.
    const input = history.map((m) => ({
      role: m.role === "system" ? "system" : m.role,
      content: [{ type: "input_text", text: m.content }],
    }));

    sseSend(res, "meta", safeJson({ model: OPENAI_MODEL, conversationId, title: conv.title, folder: conv.folder }));

    let assistantFinal = "";

    await openaiStreamToSSE({
      res,
      requestBody: {
        model: OPENAI_MODEL,
        instructions,
        input,
        // Se vuoi più output: aumenta max_output_tokens
        max_output_tokens: 1800,
      },
      onTextDelta: (delta) => {
        sseSend(res, "delta", delta);
      },
      onDone: (fullText) => {
        assistantFinal = fullText || "";
      },
    });

    // Save assistant message
    if (assistantFinal.trim()) {
      stmtInsertMsg.run({
        conversation_id: conversationId,
        role: "assistant",
        content: assistantFinal,
      });
      stmtTouchConv.run({ id: conversationId, user_id: userId });
    }

    sseSend(res, "done", safeJson({ ok: true }));
    res.end();
  } catch (e) {
    console.error(e);

    // Quota / billing / rate limit
    const msg =
      (e?.message || "").includes("insufficient_quota") || (e?.message || "").includes("quota")
        ? "Quota API esaurita o billing non attivo. Verifica credits e piano OpenAI."
        : "Errore di connessione o richiesta. Riprova tra pochi secondi.";

    try {
      sseInit(res);
      sseSend(res, "error", msg);
      sseSend(res, "done", safeJson({ ok: false }));
      res.end();
    } catch {
      res.status(500).json({ error: msg });
    }
  }
});

// ANALYZE stream (SSE) — multi-file + compare
app.post("/api/analyze/stream", async (req, res) => {
  const userId = req.userId;

  try {
    const conversationId = String(req.body?.conversationId || "").trim();
    let fileIds = req.body?.fileIds;
    if (!conversationId) return res.status(400).json({ error: "conversationId required" });

    getConversationOrThrow(userId, conversationId);

    // If no fileIds provided -> analyze all files in conversation
    if (!Array.isArray(fileIds) || !fileIds.length) {
      const all = stmtListFilesForConv.all({ user_id: userId, conversation_id: conversationId });
      fileIds = all.map((f) => f.id);
    }

    // Fetch files with extracted text
    const docs = fileIds
      .map((id) => stmtGetFileById.get({ id, user_id: userId }))
      .filter(Boolean)
      .map((f) => ({
        id: f.id,
        name: f.original_name,
        text: (f.extracted_text || "").trim(),
      }))
      .filter((d) => d.text.length > 0);

    if (!docs.length) return res.status(400).json({ error: "No readable PDF text found (extraction empty)" });

    sseInit(res);
    sseSend(res, "meta", safeJson({ model: OPENAI_MODEL, conversationId, files: docs.map((d) => ({ id: d.id, name: d.name, chars: d.text.length })) }));

    const baseInstructions = buildSeniorLawyerInstructions(`
Obiettivo:
- Analizza professionalmente atti giudiziari civili italiani.
- Output strutturato e operativo.
- Se più atti: analisi singola + confronto finale + suggerimenti strategici per entrambe le parti.

Formato richiesto:
A) Analisi per singolo documento (uno per volta):
   1. Struttura e sintesi
   2. Punti di forza (normativa, giurisprudenza, logica, prova, tecnica)
   3. Debolezze/contraddizioni/omissioni
   4. Istituti e fattispecie (forti vs deboli) con commento
   5. Mosse/contromosse (cosa aspettarsi e come reagire)

B) Confronto complessivo (solo se >1 documento):
   - quale impianto appare prevalente e perché
   - dove uno “buca” l’altro (punti d’attacco)
   - argomenti discordanti e come comporli o sfruttarli
   - suggerimenti per atti successivi: repliche, memorie, istruttoria, prova, testimoni
`);

    // 1) Per ogni documento: chunk -> mini-sintesi di chunk -> analisi unica per documento
    const perDocAnalyses = [];
    for (const doc of docs) {
      sseSend(res, "status", `Analizzo: ${doc.name}`);

      const chunks = chunkText(doc.text, 14000, 1200);
      const chunkSummaries = [];

      // Chunk summarization (non-stream per velocità/costi)
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const prompt = `
Documento: ${doc.name}
Chunk ${i + 1}/${chunks.length}

Testo:
${chunk}

Produci una sintesi tecnica fedele (max 12 bullet) con:
- fatti e richieste
- argomenti principali
- norme/istituti citati
- elementi probatori
- eventuali contraddizioni interne nel chunk
`.trim();

        const r = await client.responses.create({
          model: OPENAI_MODEL,
          instructions: baseInstructions,
          input: [{ role: "user", content: [{ type: "input_text", text: prompt }] }],
          max_output_tokens: 900,
        });

        const out = (r.output_text || "").trim();
        chunkSummaries.push(out);
        sseSend(res, "progress", safeJson({ file: doc.name, chunk: i + 1, totalChunks: chunks.length }));
      }

      // Now full per-doc analysis (stream)
      const docPrompt = `
DOCUMENTO: ${doc.name}

SINTESI DEI CHUNK (fedeli al testo):
${chunkSummaries.map((s, idx) => `--- Chunk ${idx + 1} ---\n${s}`).join("\n\n")}

Ora redigi l'ANALISI COMPLETA del documento, seguendo rigorosamente il formato:
1) Struttura e sintesi
2) Punti di forza
3) Debolezze/contraddizioni/omissioni
4) Istituti e fattispecie (forti vs deboli) con commento
5) Mosse/contromosse operative
`.trim();

      let docFull = "";
      await openaiStreamToSSE({
        res,
        requestBody: {
          model: OPENAI_MODEL,
          instructions: baseInstructions,
          input: [{ role: "user", content: [{ type: "input_text", text: docPrompt }] }],
          max_output_tokens: 2200,
        },
        onTextDelta: (delta) => {
          docFull += delta;
          sseSend(res, "delta", delta);
        },
        onDone: () => {},
      });

      perDocAnalyses.push({ id: doc.id, name: doc.name, analysis: docFull.trim() });
      sseSend(res, "separator", "\n\n────────────\n\n");
    }

    // 2) Confronto finale se multi-file
    let compareText = "";
    if (perDocAnalyses.length > 1) {
      sseSend(res, "status", "Confronto strategico tra gli atti…");

      const comparePrompt = `
Hai analizzato i seguenti atti (analisi già effettuate):

${perDocAnalyses.map((d) => `### ${d.name}\n${d.analysis}`).join("\n\n")}

Ora:
A) Mettili a confronto in modo strategico e processuale.
B) Valuta quale impianto tende a prevalere e perché (logica, norme, prova, onere, coerenza).
C) Evidenzia gli argomenti discordanti e dove ciascuno è vulnerabile.
D) Suggerisci ad entrambe le parti cosa fare nei prossimi atti (repliche/memorie/istanze/istruttoria/testi).
E) Chiudi con una “checklist operativa” per la redazione del prossimo atto per ciascuna parte.
`.trim();

      await openaiStreamToSSE({
        res,
        requestBody: {
          model: OPENAI_MODEL,
          instructions: baseInstructions,
          input: [{ role: "user", content: [{ type: "input_text", text: comparePrompt }] }],
          max_output_tokens: 2400,
        },
        onTextDelta: (delta) => {
          compareText += delta;
          sseSend(res, "delta", delta);
        },
        onDone: () => {},
      });
    }

    // 3) Save final report into conversation as assistant message
    const finalReport = [
      "=== REPORT ATTO PERFETTO — ESAME E ANALISI ATTI ===",
      "",
      ...perDocAnalyses.map((d) => `## ANALISI: ${d.name}\n\n${d.analysis}`),
      perDocAnalyses.length > 1 ? `\n## CONFRONTO STRATEGICO\n\n${compareText.trim()}` : "",
    ]
      .filter(Boolean)
      .join("\n");

    stmtInsertMsg.run({
      conversation_id: conversationId,
      role: "assistant",
      content: finalReport,
    });
    stmtTouchConv.run({ id: conversationId, user_id: userId });

    sseSend(res, "done", safeJson({ ok: true, saved: true }));
    res.end();
  } catch (e) {
    console.error(e);

    const msg =
      (e?.message || "").includes("insufficient_quota") || (e?.message || "").includes("quota")
        ? "Quota API esaurita o billing non attivo. Verifica credits e piano OpenAI."
        : "Errore durante l’analisi. Riprova o carica un PDF diverso.";

    try {
      sseInit(res);
      sseSend(res, "error", msg);
      sseSend(res, "done", safeJson({ ok: false }));
      res.end();
    } catch {
      res.status(500).json({ error: msg });
    }
  }
});

// -------------------------
// Start
// -------------------------
app.listen(PORT, () => {
  console.log(`✅ Server running on :${PORT}`);
  console.log(`DB: ${DB_PATH}`);
  console.log(`Uploads: ${UPLOAD_DIR}`);
});
