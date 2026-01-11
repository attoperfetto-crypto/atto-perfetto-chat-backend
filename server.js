import express from "express";
import cors from "cors";
import OpenAI from "openai";
import multer from "multer";
import pdfParse from "pdf-parse";
import Database from "better-sqlite3";

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 20 * 1024 * 1024 } });

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";

/** ========= DB (SQLite) =========
 * NB: su Render la persistenza file dipende dallo storage.
 * Per salvataggio garantito: usa Render Disk o Postgres (possiamo farlo dopo).
 */
const db = new Database("atto.db");
db.pragma("journal_mode = WAL");

db.exec(`
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL
);
`);

const nowISO = () => new Date().toISOString();

function requireUserId(req, res) {
  const userId = req.header("X-User-Id");
  if (!userId) {
    res.status(400).json({ error: "Missing X-User-Id" });
    return null;
  }
  return userId;
}

function ensureConversation(userId, conversationId, titleIfNew = "Conversazione") {
  const row = db.prepare("SELECT id FROM conversations WHERE id=? AND user_id=?").get(conversationId, userId);
  if (!row) {
    db.prepare("INSERT INTO conversations (id, user_id, title, updated_at) VALUES (?, ?, ?, ?)")
      .run(conversationId, userId, titleIfNew, nowISO());
  } else {
    db.prepare("UPDATE conversations SET updated_at=? WHERE id=? AND user_id=?").run(nowISO(), conversationId, userId);
  }
}

function saveMessage(userId, conversationId, role, content) {
  db.prepare("INSERT INTO messages (conversation_id, user_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)")
    .run(conversationId, userId, role, content, nowISO());
  db.prepare("UPDATE conversations SET updated_at=? WHERE id=? AND user_id=?")
    .run(nowISO(), conversationId, userId);
}

function systemChat() {
  return `
Sei "Atto Perfetto – Area Dinamica".
Agisci come Avvocato esperto di diritto italiano.
Tono tecnico, chiaro, professionale.
Se mancano dati essenziali, fai prima domande mirate.
Struttura sempre l’output con titoli e paragrafi.
`.trim();
}

function systemActaScan() {
  return `
Sei "Atto Perfetto – ACTA SCAN (Civile)".
Analizza atti/ documenti giuridici italiani con rigore tecnico e imparzialità.
Output SEMPRE strutturato in:
1) Punti di forza
2) Punti deboli e contraddizioni
3) Istituti giuridici e fattispecie (forti vs deboli)
4) Considerazioni finali operative (strategie e prossimi passi)
Non inventare fatti o riferimenti.
`.trim();
}

function systemCompare() {
  return `
Sei "Atto Perfetto – Comparazione Atti".
Confronta due o più atti della stessa controversia.
- Valuta quale impianto regge meglio e perché
- Evidenzia contrasti e contraddizioni reciproche
- Suggerisci prossimi atti per ciascuna parte (contesto civile)
Tono: forense, tecnico, impeccabile.
`.trim();
}

/** ===== Chunking (caratteri) ===== */
function chunkText(text, chunkSize = 6500) {
  const chunks = [];
  for (let i = 0; i < text.length; i += chunkSize) chunks.push(text.slice(i, i + chunkSize));
  return chunks;
}

async function analyzeDocumentWithChunking(docName, text, onProgress) {
  const chunks = chunkText(text, 6500);
  const partials = [];

  for (let i = 0; i < chunks.length; i++) {
    const progress = Math.round((i / Math.max(1, chunks.length)) * 70) + 10; // 10..80
    onProgress?.(progress, `Analisi chunk ${i + 1}/${chunks.length} – ${docName}`);

    const out = await client.chat.completions.create({
      model: MODEL,
      temperature: 0.2,
      messages: [
        { role: "system", content: systemActaScan() },
        { role: "user", content:
          `Documento: ${docName}\nCHUNK ${i + 1}/${chunks.length}\n\n` +
          `Analizza SOLO questo chunk ed estrai: tesi/argomenti, punti forti, punti deboli/contraddizioni, istituti richiamati.\n\n` +
          chunks[i]
        }
      ]
    });

    partials.push(out.choices?.[0]?.message?.content || "");
  }

  onProgress?.(85, `Consolidamento report – ${docName}`);

  const consolidated = await client.chat.completions.create({
    model: MODEL,
    temperature: 0.2,
    messages: [
      { role: "system", content: systemActaScan() },
      { role: "user", content:
        `Consolida in UN SOLO REPORT completo l’analisi del documento "${docName}". ` +
        `Usa esclusivamente le analisi parziali seguenti (una per chunk) e produci un report finale unico, coerente e non ripetitivo.\n\n` +
        partials.map((p,idx)=>`--- ANALISI CHUNK ${idx+1} ---\n${p}\n`).join("\n")
      }
    ]
  });

  return consolidated.choices?.[0]?.message?.content || "Nessun output.";
}

/** ====== LIST conversations ====== */
app.get("/conversations", (req, res) => {
  const userId = requireUserId(req, res);
  if (!userId) return;

  const items = db.prepare(
    "SELECT id, title, updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT 50"
  ).all(userId);

  res.json({ items });
});

/** ====== READ conversation ====== */
app.get("/conversation/:id", (req, res) => {
  const userId = requireUserId(req, res);
  if (!userId) return;

  const id = req.params.id;
  const conv = db.prepare("SELECT id FROM conversations WHERE id=? AND user_id=?").get(id, userId);
  if (!conv) return res.status(404).json({ error: "Not found" });

  const messages = db.prepare(
    "SELECT role, content, created_at FROM messages WHERE conversation_id=? AND user_id=? ORDER BY id ASC"
  ).all(id, userId);

  res.json({ id, messages });
});

/** ====== CHAT SSE + save ====== */
app.post("/chat", async (req, res) => {
  try {
    const userId = requireUserId(req, res);
    if (!userId) return;

    const { message, conversationId } = req.body;
    if (!message || typeof message !== "string") return res.status(400).json({ error: "Messaggio non valido" });
    if (!conversationId || typeof conversationId !== "string") return res.status(400).json({ error: "Missing conversationId" });

    ensureConversation(userId, conversationId, "Chat Atto Perfetto");
    saveMessage(userId, conversationId, "user", message);

    const messages = [
      { role: "system", content: systemChat() },
      { role: "user", content: message }
    ];

    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    const stream = await client.chat.completions.create({
      model: MODEL,
      messages,
      temperature: 0.2,
      stream: true
    });

    let full = "";
    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) {
        full += delta;
        res.write(`data: ${JSON.stringify({ delta })}\n\n`);
      }
    }

    saveMessage(userId, conversationId, "assistant", full);
    res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
    res.end();
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Errore server" });
  }
});

/** ====== MULTIFILE + CHUNKING + PROGRESS (SSE) + SAVE ====== */
app.post("/analyze-pdfs-stream", upload.array("files", 5), async (req, res) => {
  try {
    const userId = requireUserId(req, res);
    if (!userId) return;

    const conversationId = req.body.conversationId;
    if (!conversationId || typeof conversationId !== "string") {
      return res.status(400).json({ error: "Missing conversationId" });
    }

    ensureConversation(userId, conversationId, "Analisi documenti");
    const files = req.files || [];
    if (!files.length) return res.status(400).json({ error: "Nessun file ricevuto" });

    // SSE headers
    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);

    send({ progress: 5, stage: "Estrazione testo dai PDF…" });

    // 1) Parse PDFs
    const docs = [];
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      if (f.mimetype !== "application/pdf") {
        send({ progress: 0, stage: "Formato non supportato (solo PDF).", done: true });
        return res.end();
      }
      const parsed = await pdfParse(f.buffer);
      const text = (parsed.text || "").trim();
      docs.push({ name: f.originalname, text });
      send({ progress: 10 + Math.round((i / Math.max(1, files.length)) * 10), stage: `Testo estratto: ${f.originalname}` });
    }

    // 2) Analyze each doc with chunking
    const perDocReports = [];
    for (let d = 0; d < docs.length; d++) {
      const doc = docs[d];
      if (!doc.text) {
        perDocReports.push(`## REPORT DOCUMENTO: ${doc.name}\n\nImpossibile estrarre testo dal PDF.\n`);
        continue;
      }

      const report = await analyzeDocumentWithChunking(doc.name, doc.text, (p, stage) => {
        // ripartiamo progress 20..85 per blocchi
        send({ progress: Math.min(90, Math.max(20, p)), stage });
      });

      perDocReports.push(`## REPORT DOCUMENTO: ${doc.name}\n\n${report}\n`);
    }

    // 3) Compare if 2+ docs
    let compareSection = "";
    if (docs.length >= 2) {
      send({ progress: 92, stage: "Comparazione tra documenti…" });

      const cmp = await client.chat.completions.create({
        model: MODEL,
        temperature: 0.2,
        messages: [
          { role: "system", content: systemCompare() },
          { role: "user", content:
            `Confronta i seguenti report (uno per documento). ` +
            `Dopo aver commentato singolarmente, confrontali tra loro: prevalenza, discordanza, forza relativa, strategie per i prossimi atti.\n\n` +
            perDocReports.join("\n\n")
          }
        ]
      });

      compareSection = `# CONFRONTO TRA DOCUMENTI\n\n${cmp.choices?.[0]?.message?.content || ""}\n`;
    }

    const final = `# ANALISI COMPLESSIVA (MULTIFILE + CHUNKING)\n\n` +
      perDocReports.join("\n\n") +
      (compareSection ? `\n\n${compareSection}` : "");

    // Save as assistant message (report)
    saveMessage(userId, conversationId, "assistant", final);

    send({ progress: 100, stage: "Completato ✅ Report salvato.", result: final, done: true });
    res.end();
  } catch (e) {
    console.error(e);
    try {
      res.write(`data: ${JSON.stringify({ progress: 0, stage: "Errore server", done: true })}\n\n`);
      res.end();
    } catch {}
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Backend live on ${PORT}`));
