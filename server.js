import express from "express";
import cors from "cors";
import OpenAI from "openai";
import multer from "multer";
import pdfParse from "pdf-parse";

const app = express();

/** =========================
 *  CONFIG
 *  ========================= */
app.use(cors({
  origin: "*",
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "X-User-Id"]
}));

app.use(express.json({ limit: "2mb" }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 } // 25MB per file
});

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";

/** =========================
 *  SIMPLE IN-MEMORY STORE (TEST/VALIDATION)
 *  - Per produzione: Postgres consigliato (te lo preparo)
 *  ========================= */
const store = {
  conversations: new Map() // userId -> Map(conversationId -> {id,title,updatedAt,messages:[{role,content,createdAt}]})
};

function nowISO() { return new Date().toISOString(); }

function getUser(req, res) {
  const userId = req.header("X-User-Id");
  if (!userId) {
    res.status(400).json({ error: "Missing X-User-Id" });
    return null;
  }
  return userId;
}

function ensureUserMap(userId) {
  if (!store.conversations.has(userId)) store.conversations.set(userId, new Map());
  return store.conversations.get(userId);
}

function ensureConversation(userId, conversationId, title = "Conversazione") {
  const m = ensureUserMap(userId);
  if (!m.has(conversationId)) {
    m.set(conversationId, { id: conversationId, title, updatedAt: nowISO(), messages: [] });
  } else {
    m.get(conversationId).updatedAt = nowISO();
  }
  return m.get(conversationId);
}

function saveMsg(userId, conversationId, role, content) {
  const c = ensureConversation(userId, conversationId);
  c.messages.push({ role, content, createdAt: nowISO() });
  c.updatedAt = nowISO();
}

function getConversation(userId, conversationId) {
  const m = store.conversations.get(userId);
  return m?.get(conversationId) || null;
}

function buildOpenAIMessagesFromHistory(conv, systemPrompt) {
  const msgs = [{ role: "system", content: systemPrompt }];
  for (const m of (conv?.messages || [])) {
    const role = (m.role === "assistant" || m.role === "user") ? m.role : "user";
    msgs.push({ role, content: m.content });
  }
  return msgs;
}

/** =========================
 *  PROMPTS
 *  ========================= */
function systemChat() {
  return [
    `Sei "Atto Perfetto – Area Dinamica".`,
    `Tono professionale, tecnico e operativo. Non essere generico.`,
    `Considera TUTTO lo storico della conversazione prima di fare domande.`,
    `Se informazioni cruciali mancano, fai domande mirate (max 3) e motivate.`,
    `Produci output strutturati (titoli, elenchi, sezioni).`,
    `Non inventare norme o giurisprudenza: se non sei certo, segnala incertezza.`,
  ].join("\n");
}

function systemActaScan() {
  return [
    `Sei "Atto Perfetto – ACTA SCAN (Civile)".`,
    `Analizzi atti giudiziari civili italiani in modo professionale e imparziale.`,
    `Output OBBLIGATORIO nelle sezioni:`,
    `1) Punti di forza`,
    `2) Punti deboli e contraddizioni`,
    `3) Istituti giuridici e fattispecie (forti vs deboli)`,
    `4) Considerazioni finali operative`,
    `Non inventare: se il testo non contiene un dato, dichiaralo.`,
  ].join("\n");
}

function systemCompare() {
  return [
    `Sei "Atto Perfetto – Comparazione Atti".`,
    `Confronti più atti/posizioni: coerenza, conflitti, prevalenza argomentativa.`,
    `Indichi quale impianto appare più robusto e perché.`,
    `Suggerisci strategie operative per gli atti successivi per entrambe le parti.`,
    `Tono tecnico e professionale.`,
  ].join("\n");
}

/** =========================
 *  UTILS
 *  ========================= */
function sseHeaders(res) {
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
}

function chunkText(text, maxChars = 9000) {
  const out = [];
  for (let i = 0; i < text.length; i += maxChars) out.push(text.slice(i, i + maxChars));
  return out;
}

async function analyzeDocWithChunking(docName, text, onProgress) {
  const chunks = chunkText(text, 9000);
  const partials = [];

  for (let i = 0; i < chunks.length; i++) {
    const p = 10 + Math.round((i / Math.max(1, chunks.length)) * 70);
    onProgress?.(p, `Analisi contenuto ${i + 1}/${chunks.length} – ${docName}`);

    const r = await client.chat.completions.create({
      model: MODEL,
      temperature: 0.2,
      messages: [
        { role: "system", content: systemActaScan() },
        {
          role: "user",
          content:
            `Documento: ${docName}\n` +
            `Parte ${i + 1}/${chunks.length}\n\n` +
            `Analizza SOLO questa parte. Estrarre tesi, punti forti, debolezze/contraddizioni, istituti.\n\n` +
            chunks[i]
        }
      ]
    });
    partials.push(r.choices?.[0]?.message?.content || "");
  }

  onProgress?.(85, `Consolidamento report – ${docName}`);

  const consolidated = await client.chat.completions.create({
    model: MODEL,
    temperature: 0.2,
    messages: [
      { role: "system", content: systemActaScan() },
      {
        role: "user",
        content:
          `Consolida in UN SOLO REPORT completo per "${docName}".\n` +
          `Usa SOLO le analisi parziali seguenti:\n\n` +
          partials.map((p, idx) => `--- PARTE ${idx + 1} ---\n${p}\n`).join("\n")
      }
    ]
  });

  return consolidated.choices?.[0]?.message?.content || "Nessun output.";
}

function prettyErrorMessage(e) {
  const msg = String(e?.message || e);
  const isQuota = msg.includes("insufficient_quota") || msg.includes("quota");
  const isRate = msg.includes("Rate limit") || msg.includes("429");
  if (isQuota) return "Quota API momentaneamente esaurita. Riprova tra poco.";
  if (isRate) return "Servizio molto richiesto: riprova tra qualche secondo.";
  return msg;
}

/** =========================
 *  ROUTES
 *  ========================= */
app.get("/health", (req, res) => res.json({ ok: true, time: nowISO() }));

app.get("/conversations", (req, res) => {
  const userId = getUser(req, res); if (!userId) return;
  const m = store.conversations.get(userId);
  const items = m ? Array.from(m.values()).sort((a, b) => (b.updatedAt || "").localeCompare(a.updatedAt || "")) : [];
  res.json({ items: items.map(x => ({ id: x.id, title: x.title, updated_at: x.updatedAt })) });
});

app.get("/conversation/:id", (req, res) => {
  const userId = getUser(req, res); if (!userId) return;
  const m = store.conversations.get(userId);
  const c = m?.get(req.params.id);
  if (!c) return res.status(404).json({ error: "Not found" });
  res.json({ id: c.id, messages: c.messages });
});

/** =========================
 *  CHAT (SSE) — MEMORIA STORICA COMPLETA
 *  ========================= */
app.post("/chat", async (req, res) => {
  const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);

  try {
    const userId = getUser(req, res); if (!userId) return;

    const { message, conversationId } = req.body || {};
    if (!message || !conversationId) {
      return res.status(400).json({ error: "Missing message/conversationId" });
    }

    ensureConversation(userId, conversationId, "Chat Atto Perfetto");

    // Salva msg utente
    saveMsg(userId, conversationId, "user", message);

    // SSE
    sseHeaders(res);

    // Costruisci history completa (memoria)
    const conv = getConversation(userId, conversationId);
    const messages = buildOpenAIMessagesFromHistory(conv, systemChat());

    let full = "";

    const stream = await client.chat.completions.create({
      model: MODEL,
      temperature: 0.2,
      stream: true,
      messages
    });

    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) {
        full += delta;
        send({ delta });
      }
    }

    // Salva risposta assistant nello storico
    saveMsg(userId, conversationId, "assistant", full);

    send({ done: true });
    res.end();

  } catch (e) {
    console.error("CHAT ERROR:", e);
    sseHeaders(res);
    send({ error: prettyErrorMessage(e), done: true });
    res.end();
  }
});

/** =========================
 *  ANALYZE PDFs (SSE) — MULTIFILE + CHUNKING + REPORT + SALVATAGGIO
 *  ========================= */
app.post("/analyze-pdfs-stream", upload.array("files", 5), async (req, res) => {
  const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);

  try {
    const userId = getUser(req, res); if (!userId) return;

    const conversationId = req.body?.conversationId;
    if (!conversationId) return res.status(400).json({ error: "Missing conversationId" });

    ensureConversation(userId, conversationId, "Analisi documenti");

    sseHeaders(res);

    const files = req.files || [];
    if (!files.length) {
      send({ error: "Nessun file ricevuto.", done: true });
      return res.end();
    }

    // Salva un messaggio utente “logico” nello storico, così la chat ha contesto
    saveMsg(userId, conversationId, "user",
      `Ho caricato ${files.length} documento/i PDF per l’analisi ACTA SCAN. Procedi con esame e report.`
    );

    send({ progress: 5, stage: "Estrazione testo dai PDF…" });

    const docs = [];
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      if (f.mimetype !== "application/pdf") {
        send({ error: "Formato non supportato (solo PDF).", done: true });
        return res.end();
      }
      const parsed = await pdfParse(f.buffer);
      const text = (parsed.text || "").trim();
      docs.push({ name: f.originalname, text });

      const p = 10 + Math.round((i / Math.max(1, files.length)) * 10);
      send({ progress: p, stage: `Testo estratto: ${f.originalname}` });
    }

    const perDocReports = [];

    for (let d = 0; d < docs.length; d++) {
      const doc = docs[d];

      if (!doc.text) {
        perDocReports.push(
          `## REPORT DOCUMENTO: ${doc.name}\n\n` +
          `⚠️ Non è stato possibile estrarre testo dal PDF (potrebbe essere una scansione immagine).\n`
        );
        continue;
      }

      const rep = await analyzeDocWithChunking(doc.name, doc.text, (p, stage) => send({ progress: p, stage }));
      perDocReports.push(`## REPORT DOCUMENTO: ${doc.name}\n\n${rep}\n`);
    }

    let compareSection = "";
    if (docs.length >= 2) {
      send({ progress: 92, stage: "Comparazione tra documenti…" });

      const cmp = await client.chat.completions.create({
        model: MODEL,
        temperature: 0.2,
        messages: [
          { role: "system", content: systemCompare() },
          { role: "user", content: `Confronta i report seguenti:\n\n${perDocReports.join("\n\n")}` }
        ]
      });

      compareSection =
        `# CONFRONTO TRA DOCUMENTI\n\n` +
        (cmp.choices?.[0]?.message?.content || "") +
        `\n`;
    }

    const final =
      `# ANALISI COMPLESSIVA (ACTA SCAN)\n\n` +
      perDocReports.join("\n\n") +
      `\n\n` +
      (compareSection || "");

    // Salva report nello storico (così la chat lo ricorda)
    saveMsg(userId, conversationId, "assistant", final);

    send({ progress: 100, stage: "Completato ✅ Report salvato.", result: final, done: true });
    res.end();

  } catch (e) {
    console.error("ANALYZE ERROR:", e);
    sseHeaders(res);
    send({ error: prettyErrorMessage(e), done: true });
    res.end();
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("Live on", PORT));
