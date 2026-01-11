import express from "express";
import cors from "cors";
import OpenAI from "openai";
import multer from "multer";
import pdfParse from "pdf-parse";

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 } // 20MB per file
});

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";

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
Se mancano informazioni, segnala cosa manca.
`.trim();
}

function systemCompare() {
  return `
Sei "Atto Perfetto – Comparazione Atti".
Hai due o più atti della stessa controversia.
Obiettivi:
- confrontare argomentazioni, coerenza e forza persuasiva
- evidenziare punti discordanti e contraddizioni reciproche
- valutare quale impianto regge meglio (e perché)
- suggerire, per ciascuna parte, come impostare i prossimi atti (in base alla fase)
Tono: forense, tecnico, impeccabile.
`.trim();
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

// ---------- Chunking ----------
function chunkText(text, chunkSize = 6000) {
  // chunkSize in caratteri (semplice e robusto)
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    const slice = text.slice(i, i + chunkSize);
    chunks.push(slice);
    i += chunkSize;
  }
  return chunks;
}

async function analyzeChunksAsOneDocument(docName, fullText) {
  const chunks = chunkText(fullText, 6000);

  // 1) Analisi per chunk (sintesi/estrazione punti chiave)
  const partials = [];
  for (let idx = 0; idx < chunks.length; idx++) {
    const messages = [
      { role: "system", content: systemActaScan() },
      { role: "user", content:
        `Documento: ${docName}\n` +
        `CHUNK ${idx + 1}/${chunks.length}\n\n` +
        `Analizza SOLO questo chunk ed estrai: (a) tesi/argomenti, (b) punti forti, (c) punti deboli/contraddizioni, (d) istituti richiamati.\n\n` +
        chunks[idx]
      }
    ];

    const out = await client.chat.completions.create({
      model: MODEL,
      messages,
      temperature: 0.2
    });

    partials.push(out.choices?.[0]?.message?.content || "");
  }

  // 2) Consolidamento finale del documento (report unico)
  const consolidate = await client.chat.completions.create({
    model: MODEL,
    temperature: 0.2,
    messages: [
      { role: "system", content: systemActaScan() },
      { role: "user", content:
        `Ora consolida in UN SOLO REPORT completo l’analisi del documento "${docName}".\n` +
        `Usa esclusivamente le analisi parziali seguenti (una per chunk) e produci un report finale unico, coerente e non ripetitivo.\n\n` +
        partials.map((p,i)=>`--- ANALISI CHUNK ${i+1} ---\n${p}\n`).join("\n")
      }
    ]
  });

  return consolidate.choices?.[0]?.message?.content || "Nessun output.";
}

// ---------- Chat endpoint ----------
app.post("/chat", async (req, res) => {
  try {
    const { message, history } = req.body;
    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "Messaggio non valido" });
    }

    const messages = [
      { role: "system", content: systemChat() },
      ...(Array.isArray(history) ? history.slice(-20) : []),
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

    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) res.write(`data: ${JSON.stringify({ delta })}\n\n`);
    }

    res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
    res.end();
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Errore server" });
  }
});

// ---------- MULTIFILE PDF + CHUNKING + COMPARAZIONE ----------
app.post("/analyze-pdfs", upload.array("files", 5), async (req, res) => {
  try {
    const files = req.files || [];
    if (!files.length) return res.status(400).json({ error: "Nessun file ricevuto" });

    // 1) Estrazione testo per ogni PDF
    const docs = [];
    for (const f of files) {
      if (f.mimetype !== "application/pdf") {
        return res.status(400).json({ error: "Formato non supportato (solo PDF)" });
      }
      const parsed = await pdfParse(f.buffer);
      const text = (parsed.text || "").trim();
      if (!text) {
        docs.push({ name: f.originalname, text: "", report: "Impossibile estrarre testo dal PDF." });
      } else {
        docs.push({ name: f.originalname, text });
      }
    }

    // 2) Analisi con chunking per ogni documento
    const perDocReports = [];
    for (const d of docs) {
      if (!d.text) {
        perDocReports.push(`## ${d.name}\n\n${d.report || "Nessun testo."}\n`);
        continue;
      }
      const report = await analyzeChunksAsOneDocument(d.name, d.text);
      perDocReports.push(`## REPORT DOCUMENTO: ${d.name}\n\n${report}\n`);
    }

    // 3) Se 2+ documenti: confronto e strategia
    let compareSection = "";
    if (docs.length >= 2) {
      const compareOut = await client.chat.completions.create({
        model: MODEL,
        temperature: 0.2,
        messages: [
          { role: "system", content: systemCompare() },
          { role: "user", content:
            `Confronta i seguenti report (uno per documento). Dopo aver commentato singolarmente, confrontali tra loro.\n` +
            `Valuta se uno tende a prevalere sull’altro, evidenzia argomentazioni discordanti, punti di attrito e suggerisci prossimi atti per ciascuna parte (in base a una controversia civile).\n\n` +
            perDocReports.join("\n\n")
          }
        ]
      });

      compareSection = `# CONFRONTO TRA DOCUMENTI\n\n${compareOut.choices?.[0]?.message?.content || ""}\n`;
    }

    const final = `# ANALISI COMPLESSIVA (MULTIFILE + CHUNKING)\n\n` +
                  perDocReports.join("\n\n") +
                  (compareSection ? `\n\n${compareSection}` : "");

    res.json({ result: final });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Errore server multifile" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Backend live on ${PORT}`));
