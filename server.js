import express from "express";
import cors from "cors";
import OpenAI from "openai";
import multer from "multer";
import pdfParse from "pdf-parse";

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 15 * 1024 * 1024 } }); // 15MB

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";

function buildSystemPromptChat() {
  return `
Sei "Atto Perfetto – Area Dinamica".
Agisci come Avvocato esperto di diritto italiano.
Tono tecnico, chiaro, professionale.
Se mancano informazioni essenziali, chiedile prima di procedere.
Struttura sempre l’output con titoli e paragrafi.
`.trim();
}

function buildSystemPromptActaScan() {
  return `
Sei "Atto Perfetto – ACTA SCAN".
Analizza un atto/ documento giuridico civile italiano in modo professionale e imparziale.
Produci un report strutturato in:
1) Punti di forza
2) Punti deboli e contraddizioni
3) Istituti giuridici e fattispecie (forti vs deboli)
4) Considerazioni finali operative
Non inventare fatti: se un passaggio è ambiguo, segnala l’ambiguità.
`.trim();
}

// Chat streaming (SSE)
app.post("/chat", async (req, res) => {
  try {
    const { message, history } = req.body;
    if (!message || typeof message !== "string") return res.status(400).json({ error: "Messaggio non valido" });

    const messages = [
      { role: "system", content: buildSystemPromptChat() },
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

// PDF analysis (JSON response)
app.post("/analyze-pdf", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "File mancante" });
    if (req.file.mimetype !== "application/pdf") return res.status(400).json({ error: "Formato non supportato (solo PDF)" });

    const pdfText = (await pdfParse(req.file.buffer)).text || "";
    if (!pdfText.trim()) return res.status(400).json({ error: "Impossibile estrarre testo dal PDF" });

    // Se troppo lungo, tagliamo (poi possiamo fare chunking avanzato)
    const limited = pdfText.slice(0, 18000);

    const messages = [
      { role: "system", content: buildSystemPromptActaScan() },
      { role: "user", content: `Analizza questo documento:\n\n${limited}` }
    ];

    const out = await client.chat.completions.create({
      model: MODEL,
      messages,
      temperature: 0.2
    });

    const result = out.choices?.[0]?.message?.content || "Nessuna risposta.";
    res.json({ result });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Errore server PDF" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Backend live on ${PORT}`));
