import express from "express";
import cors from "cors";
import OpenAI from "openai";

const app = express();

// CORS – in produzione puoi limitare al tuo dominio
app.use(cors());

// Body JSON
app.use(express.json({ limit: "1mb" }));

// Client OpenAI
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Modello (impostabile da variabile ambiente)
const MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";

// Prompt di sistema Atto Perfetto
function buildSystemPrompt() {
  return `
Sei "Atto Perfetto – Area Dinamica".
Agisci come Avvocato esperto di diritto italiano.
Usa un linguaggio tecnico, chiaro, professionale e strutturato.

Obiettivi:
- Guidare l’utente nella redazione di atti giuridici impeccabili
- Evidenziare struttura, strategia e coerenza logico-giuridica
- Suggerire miglioramenti senza inventare fatti

Regole operative:
- Se mancano informazioni essenziali, chiedile prima di procedere
- Struttura sempre l’output con titoli e paragrafi
- Non citare giurisprudenza o norme se non fornite o chiaramente richiedibili
- Mantieni sempre un tono professionale forense
`.trim();
}

// Endpoint CHAT
app.post("/chat", async (req, res) => {
  try {
    const { message, history } = req.body;

    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "Messaggio non valido" });
    }

    const messages = [
      { role: "system", content: buildSystemPrompt() },
      ...(Array.isArray(history) ? history.slice(-20) : []),
      { role: "user", content: message }
    ];

    // Headers per streaming
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
      if (delta) {
        res.write(`data: ${JSON.stringify({ delta })}\n\n`);
      }
    }

    res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
    res.end();

  } catch (error) {
    console.error(error);
    res.write(`data: ${JSON.stringify({ error: "Errore server" })}\n\n`);
    res.end();
  }
});

// Avvio server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Atto Perfetto backend attivo sulla porta ${PORT}`);
});
