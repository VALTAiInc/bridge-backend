import express from "express";
import multer from "multer";
import cors from "cors";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import FormData from "form-data";
import fetch from "node-fetch";
import Anthropic from "@anthropic-ai/sdk";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;

const LANGUAGE_NAMES = {
  en: "English", es: "Spanish", fr: "French", de: "German",
  pt: "Portuguese", zh: "Mandarin Chinese", ja: "Japanese",
  ko: "Korean", ar: "Arabic", hi: "Hindi",
  fil: "Filipino", "ar-LB": "Arabic",
};

const ELEVENLABS_VOICES = {
  en: process.env.VOICE_EN || "21m00Tcm4TlvDq8ikWAM",
  es: process.env.VOICE_ES || "AZnzlk1XvdvUeBnXmlld",
  fr: process.env.VOICE_FR || "EXAVITQu4vr4xnSDxMaL",
  de: process.env.VOICE_DE || "ErXwobaYiN019PkySvjV",
  pt: process.env.VOICE_PT || "VR6AewLTigWG4xSOukaG",
  zh: process.env.VOICE_ZH || "pNInz6obpgDQGcFmaJgB",
  ja: process.env.VOICE_JA || "yoZ06aMxZJJ28mfd3POQ",
  ko: process.env.VOICE_KO || "pMsXgVXv3BLzUgSXRplE",
  ar: process.env.VOICE_AR || "jsCqWAovK2LkecY7zXl4",
  hi: process.env.VOICE_HI || "ThT5KcBeYPX3keUQqHPh",
  fil: process.env.VOICE_EN || "21m00Tcm4TlvDq8ikWAM",
};

const anthropic = new Anthropic({ apiKey: ANTHROPIC_API_KEY });
const app = express();

app.use(helmet());
app.use(cors({ origin: "*" }));
app.use(express.json());

const limiter = rateLimit({ windowMs: 60 * 1000, max: 60 });
app.use("/api/", limiter);

const upload = multer({
  dest: "/tmp/bridge-uploads/",
  limits: { fileSize: 25 * 1024 * 1024 },
});

async function transcribeAudio(filePath, language) {
  const whisperLang = language === "ar-LB" ? "ar"
    : language === "fil" ? "tl"
    : language;

  const form = new FormData();
  form.append("file", fs.createReadStream(filePath), {
    filename: "recording.m4a",
    contentType: "audio/m4a",
  });
  form.append("model", "whisper-1");
  form.append("response_format", "text");
  if (whisperLang) form.append("language", whisperLang);

  const response = await fetch("https://api.openai.com/v1/audio/transcriptions", {
    method: "POST",
    headers: { Authorization: `Bearer ${OPENAI_API_KEY}`, ...form.getHeaders() },
    body: form,
  });

  if (!response.ok) throw new Error(`Whisper error ${response.status}: ${await response.text()}`);
  return (await response.text()).trim();
}

async function translateText(text, sourceLanguage, targetLanguage) {
  const sourceName = LANGUAGE_NAMES[sourceLanguage] || sourceLanguage;
  const targetName = LANGUAGE_NAMES[targetLanguage] || targetLanguage;

  const message = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    system: `You are a professional translator for workplace communication in trades, hospitality, and industrial settings.
Translate naturally and accurately. Preserve the speaker's tone and intent.
Output ONLY the translated text — no explanations, no quotes, no preamble.`,
    messages: [{ role: "user", content: `Translate from ${sourceName} to ${targetName}:\n\n${text}` }],
  });

  return message.content[0].text.trim();
}

async function synthesizeSpeech(text, language) {
  const voiceId = ELEVENLABS_VOICES[language] || ELEVENLABS_VOICES["en"];
  const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
    method: "POST",
    headers: {
      "xi-api-key": ELEVENLABS_API_KEY,
      "Content-Type": "application/json",
      Accept: "audio/mpeg",
    },
    body: JSON.stringify({
      text,
      model_id: "eleven_turbo_v2",
      voice_settings: { stability: 0.5, similarity_boost: 0.8, style: 0.2, use_speaker_boost: true },
    }),
  });

  if (!response.ok) throw new Error(`ElevenLabs error ${response.status}: ${await response.text()}`);
  return Buffer.from(await response.arrayBuffer());
}

app.post("/api/translate", upload.single("audio"), async (req, res) => {
  const filePath = req.file?.path;
  try {
    const { sourceLanguage, targetLanguage, callerApp = "Bridge", text: textInput } = req.body;
    if (!req.file && !textInput) return res.status(400).json({ error: "No audio file or text provided." });
    if (!sourceLanguage || !targetLanguage) return res.status(400).json({ error: "sourceLanguage and targetLanguage required." });
    if (sourceLanguage === targetLanguage) return res.status(400).json({ error: "Languages must be different." });

    console.log(`[${callerApp}] ${sourceLanguage}→${targetLanguage}`);

    const transcript = textInput ? textInput.trim() : await transcribeAudio(filePath, sourceLanguage);
    if (!transcript) return res.status(422).json({ error: "No speech detected." });
    console.log(`  STT: "${transcript}"`);

    const translation = await translateText(transcript, sourceLanguage, targetLanguage);
    console.log(`  Translation: "${translation}"`);

    const audioBuffer = await synthesizeSpeech(translation, targetLanguage);
    console.log(`  TTS: ${audioBuffer.length} bytes`);

    return res.json({
      transcript,
      translation,
      audioBase64: audioBuffer.toString("base64"),
      audioMimeType: "audio/mpeg",
      sourceLanguage,
      targetLanguage,
    });
  } catch (err) {
    console.error("[Bridge] Error:", err.message);
    return res.status(500).json({ error: "Translation pipeline failed.", detail: err.message });
  } finally {
    if (filePath) fs.unlink(filePath, () => {});
  }
});

app.post("/api/transcribe", upload.single("audio"), async (req, res) => {
  const filePath = req.file?.path;
  try {
    if (!req.file) return res.status(400).json({ error: "No audio file provided." });
    const { language } = req.body;

    const start = Date.now();
    const transcript = await transcribeAudio(filePath, language);
    const durationMs = Date.now() - start;

    if (!transcript) return res.status(422).json({ error: "No speech detected." });
    console.log(`[Transcribe] "${transcript}" (${durationMs}ms)`);

    return res.json({ transcript, language: language || null, durationMs });
  } catch (err) {
    console.error("[Transcribe] Error:", err.message);
    return res.status(500).json({ error: "Transcription failed.", detail: err.message });
  } finally {
    if (filePath) fs.unlink(filePath, () => {});
  }
});

app.get("/api/health", (req, res) => {
  res.json({ status: "ok", service: "Bridge by VALT", version: "2.0.0", timestamp: new Date().toISOString() });
});

app.use((err, req, res, _next) => {
  console.error("[Bridge] Unhandled:", err);
  res.status(500).json({ error: err.message || "Internal server error" });
});

fs.mkdirSync("/tmp/bridge-uploads", { recursive: true });

app.listen(PORT, () => {
  console.log(`\n🌉 Bridge by VALT — Translation Service`);
  console.log(`   Port: ${PORT}`);
  console.log(`   Pipeline: Whisper → Claude → ElevenLabs\n`);
});
