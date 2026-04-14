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
import { YoutubeTranscript } from "youtube-transcript/dist/youtube-transcript.esm.js";
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
  ja: process.env.VOICE_JA || "WQz3clzUdMqvBf0jswZQ",
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
  fileFilter: (req, file, cb) => {
    const allowed = [
      "audio/", "video/mp4", "video/quicktime", "video/mov",
      "application/octet-stream",
    ];
    if (allowed.some((t) => file.mimetype.startsWith(t))) cb(null, true);
    else cb(new Error(`Unsupported file type: ${file.mimetype}`));
  },
});

async function transcribeAudio(filePath, language, originalMime) {
  const whisperLang = language === "ar-LB" ? "ar"
    : language === "fil" ? "tl"
    : language;

  const isVideo = originalMime && (originalMime.startsWith("video/") || originalMime === "video/quicktime");
  const filename = isVideo ? "recording.mp4" : "recording.m4a";
  const contentType = isVideo ? "video/mp4" : "audio/m4a";

  const form = new FormData();
  form.append("file", fs.createReadStream(filePath), {
    filename,
    contentType,
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

  let systemPrompt = `You are a professional translator for workplace communication in trades, hospitality, and industrial settings.
Translate naturally and accurately. Preserve the speaker's tone and intent.
Output ONLY the translated text — no explanations, no quotes, no preamble.`;

  if (targetLanguage === "ja") {
    systemPrompt += `\n\nJapanese punctuation rules — you MUST follow these:
- Use 。for sentence endings (never a period).
- Use ？for questions (never ?).
- Use ！for exclamations (never !).
- Use 、for natural breath pauses mid-sentence.
- Questions must end with ですか？ or か？ patterns where grammatically natural.`;
  }

  const message = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    system: systemPrompt,
    messages: [{ role: "user", content: `Translate from ${sourceName} to ${targetName}:\n\n${text}` }],
  });

  return message.content[0].text.trim();
}

async function synthesizeSpeech(text, language) {
  const voiceId = ELEVENLABS_VOICES[language] || ELEVENLABS_VOICES["en"];
  const useMultilingual = ["ja", "ko", "zh"].includes(language);
  const modelId = useMultilingual ? "eleven_multilingual_v2" : "eleven_turbo_v2";
  const voiceSettings = language === "ja"
    ? { stability: 0.35, similarity_boost: 0.80, style: 0.50, use_speaker_boost: true }
    : { stability: 0.5, similarity_boost: 0.8, style: 0.2, use_speaker_boost: true };

  const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
    method: "POST",
    headers: {
      "xi-api-key": ELEVENLABS_API_KEY,
      "Content-Type": "application/json",
      Accept: "audio/mpeg",
    },
    body: JSON.stringify({
      text,
      model_id: modelId,
      voice_settings: voiceSettings,
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

app.post("/api/translate-text", async (req, res) => {
  try {
    const { text, sourceLanguage, targetLanguage, callerApp = "Bridge" } = req.body || {};
    if (!text || !sourceLanguage || !targetLanguage) {
      return res.status(400).json({ error: "text, sourceLanguage, and targetLanguage required." });
    }
    if (sourceLanguage === targetLanguage) {
      return res.status(400).json({ error: "Languages must be different." });
    }

    console.log(`[${callerApp}] (text) ${sourceLanguage}→${targetLanguage}`);

    const start = Date.now();
    const transcript = text.trim();
    console.log(`  Input: "${transcript}"`);

    const translation = await translateText(transcript, sourceLanguage, targetLanguage);
    console.log(`  Translation: "${translation}"`);

    const audioBuffer = await synthesizeSpeech(translation, targetLanguage);
    console.log(`  TTS: ${audioBuffer.length} bytes`);

    const durationMs = Date.now() - start;

    return res.json({
      transcript,
      translation,
      audioBase64: audioBuffer.toString("base64"),
      audioMimeType: "audio/mpeg",
      sourceLanguage,
      targetLanguage,
      durationMs,
    });
  } catch (err) {
    console.error("[Bridge] translate-text error:", err.message);
    return res.status(500).json({ error: "Translation pipeline failed.", detail: err.message });
  }
});

app.post("/api/clean-japanese", async (req, res) => {
  try {
    const { text } = req.body || {};
    if (!text) {
      return res.status(400).json({ error: "text is required." });
    }

    console.log(`[CleanJP] Input: "${text.slice(0, 80)}..."`);

    const message = await anthropic.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      system: `You are a text prep assistant for Japanese text-to-speech. The input is English text that will be read aloud by a Japanese voice. Add natural punctuation to improve TTS delivery — use ? for questions and ... for natural pauses. Do not add exclamation marks. Never duplicate punctuation that already exists in the text. Do not translate. Do not change any words. Return ONLY the corrected text.`,
      messages: [{ role: "user", content: text }],
    });

    const cleaned = message.content[0].text.trim();
    console.log(`[CleanJP] Output: "${cleaned.slice(0, 80)}..."`);

    return res.json({ text: cleaned });
  } catch (err) {
    console.error("[CleanJP] Error:", err.message);
    return res.status(500).json({ error: "Japanese punctuation cleanup failed.", detail: err.message });
  }
});

app.post("/api/speak", async (req, res) => {
  try {
    const { text, language, voiceSettings, customVoiceId } = req.body || {};
    if (!text || !language) {
      return res.status(400).json({ error: "text and language are required." });
    }

    const voiceId = customVoiceId || ELEVENLABS_VOICES[language] || ELEVENLABS_VOICES["en"];
    const useMultilingual = ["ja", "ko", "zh"].includes(language);
    const modelId = useMultilingual ? "eleven_multilingual_v2" : "eleven_multilingual_v2";
    const defaults = language === "ja"
      ? { stability: 0.35, similarity_boost: 0.80, style: 0.25, use_speaker_boost: true }
      : { stability: 0.5, similarity_boost: 1.0, style: 0.2, use_speaker_boost: true, speed: 1.0 };
    const mergedSettings = voiceSettings
      ? { ...defaults, ...voiceSettings, use_speaker_boost: true }
      : defaults;

    console.log(`[Speak] lang=${language}, text="${text.slice(0, 80)}..."`);

    const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}?output_format=mp3_44100_192`, {
      method: "POST",
      headers: {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        Accept: "audio/mpeg",
      },
      body: JSON.stringify({
        text,
        model_id: modelId,
        voice_settings: mergedSettings,
      }),
    });

    if (!response.ok) {
      const errBody = await response.text();
      throw new Error(`ElevenLabs error ${response.status}: ${errBody}`);
    }

    const audioBuffer = Buffer.from(await response.arrayBuffer());
    console.log(`[Speak] TTS: ${audioBuffer.length} bytes`);

    return res.json({
      audioBase64: audioBuffer.toString("base64"),
      audioMimeType: "audio/mpeg",
    });
  } catch (err) {
    console.error("[Speak] Error:", err.message);
    return res.status(500).json({ error: "Speech synthesis failed.", detail: err.message });
  }
});

app.post("/api/clone-voice", upload.single("audio"), async (req, res) => {
  const filePath = req.file?.path;
  try {
    if (!req.file) return res.status(400).json({ error: "No audio file provided." });
    const { voiceName } = req.body || {};
    if (!voiceName) return res.status(400).json({ error: "voiceName is required." });

    const fileSize = fs.statSync(filePath).size;
    const ext = (req.file.originalname || "voice.m4a").split(".").pop()?.toLowerCase() ?? "m4a";
    const mimeMap = { m4a: "audio/mp4", mp3: "audio/mpeg", wav: "audio/wav", mp4: "audio/mp4" };
    const contentType = mimeMap[ext] || "audio/mp4";
    const filename = ext === "m4a" ? req.file.originalname || "voice.m4a" : `voice.${ext}`;

    console.log(`[CloneVoice] name="${voiceName}", file=${req.file.originalname}, uploadedMime=${req.file.mimetype}, resolvedMime=${contentType}, size=${fileSize} bytes`);

    const form = new FormData();
    form.append("name", voiceName);
    form.append("files", fs.createReadStream(filePath), {
      filename,
      contentType,
    });

    const response = await fetch("https://api.elevenlabs.io/v1/voices/add", {
      method: "POST",
      headers: { "xi-api-key": ELEVENLABS_API_KEY, ...form.getHeaders() },
      body: form,
    });

    if (!response.ok) {
      const errBody = await response.text();
      console.error(`[CloneVoice] ElevenLabs responded ${response.status}: ${errBody}`);
      throw new Error(`ElevenLabs clone error ${response.status}: ${errBody}`);
    }

    const data = await response.json();
    console.log(`[CloneVoice] Created voice: ${data.voice_id}`);

    return res.json({ voiceId: data.voice_id });
  } catch (err) {
    console.error("[CloneVoice] Error:", err.message);
    return res.status(500).json({ error: "Voice cloning failed.", detail: err.message });
  } finally {
    if (filePath) fs.unlink(filePath, () => {});
  }
});

app.post("/api/transcribe", upload.single("audio"), async (req, res) => {
  const filePath = req.file?.path;
  try {
    if (!req.file) return res.status(400).json({ error: "No audio or video file provided." });
    const { language } = req.body;

    const start = Date.now();
    const transcript = await transcribeAudio(filePath, language, req.file.mimetype);
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

app.post("/api/transcribe-youtube", async (req, res) => {
  try {
    const { youtubeUrl } = req.body;
    if (!youtubeUrl) return res.status(400).json({ error: "youtubeUrl is required." });

    console.log(`[YouTube] Fetching captions: ${youtubeUrl}`);
    const items = await YoutubeTranscript.fetchTranscript(youtubeUrl);

    if (!items || items.length === 0) {
      return res.status(422).json({ error: "This video has no closed captions available." });
    }

    const transcript = items.map((i) => i.text).join(" ");
    console.log(`[YouTube] Got ${items.length} segments, ${transcript.length} chars`);

    return res.json({ transcript });
  } catch (err) {
    console.error("[YouTube] Error:", err.message);
    if (err.message?.includes("disabled") || err.message?.includes("not available")) {
      return res.status(422).json({ error: "This video has no closed captions available." });
    }
    return res.status(500).json({ error: "YouTube transcription failed.", detail: err.message });
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
