// Text-only Gemini bridge for tools/text/gemini_text.py.
//
// Protocol: one JSON request on stdin, one JSON response on stdout.
//   request:  { prompt, model?, system_instruction?, temperature?,
//               max_output_tokens?, timeout_ms? }
//   success:  { ok: true, text, model, model_version, finish_reason, usage }
//   failure:  { ok: false, error, status? }   (exit code 1, or 2 for bad setup)
//
// The API key is read ONLY from process.env.GEMINI_API_KEY. It is never
// accepted on the command line or in the request, never printed, and any
// accidental occurrence of it in an error message is replaced with "***".

import { GoogleGenAI } from "@google/genai";

const DEFAULT_MODEL = "gemini-3-flash-preview";
const DEFAULT_TIMEOUT_MS = 120_000;

const apiKey = process.env.GEMINI_API_KEY || "";

function redact(message) {
  let text = String(message ?? "");
  if (apiKey) text = text.split(apiKey).join("***");
  return text;
}

function reply(payload, exitCode = 0) {
  process.stdout.write(JSON.stringify(payload));
  process.exitCode = exitCode;
}

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
}

async function main() {
  if (!apiKey) {
    return reply({ ok: false, error: "GEMINI_API_KEY is not set." }, 2);
  }

  let request;
  try {
    request = JSON.parse(await readStdin());
  } catch {
    return reply({ ok: false, error: "Request on stdin is not valid JSON." }, 2);
  }

  const prompt = typeof request.prompt === "string" ? request.prompt.trim() : "";
  if (!prompt) {
    return reply({ ok: false, error: "'prompt' must be a non-empty string." }, 2);
  }

  const model = request.model || DEFAULT_MODEL;
  const config = {
    // Text only: never ask for image or audio output.
    responseModalities: ["TEXT"],
    httpOptions: { timeout: Number(request.timeout_ms) || DEFAULT_TIMEOUT_MS },
  };
  if (request.system_instruction) config.systemInstruction = String(request.system_instruction);
  if (request.temperature !== undefined && request.temperature !== null) {
    config.temperature = Number(request.temperature);
  }
  if (request.max_output_tokens) config.maxOutputTokens = Number(request.max_output_tokens);

  const ai = new GoogleGenAI({ apiKey, vertexai: false });

  try {
    const response = await ai.models.generateContent({ model, contents: prompt, config });
    const text = response.text ?? "";
    const candidate = response.candidates?.[0];
    if (!text) {
      const reason =
        response.promptFeedback?.blockReason || candidate?.finishReason || "unknown";
      return reply({ ok: false, error: `Gemini returned no text (reason: ${reason}).` }, 1);
    }
    const usage = response.usageMetadata || {};
    return reply({
      ok: true,
      text,
      model,
      model_version: response.modelVersion ?? null,
      finish_reason: candidate?.finishReason ?? null,
      usage: {
        prompt_tokens: usage.promptTokenCount ?? null,
        output_tokens: usage.candidatesTokenCount ?? null,
        thoughts_tokens: usage.thoughtsTokenCount ?? null,
        total_tokens: usage.totalTokenCount ?? null,
      },
    });
  } catch (err) {
    return reply(
      { ok: false, error: redact(err?.message || err), status: err?.status ?? null },
      1,
    );
  }
}

main().catch((err) => reply({ ok: false, error: redact(err?.message || err) }, 1));
