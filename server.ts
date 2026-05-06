import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Lazy-initialized Gemini Client
let ai: GoogleGenAI | null = null;
function getAI() {
  if (!ai) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is not set in environment variables.");
    }
    ai = new GoogleGenAI({ apiKey });
  }
  return ai;
}

// Prediction API
app.post("/api/predict", async (req, res) => {
  try {
    const { stream, percentage, age, skills } = req.body;

    const result = await getAI().models.generateContent({
      model: "gemini-2.0-flash",
      contents: `
      You are an expert career advisor and job market analyst. 
      Analyze the following candidate profile and predict the top job opportunity for them.
      
      Candidate Profile:
      - Stream: ${stream}
      - Academic Percentage: ${percentage}%
      - Age: ${age}
      - Additional Skills: ${skills}
      
      Return a JSON object with EXACTLY this structure:
      {
        "prediction": "Job Title",
        "confidence": 0.0-1.0,
        "explanation": "Brief 1-2 sentence reason for this prediction based on their background.",
        "skillsToImprove": ["Skill 1", "Skill 2"]
      }
    `});

    const responseText = result.text;
    
    // Extract JSON from potential markdown blocks
    const jsonMatch = responseText.match(/\{[\s\S]*\}/);
    if (!jsonMatch) throw new Error("Failed to parse prediction results.");
    
    const predictionData = JSON.parse(jsonMatch[0]);
    res.json(predictionData);
  } catch (error) {
    console.error("Prediction error:", error);
    res.status(500).json({ error: "Failed to generate prediction. Please ensure GEMINI_API_KEY is configured." });
  }
});

async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
