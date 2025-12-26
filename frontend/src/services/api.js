// src/services/api.js
// get from .env file
const API_BASE = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

function getSearchMode() {
  const mode = window.localStorage.getItem("searchMode");
  // Default fallback is HNSW
  return mode === "bruteforce" ? "bruteforce" : "hnsw";
}

export async function recognizeImage(formData) {
  const mode = getSearchMode();

  const res = await fetch(`${API_BASE}/recognize_image?mode=${mode}`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Upload API error");
  }

  const data = await res.json();
  // Optionally attach client-side timing info
  return data;
}

export async function recognizeFrame(base64Image) {
  const mode = getSearchMode();

  const payload = { image: base64Image };

  const res = await fetch(`${API_BASE}/recognize_frame?mode=${mode}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error("Webcam API error");
  }

  const data = await res.json();
  return data;
}
