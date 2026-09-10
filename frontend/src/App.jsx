import { useEffect, useState } from "react";
import axios from "axios";
import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  HeartPulse,
  LoaderCircle,
  RefreshCw,
  ShieldCheck,
  Zap,
} from "lucide-react";

import ImageUploader from "./components/ImageUploader";
import PredictionCard from "./components/PredictionCard";
import ProbabilityChart from "./components/ProbabilityChart";

const API_URL = import.meta.env.VITE_API_URL || "https://ecg-api-production.up.railway.app";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/health`);
      setHealth(response.data);
    } catch {
      setHealth({
        status: "offline",
        model_loaded: false,
        error: "Cannot connect to FastAPI backend.",
      });
    }
  };

  const handleFile = (selected) => {
    if (!selected.type.startsWith("image/")) {
      setError("Please select a valid ECG image.");
      return;
    }

    setError("");
    setResult(null);
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  const clearFile = () => {
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
  };

  const predict = async () => {
    if (!file) {
      setError("Please upload an ECG image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(
        `${API_URL}/api/predict`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setResult(response.data);
    } catch (err) {
      const message =
        err.response?.data?.detail ||
        "Prediction failed. Make sure the FastAPI backend and trained model are running.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const modelOnline = health?.model_loaded === true;

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-slate-950 p-2.5 text-white">
              <HeartPulse size={22} />
            </div>
            <div>
              <p className="font-black tracking-tight text-slate-950">
                ECG Vision AI
              </p>
              <p className="text-xs text-slate-500">
                CNN image classification
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-2">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                modelOnline ? "bg-emerald-500" : "bg-red-500"
              }`}
            />
            <span className="text-xs font-semibold text-slate-600">
              {modelOnline ? "Model Online" : "Model Offline"}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-5 py-8 lg:px-8 lg:py-12">
        <section className="mb-9">
          <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 shadow-sm">
            <Zap size={14} />
            TensorFlow · Keras · FastAPI · React
          </div>

          <h1 className="mt-5 max-w-4xl text-4xl font-black tracking-tight text-slate-950 sm:text-5xl">
            ECG classification dashboard
          </h1>

          <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600">
            Upload an ECG image and send it through your trained 224 × 224 CNN.
            The dashboard displays the predicted class, confidence, and the
            complete softmax probability distribution.
          </p>
        </section>

        <section className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-3">
          {[
            {
              icon: BrainCircuit,
              title: "CNN inference",
              text: "Your trained Keras model",
            },
            {
              icon: Activity,
              title: "224 × 224 input",
              text: "RGB + 1/255 scaling",
            },
            {
              icon: ShieldCheck,
              title: "API protected",
              text: "File validation + size limit",
            },
          ].map((item) => (
            <div
              key={item.title}
              className="glass rounded-2xl border border-slate-200 p-4 shadow-sm"
            >
              <item.icon size={20} className="text-slate-700" />
              <p className="mt-3 text-sm font-bold text-slate-900">
                {item.title}
              </p>
              <p className="mt-1 text-xs text-slate-500">{item.text}</p>
            </div>
          ))}
        </section>

        {error && (
          <div className="mb-6 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.15fr_0.85fr]">
          <ImageUploader
            file={file}
            preview={preview}
            onFile={handleFile}
            onClear={clearFile}
            disabled={loading}
          />

          <PredictionCard result={result} />
        </div>

        <div className="mt-6">
          {result ? (
            <ProbabilityChart result={result} />
          ) : (
            <div className="rounded-3xl border border-dashed border-slate-300 bg-white p-10 text-center">
              <p className="text-sm font-semibold text-slate-700">
                Probability chart will appear after inference
              </p>
              <p className="mt-1 text-xs text-slate-500">
                The backend returns all five softmax class probabilities.
              </p>
            </div>
          )}
        </div>

        <div className="mt-6 flex flex-col gap-4 rounded-3xl border border-slate-200 bg-white p-5 shadow-soft sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-sm font-bold text-slate-900">
              Ready to classify an ECG?
            </p>
            <p className="mt-1 text-xs text-slate-500">
              {file
                ? `Selected: ${file.name}`
                : "Choose an ECG image to begin."}
            </p>
          </div>

          <div className="flex gap-3">
            <button
              onClick={checkHealth}
              className="inline-flex items-center justify-center gap-2 rounded-xl border border-slate-200 px-4 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
            >
              <RefreshCw size={16} />
              Check API
            </button>

            <button
              onClick={predict}
              disabled={!file || loading}
              className="inline-flex min-w-40 items-center justify-center gap-2 rounded-xl bg-slate-950 px-5 py-3 text-sm font-bold text-white shadow-lg transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? (
                <>
                  <LoaderCircle className="animate-spin" size={17} />
                  Predicting...
                </>
              ) : (
                <>
                  <CheckCircle2 size={17} />
                  Run prediction
                </>
              )}
            </button>
          </div>
        </div>

        <footer className="mt-10 border-t border-slate-200 pt-6 text-center text-xs leading-6 text-slate-500">
          AI classification is for demonstration and decision-support purposes
          only. It is not a substitute for professional medical diagnosis or
          clinical judgment.
        </footer>
      </main>
    </div>
  );
}
