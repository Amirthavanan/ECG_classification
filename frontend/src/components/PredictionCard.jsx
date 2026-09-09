import { Activity, CheckCircle2, TriangleAlert } from "lucide-react";

export default function PredictionCard({ result }) {
  if (!result) {
    return (
      <div className="flex min-h-[300px] flex-col items-center justify-center rounded-3xl border border-slate-200 bg-white p-8 text-center shadow-soft">
        <div className="mb-4 rounded-full bg-slate-100 p-4 text-slate-700">
          <Activity size={30} />
        </div>
        <h3 className="text-lg font-bold text-slate-900">No prediction yet</h3>
        <p className="mt-2 max-w-sm text-sm leading-6 text-slate-500">
          Upload an ECG image and run the CNN model to see the predicted class
          and confidence score.
        </p>
      </div>
    );
  }

  const confidencePercent = (result.confidence * 100).toFixed(2);

  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-soft">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold text-slate-500">AI prediction</p>
        <CheckCircle2 className="text-emerald-600" size={21} />
      </div>

      <div className="mt-7">
        <p className="text-xs font-medium uppercase tracking-widest text-slate-400">
          Predicted class
        </p>
        <h2 className="mt-2 break-words text-3xl font-black tracking-tight text-slate-950">
          {result.predicted_class}
        </h2>
      </div>

      <div className="mt-7 rounded-2xl bg-slate-50 p-5">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-slate-600">
            Confidence
          </span>
          <span className="text-2xl font-black text-slate-950">
            {confidencePercent}%
          </span>
        </div>

        <div className="mt-4 h-3 overflow-hidden rounded-full bg-slate-200">
          <div
            className="h-full rounded-full bg-slate-900 transition-all duration-700"
            style={{ width: `${confidencePercent}%` }}
          />
        </div>
      </div>

      <div className="mt-5 flex gap-3 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-900">
        <TriangleAlert className="mt-0.5 shrink-0" size={18} />
        <p>
          This is an AI classification result for decision support and is not
          a medical diagnosis.
        </p>
      </div>
    </div>
  );
}
