import { useRef } from "react";
import { ImagePlus, UploadCloud, X } from "lucide-react";

export default function ImageUploader({
  file,
  preview,
  onFile,
  onClear,
  disabled,
}) {
  const inputRef = useRef(null);

  const handleDrop = (event) => {
    event.preventDefault();
    if (disabled) return;
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) onFile(dropped);
  };

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      className="rounded-3xl border border-slate-200 bg-white p-6 shadow-soft"
    >
      <div className="mb-5 flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold text-slate-900">ECG image</p>
          <p className="text-xs text-slate-500">
            PNG, JPG, WEBP or BMP · max 10 MB
          </p>
        </div>

        {file && (
          <button
            onClick={onClear}
            className="rounded-full p-2 text-slate-500 hover:bg-slate-100"
            title="Remove image"
          >
            <X size={18} />
          </button>
        )}
      </div>

      {preview ? (
        <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50">
          <img
            src={preview}
            alt="ECG preview"
            className="max-h-[420px] w-full object-contain"
          />
        </div>
      ) : (
        <button
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
          className="group flex min-h-[300px] w-full flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 px-6 transition hover:border-slate-500 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <div className="mb-4 rounded-2xl bg-slate-900 p-4 text-white">
            <UploadCloud size={28} />
          </div>
          <p className="font-semibold text-slate-900">
            Drop your ECG image here
          </p>
          <p className="mt-1 text-sm text-slate-500">
            or click to browse your computer
          </p>
          <span className="mt-5 inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700">
            <ImagePlus size={16} />
            Choose image
          </span>
        </button>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/bmp"
        className="hidden"
        onChange={(e) => {
          const selected = e.target.files?.[0];
          if (selected) onFile(selected);
        }}
      />
    </div>
  );
}
