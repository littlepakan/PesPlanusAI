"use client";
import { useState, useEffect, useMemo, useRef } from "react";

type ImageTask = {
  id: string;
  file: File;
  previewUrl: string;
  status: "idle" | "processing" | "success" | "cancelled" | "error";
  result?: any;
  controller?: AbortController;
};

export default function Home() {
  const [tasks, setTasks] = useState<ImageTask[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  // File states
  const [weightFile, setWeightFile] = useState<File | null>(null);
  const [clfFile, setClfFile] = useState<File | null>(null);
  const [rfeFile, setRfeFile] = useState<File | null>(null);
  const [csvFile, setCsvFile] = useState<File | null>(null);

  // Drag-and-drop active states for custom file boxes
  const [isDraggingWeight, setIsDraggingWeight] = useState(false);
  const [isDraggingClf, setIsDraggingClf] = useState(false);
  const [isDraggingRfe, setIsDraggingRfe] = useState(false);
  const [isDraggingCsv, setIsDraggingCsv] = useState(false);

  // Refs for hidden file inputs
  const weightInputRef = useRef<HTMLInputElement>(null);
  const clfInputRef = useRef<HTMLInputElement>(null);
  const rfeInputRef = useRef<HTMLInputElement>(null);
  const csvInputRef = useRef<HTMLInputElement>(null);

  // Config states
  const [backbone, setBackbone] = useState("GoogleNet");
  const [classifier, setClassifier] = useState("Fine-Tuning");
  const [filterType, setFilterType] = useState("Median Filter");
  const [useGroundTruth, setUseGroundTruth] = useState(false);

  // UX states
  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"full" | "compact">("full");
  const [theme, setTheme] = useState<"light" | "dark" | "system">("system");

  const safeClassifier = classifier.trim().toLowerCase();
  const isFineTuning =
    safeClassifier === "fine-tuning" ||
    safeClassifier === "finetuning" ||
    safeClassifier === "ft";
  const isPklModel = !isFineTuning;

  // Theme Management
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "dark") {
      root.classList.add("dark");
    } else if (theme === "light") {
      root.classList.remove("dark");
    } else {
      if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
        root.classList.add("dark");
      } else {
        root.classList.remove("dark");
      }
    }
  }, [theme]);

  // Confusion Matrix Calculations
  const metrics = useMemo(() => {
    const evaluatedTasks = tasks.filter(
      (t) =>
        t.status === "success" &&
        t.result &&
        t.result.eval_status !== "ไม่มีเฉลย",
    );

    const tp = evaluatedTasks.filter((t) =>
      t.result.eval_status.includes("TP"),
    ).length;
    const tn = evaluatedTasks.filter((t) =>
      t.result.eval_status.includes("TN"),
    ).length;
    const fp = evaluatedTasks.filter((t) =>
      t.result.eval_status.includes("FP"),
    ).length;
    const fn = evaluatedTasks.filter((t) =>
      t.result.eval_status.includes("FN"),
    ).length;
    const total = tp + tn + fp + fn;

    const accuracy = total > 0 ? ((tp + tn) / total) * 100 : 0;
    const precision = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
    const recall = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
    const f1 =
      precision + recall > 0
        ? (2 * (precision * recall)) / (precision + recall)
        : 0;

    return { tp, tn, fp, fn, total, accuracy, precision, recall, f1 };
  }, [tasks]);

  const handleFiles = (files: FileList | File[]) => {
    const newTasks: ImageTask[] = Array.from(files)
      .filter((file) => file.type.startsWith("image/"))
      .map((file) => ({
        id: `${file.name}-${Date.now()}`,
        file,
        previewUrl: URL.createObjectURL(file),
        status: "idle",
      }));
    setTasks((prev) => [...prev, ...newTasks]);
  };

  const removeTask = (id: string) => {
    setTasks((prev) => {
      const task = prev.find((t) => t.id === id);
      if (task?.previewUrl) URL.revokeObjectURL(task.previewUrl);
      return prev.filter((t) => t.id !== id);
    });
  };

  const clearAllTasks = () => {
    tasks.forEach((t) => {
      if (t.previewUrl) URL.revokeObjectURL(t.previewUrl);
    });
    setTasks([]);
  };

  const processAllTasks = async (forceReprocess = false) => {
    const targetTasks = forceReprocess
      ? tasks
      : tasks.filter(
          (t) =>
            t.status === "idle" ||
            t.status === "error" ||
            t.status === "cancelled",
        );

    if (targetTasks.length === 0) return;

    if (isFineTuning && !weightFile) {
      alert(
        "⚠️ กรุณาอัปโหลดไฟล์ Weights (.pth) สำหรับ Fine-Tuning ก่อนประมวลผล",
      );
      return;
    }
    if (isPklModel && (!clfFile || !rfeFile)) {
      alert(
        "⚠️ กรุณาอัปโหลดไฟล์ Classifier (.pkl) และ RFE Selector (.pkl) ให้ครบถ้วน",
      );
      return;
    }
    if (useGroundTruth && !csvFile) {
      alert("⚠️ กรุณาอัปโหลดไฟล์เฉลย CSV (Ground Truth)");
      return;
    }

    const controller = new AbortController();

    setTasks((prev) =>
      prev.map((t) =>
        targetTasks.some((tt) => tt.id === t.id)
          ? { ...t, status: "processing", controller, result: undefined }
          : t,
      ),
    );

    const formData = new FormData();
    targetTasks.forEach((task) => {
      formData.append("files", task.file);
    });

    formData.append("backbone", backbone);
    formData.append("classifier", classifier);
    formData.append("filter_type", filterType);
    formData.append("gt_option", useGroundTruth ? "upload" : "none");

    if (isFineTuning && weightFile) formData.append("weight_file", weightFile);
    if (isPklModel && clfFile) formData.append("clf_file", clfFile);
    if (isPklModel && rfeFile) formData.append("rfe_file", rfeFile);
    if (useGroundTruth && csvFile) formData.append("csv_file", csvFile);

    try {
      const res = await fetch("https://pesplanusai.onrender.com/api/predict", {
        // const res = await fetch("http://127.0.0.1:8000/api/predict", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(
          errorData.detail || `ข้อผิดพลาดจากเซิร์ฟเวอร์ (HTTP ${res.status})`,
        );
      }

      const results = await res.json();
      setTasks((prev) =>
        prev.map((t) => {
          const matchResult = results.find((r: any) => r.id === t.file.name);
          if (matchResult) {
            return { ...t, status: "success", result: matchResult };
          }
          return t;
        }),
      );
    } catch (error: any) {
      if (error.name === "AbortError") {
        setTasks((prev) =>
          prev.map((t) =>
            targetTasks.some((tt) => tt.id === t.id)
              ? { ...t, status: "cancelled" }
              : t,
          ),
        );
      } else {
        alert(
          "ข้อผิดพลาดจากระบบ: \n" +
            (error.message || "เกิดข้อผิดพลาดในการเชื่อมต่อ"),
        );
        setTasks((prev) =>
          prev.map((t) =>
            targetTasks.some((tt) => tt.id === t.id)
              ? {
                  ...t,
                  status: "error",
                  result: { error: error.message || "เกิดข้อผิดพลาด" },
                }
              : t,
          ),
        );
      }
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors duration-200">
      <header className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-800 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="text-2xl">🦶</span>
            <div>
              <h1 className="font-bold text-lg leading-tight">
                Pes Planus AI Diagnosis
              </h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                ระบบจำแนกภาวะเท้าแบนจากภาพรังสี
              </p>
            </div>
          </div>
          <select
            value={theme}
            onChange={(e: any) => setTheme(e.target.value)}
            className="text-xs bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg px-2.5 py-1.5 focus:outline-none"
          >
            <option value="system">💻 ธีมระบบ</option>
            <option value="light">☀️ โหมดสว่าง</option>
            <option value="dark">🌙 โหมดมืด</option>
          </select>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar Settings Panel */}
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700 space-y-4">
              <h2 className="font-bold text-sm border-b border-gray-100 dark:border-gray-700 pb-2 flex items-center justify-between">
                <span>⚙️ การตั้งค่าโมเดล</span>
              </h2>

              <div>
                <label className="block text-xs font-semibold mb-1">
                  Backbone Network
                </label>
                <select
                  value={backbone}
                  onChange={(e) => setBackbone(e.target.value)}
                  className="w-full text-xs bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg p-2"
                >
                  <option value="GoogleNet">GoogleNet (224x224)</option>
                  <option value="SqueezeNet">SqueezeNet (227x227)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold mb-1">
                  Algorithm
                </label>
                <select
                  value={classifier}
                  onChange={(e) => setClassifier(e.target.value)}
                  className="w-full text-xs bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg p-2"
                >
                  <option value="Fine-Tuning">Fine-Tuning (End-to-End)</option>
                  <option value="Neural Network">Neural Network (MLP)</option>
                  <option value="RandomForest">Random Forest</option>
                  <option value="SVM">Support Vector Machine (SVM)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold mb-1">
                  Preprocessing Filter
                </label>
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="w-full text-xs bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg p-2"
                >
                  <option value="Median Filter">Median Filter (3x3)</option>
                  <option value="Gaussian Blur">Gaussian Blur (3x3)</option>
                  <option value="None">None (ภาพต้นฉบับ)</option>
                </select>
              </div>

              {/* Permanent File Uploads Section (No hide/show toggle) */}
              <div className="border border-blue-200 dark:border-blue-900 bg-blue-50/50 dark:bg-blue-900/10 rounded-lg p-3 space-y-3">
                <div className="text-xs font-bold text-blue-700 dark:text-blue-400 border-b border-blue-100 dark:border-blue-900/50 pb-2">
                  📁 จัดการไฟล์โมเดล & เฉลย
                </div>

                {isFineTuning ? (
                  <div>
                    <label className="block font-semibold mb-1 text-xs text-gray-700 dark:text-gray-300">
                      ไฟล์ Weights (.pth) *
                    </label>
                    <input
                      type="file"
                      accept=".pth"
                      ref={weightInputRef}
                      onChange={(e) =>
                        setWeightFile(e.target.files?.[0] || null)
                      }
                      className="hidden"
                    />
                    <div
                      onDragOver={(e) => {
                        e.preventDefault();
                        setIsDraggingWeight(true);
                      }}
                      onDragLeave={() => setIsDraggingWeight(false)}
                      onDrop={(e) => {
                        e.preventDefault();
                        setIsDraggingWeight(false);
                        if (e.dataTransfer.files?.[0])
                          setWeightFile(e.dataTransfer.files[0]);
                      }}
                      onClick={() => weightInputRef.current?.click()}
                      className={`border-2 border-dashed rounded-lg p-2.5 text-center cursor-pointer transition text-xs ${
                        isDraggingWeight
                          ? "border-blue-500 bg-blue-100 dark:bg-blue-900/40"
                          : "border-blue-200 dark:border-blue-800 bg-white dark:bg-gray-800 hover:border-blue-400"
                      }`}
                    >
                      <p className="font-semibold text-blue-600 dark:text-blue-400 truncate">
                        {weightFile
                          ? weightFile.name
                          : "📂 คลิกหรือลากไฟล์ .pth มาวาง"}
                      </p>
                      <p className="text-[10px] text-gray-400 mt-0.5">
                        PyTorch model weights
                      </p>
                    </div>
                  </div>
                ) : (
                  <>
                    <div>
                      <label className="block font-semibold mb-1 text-xs text-gray-700 dark:text-gray-300">
                        Classifier Model (.pkl) *
                      </label>
                      <input
                        type="file"
                        accept=".pkl"
                        ref={clfInputRef}
                        onChange={(e) =>
                          setClfFile(e.target.files?.[0] || null)
                        }
                        className="hidden"
                      />
                      <div
                        onDragOver={(e) => {
                          e.preventDefault();
                          setIsDraggingClf(true);
                        }}
                        onDragLeave={() => setIsDraggingClf(false)}
                        onDrop={(e) => {
                          e.preventDefault();
                          setIsDraggingClf(false);
                          if (e.dataTransfer.files?.[0])
                            setClfFile(e.dataTransfer.files[0]);
                        }}
                        onClick={() => clfInputRef.current?.click()}
                        className={`border-2 border-dashed rounded-lg p-2.5 text-center cursor-pointer transition text-xs ${
                          isDraggingClf
                            ? "border-blue-500 bg-blue-100 dark:bg-blue-900/40"
                            : "border-blue-200 dark:border-blue-800 bg-white dark:bg-gray-800 hover:border-blue-400"
                        }`}
                      >
                        <p className="font-semibold text-blue-600 dark:text-blue-400 truncate">
                          {clfFile
                            ? clfFile.name
                            : "📂 คลิกหรือลากไฟล์ Model (.pkl)"}
                        </p>
                        <p className="text-[10px] text-gray-400 mt-0.5">
                          Classifier object
                        </p>
                      </div>
                    </div>

                    <div>
                      <label className="block font-semibold mb-1 text-xs text-gray-700 dark:text-gray-300">
                        RFE Selector (.pkl) *
                      </label>
                      <input
                        type="file"
                        accept=".pkl"
                        ref={rfeInputRef}
                        onChange={(e) =>
                          setRfeFile(e.target.files?.[0] || null)
                        }
                        className="hidden"
                      />
                      <div
                        onDragOver={(e) => {
                          e.preventDefault();
                          setIsDraggingRfe(true);
                        }}
                        onDragLeave={() => setIsDraggingRfe(false)}
                        onDrop={(e) => {
                          e.preventDefault();
                          setIsDraggingRfe(false);
                          if (e.dataTransfer.files?.[0])
                            setRfeFile(e.dataTransfer.files[0]);
                        }}
                        onClick={() => rfeInputRef.current?.click()}
                        className={`border-2 border-dashed rounded-lg p-2.5 text-center cursor-pointer transition text-xs ${
                          isDraggingRfe
                            ? "border-blue-500 bg-blue-100 dark:bg-blue-900/40"
                            : "border-blue-200 dark:border-blue-800 bg-white dark:bg-gray-800 hover:border-blue-400"
                        }`}
                      >
                        <p className="font-semibold text-blue-600 dark:text-blue-400 truncate">
                          {rfeFile
                            ? rfeFile.name
                            : "📂 คลิกหรือลากไฟล์ RFE (.pkl)"}
                        </p>
                        <p className="text-[10px] text-gray-400 mt-0.5">
                          RFE feature selector
                        </p>
                      </div>
                    </div>
                  </>
                )}

                <div className="pt-2 border-t border-blue-100 dark:border-blue-900/50 space-y-2 text-xs">
                  <label className="flex items-center space-x-2 font-semibold cursor-pointer">
                    <input
                      type="checkbox"
                      checked={useGroundTruth}
                      onChange={(e) => setUseGroundTruth(e.target.checked)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span>เปรียบเทียบผล (Ground Truth)</span>
                  </label>
                  {useGroundTruth && (
                    <div>
                      <input
                        type="file"
                        accept=".csv"
                        ref={csvInputRef}
                        onChange={(e) =>
                          setCsvFile(e.target.files?.[0] || null)
                        }
                        className="hidden"
                      />
                      <div
                        onDragOver={(e) => {
                          e.preventDefault();
                          setIsDraggingCsv(true);
                        }}
                        onDragLeave={() => setIsDraggingCsv(false)}
                        onDrop={(e) => {
                          e.preventDefault();
                          setIsDraggingCsv(false);
                          if (e.dataTransfer.files?.[0])
                            setCsvFile(e.dataTransfer.files[0]);
                        }}
                        onClick={() => csvInputRef.current?.click()}
                        className={`border-2 border-dashed rounded-lg p-2.5 text-center cursor-pointer transition mt-1 ${
                          isDraggingCsv
                            ? "border-blue-500 bg-blue-100 dark:bg-blue-900/40"
                            : "border-blue-200 dark:border-blue-800 bg-white dark:bg-gray-800 hover:border-blue-400"
                        }`}
                      >
                        <p className="font-semibold text-blue-600 dark:text-blue-400 truncate">
                          {csvFile
                            ? csvFile.name
                            : "📂 คลิกหรือลากไฟล์ CSV เฉลย"}
                        </p>
                        <p className="text-[10px] text-gray-400 mt-0.5">
                          Ground truth dataset
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Process Button */}
              <button
                onClick={() => processAllTasks()}
                disabled={tasks.length === 0}
                className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 dark:disabled:bg-gray-700 text-white font-semibold text-xs rounded-xl shadow transition-all duration-150"
              >
                🚀 ประมวลผล ({tasks.length})
              </button>
            </div>
          </div>

          {/* Main Area */}
          <div className="lg:col-span-3 space-y-4">
            {/* Drag & Drop Zone */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                if (e.dataTransfer.files) handleFiles(e.dataTransfer.files);
              }}
              className={`border-2 border-dashed rounded-2xl p-6 text-center transition-all duration-200 cursor-pointer bg-white dark:bg-gray-800 ${
                isDragging
                  ? "border-blue-500 bg-blue-50/50 dark:bg-blue-900/20 scale-[0.99]"
                  : "border-gray-300 dark:border-gray-700 hover:border-blue-400"
              }`}
              onClick={() => {
                const input = document.createElement("input");
                input.type = "file";
                input.multiple = true;
                input.accept = "image/*";
                input.onchange = (e: any) => {
                  if (e.target.files) handleFiles(e.target.files);
                };
                input.click();
              }}
            >
              <div className="text-3xl mb-2">📸</div>
              <p className="text-sm font-semibold">
                ลากรูปภาพมาวางที่นี่ หรือคลิกเพื่ออัปโหลด
              </p>
              <p className="text-xs text-gray-400 mt-1">
                รองรับ .png, .jpg, .jpeg
              </p>
            </div>

            {/* Confusion Matrix Dashboard */}
            {useGroundTruth && metrics.total > 0 && (
              <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700 shadow-sm animate-fade-in space-y-3">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-gray-100 dark:border-gray-700 pb-3">
                  <h3 className="font-bold text-sm text-gray-800 dark:text-gray-100">
                    📊 Confusion Matrix & Evaluation Metrics
                  </h3>
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800">
                      🧬 Backbone: {backbone}
                    </span>
                    <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-800">
                      ⚙️ Algo: {classifier}
                    </span>
                    <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800">
                      🔍 Filter: {filterType}
                    </span>
                    <button
                      onClick={() => processAllTasks(true)}
                      className="ml-auto px-2.5 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg text-xs font-semibold hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                    >
                      🔄 รันใหม่
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-1">
                  <div className="overflow-x-auto text-xs">
                    <table className="min-w-full border-collapse border border-gray-200 dark:border-gray-700 text-center">
                      <thead>
                        <tr className="bg-gray-50 dark:bg-gray-700/50">
                          <th className="border border-gray-200 dark:border-gray-700 p-2 text-gray-500">
                            n = {metrics.total}
                          </th>
                          <th className="border border-gray-200 dark:border-gray-700 p-2 font-semibold">
                            Predicted Pes Planus
                          </th>
                          <th className="border border-gray-200 dark:border-gray-700 p-2 font-semibold">
                            Predicted Normal
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td className="border border-gray-200 dark:border-gray-700 p-2 font-semibold bg-gray-50 dark:bg-gray-700/50">
                            Actual Pes Planus
                          </td>
                          <td className="border border-gray-200 dark:border-gray-700 p-2 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 font-bold">
                            TP: {metrics.tp}
                          </td>
                          <td className="border border-gray-200 dark:border-gray-700 p-2 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 font-bold">
                            FN: {metrics.fn}
                          </td>
                        </tr>
                        <tr>
                          <td className="border border-gray-200 dark:border-gray-700 p-2 font-semibold bg-gray-50 dark:bg-gray-700/50">
                            Actual Normal
                          </td>
                          <td className="border border-gray-200 dark:border-gray-700 p-2 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 font-bold">
                            FP: {metrics.fp}
                          </td>
                          <td className="border border-gray-200 dark:border-gray-700 p-2 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 font-bold">
                            TN: {metrics.tn}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-center">
                    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
                      <p className="text-[10px] uppercase font-bold text-gray-500">
                        Accuracy
                      </p>
                      <p className="text-xl font-bold text-blue-600">
                        {metrics.accuracy.toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
                      <p className="text-[10px] uppercase font-bold text-gray-500">
                        Precision
                      </p>
                      <p className="text-xl font-bold text-blue-600">
                        {metrics.precision.toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
                      <p className="text-[10px] uppercase font-bold text-gray-500">
                        Recall (Sensitivity)
                      </p>
                      <p className="text-xl font-bold text-blue-600">
                        {metrics.recall.toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
                      <p className="text-[10px] uppercase font-bold text-gray-500">
                        F1-Score
                      </p>
                      <p className="text-xl font-bold text-blue-600">
                        {metrics.f1.toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Task Controls Header */}
            {tasks.length > 0 && (
              <div className="flex items-center justify-between bg-white dark:bg-gray-800 p-3 rounded-xl border border-gray-200 dark:border-gray-700 text-xs">
                <span className="font-semibold text-gray-600 dark:text-gray-300">
                  รายการภาพทั้งหมด ({tasks.length} รายการ)
                </span>
                <div className="flex items-center space-x-3">
                  <div className="flex items-center space-x-1 border border-gray-200 dark:border-gray-700 rounded-lg p-0.5">
                    <button
                      onClick={() => setViewMode("full")}
                      className={`px-2 py-1 rounded ${viewMode === "full" ? "bg-blue-100 dark:bg-blue-900/40 text-blue-600 font-semibold" : "text-gray-500"}`}
                    >
                      แบบละเอียด
                    </button>
                    <button
                      onClick={() => setViewMode("compact")}
                      className={`px-2 py-1 rounded ${viewMode === "compact" ? "bg-blue-100 dark:bg-blue-900/40 text-blue-600 font-semibold" : "text-gray-500"}`}
                    >
                      แบบกะทัดรัด
                    </button>
                  </div>
                  <button
                    onClick={clearAllTasks}
                    className="text-red-500 hover:text-red-700 font-semibold"
                  >
                    ลบทั้งหมด
                  </button>
                </div>
              </div>
            )}

            {/* Image Task List */}
            <div
              className={`grid gap-3 ${viewMode === "compact" ? "grid-cols-2 sm:grid-cols-3" : "grid-cols-1"}`}
            >
              {tasks.map((task) => (
                <div
                  key={task.id}
                  className="bg-white dark:bg-gray-800 rounded-xl p-3 border border-gray-200 dark:border-gray-700 shadow-sm flex items-center space-x-3"
                >
                  <img
                    src={task.previewUrl}
                    alt="preview"
                    onClick={() => setFullscreenImage(task.previewUrl)}
                    className="w-16 h-16 object-cover rounded-lg border border-gray-200 dark:border-gray-700 cursor-pointer hover:opacity-90 flex-shrink-0"
                  />
                  <div className="flex-1 min-w-0">
                    <p className="font-semibold text-xs truncate">
                      {task.file.name}
                    </p>
                    <div className="mt-1">
                      {task.status === "idle" && (
                        <span className="text-[11px] text-gray-400">
                          ⏳ รอดำเนินการ
                        </span>
                      )}
                      {task.status === "processing" && (
                        <span className="text-[11px] text-blue-500 animate-pulse font-semibold">
                          ⚙️ กำลังประมวลผล...
                        </span>
                      )}
                      {task.status === "cancelled" && (
                        <span className="text-[11px] text-orange-500 font-semibold">
                          ยกเลิกแล้ว
                        </span>
                      )}
                      {task.status === "error" && (
                        <span className="text-[11px] text-red-500 font-semibold">
                          ❌ เกิดข้อผิดพลาด
                        </span>
                      )}
                      {task.status === "success" && task.result && (
                        <div className="space-y-0.5">
                          <p className="font-bold text-xs text-blue-600 dark:text-blue-400">
                            ผล: {task.result.prediction_class}
                          </p>
                          <p className="text-[11px] text-gray-500">
                            Confidence:{" "}
                            {(task.result.confidence * 100).toFixed(2)}%
                          </p>
                          {task.result.eval_status !== "ไม่มีเฉลย" && (
                            <p
                              className={`font-semibold text-[11px] ${task.result.eval_status.includes("True") ? "text-green-600" : "text-red-500"}`}
                            >
                              สถานะ: {task.result.eval_status}
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => removeTask(task.id)}
                    className="text-gray-400 hover:text-red-500 text-sm p-1"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>

      {/* Fullscreen Modal Preview */}
      {fullscreenImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
          onClick={() => setFullscreenImage(null)}
        >
          <img
            src={fullscreenImage}
            alt="full"
            className="max-w-full max-h-[90vh] rounded-xl shadow-2xl cursor-pointer"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}
