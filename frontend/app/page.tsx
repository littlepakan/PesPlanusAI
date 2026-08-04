"use client";
import { useState, useEffect } from "react";

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

  const [weightFile, setWeightFile] = useState<File | null>(null);
  const [clfFile, setClfFile] = useState<File | null>(null);
  const [rfeFile, setRfeFile] = useState<File | null>(null);
  const [csvFile, setCsvFile] = useState<File | null>(null);

  const [backbone, setBackbone] = useState("GoogleNet");
  const [classifier, setClassifier] = useState("XGBoost");
  const [filterType, setFilterType] = useState("Median Filter");
  const [useGroundTruth, setUseGroundTruth] = useState(false);

  const [lastRunConfig, setLastRunConfig] = useState<{
    backbone: string;
    classifier: string;
    fileName: string;
  } | null>(null);
  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"full" | "compact">("full");

  const [theme, setTheme] = useState<"light" | "dark" | "system">("system");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const savedTheme = localStorage.getItem("theme") as
      | "light"
      | "dark"
      | "system"
      | null;
    if (savedTheme) setTheme(savedTheme);
  }, []);

  useEffect(() => {
    if (!mounted) return;
    const root = document.documentElement;
    if (theme === "dark") {
      root.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else if (theme === "light") {
      root.classList.remove("dark");
      localStorage.setItem("theme", "light");
    } else {
      localStorage.removeItem("theme");
      if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
        root.classList.add("dark");
      } else {
        root.classList.remove("dark");
      }
    }
  }, [theme, mounted]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const files = Array.from(e.target.files);
    const newTasks: ImageTask[] = files.map((file) => ({
      id: `${file.name}-${Date.now()}-${Math.random()}`,
      file,
      previewUrl: URL.createObjectURL(file),
      status: "idle",
    }));
    setTasks((prev) => [...prev, ...newTasks]);
    e.target.value = "";
  };

  const handleDragOver = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const files = Array.from(e.dataTransfer.files).filter((file) =>
        file.type.startsWith("image/"),
      );
      const newTasks: ImageTask[] = files.map((file) => ({
        id: `${file.name}-${Date.now()}-${Math.random()}`,
        file,
        previewUrl: URL.createObjectURL(file),
        status: "idle",
      }));
      setTasks((prev) => [...prev, ...newTasks]);
    }
  };

  const removeTask = (id: string) => {
    setTasks((prev) => {
      const target = prev.find((t) => t.id === id);
      if (target?.controller) target.controller.abort();
      return prev.filter((t) => t.id !== id);
    });
  };

  const clearAllTasks = () => {
    tasks.forEach((t) => t.controller?.abort());
    setTasks([]);
  };

  const isFineTuning = classifier === "Fine-Tuning";
  const isNeuralNetwork = classifier === "NeuralNetwork";
  const isHybridML = !isFineTuning && !isNeuralNetwork;

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
      alert("กรุณาอัปโหลดไฟล์ Weights (.pth) สำหรับ Fine-Tuning");
      return;
    }
    if (isNeuralNetwork && (!weightFile || !rfeFile)) {
      alert(
        "กรุณาอัปโหลดไฟล์ Weights (.pth) และ RFE Selector (.pkl) ให้ครบถ้วน",
      );
      return;
    }
    if (isHybridML && (!clfFile || !rfeFile)) {
      alert(
        "กรุณาอัปโหลดไฟล์ Classifier (.pkl) และ RFE Selector (.pkl) ให้ครบถ้วน",
      );
      return;
    }

    if (useGroundTruth && !csvFile) {
      alert("กรุณาอัปโหลดไฟล์เฉลย CSV");
      return;
    }

    const currentFileName =
      isFineTuning || isNeuralNetwork
        ? weightFile?.name || ""
        : clfFile?.name || "";
    setLastRunConfig({ backbone, classifier, fileName: currentFileName });

    const controller = new AbortController();

    // ตั้งสถานะเป็น processing ทั้งหมดทันที
    setTasks((prev) =>
      prev.map((t) =>
        targetTasks.some((tt) => tt.id === t.id)
          ? { ...t, status: "processing", controller, result: undefined }
          : t,
      ),
    );

    const formData = new FormData();

    // 🌟 ยัดไฟล์ภาพทั้งหมดลงไปใน FormData ครั้งเดียว!
    targetTasks.forEach((task) => {
      formData.append("files", task.file);
    });

    formData.append("backbone", backbone);
    formData.append("classifier", classifier);
    formData.append("filter_type", filterType);
    formData.append("gt_option", useGroundTruth ? "upload" : "none");

    if ((isFineTuning || isNeuralNetwork) && weightFile)
      formData.append("weight_file", weightFile);
    if (isHybridML && clfFile) formData.append("clf_file", clfFile);
    if (!isFineTuning && rfeFile) formData.append("rfe_file", rfeFile);
    if (useGroundTruth && csvFile) formData.append("csv_file", csvFile);

    // 🌟 สลับ URL ตามการใช้งาน (เลือกเปิดอันนึง ปิดอันนึง)
    // const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"; // สำหรับรันเทสบนเครื่อง
    const API_URL =
      process.env.NEXT_PUBLIC_API_URL || "https://pesplanusai.onrender.com"; // สำหรับขึ้น Deploy จริง

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      // 🌟 แก้ไขบล็อกนี้เพื่อแปลง Object Error ให้เป็นข้อความที่อ่านได้
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        let errMsg = errData.detail || `HTTP error! status: ${response.status}`;

        // ถ้า Error ที่ส่งมาเป็น Object/Array ให้แปลงเป็น String
        if (typeof errMsg !== "string") {
          errMsg = JSON.stringify(errMsg);
        }
        throw new Error(errMsg);
      }

      // 🌟 รับผลลัพธ์มาเป็น Array แล้วจับคู่ด้วยชื่อไฟล์
      const dataArray = await response.json();

      setTasks((prev) =>
        prev.map((t) => {
          if (targetTasks.some((tt) => tt.id === t.id)) {
            const res = dataArray.find((r: any) => r.filename === t.file.name);
            if (res)
              return {
                ...t,
                status: "success",
                result: res,
                controller: undefined,
              };
            return {
              ...t,
              status: "error",
              result: { error: "ไม่ได้รับผลลัพธ์จากเซิร์ฟเวอร์" },
              controller: undefined,
            };
          }
          return t;
        }),
      );
    } catch (error: any) {
      if (error.name === "AbortError") {
        setTasks((prev) =>
          prev.map((t) =>
            targetTasks.some((tt) => tt.id === t.id)
              ? { ...t, status: "cancelled", controller: undefined }
              : t,
          ),
        );
      } else {
        setTasks((prev) =>
          prev.map((t) =>
            targetTasks.some((tt) => tt.id === t.id)
              ? {
                  ...t,
                  status: "error",
                  result: { error: error.message },
                  controller: undefined,
                }
              : t,
          ),
        );
      }
    }
  };

  const completedTasks = tasks.filter(
    (t) => t.status === "success" && t.result?.eval_status,
  );
  const tp = completedTasks.filter(
    (t) => t.result.eval_status === "True Positive (TP)",
  ).length;
  const tn = completedTasks.filter(
    (t) => t.result.eval_status === "True Negative (TN)",
  ).length;
  const fp = completedTasks.filter(
    (t) => t.result.eval_status === "False Positive (FP)",
  ).length;
  const fn = completedTasks.filter(
    (t) => t.result.eval_status === "False Negative (FN)",
  ).length;
  const totalEval = tp + tn + fp + fn;

  const accuracy = totalEval > 0 ? ((tp + tn) / totalEval) * 100 : 0;
  const sensitivity = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
  const specificity = tn + fp > 0 ? (tn / (tn + fp)) * 100 : 0;

  const idleCount = tasks.filter(
    (t) =>
      t.status === "idle" || t.status === "error" || t.status === "cancelled",
  ).length;
  const isProcessingAny = tasks.some((t) => t.status === "processing");

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-100 transition-colors duration-200">
      <header className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950 sticky top-0 z-30 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="text-2xl">🦶</span>
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              Pes Planus AI Analyzer
            </h1>
          </div>

          <div className="flex items-center space-x-2 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg">
            <button
              onClick={() => setTheme("light")}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                theme === "light"
                  ? "bg-white dark:bg-gray-700 shadow text-blue-600 dark:text-blue-400"
                  : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              }`}
            >
              ☀️ Light
            </button>
            <button
              onClick={() => setTheme("dark")}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                theme === "dark"
                  ? "bg-white dark:bg-gray-700 shadow text-blue-600 dark:text-blue-400"
                  : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              }`}
            >
              🌙 Dark
            </button>
            <button
              onClick={() => setTheme("system")}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                theme === "system"
                  ? "bg-white dark:bg-gray-700 shadow text-blue-600 dark:text-blue-400"
                  : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              }`}
            >
              💻 System
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 space-y-5">
            <h2 className="text-lg font-semibold flex items-center gap-2 border-b border-gray-100 dark:border-gray-700 pb-3">
              ⚙️ การตั้งค่าระบบ
            </h2>

            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Backbone Architecture
              </label>
              <select
                value={backbone}
                onChange={(e) => setBackbone(e.target.value)}
                className="w-full bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg p-2.5 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
              >
                <option value="GoogleNet">GoogleNet (224x224)</option>
                <option value="SqueezeNet">SqueezeNet (227x227)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Classifier Algorithm / Approach
              </label>
              <select
                value={classifier}
                onChange={(e) => setClassifier(e.target.value)}
                className="w-full bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg p-2.5 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
              >
                <option value="Fine-Tuning">Fine-Tuning (End-to-End)</option>
                <option value="XGBoost">XGBoost (Hybrid)</option>
                <option value="SVM">SVM (Hybrid)</option>
                <option value="RandomForest">Random Forest (Hybrid)</option>
                <option value="DecisionTree">Decision Tree (Hybrid)</option>
                <option value="NeuralNetwork">Neural Network (Hybrid)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Image Preprocessing Filter
              </label>
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="w-full bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg p-2.5 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
              >
                <option value="Median Filter">Median Filter (3x3)</option>
                <option value="Gaussian Blur">Gaussian Blur (3x3)</option>
                <option value="None">None (Original)</option>
              </select>
            </div>

            <div className="pt-2 border-t border-gray-100 dark:border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
                  เปรียบเทียบกับเฉลย (Ground Truth)
                </span>
                <button
                  type="button"
                  onClick={() => setUseGroundTruth(!useGroundTruth)}
                  className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                    useGroundTruth
                      ? "bg-blue-600"
                      : "bg-gray-300 dark:bg-gray-700"
                  }`}
                >
                  <span
                    className={`inline-block h-5 w-5 transform rounded-full bg-white shadow-lg ring-0 transition duration-200 ease-in-out ${
                      useGroundTruth ? "translate-x-5" : "translate-x-0"
                    }`}
                  />
                </button>
              </div>

              {useGroundTruth && (
                <div className="mt-3 p-3 bg-gray-50 dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 space-y-1.5 animate-fadeIn">
                  <label className="block text-[11px] font-medium text-gray-500 dark:text-gray-400">
                    อัปโหลดไฟล์ CSV เฉลย *
                  </label>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => setCsvFile(e.target.files?.[0] || null)}
                    className="w-full text-xs text-gray-500 file:mr-2 file:py-1 file:px-2.5 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-green-50 file:text-green-700 dark:file:bg-green-900/30 dark:file:text-green-400 hover:file:bg-green-100 cursor-pointer"
                  />
                </div>
              )}
            </div>

            <div className="space-y-3 pt-2 border-t border-gray-100 dark:border-gray-700">
              {(isFineTuning || isNeuralNetwork) && (
                <div>
                  <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                    ไฟล์ Weights (.pth) *
                  </label>
                  <input
                    type="file"
                    accept=".pth"
                    onChange={(e) => setWeightFile(e.target.files?.[0] || null)}
                    className="w-full text-xs text-gray-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 dark:file:bg-blue-900/30 dark:file:text-blue-400 hover:file:bg-blue-100 cursor-pointer"
                  />
                </div>
              )}

              {isHybridML && (
                <div>
                  <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                    ไฟล์ Classifier (.pkl) *
                  </label>
                  <input
                    type="file"
                    accept=".pkl"
                    onChange={(e) => setClfFile(e.target.files?.[0] || null)}
                    className="w-full text-xs text-gray-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 dark:file:bg-blue-900/30 dark:file:text-blue-400 hover:file:bg-blue-100 cursor-pointer"
                  />
                </div>
              )}

              {!isFineTuning && (
                <div>
                  <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                    ไฟล์ RFE Selector (.pkl) *
                  </label>
                  <input
                    type="file"
                    accept=".pkl"
                    onChange={(e) => setRfeFile(e.target.files?.[0] || null)}
                    className="w-full text-xs text-gray-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 dark:file:bg-blue-900/30 dark:file:text-blue-400 hover:file:bg-blue-100 cursor-pointer"
                  />
                </div>
              )}
            </div>
          </div>

          <div className="space-y-4">
            <label
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-2xl cursor-pointer transition-all ${
                isDragging
                  ? "border-blue-500 bg-blue-50 dark:border-blue-500 dark:bg-blue-900/20 scale-[1.02]"
                  : "border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-750"
              }`}
            >
              <div className="flex flex-col items-center justify-center pt-5 pb-6 pointer-events-none">
                <span
                  className={`text-3xl mb-1 transition-transform ${isDragging ? "animate-bounce" : ""}`}
                >
                  📸
                </span>
                <p
                  className={`text-xs font-medium ${isDragging ? "text-blue-600 dark:text-blue-400" : "text-gray-600 dark:text-gray-300"}`}
                >
                  {isDragging
                    ? "ปล่อยเมาส์เพื่อวางรูปเลย!"
                    : "คลิก หรือ ลากรูป X-Ray มาวางที่นี่"}
                </p>
                <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-1">
                  รองรับ JPG, PNG, BMP (เลือกได้หลายรูปพร้อมกัน)
                </p>
              </div>
              <input
                type="file"
                multiple
                accept="image/*"
                className="hidden"
                onChange={handleImageUpload}
              />
            </label>

            <button
              onClick={() => processAllTasks(idleCount === 0)}
              disabled={tasks.length === 0 || isProcessingAny}
              className={`w-full py-3.5 font-bold text-[15px] rounded-xl shadow-md transition-all flex items-center justify-center gap-2 text-white
                ${
                  tasks.length === 0
                    ? "bg-gray-300 dark:bg-gray-700 cursor-not-allowed text-gray-500 shadow-none"
                    : isProcessingAny
                      ? "bg-blue-400 dark:bg-blue-600 cursor-wait animate-pulse"
                      : idleCount > 0
                        ? "bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 hover:scale-[1.02]"
                        : "bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 hover:scale-[1.02]"
                }
              `}
            >
              {isProcessingAny ? (
                <>
                  <span className="animate-spin">⏳</span> กำลังประมวลผลภาพ...
                </>
              ) : idleCount > 0 ? (
                <>
                  <span>🚀</span> เริ่มการวิเคราะห์ ({idleCount} รูป)
                </>
              ) : (
                <>
                  <span>🔄</span> ประเมินใหม่ทั้งหมด ({tasks.length} รูป)
                </>
              )}
            </button>
          </div>
        </div>

        <div className="lg:col-span-8 space-y-6">
          {totalEval > 0 && (
            <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 space-y-4">
              <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-3 gap-3">
                <h3 className="text-md font-semibold flex items-center gap-2 whitespace-nowrap">
                  📊 Confusion Matrix & Metrics
                </h3>

                <div className="flex flex-wrap items-center justify-end gap-2 text-xs">
                  <span
                    className="bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 px-2.5 py-1 rounded-lg border border-amber-200 dark:border-amber-800 font-medium flex items-center gap-1 max-w-[150px] sm:max-w-[200px]"
                    title={lastRunConfig?.fileName}
                  >
                    📁{" "}
                    <span className="truncate">
                      {lastRunConfig?.fileName || "ยังไม่มีไฟล์"}
                    </span>
                  </span>
                  <span className="bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 px-2.5 py-1 rounded-lg border border-blue-100 dark:border-blue-800 font-medium flex items-center gap-1">
                    🧠 {lastRunConfig?.backbone || backbone}
                  </span>
                  <span className="bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 px-2.5 py-1 rounded-lg border border-indigo-100 dark:border-indigo-800 font-medium flex items-center gap-1">
                    ⚙️ {lastRunConfig?.classifier || classifier}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-xl border border-blue-100 dark:border-blue-800/30 flex flex-col justify-center">
                  <p className="text-[10px] text-blue-600 dark:text-blue-400 font-medium">
                    Accuracy
                  </p>
                  <p className="text-xl font-bold text-blue-700 dark:text-blue-300">
                    {accuracy.toFixed(1)}%
                  </p>
                </div>
                <div className="bg-emerald-50 dark:bg-emerald-900/20 p-3 rounded-xl border border-emerald-100 dark:border-emerald-800/30 flex flex-col justify-center">
                  <p className="text-[10px] text-emerald-600 dark:text-emerald-400 font-medium">
                    Sensitivity (Recall)
                  </p>
                  <p className="text-xl font-bold text-emerald-700 dark:text-emerald-300">
                    {sensitivity.toFixed(1)}%
                  </p>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-xl border border-purple-100 dark:border-purple-800/30 flex flex-col justify-center">
                  <p className="text-[10px] text-purple-600 dark:text-purple-400 font-medium">
                    Specificity
                  </p>
                  <p className="text-xl font-bold text-purple-700 dark:text-purple-300">
                    {specificity.toFixed(1)}%
                  </p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-900 p-3 rounded-xl border border-gray-200 dark:border-gray-700 flex flex-col justify-center">
                  <p className="text-[10px] text-gray-500 font-medium">
                    ประเมินแล้วทั้งหมด
                  </p>
                  <p className="text-xl font-bold text-gray-700 dark:text-gray-300">
                    {totalEval} รูป
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-2 text-center text-xs">
                <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded-lg border border-green-200 dark:border-green-800">
                  <p className="text-green-800 dark:text-green-300 font-bold">
                    True Positive (TP)
                  </p>
                  <p className="text-2xl font-black text-green-600 dark:text-green-400 mt-1">
                    {tp}
                  </p>
                  <p className="text-[10px] text-green-600 dark:text-green-500">
                    ทาย Pes Planus ถูก
                  </p>
                </div>
                <div className="bg-red-100 dark:bg-red-900/30 p-3 rounded-lg border border-red-200 dark:border-red-800">
                  <p className="text-red-800 dark:text-red-300 font-bold">
                    False Positive (FP)
                  </p>
                  <p className="text-2xl font-black text-red-600 dark:text-red-400 mt-1">
                    {fp}
                  </p>
                  <p className="text-[10px] text-red-600 dark:text-red-500">
                    ปกติ แต่นึกว่า Pes Planus
                  </p>
                </div>
                <div className="bg-orange-100 dark:bg-orange-900/30 p-3 rounded-lg border border-orange-200 dark:border-orange-800">
                  <p className="text-orange-800 dark:text-orange-300 font-bold">
                    False Negative (FN)
                  </p>
                  <p className="text-2xl font-black text-orange-600 dark:text-orange-400 mt-1">
                    {fn}
                  </p>
                  <p className="text-[10px] text-orange-600 dark:text-orange-500">
                    Pes Planus แต่นึกว่าปกติ
                  </p>
                </div>
                <div className="bg-blue-100 dark:bg-blue-900/30 p-3 rounded-lg border border-blue-200 dark:border-blue-800">
                  <p className="text-blue-800 dark:text-blue-300 font-bold">
                    True Negative (TN)
                  </p>
                  <p className="text-2xl font-black text-blue-600 dark:text-blue-400 mt-1">
                    {tn}
                  </p>
                  <p className="text-[10px] text-blue-600 dark:text-blue-500">
                    ทาย Normal ถูก
                  </p>
                </div>
              </div>
            </div>
          )}

          <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 space-y-4">
            <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-3">
              <h3 className="text-md font-semibold flex items-center gap-2">
                🖼️ รายการรูปภาพ ({tasks.length})
              </h3>

              <div className="flex items-center gap-2">
                <button
                  onClick={() =>
                    setViewMode(viewMode === "full" ? "compact" : "full")
                  }
                  className="px-2.5 py-1 text-xs bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  {viewMode === "full" ? "📑 มุมมองย่อ" : "🖼️ มุมมองขยาย"}
                </button>
                {tasks.length > 0 && (
                  <button
                    onClick={clearAllTasks}
                    disabled={isProcessingAny}
                    className="px-2.5 py-1 text-xs bg-red-50 text-red-600 hover:bg-red-100 rounded-lg transition-colors disabled:opacity-50"
                  >
                    🗑️ ลบทั้งหมด
                  </button>
                )}
              </div>
            </div>

            {tasks.length === 0 ? (
              <div className="text-center py-12 text-gray-400 dark:text-gray-500">
                <span className="text-4xl block mb-2">📂</span>
                <p className="text-xs">ยังไม่มีรูปภาพในรายการ</p>
              </div>
            ) : (
              <div
                className={
                  viewMode === "compact"
                    ? "grid grid-cols-1 sm:grid-cols-2 gap-3"
                    : "space-y-4"
                }
              >
                {tasks.map((task) => (
                  <div
                    key={task.id}
                    className="p-3 bg-gray-50 dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 flex gap-3 items-center"
                  >
                    <img
                      src={task.previewUrl}
                      alt="preview"
                      className="w-16 h-16 object-cover rounded-lg cursor-pointer hover:opacity-80 transition-opacity"
                      onClick={() => setFullscreenImage(task.previewUrl)}
                    />

                    <div className="flex-1 min-w-0 text-xs space-y-1">
                      <p className="font-semibold text-gray-800 dark:text-gray-200 truncate">
                        {task.file.name}
                      </p>

                      {task.status === "error" && (
                        <p className="text-red-500">
                          ❌ เกิดข้อผิดพลาด: {task.result?.error}
                        </p>
                      )}

                      {task.status === "success" && task.result && (
                        <div className="space-y-0.5">
                          <p className="font-bold text-blue-600 dark:text-blue-400">
                            ผลทำนาย: {task.result.prediction_class} (ความมั่นใจ{" "}
                            {(task.result.confidence * 100).toFixed(2)}%)
                          </p>

                          {task.result.eval_status !== "ไม่มีเฉลย" && (
                            <p
                              className={`font-semibold text-[11px] ${
                                task.result.eval_status.includes("True")
                                  ? "text-green-600"
                                  : "text-red-500"
                              }`}
                            >
                              สถานะ: {task.result.eval_status} (เฉลย:{" "}
                              {task.result.ground_truth})
                            </p>
                          )}
                        </div>
                      )}
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
            )}
          </div>
        </div>
      </main>

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
