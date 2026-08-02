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

  const [weightFile, setWeightFile] = useState<File | null>(null);
  const [clfFile, setClfFile] = useState<File | null>(null);
  const [rfeFile, setRfeFile] = useState<File | null>(null);
  const [csvFile, setCsvFile] = useState<File | null>(null);

  const [backbone, setBackbone] = useState("SqueezeNet");
  const [classifier, setClassifier] = useState("Fine-Tuning");
  const [filterType, setFilterType] = useState("Median Filter");
  const [gtOption, setGtOption] = useState<"none" | "default" | "upload">(
    "none",
  );

  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"full" | "compact">("full");

  // ระบบ Theme (Light / Dark / System)
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
    const root = window.document.documentElement;
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

    const applyTheme = () => {
      root.classList.remove("light", "dark");
      if (theme === "system") {
        root.classList.add(mediaQuery.matches ? "dark" : "light");
      } else {
        root.classList.add(theme);
      }
    };

    applyTheme();
    localStorage.setItem("theme", theme);

    if (theme === "system") {
      const listener = () => applyTheme();
      mediaQuery.addEventListener("change", listener);
      return () => mediaQuery.removeEventListener("change", listener);
    }
  }, [theme, mounted]);

  const isFineTuning = classifier === "Fine-Tuning";
  const isProcessing = tasks.some((t) => t.status === "processing");

  const evalTasks = tasks.filter(
    (t) =>
      t.status === "success" &&
      t.result &&
      t.result.eval_status !== "ไม่มีเฉลย",
  );
  const showMatrix = evalTasks.length > 0;

  let tp = 0,
    tn = 0,
    fp = 0,
    fn = 0;
  evalTasks.forEach((t) => {
    if (t.result.eval_status.includes("TP")) tp++;
    if (t.result.eval_status.includes("TN")) tn++;
    if (t.result.eval_status.includes("FP")) fp++;
    if (t.result.eval_status.includes("FN")) fn++;
  });

  const totalEval = tp + tn + fp + fn;
  const accuracy =
    totalEval > 0 ? (((tp + tn) / totalEval) * 100).toFixed(2) : "0.00";
  const sensitivity =
    tp + fn > 0 ? ((tp / (tp + fn)) * 100).toFixed(2) : "0.00";
  const specificity =
    tn + fp > 0 ? ((tn / (tn + fp)) * 100).toFixed(2) : "0.00";

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const newFiles = Array.from(e.target.files);
    const newTasks = newFiles.map((file) => ({
      id: crypto.randomUUID(),
      file: file,
      previewUrl: URL.createObjectURL(file),
      status: "idle" as const,
    }));
    setTasks((prev) => [...prev, ...newTasks]);
    e.target.value = "";
  };

  const removeTask = (task: ImageTask) => {
    URL.revokeObjectURL(task.previewUrl);
    setTasks((prev) => prev.filter((t) => t.id !== task.id));
  };

  const cancelTask = (taskId: string) => {
    setTasks((prev) =>
      prev.map((t) => {
        if (t.id === taskId && t.status === "processing") {
          t.controller?.abort();
          return { ...t, status: "cancelled" };
        }
        return t;
      }),
    );
  };

  const clearAllTasks = () => {
    tasks.forEach((t) => URL.revokeObjectURL(t.previewUrl));
    setTasks([]);
  };

  const startAnalysis = () => {
    if (tasks.filter((t) => t.status === "idle").length === 0)
      return alert("ไม่มีรูปภาพใหม่ให้ประมวลผลครับ");
    if (classifier === "Fine-Tuning" && !weightFile)
      return alert("กรุณาอัปโหลดไฟล์ Weights (.pth) ก่อนครับ");
    if (classifier !== "Fine-Tuning" && (!clfFile || !rfeFile))
      return alert("กรุณาอัปโหลดไฟล์ Classifier และ RFE ให้ครบก่อนครับ");
    if (gtOption === "upload" && !csvFile)
      return alert("กรุณาเลือกไฟล์เฉลย CSV ที่ต้องการอัปโหลดก่อนครับ");

    const tasksToProcess = tasks.filter((t) => t.status === "idle");

    tasksToProcess.forEach((task) => {
      const abortController = new AbortController();
      setTasks((prev) =>
        prev.map((t) =>
          t.id === task.id
            ? { ...t, status: "processing", controller: abortController }
            : t,
        ),
      );

      const formData = new FormData();
      formData.append("file", task.file);
      formData.append("backbone", backbone);
      formData.append("classifier", classifier);
      formData.append("filter_type", filterType);
      formData.append("gt_option", gtOption);

      if (classifier === "Fine-Tuning" && weightFile)
        formData.append("weight_file", weightFile);
      else if (clfFile && rfeFile) {
        formData.append("clf_file", clfFile);
        formData.append("rfe_file", rfeFile);
      }
      if (gtOption === "upload" && csvFile)
        formData.append("csv_file", csvFile);
      // fetch("https://pesplanus.onrender.com/api/predict", {
      fetch("http://localhost:8000/api/predict", {
        method: "POST",
        body: formData,
        signal: abortController.signal,
      })
        .then((res) => {
          if (!res.ok) throw new Error("API Error");
          return res.json();
        })
        .then((data) => {
          setTasks((prev) =>
            prev.map((t) =>
              t.id === task.id ? { ...t, status: "success", result: data } : t,
            ),
          );
        })
        .catch((err) => {
          if (err.name !== "AbortError") {
            setTasks((prev) =>
              prev.map((t) =>
                t.id === task.id ? { ...t, status: "error" } : t,
              ),
            );
          }
        });
    });
  };

  if (!mounted) return null;

  return (
    <>
      <main className="min-h-screen bg-gray-100 dark:bg-gray-900 p-4 md:p-8 font-sans pb-24 transition-colors duration-300">
        <div className="max-w-6xl mx-auto space-y-6">
          <div className="bg-white dark:bg-gray-800 p-6 md:p-8 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 transition-colors duration-300">
            {/* Header & Theme Switcher */}
            <div className="flex flex-col md:flex-row justify-between items-center mb-6 gap-4">
              <div className="flex-1"></div>
              <h1 className="text-3xl font-extrabold text-blue-900 dark:text-blue-400 text-center flex-1">
                🦶 Pes Planus AI
              </h1>

              <div className="flex-1 flex justify-end">
                <div className="flex bg-gray-100 dark:bg-gray-900 p-1 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                  <button
                    onClick={() => setTheme("light")}
                    className={`px-3 py-1.5 text-xs font-bold rounded-md transition-all ${theme === "light" ? "bg-white dark:bg-gray-700 shadow text-yellow-500" : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"}`}
                  >
                    ☀️ สว่าง
                  </button>
                  <button
                    onClick={() => setTheme("dark")}
                    className={`px-3 py-1.5 text-xs font-bold rounded-md transition-all ${theme === "dark" ? "bg-white dark:bg-gray-700 shadow text-blue-400" : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"}`}
                  >
                    🌙 มืด
                  </button>
                  <button
                    onClick={() => setTheme("system")}
                    className={`px-3 py-1.5 text-xs font-bold rounded-md transition-all ${theme === "system" ? "bg-white dark:bg-gray-700 shadow text-gray-800 dark:text-gray-200" : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"}`}
                  >
                    💻 ระบบ
                  </button>
                </div>
              </div>
            </div>

            <p className="text-gray-600 dark:text-gray-400 text-center mb-8 font-medium">
              Real-time Batch Processing & Evaluation
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8 border-b border-gray-200 dark:border-gray-700 pb-8">
              <div className="space-y-5">
                <h3 className="font-bold text-lg text-blue-900 dark:text-blue-400 border-b border-gray-200 dark:border-gray-700 pb-2">
                  1. ตั้งค่าภาพและการประมวลผล
                </h3>

                <div>
                  <label className="block text-sm font-bold text-gray-900 dark:text-gray-200 mb-1">
                    Backbone Model
                  </label>
                  <select
                    className="w-full border border-gray-300 dark:border-gray-600 p-2.5 rounded-md focus:ring-2 focus:ring-blue-500 font-medium text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-700"
                    value={backbone}
                    onChange={(e) => setBackbone(e.target.value)}
                  >
                    <option value="SqueezeNet">SqueezeNet</option>
                    <option value="GoogleNet">GoogleNet</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-bold text-gray-900 dark:text-gray-200 mb-1">
                    Classifier Mode
                  </label>
                  <select
                    className="w-full border border-gray-300 dark:border-gray-600 p-2.5 rounded-md focus:ring-2 focus:ring-blue-500 font-medium text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-700"
                    value={classifier}
                    onChange={(e) => setClassifier(e.target.value)}
                  >
                    <option value="Fine-Tuning">
                      Fine-Tuning (End-to-End)
                    </option>
                    <option value="SVM">SVM</option>
                    <option value="RandomForest">Random Forest</option>
                    <option value="XGBoost">XGBoost</option>
                    <option value="NeuralNetwork">Neural Network (MLP)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-bold text-gray-900 dark:text-gray-200 mb-1">
                    Preprocessing Filter
                  </label>
                  <select
                    className="w-full border border-gray-300 dark:border-gray-600 p-2.5 rounded-md focus:ring-2 focus:ring-blue-500 font-medium text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-700"
                    value={filterType}
                    onChange={(e) => setFilterType(e.target.value)}
                  >
                    <option value="Median Filter">
                      Median Filter (ลดสัญญาณรบกวน)
                    </option>
                    <option value="Gaussian Blur">
                      Gaussian Blur (เบลอเกาส์เซียน)
                    </option>
                    <option value="None">None (ไม่ใช้ตัวกรอง)</option>
                  </select>
                </div>

                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-100 dark:border-blue-900/50">
                  <label className="block text-sm font-bold text-gray-900 dark:text-gray-200 mb-2">
                    อัปโหลดภาพเอกซเรย์เท้า{" "}
                    <span className="text-blue-600 dark:text-blue-400 font-normal">
                      (เลือกได้หลายรูป)
                    </span>
                  </label>
                  <input
                    type="file"
                    accept="image/*"
                    multiple
                    onChange={handleImageSelect}
                    className="w-full text-sm text-gray-700 dark:text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 cursor-pointer"
                  />
                </div>
              </div>

              <div className="space-y-5 bg-gray-50 dark:bg-gray-800/80 p-5 rounded-lg border border-gray-200 dark:border-gray-700">
                <h3 className="font-bold text-lg text-blue-900 dark:text-blue-400 border-b border-gray-200 dark:border-gray-700 pb-2">
                  2. อัปโหลดโมเดลและจัดการเฉลย
                </h3>
                {isFineTuning ? (
                  <div>
                    <label className="block text-sm font-bold text-gray-900 dark:text-gray-200 mb-1">
                      ไฟล์ Weights (.pth) สำหรับ {backbone}
                    </label>
                    <input
                      type="file"
                      accept=".pth"
                      onChange={(e) =>
                        setWeightFile(e.target.files?.[0] || null)
                      }
                      className="w-full text-sm text-gray-700 dark:text-gray-300 file:mr-4 file:py-1.5 file:px-4 file:rounded-md file:border-0 file:bg-indigo-50 dark:file:bg-indigo-900/50 file:text-indigo-700 dark:file:text-indigo-300 hover:file:bg-indigo-100 cursor-pointer"
                    />
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-bold text-gray-900 dark:text-gray-200 mb-1">
                        ไฟล์ Classifier (.pkl) - {classifier}
                      </label>
                      <input
                        type="file"
                        accept=".pkl"
                        onChange={(e) =>
                          setClfFile(e.target.files?.[0] || null)
                        }
                        className="w-full text-sm text-gray-700 dark:text-gray-300 file:mr-4 file:py-1.5 file:px-4 file:rounded-md file:border-0 file:bg-green-50 dark:file:bg-green-900/50 file:text-green-700 dark:file:text-green-300 hover:file:bg-green-100 cursor-pointer"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-bold text-gray-900 dark:text-gray-200 mb-1">
                        ไฟล์ RFE Selector (.pkl)
                      </label>
                      <input
                        type="file"
                        accept=".pkl"
                        onChange={(e) =>
                          setRfeFile(e.target.files?.[0] || null)
                        }
                        className="w-full text-sm text-gray-700 dark:text-gray-300 file:mr-4 file:py-1.5 file:px-4 file:rounded-md file:border-0 file:bg-green-50 dark:file:bg-green-900/50 file:text-green-700 dark:file:text-green-300 hover:file:bg-green-100 cursor-pointer"
                      />
                    </div>
                  </div>
                )}

                <div className="pt-3 border-t border-gray-200 dark:border-gray-700 space-y-3">
                  <label className="block text-sm font-bold text-gray-900 dark:text-gray-200">
                    การตรวจคำตอบ (Ground Truth)
                  </label>

                  <div className="space-y-2 text-sm">
                    <label className="flex items-center gap-2 cursor-pointer font-medium text-gray-800 dark:text-gray-300">
                      <input
                        type="radio"
                        name="gt_mode"
                        value="none"
                        checked={gtOption === "none"}
                        onChange={() => setGtOption("none")}
                        className="w-4 h-4 text-blue-600"
                      />
                      <span>1. ไม่ใช้เฉลย (วิเคราะห์อย่างเดียว)</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer font-medium text-gray-800 dark:text-gray-300">
                      <input
                        type="radio"
                        name="gt_mode"
                        value="default"
                        checked={gtOption === "default"}
                        onChange={() => setGtOption("default")}
                        className="w-4 h-4 text-blue-600"
                      />
                      <span>
                        2. ใช้เฉลย Default (อ่านไฟล์ CSV ในโฟลเดอร์ backend)
                      </span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer font-medium text-gray-800 dark:text-gray-300">
                      <input
                        type="radio"
                        name="gt_mode"
                        value="upload"
                        checked={gtOption === "upload"}
                        onChange={() => setGtOption("upload")}
                        className="w-4 h-4 text-blue-600"
                      />
                      <span>3. อัปโหลดไฟล์เฉลย CSV เอง</span>
                    </label>
                  </div>

                  {gtOption === "upload" && (
                    <div className="pt-2 animate-fade-in">
                      <input
                        type="file"
                        accept=".csv"
                        onChange={(e) =>
                          setCsvFile(e.target.files?.[0] || null)
                        }
                        className="w-full text-sm text-gray-700 dark:text-gray-300 file:mr-4 file:py-1.5 file:px-4 file:rounded-md file:border-0 file:bg-amber-50 dark:file:bg-amber-900/50 file:text-amber-800 dark:file:text-amber-300 hover:file:bg-amber-100 cursor-pointer"
                      />
                    </div>
                  )}
                </div>
              </div>
            </div>

            <button
              onClick={startAnalysis}
              disabled={isProcessing || tasks.length === 0}
              className={`w-full py-4 text-white font-bold text-lg rounded-lg shadow-md transition-all ${
                isProcessing || tasks.length === 0
                  ? "bg-gray-400 dark:bg-gray-600 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-700 active:scale-[0.99]"
              }`}
            >
              {isProcessing
                ? "⏳ กำลังประมวลผลภาพในคิว..."
                : "🚀 เริ่มการวิเคราะห์"}
            </button>
          </div>

          {/* Confusion Matrix Dashboard */}
          {showMatrix && (
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 animate-fade-in transition-colors duration-300">
              <h2 className="text-2xl font-bold text-blue-900 dark:text-blue-400 mb-6 flex items-center gap-2">
                <span>📈</span> สรุปผลการประเมิน (Confusion Matrix)
              </h2>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 overflow-x-auto">
                  <table className="w-full text-sm text-center border-collapse border border-gray-300 dark:border-gray-600">
                    <thead>
                      <tr>
                        <th className="border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 p-3 w-1/3">
                          Actual \ Predicted
                        </th>
                        <th className="border border-gray-300 dark:border-gray-600 bg-blue-50 dark:bg-blue-900/30 p-3 w-1/3 font-bold text-blue-900 dark:text-blue-300">
                          ทายว่า "เป็นเท้าแบน" (1)
                        </th>
                        <th className="border border-gray-300 dark:border-gray-600 bg-green-50 dark:bg-green-900/30 p-3 w-1/3 font-bold text-green-900 dark:text-green-300">
                          ทายว่า "ปกติ" (0)
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700/50 p-3 font-bold text-gray-800 dark:text-gray-200">
                          เป็นเท้าแบน (1)
                        </td>
                        <td className="border border-gray-300 dark:border-gray-600 p-4 bg-green-50 dark:bg-green-900/20">
                          <div className="text-3xl font-extrabold text-green-600 dark:text-green-400">
                            {tp}
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400 font-bold mt-1">
                            True Positive (TP)
                          </div>
                        </td>
                        <td className="border border-gray-300 dark:border-gray-600 p-4 bg-red-50 dark:bg-red-900/20">
                          <div className="text-3xl font-extrabold text-red-600 dark:text-red-400">
                            {fn}
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400 font-bold mt-1">
                            False Negative (FN)
                          </div>
                        </td>
                      </tr>
                      <tr>
                        <td className="border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700/50 p-3 font-bold text-gray-800 dark:text-gray-200">
                          ปกติ (0)
                        </td>
                        <td className="border border-gray-300 dark:border-gray-600 p-4 bg-red-50 dark:bg-red-900/20">
                          <div className="text-3xl font-extrabold text-red-600 dark:text-red-400">
                            {fp}
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400 font-bold mt-1">
                            False Positive (FP)
                          </div>
                        </td>
                        <td className="border border-gray-300 dark:border-gray-600 p-4 bg-green-50 dark:bg-green-900/20">
                          <div className="text-3xl font-extrabold text-green-600 dark:text-green-400">
                            {tn}
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400 font-bold mt-1">
                            True Negative (TN)
                          </div>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div className="flex flex-col justify-center gap-4">
                  <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 text-center shadow-sm">
                    <p className="text-blue-900 dark:text-blue-300 font-bold text-sm">
                      Accuracy (ความแม่นยำรวม)
                    </p>
                    <p className="text-3xl font-black text-blue-700 dark:text-blue-400 mt-1">
                      {accuracy}%
                    </p>
                  </div>
                  <div className="bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 rounded-lg p-4 text-center shadow-sm">
                    <p className="text-indigo-900 dark:text-indigo-300 font-bold text-sm">
                      Sensitivity (ความไว / หาโรคเจอ)
                    </p>
                    <p className="text-2xl font-black text-indigo-700 dark:text-indigo-400 mt-1">
                      {sensitivity}%
                    </p>
                  </div>
                  <div className="bg-teal-50 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-800 rounded-lg p-4 text-center shadow-sm">
                    <p className="text-teal-900 dark:text-teal-300 font-bold text-sm">
                      Specificity (คัดคนปกติถูก)
                    </p>
                    <p className="text-2xl font-black text-teal-700 dark:text-teal-400 mt-1">
                      {specificity}%
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* รายการรูปภาพ (คิว) */}
          {tasks.length > 0 && (
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 transition-colors duration-300">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
                <h2 className="text-xl font-bold text-blue-900 dark:text-blue-400">
                  รายการรูปภาพ ({tasks.length} รูป)
                </h2>

                <div className="flex items-center gap-4 w-full md:w-auto justify-between md:justify-end">
                  <div className="bg-gray-100 dark:bg-gray-900 p-1 rounded-lg flex border border-gray-200 dark:border-gray-700">
                    <button
                      onClick={() => setViewMode("full")}
                      className={`px-4 py-1.5 text-sm font-bold rounded-md transition-all ${viewMode === "full" ? "bg-white dark:bg-gray-700 shadow text-blue-700 dark:text-blue-400" : "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"}`}
                    >
                      แบบเต็ม
                    </button>
                    <button
                      onClick={() => setViewMode("compact")}
                      className={`px-4 py-1.5 text-sm font-bold rounded-md transition-all ${viewMode === "compact" ? "bg-white dark:bg-gray-700 shadow text-blue-700 dark:text-blue-400" : "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"}`}
                    >
                      แบบย่อ
                    </button>
                  </div>

                  <button
                    onClick={clearAllTasks}
                    disabled={isProcessing}
                    className="text-sm font-bold text-red-600 dark:text-red-400 hover:underline disabled:text-gray-400 dark:disabled:text-gray-600"
                  >
                    ล้างทั้งหมด
                  </button>
                </div>
              </div>

              <div className="space-y-2">
                {tasks.map((task) => (
                  <div
                    key={task.id}
                    className={`bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700 rounded-lg flex items-center gap-3 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors overflow-hidden
                      ${viewMode === "full" ? "p-3 md:p-4" : "p-2"}`}
                  >
                    {viewMode === "full" && (
                      <div
                        className="w-14 h-14 md:w-16 md:h-16 relative flex-shrink-0 bg-gray-200 dark:bg-gray-700 rounded-md overflow-hidden border border-gray-300 dark:border-gray-600 cursor-pointer hover:ring-2 hover:ring-blue-500 transition-all group"
                        onClick={() => setFullscreenImage(task.previewUrl)}
                        title="คลิกเพื่อขยายรูปภาพ"
                      >
                        <img
                          src={task.previewUrl}
                          alt={task.file.name}
                          className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                        />
                        <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                          <span className="text-white text-xs font-bold">
                            🔍
                          </span>
                        </div>
                      </div>
                    )}

                    <div
                      className={`flex-1 font-bold text-gray-800 dark:text-gray-200 truncate ${viewMode === "compact" ? "text-sm" : ""}`}
                      title={task.file.name}
                    >
                      {task.file.name}
                    </div>

                    <div className="flex items-center gap-3 md:gap-4 flex-shrink-0">
                      <div className="flex justify-end">
                        {task.status === "idle" && (
                          <span
                            className={`text-gray-500 dark:text-gray-400 font-bold bg-gray-200 dark:bg-gray-700 rounded ${viewMode === "compact" ? "text-xs px-2 py-0.5" : "text-sm px-3 py-1"}`}
                          >
                            รอคิว
                          </span>
                        )}
                        {task.status === "processing" && (
                          <span
                            className={`text-blue-600 dark:text-blue-400 font-bold animate-pulse ${viewMode === "compact" ? "text-xs" : "text-sm"}`}
                          >
                            ⏳ กำลังวิเคราะห์...
                          </span>
                        )}
                        {task.status === "cancelled" && (
                          <span
                            className={`text-red-500 dark:text-red-400 font-bold bg-red-100 dark:bg-red-900/30 rounded ${viewMode === "compact" ? "text-xs px-2 py-0.5" : "text-sm px-3 py-1"}`}
                          >
                            ยกเลิกแล้ว
                          </span>
                        )}
                        {task.status === "error" && (
                          <span
                            className={`text-red-700 dark:text-red-400 font-bold bg-red-200 dark:bg-red-900/50 rounded ${viewMode === "compact" ? "text-xs px-2 py-0.5" : "text-sm px-3 py-1"}`}
                          >
                            เกิดข้อผิดพลาด
                          </span>
                        )}

                        {task.status === "success" && task.result && (
                          <span
                            className={`rounded font-bold border ${task.result.prediction_code === 1 ? "bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800" : "bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800"} ${viewMode === "compact" ? "text-[11px] px-2 py-0.5" : "text-sm px-3 py-1"}`}
                          >
                            {task.result.prediction_class}{" "}
                            <span className="font-normal opacity-80 ml-1">
                              (
                              {(
                                parseFloat(task.result.confidence) * 100
                              ).toFixed(1)}
                              %)
                            </span>
                          </span>
                        )}
                      </div>

                      {/* 🌟 ป้ายประเมินผล TP/TN เป็นคำเต็ม ย้ายมาชิดขวา */}
                      {task.status === "success" &&
                        task.result &&
                        task.result.eval_status !== "ไม่มีเฉลย" && (
                          <div
                            className={`border-l border-gray-300 dark:border-gray-600 pl-3 flex items-center justify-center ${viewMode === "compact" ? "h-5" : "h-8"}`}
                          >
                            <span
                              className={`font-bold whitespace-nowrap rounded-md shadow-sm border ${
                                task.result.eval_status.includes("T")
                                  ? "border-green-500 dark:border-green-400 text-green-800 dark:text-green-300 bg-green-200 dark:bg-green-900/50"
                                  : "border-red-500 dark:border-red-400 text-red-800 dark:text-red-300 bg-red-200 dark:bg-red-900/50"
                              } ${viewMode === "compact" ? "text-[11px] px-2 py-0.5" : "text-xs px-3 py-1"}`}
                            >
                              {task.result.eval_status}
                            </span>
                          </div>
                        )}

                      <div className="border-l border-gray-300 dark:border-gray-600 pl-3 flex-shrink-0">
                        {task.status === "processing" ? (
                          <button
                            onClick={() => cancelTask(task.id)}
                            className={`text-red-500 dark:text-red-400 hover:bg-red-100 dark:hover:bg-red-900/50 font-bold bg-red-50 dark:bg-red-900/20 rounded-full transition-colors ${viewMode === "compact" ? "text-xs p-1" : "text-sm p-1.5"}`}
                            title="ยกเลิก"
                          >
                            ✕
                          </button>
                        ) : (
                          <button
                            onClick={() => removeTask(task)}
                            className={`text-gray-400 dark:text-gray-500 hover:bg-gray-200 dark:hover:bg-gray-600 hover:text-gray-800 dark:hover:text-gray-200 rounded-full font-bold transition-colors ${viewMode === "compact" ? "text-xs p-1" : "text-sm p-1.5"}`}
                            title="ลบออก"
                          >
                            ✕
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>

      {fullscreenImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm p-4 transition-opacity duration-300"
          onClick={() => setFullscreenImage(null)}
        >
          <div className="relative max-w-5xl w-full flex flex-col items-center">
            <button
              className="absolute -top-12 right-0 md:-right-8 text-white text-4xl hover:text-gray-300 transition-colors focus:outline-none"
              onClick={() => setFullscreenImage(null)}
            >
              &times;
            </button>
            <img
              src={fullscreenImage}
              alt="Full Size Preview"
              className="max-w-full max-h-[85vh] rounded-lg shadow-2xl border-4 border-white/10"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        </div>
      )}
    </>
  );
}
