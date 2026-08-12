"use client";
import { useState, useEffect, useRef } from "react";

type PredictionResult = {
  id: string;
  filename: string;
  prediction_class: string;
  prediction_code: number;
  confidence: number;
  ground_truth: string;
  eval_status: string;
};

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const [isServerReady, setIsServerReady] = useState(false);
  const [isWakingUp, setIsWakingUp] = useState(true);
  const [showSuccessBanner, setShowSuccessBanner] = useState(false);

  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [isDraggingCsv, setIsDraggingCsv] = useState(false);
  const csvInputRef = useRef<HTMLInputElement>(null);

  const [useGroundTruth, setUseGroundTruth] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);
  const [theme, setTheme] = useState<"light" | "dark" | "system">("system");

  // 1. สำหรับปลุก Backend ตอนโหลดหน้าเว็บครั้งแรก (Run once)
  useEffect(() => {
    const wakeUpServer = async () => {
      try {
        const response = await fetch("https://pesplanusai.onrender.com/");
        if (response.ok) {
          setIsServerReady(true);
          setShowSuccessBanner(true); // ✨ 1. เปิดโชว์ป้ายสีเขียว

          // ✨ 2. ตั้งเวลาให้ซ่อนป้ายอัตโนมัติใน 4 วินาที (4000 ms)
          setTimeout(() => {
            setShowSuccessBanner(false);
          }, 4000);
        }
      } catch (error) {
        console.error("Backend is sleeping or not reachable:", error);
      } finally {
        setIsWakingUp(false);
      }
    };

    wakeUpServer();
  }, []);

  // 2. สำหรับสลับธีม Dark / Light (Run when 'theme' changes)
  useEffect(() => {
    const isDark =
      theme === "dark" ||
      (theme === "system" &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);

    document.documentElement.classList.toggle("dark", isDark);
  }, [theme]);

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith("image/")) {
      alert("⚠️ กรุณาเลือกไฟล์รูปภาพเท่านั้น (.png, .jpg, .jpeg)");
      return;
    }
    setSelectedFile(file);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
  };

  const removeSelectedFile = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
  };

  const processImage = async () => {
    if (!selectedFile) {
      alert("⚠️ กรุณาเลือกรูปภาพเอ็กซ์เรย์ก่อนประมวลผล");
      return;
    }
    if (useGroundTruth && !csvFile) {
      alert("⚠️ กรุณาอัปโหลดไฟล์เฉลย CSV (Ground Truth)");
      return;
    }

    setIsProcessing(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("gt_option", useGroundTruth ? "upload" : "none");

    if (useGroundTruth && csvFile) formData.append("csv_file", csvFile);

    try {
      // const res = await fetch("http://127.0.0.1:8000/api/predict", {
      const res = await fetch("https://pesplanusai.onrender.com/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        let errorMessage = `HTTP ${res.status}:\n`;

        if (errorData.detail) {
          if (typeof errorData.detail === "string") {
            errorMessage += errorData.detail;
          } else {
            errorMessage += JSON.stringify(errorData.detail, null, 2);
          }
        } else {
          errorMessage += "เกิดข้อผิดพลาดฝั่งเซิร์ฟเวอร์ (ไม่ทราบรายละเอียด)";
        }

        throw new Error(errorMessage);
      }

      const data = await res.json();
      setResult(data);
    } catch (error: any) {
      console.error("System Error: ", error);
      alert(
        `ข้อผิดพลาดจากระบบ:\n${error.message || "เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ"}`,
      );
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors duration-200 font-sans">
      <header className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-800 sticky top-0 z-30 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div
            onClick={() => window.location.reload()}
            className="flex items-center space-x-3 cursor-pointer hover:opacity-80 transition-opacity select-none"
          >
            <span className="text-2xl drop-shadow-sm">🦶</span>
            <div>
              <h1 className="font-bold text-sm leading-tight text-gray-900 dark:text-white">
                เว็บแอปพลิเคชันเพื่อจำแนกโรคเท้าแบนจากภาพเอ็กซเรย์ด้วยตัวแบบการเรียนรู้ด้วยเครื่องและการเรียนรู้เชิงลึก
              </h1>
              <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">
                Web-based Application for Flatfoot Classification from X-ray
                Images Using Machine Learning and Deep Learning Models
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="hidden md:flex items-center space-x-3 mr-4">
              <img
                src="cslogo.png"
                alt="CS Logo"
                className="h-9 object-contain"
              />
              <img
                src="scilogo.png"
                alt="Sci Logo"
                className="h-9 object-contain"
              />
              <img
                src="nprulogo.png"
                alt="NPRU Logo"
                className="h-9 object-contain"
              />
            </div>

            {typeof theme !== "undefined" && (
              <select
                value={theme}
                onChange={(e: any) => setTheme && setTheme(e.target.value)}
                className="text-xs bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 border border-gray-300 dark:border-gray-600 rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer font-medium"
              >
                <option value="system">💻 ธีมระบบ</option>
                <option value="light">☀️ สว่าง</option>
                <option value="dark">🌙 มืด</option>
              </select>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* ✨ เพิ่ม UI แจ้งเตือนสถานะเซิร์ฟเวอร์ไว้ด้านบนสุด หรือใต้ Header */}
        <div className="max-w-4xl mx-auto pt-4 px-4">
          {
            isWakingUp ? (
              // ป้ายสีเหลือง: กำลังปลุก (แสดงค้างไว้จนกว่าจะตื่น)
              <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 rounded shadow-sm flex items-center animate-pulse">
                <span className="text-xl mr-3">⏳</span>
                <p className="text-sm font-medium">
                  กำลังปลุกเซิร์ฟเวอร์ AI... (อาจใช้เวลา 1-2
                  นาทีเนื่องจากโหมดประหยัดพลังงาน)
                </p>
              </div>
            ) : !isServerReady ? (
              // ป้ายสีแดง: เชื่อมต่อไม่ได้
              <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-3 rounded shadow-sm flex items-center">
                <span className="text-xl mr-3">⚠️</span>
                <p className="text-sm font-medium">
                  ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้
                  กรุณาลองรีเฟรชหน้าเว็บอีกครั้ง
                </p>
              </div>
            ) : showSuccessBanner ? (
              // ✨ ป้ายสีเขียว: พร้อมใช้งาน (จะแสดงแค่ 4 วินาที แล้วหายวับไป)
              <div className="bg-green-100 border-l-4 border-green-500 text-green-700 p-3 rounded shadow-sm flex items-center transition-all duration-500 ease-in-out">
                <span className="text-xl mr-3">🚀</span>
                <p className="text-sm font-medium">
                  เซิร์ฟเวอร์ AI พร้อมใช้งานแล้ว!
                </p>
              </div>
            ) : null /* ✨ คืนค่า null เพื่อให้พื้นที่ตรงนี้หายไปเนียนๆ เมื่อหมดเวลา */
          }
        </div>
        {!previewUrl ? (
          // ================= STATE 1: ยังไม่อัปโหลดรูปภาพ =================
          <div className="max-w-2xl mx-auto mt-10">
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                if (e.dataTransfer.files?.[0])
                  handleFileSelect(e.dataTransfer.files[0]);
              }}
              className={`border-2 border-dashed rounded-3xl p-16 text-center transition-all duration-300 cursor-pointer bg-white dark:bg-gray-800 shadow-sm flex flex-col items-center justify-center ${
                isDragging
                  ? "border-blue-500 bg-blue-50/50 dark:bg-blue-900/20 scale-[0.98]"
                  : "border-gray-300 dark:border-gray-700 hover:border-blue-400 hover:shadow-md"
              }`}
              onClick={() => {
                const input = document.createElement("input");
                input.type = "file";
                input.accept = "image/*";
                input.onchange = (e: any) => {
                  if (e.target.files?.[0]) handleFileSelect(e.target.files[0]);
                };
                input.click();
              }}
            >
              <div className="w-20 h-20 bg-blue-100 dark:bg-blue-900/50 rounded-full flex items-center justify-center mb-6">
                <span className="text-4xl text-blue-600 drop-shadow-sm">
                  📸
                </span>
              </div>
              <h3 className="text-xl font-bold text-gray-800 dark:text-gray-100 mb-2">
                อัปโหลดภาพเอ็กซ์เรย์เท้า
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400 max-w-sm">
                ลากรูปภาพมาวางที่นี่ หรือคลิกเพื่อเลือกไฟล์จากอุปกรณ์ของคุณ
              </p>
              <div className="mt-6 flex items-center gap-2 text-xs text-gray-400 font-medium">
                <span className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                  PNG
                </span>
                <span className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                  JPG
                </span>
                <span className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                  JPEG
                </span>
              </div>
            </div>
          </div>
        ) : (
          // ================= STATE 2: อัปโหลดรูปภาพแล้ว =================
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8">
            {/* ---------------- ฝั่งซ้าย: รูปภาพและแผงควบคุม ---------------- */}
            <div className="lg:col-span-5 flex flex-col space-y-4">
              <div className="bg-white dark:bg-gray-800 rounded-2xl p-4 border border-gray-200 dark:border-gray-700 shadow-sm">
                <div className="relative w-full aspect-square rounded-xl overflow-hidden bg-gray-100 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 group flex items-center justify-center">
                  <img
                    src={previewUrl}
                    alt="X-ray Preview"
                    className="w-full h-full object-contain cursor-pointer transition-transform duration-300 group-hover:scale-105"
                    onClick={() => setFullscreenImage(previewUrl)}
                  />
                  <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors pointer-events-none" />
                  <button
                    onClick={() => setFullscreenImage(previewUrl)}
                    className="absolute bottom-3 right-3 bg-black/60 hover:bg-black/80 text-white p-2 rounded-lg text-xs backdrop-blur-md transition shadow-lg opacity-0 group-hover:opacity-100"
                  >
                    🔍 ขยายภาพ
                  </button>
                </div>
                <div className="w-full flex items-center justify-between text-xs text-gray-500 mt-4 px-1">
                  <span className="truncate font-medium max-w-[200px]">
                    {selectedFile?.name}
                  </span>
                  <button
                    onClick={removeSelectedFile}
                    className="text-red-500 hover:text-red-700 font-bold cursor-pointer bg-red-50 dark:bg-red-900/20 px-3 py-1.5 rounded-lg transition"
                  >
                    ✕ นำออก
                  </button>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 border border-gray-200 dark:border-gray-700 shadow-sm flex flex-col gap-4">
                <div className="flex flex-col gap-3 text-sm">
                  <label className="flex items-center space-x-2 font-semibold cursor-pointer text-gray-700 dark:text-gray-300 select-none">
                    <input
                      type="checkbox"
                      checked={useGroundTruth}
                      onChange={(e) => setUseGroundTruth(e.target.checked)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 w-4 h-4 cursor-pointer"
                    />
                    <span>เปรียบเทียบกับเฉลย (Ground Truth CSV)</span>
                  </label>

                  {useGroundTruth && (
                    <div className="animate-in fade-in slide-in-from-top-2 duration-200">
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
                        className={`border-2 border-dashed rounded-xl p-3 text-center cursor-pointer transition ${
                          isDraggingCsv
                            ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                            : "border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700/50 hover:border-blue-400"
                        }`}
                      >
                        <p className="font-semibold text-blue-600 dark:text-blue-400 text-xs truncate">
                          {csvFile
                            ? `✅ ${csvFile.name}`
                            : "📂 อัปโหลดไฟล์ CSV เฉลย"}
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                <button
                  onClick={processImage}
                  disabled={isProcessing}
                  className="w-full py-3.5 px-4 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 disabled:from-gray-400 disabled:to-gray-500 text-white font-bold text-sm rounded-xl shadow-lg shadow-blue-500/30 transition-all duration-200 flex items-center justify-center space-x-2 cursor-pointer disabled:cursor-not-allowed transform active:scale-[0.98]"
                >
                  {isProcessing ? (
                    <span className="flex items-center gap-2">
                      <svg
                        className="animate-spin h-4 w-4 text-white"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                      >
                        <circle
                          className="opacity-25"
                          cx="12"
                          cy="12"
                          r="10"
                          stroke="currentColor"
                          strokeWidth="4"
                        ></circle>
                        <path
                          className="opacity-75"
                          fill="currentColor"
                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                        ></path>
                      </svg>
                      กำลังประมวลผล...
                    </span>
                  ) : (
                    <span>🚀 ประมวลผลวิเคราะห์</span>
                  )}
                </button>
              </div>
            </div>

            {/* ---------------- ฝั่งขวา: พื้นที่แสดงผลลัพธ์ ---------------- */}
            <div className="lg:col-span-7 flex flex-col">
              <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 lg:p-8 border border-gray-200 dark:border-gray-700 shadow-sm flex-grow flex flex-col">
                <h3 className="font-bold text-lg border-b border-gray-100 dark:border-gray-700 pb-4 mb-6 flex items-center gap-2 text-gray-800 dark:text-gray-100">
                  <span className="text-xl">📊</span> ผลการวินิจฉัยภาวะเท้าแบน
                </h3>

                {result ? (
                  <div className="flex-grow flex flex-col justify-start animate-in fade-in duration-300">
                    <div className="space-y-8">
                      <div>
                        <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">
                          ผลการทำนาย (Prediction)
                        </p>
                        <div
                          className={`p-6 rounded-2xl border-2 flex items-center justify-center ${
                            result.prediction_code === 1
                              ? "bg-red-50 border-red-200 dark:bg-red-900/10 dark:border-red-900/50 text-red-600 dark:text-red-400"
                              : "bg-green-50 border-green-200 dark:bg-green-900/10 dark:border-green-900/50 text-green-600 dark:text-green-400"
                          }`}
                        >
                          <span className="text-3xl lg:text-4xl font-black tracking-tight">
                            {result.prediction_class}
                          </span>
                        </div>
                      </div>

                      <div className="bg-gray-50 dark:bg-gray-900/50 p-5 rounded-2xl border border-gray-100 dark:border-gray-700">
                        <div className="flex justify-between items-end mb-2">
                          <span className="text-sm font-bold text-gray-600 dark:text-gray-300">
                            ความเชื่อมั่น (Confidence)
                          </span>
                          <span className="text-xl font-black text-blue-600 dark:text-blue-400">
                            {(result.confidence * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 h-4 rounded-full overflow-hidden shadow-inner">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-blue-600 h-full rounded-full transition-all duration-1000 ease-out"
                            style={{
                              width: `${Math.min(result.confidence * 100, 100)}%`,
                            }}
                          ></div>
                        </div>
                      </div>

                      {useGroundTruth && (
                        <div className="pt-2">
                          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">
                            การประเมินความแม่นยำ
                          </p>
                          <div className="grid grid-cols-2 gap-4">
                            <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded-xl border border-gray-100 dark:border-gray-700">
                              <span className="block text-xs text-gray-500 mb-1">
                                เฉลยจริง (Ground Truth)
                              </span>
                              <span className="font-bold text-sm">
                                {result.ground_truth}
                              </span>
                            </div>
                            <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded-xl border border-gray-100 dark:border-gray-700">
                              <span className="block text-xs text-gray-500 mb-1">
                                สถานะการประเมิน
                              </span>
                              <span
                                className={`font-bold text-sm ${
                                  result.eval_status.includes("True")
                                    ? "text-green-600 dark:text-green-400"
                                    : result.eval_status.includes("False")
                                      ? "text-red-500 dark:text-red-400"
                                      : "text-gray-500"
                                }`}
                              >
                                {result.eval_status}
                              </span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center flex-grow text-gray-400 dark:text-gray-500 space-y-4 min-h-[300px]">
                    <div className="w-24 h-24 bg-gray-50 dark:bg-gray-800 rounded-full flex items-center justify-center border-4 border-gray-100 dark:border-gray-700/50">
                      <span className="text-4xl opacity-50">⏳</span>
                    </div>
                    <p className="text-sm font-medium">รอการประมวลผล</p>
                    <p className="text-xs">
                      คลิกปุ่ม "🚀 ประมวลผลวิเคราะห์" ด้านซ้ายมือเพื่อดูผลลัพธ์
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Fullscreen Image Modal */}
      {fullscreenImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm p-4 animate-in fade-in duration-200"
          onClick={() => setFullscreenImage(null)}
        >
          <div className="relative max-w-5xl w-full h-full flex items-center justify-center">
            <button
              className="absolute top-4 right-4 bg-white/10 hover:bg-white/25 text-white rounded-full p-2 backdrop-blur-md transition z-50"
              onClick={() => setFullscreenImage(null)}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
            <img
              src={fullscreenImage}
              alt="Full Preview"
              className="max-w-full max-h-[90vh] rounded-xl shadow-2xl object-contain"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        </div>
      )}
    </div>
  );
}
