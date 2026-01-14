/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

type CvType = any;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState<string>("Initializing...");
  const [isReady, setIsReady] = useState(false); // เช็คว่าโหลด Model เสร็จหรือยัง
  const [isRunning, setIsRunning] = useState(false); // เช็คว่ากล้องเปิดหรือยัง
  const [emotion, setEmotion] = useState<string>("-");
  const [conf, setConf] = useState<number>(0);

  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);

  // --- Logic เดิม (Load OpenCV, Model, Cascade) ---
  async function loadOpenCV() {
    if (typeof window === "undefined") return;
    if ((window as any).cv?.Mat) {
      cvRef.current = (window as any).cv;
      return;
    }
    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/opencv/opencv.js";
      script.async = true;
      script.onload = () => {
        const cv = (window as any).cv;
        if (!cv) return reject(new Error("OpenCV โหลดแล้วแต่ window.cv ไม่มีค่า"));
        const waitReady = () => {
          if ((window as any).cv?.Mat) {
            cvRef.current = (window as any).cv;
            resolve();
          } else {
            setTimeout(waitReady, 50);
          }
        };
        if ("onRuntimeInitialized" in cv) {
          cv.onRuntimeInitialized = () => waitReady();
        } else {
          waitReady();
        }
      };
      script.onerror = () => reject(new Error("โหลด /opencv/opencv.js ไม่สำเร็จ"));
      document.body.appendChild(script);
    });
  }

  async function loadCascade() {
    const cv = cvRef.current;
    if (!cv) throw new Error("cv ยังไม่พร้อม");
    const cascadeUrl = "/opencv/haarcascade_frontalface_default.xml";
    const res = await fetch(cascadeUrl);
    if (!res.ok) throw new Error("โหลด cascade ไม่สำเร็จ");
    const data = new Uint8Array(await res.arrayBuffer());
    const cascadePath = "haarcascade_frontalface_default.xml";
    try {
      cv.FS_unlink(cascadePath);
    } catch {}
    cv.FS_createDataFile("/", cascadePath, data, true, false, false);
    const faceCascade = new cv.CascadeClassifier();
    const loaded = faceCascade.load(cascadePath);
    if (!loaded) throw new Error("cascade load() ไม่สำเร็จ");
    faceCascadeRef.current = faceCascade;
  }

  async function loadModel() {
    const session = await ort.InferenceSession.create(
      "/models/emotion_yolo11n_cls.onnx",
      { executionProviders: ["wasm"] }
    );
    sessionRef.current = session;
    const clsRes = await fetch("/models/classes.json");
    if (!clsRes.ok) throw new Error("โหลด classes.json ไม่สำเร็จ");
    classesRef.current = await clsRes.json();
  }

  async function startCamera() {
    if (!isReady) return;
    try {
      setStatus("Requesting camera access...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setStatus("Active");
      setIsRunning(true);
      requestAnimationFrame(loop);
    } catch (e: any) {
      setStatus(`Camera Error: ${e?.message}`);
    }
  }

  function preprocessToTensor(faceCanvas: HTMLCanvasElement) {
    const size = 64;
    const tmp = document.createElement("canvas");
    tmp.width = size;
    tmp.height = size;
    const ctx = tmp.getContext("2d")!;
    ctx.drawImage(faceCanvas, 0, 0, size, size);
    const imgData = ctx.getImageData(0, 0, size, size).data;
    const float = new Float32Array(1 * 3 * size * size);
    let idx = 0;
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < size * size; i++) {
        const r = imgData[i * 4 + 0] / 255;
        const g = imgData[i * 4 + 1] / 255;
        const b = imgData[i * 4 + 2] / 255;
        float[idx++] = c === 0 ? r : c === 1 ? g : b;
      }
    }
    return new ort.Tensor("float32", float, [1, 3, size, size]);
  }

  function softmax(logits: Float32Array) {
    let max = -Infinity;
    for (const v of logits) max = Math.max(max, v);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  async function loop() {
    try {
      const cv = cvRef.current;
      const faceCascade = faceCascadeRef.current;
      const session = sessionRef.current;
      const classes = classesRef.current;
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!cv || !faceCascade || !session || !classes || !video || !canvas) {
        requestAnimationFrame(loop);
        return;
      }

      if (video.paused || video.ended) return;

      const ctx = canvas.getContext("2d")!;
      // Sync canvas size with video
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
      }
      
      ctx.drawImage(video, 0, 0);

      const src = cv.imread(canvas);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      const faces = new cv.RectVector();
      const msize = new cv.Size(0, 0);
      faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);

      let bestRect: any = null;
      let bestArea = 0;

      for (let i = 0; i < faces.size(); i++) {
        const r = faces.get(i);
        const area = r.width * r.height;
        if (area > bestArea) {
          bestArea = area;
          bestRect = r;
        }
        // Draw secondary faces lightly
        ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
        ctx.lineWidth = 1;
        ctx.strokeRect(r.x, r.y, r.width, r.height);
      }

      if (bestRect) {
        // Highlighting best face
        ctx.strokeStyle = "#00FF00"; // Lime Green
        ctx.lineWidth = 3;
        ctx.shadowColor = "#00FF00";
        ctx.shadowBlur = 10;
        ctx.strokeRect(bestRect.x, bestRect.y, bestRect.width, bestRect.height);
        ctx.shadowBlur = 0; // reset shadow

        // Crop & Predict
        const faceCanvas = document.createElement("canvas");
        faceCanvas.width = bestRect.width;
        faceCanvas.height = bestRect.height;
        const fctx = faceCanvas.getContext("2d")!;
        fctx.drawImage(
          canvas,
          bestRect.x, bestRect.y, bestRect.width, bestRect.height,
          0, 0, bestRect.width, bestRect.height
        );

        const input = preprocessToTensor(faceCanvas);
        const feeds: Record<string, ort.Tensor> = {};
        feeds[session.inputNames[0]] = input;

        const out = await session.run(feeds);
        const outName = session.outputNames[0];
        const logits = out[outName].data as Float32Array;

        const probs = softmax(logits);
        let maxIdx = 0;
        for (let i = 1; i < probs.length; i++) {
          if (probs[i] > probs[maxIdx]) maxIdx = i;
        }

        const detectedEmotion = classes[maxIdx] ?? `class_${maxIdx}`;
        const confidence = probs[maxIdx] ?? 0;

        setEmotion(detectedEmotion);
        setConf(confidence);
      } else {
        // Reset if no face found for a while (optional)
        // setEmotion("-");
        // setConf(0);
      }

      src.delete();
      gray.delete();
      faces.delete();

      requestAnimationFrame(loop);
    } catch (e: any) {
      setStatus(`Loop Error: ${e?.message}`);
    }
  }

  // Boot sequence
  useEffect(() => {
    (async () => {
      try {
        setStatus("Loading OpenCV...");
        await loadOpenCV();
        setStatus("Loading Haar Cascade...");
        await loadCascade();
        setStatus("Loading ONNX Model...");
        await loadModel();
        setStatus("Ready");
        setIsReady(true);
      } catch (e: any) {
        setStatus(`Init Failed: ${e?.message ?? e}`);
      }
    })();
  }, []);

  return (
    <main className="min-h-screen bg-neutral-950 text-white selection:bg-indigo-500/30 flex flex-col items-center justify-center p-4 md:p-8">
      
      {/* Header */}
      <header className="mb-8 text-center space-y-2">
        <h1 className="text-3xl md:text-5xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">
          Emotion Recognition
        </h1>
        <p className="text-neutral-400 text-sm md:text-base">
          Real-time analysis using OpenCV & YOLO11 (ONNX)
        </p>
      </header>

      {/* Main Grid Layout */}
      <div className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Video Feed */}
        <div className="lg:col-span-2 relative aspect-[4/3] bg-neutral-900 rounded-2xl overflow-hidden shadow-2xl border border-neutral-800 ring-1 ring-white/5">
            {/* Hidden Video Source */}
            <video ref={videoRef} className="hidden" playsInline muted />
            
            {/* Main Canvas */}
            <canvas 
                ref={canvasRef} 
                className={`w-full h-full object-cover transition-opacity duration-500 ${isRunning ? 'opacity-100' : 'opacity-30'}`} 
            />

            {/* Overlay: Start Button / Loading */}
            {!isRunning && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm z-10">
                    <button
                        onClick={startCamera}
                        disabled={!isReady}
                        className={`group relative px-8 py-4 rounded-full font-bold text-lg transition-all transform hover:scale-105 active:scale-95 shadow-[0_0_20px_rgba(79,70,229,0.5)] 
                        ${isReady 
                            ? "bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:shadow-[0_0_30px_rgba(79,70,229,0.7)]" 
                            : "bg-neutral-700 text-neutral-400 cursor-not-allowed"
                        }`}
                    >
                        {isReady ? (
                             <span className="flex items-center gap-2">
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>
                                Start Camera
                             </span>
                        ) : (
                            <span className="flex items-center gap-2">
                                <svg className="animate-spin h-5 w-5 text-neutral-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                System Preparing...
                            </span>
                        )}
                    </button>
                </div>
            )}
            
            {/* Status Badge in Video Corner */}
            <div className="absolute top-4 left-4 bg-black/60 backdrop-blur text-xs px-3 py-1 rounded-full border border-white/10 text-neutral-300">
                System: <span className={isReady ? "text-green-400" : "text-yellow-400"}>{status}</span>
            </div>
        </div>

        {/* Right Column: Analysis Dashboard */}
        <div className="flex flex-col gap-4">
            
            {/* Emotion Card */}
            <div className="flex-1 bg-neutral-900 rounded-2xl p-6 border border-neutral-800 shadow-xl flex flex-col items-center justify-center text-center relative overflow-hidden group">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500"></div>
                
                <h2 className="text-neutral-400 text-sm font-medium uppercase tracking-wider mb-2">Detected Emotion</h2>
                <div className="text-5xl font-black bg-clip-text text-transparent bg-gradient-to-br from-white to-neutral-400 py-2 capitalize transition-all duration-300 group-hover:scale-110">
                    {emotion}
                </div>
                
                {/* Confidence Bar */}
                <div className="w-full mt-6">
                    <div className="flex justify-between text-xs text-neutral-400 mb-1">
                        <span>Confidence</span>
                        <span>{(conf * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-3 w-full bg-neutral-800 rounded-full overflow-hidden">
                        <div 
                            className="h-full bg-indigo-500 rounded-full transition-all duration-300 ease-out"
                            style={{ width: `${conf * 100}%` }}
                        ></div>
                    </div>
                </div>
            </div>

            {/* Instruction / Info Card */}
            <div className="bg-neutral-900/50 rounded-2xl p-6 border border-neutral-800/50 text-sm text-neutral-400">
                <h3 className="text-white font-semibold mb-2 flex items-center gap-2">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
                    How it works
                </h3>
                <ul className="space-y-2 list-disc list-inside opacity-80">
                    <li>Allow camera access to start.</li>
                    <li>Ensure your face is well-lit.</li>
                    <li>Processing happens locally on your device (Privacy focused).</li>
                </ul>
            </div>

        </div>
      </div>
    </main>
  );
}