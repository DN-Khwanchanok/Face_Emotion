/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

type CvType = any;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState("Initializing...");
  const [isReady, setIsReady] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [emotion, setEmotion] = useState("-");
  const [conf, setConf] = useState(0);

  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);

  // ================= OpenCV =================
  async function loadOpenCV() {
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
        cv.onRuntimeInitialized = () => {
          cvRef.current = cv;
          resolve();
        };
      };
      script.onerror = () => reject("OpenCV load failed");
      document.body.appendChild(script);
    });
  }

  // ================= Cascade =================
  async function loadCascade() {
    const cv = cvRef.current;
    if (!cv) return;

    const res = await fetch("/opencv/haarcascade_frontalface_default.xml");
    const data = new Uint8Array(await res.arrayBuffer());
    const path = "face.xml";

    try { cv.FS_unlink(path); } catch {}
    cv.FS_createDataFile("/", path, data, true, false, false);

    const classifier = new cv.CascadeClassifier();
    classifier.load(path);
    faceCascadeRef.current = classifier;
  }

  // ================= Model =================
  async function loadModel() {
    sessionRef.current = await ort.InferenceSession.create(
      "/models/emotion_yolo11n_cls.onnx",
      { executionProviders: ["wasm"] }
    );
    const res = await fetch("/models/classes.json");
    classesRef.current = await res.json();
  }

  // ================= Camera =================
  async function startCamera() {
    if (!isReady) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });

    videoRef.current!.srcObject = stream;
    await videoRef.current!.play();
    setIsRunning(true);
    requestAnimationFrame(loop);
  }

  // ================= Preprocess =================
  function preprocessToTensor(faceCanvas: HTMLCanvasElement) {
    const size = 64;
    const tmp = document.createElement("canvas");
    tmp.width = size;
    tmp.height = size;

    const ctx = tmp.getContext("2d")!;
    ctx.drawImage(faceCanvas, 0, 0, size, size);

    const data = ctx.getImageData(0, 0, size, size).data;
    const input = new Float32Array(1 * 3 * size * size);

    let idx = 0;
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < size * size; i++) {
        input[idx++] = data[i * 4 + c] / 255;
      }
    }

    return new ort.Tensor("float32", input, [1, 3, size, size]);
  }

  function softmax(arr: Float32Array) {
    const max = Math.max(...arr);
    const exps = arr.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
  }

  // ================= Loop =================
  async function loop() {
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

    const ctx = canvas.getContext("2d")!;
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    ctx.drawImage(video, 0, 0);

    const src = cv.imread(canvas);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    const faces = new cv.RectVector();
    faceCascade.detectMultiScale(gray, faces, 1.1, 3);

    let best: any = null;
    let bestArea = 0;

    for (let i = 0; i < faces.size(); i++) {
      const r = faces.get(i);
      const area = r.width * r.height;
      if (area > bestArea) {
        bestArea = area;
        best = r;
      }
      ctx.strokeStyle = "rgba(255,255,255,0.3)";
      ctx.lineWidth = 1;
      ctx.strokeRect(r.x, r.y, r.width, r.height);
    }

    if (best) {
      const faceCanvas = document.createElement("canvas");
      faceCanvas.width = best.width;
      faceCanvas.height = best.height;
      faceCanvas
        .getContext("2d")!
        .drawImage(canvas, best.x, best.y, best.width, best.height, 0, 0, best.width, best.height);

      const input = preprocessToTensor(faceCanvas);
      const feeds: any = {};
      feeds[session.inputNames[0]] = input;

      const out = await session.run(feeds);
      const probs = softmax(out[session.outputNames[0]].data as Float32Array);

      const idx = probs.indexOf(Math.max(...probs));
      const detectedEmotion = classes[idx];
      const confidence = probs[idx];

      setEmotion(detectedEmotion);
      setConf(confidence);

      // ===== Face Box (เหมือนเดิม) =====
      const accent = "#00FF88";
      ctx.strokeStyle = accent;
      ctx.lineWidth = 3;
      ctx.shadowColor = accent;
      ctx.shadowBlur = 12;
      ctx.strokeRect(best.x, best.y, best.width, best.height);
      ctx.shadowBlur = 0;

      // ===== Label =====
      const label = `${detectedEmotion} ${(confidence * 100).toFixed(1)}%`;
      const fontSize = 14;
      ctx.font = `600 ${fontSize}px sans-serif`;
      ctx.textBaseline = "middle";

      const padX = 10;
      const padY = 6;
      const textW = ctx.measureText(label).width;
      const boxW = textW + padX * 2;
      const boxH = fontSize + padY * 2;

      const boxX = best.x;
      const boxY = Math.max(0, best.y - boxH - 8);

      ctx.fillStyle = "rgba(0,255,136,0.85)";
      ctx.fillRect(boxX, boxY, boxW, boxH);

      ctx.strokeStyle = accent;
      ctx.lineWidth = 1;
      ctx.strokeRect(boxX, boxY, boxW, boxH);

      ctx.fillStyle = "#001a12";
      ctx.fillText(label, boxX + padX, boxY + boxH / 2);
    }

    src.delete();
    gray.delete();
    faces.delete();
    requestAnimationFrame(loop);
  }

  // ================= Boot =================
  useEffect(() => {
    (async () => {
      setStatus("Loading OpenCV...");
      await loadOpenCV();
      setStatus("Loading Haar Cascade...");
      await loadCascade();
      setStatus("Loading ONNX Model...");
      await loadModel();
      setStatus("Ready");
      setIsReady(true);
    })();
  }, []);

  return (
    <main className="min-h-screen bg-neutral-950 text-white flex flex-col items-center justify-center p-4 md:p-8">
      <header className="mb-8 text-center">
        <h1 className="text-3xl md:text-5xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">
          Emotion Recognition
        </h1>
        <p className="text-neutral-400 text-sm">
          Real-time analysis using OpenCV & YOLO11 (ONNX)
        </p>
      </header>

      <div className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 relative aspect-[4/3] bg-neutral-900 rounded-2xl overflow-hidden border border-neutral-800">
          <video ref={videoRef} className="hidden" />
          <canvas ref={canvasRef} className={`w-full h-full ${isRunning ? "opacity-100" : "opacity-30"}`} />

          {!isRunning && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/40">
              <button
                onClick={startCamera}
                disabled={!isReady}
                className={`px-8 py-4 rounded-full font-bold ${
                  isReady ? "bg-indigo-600" : "bg-neutral-700"
                }`}
              >
                {isReady ? "Start Camera" : "System Preparing..."}
              </button>
            </div>
          )}
        </div>

        <div className="bg-neutral-900 rounded-2xl p-6 border border-neutral-800 flex flex-col items-center">
          <h2 className="text-neutral-400 text-sm mb-2">Detected Emotion</h2>
          <div className="text-5xl font-black capitalize">{emotion}</div>

          <div className="w-full mt-6">
            <div className="flex justify-between text-xs text-neutral-400 mb-1">
              <span>Confidence</span>
              <span>{(conf * 100).toFixed(1)}%</span>
            </div>
            <div className="h-3 w-full bg-neutral-800 rounded-full">
              <div
                className="h-full bg-indigo-500 rounded-full"
                style={{ width: `${conf * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 text-xs text-neutral-400">Status: {status}</div>
    </main>
  );
}
