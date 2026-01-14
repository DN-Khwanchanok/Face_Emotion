/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

type CvType = any;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState<string>("Initializing AI Pipeline...");
  const [isReady, setIsReady] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [emotion, setEmotion] = useState<string>("-");
  const [conf, setConf] = useState<number>(0);

  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);

  const emotionEmoji: Record<string, string> = {
    happy: "😄",
    sad: "😢",
    angry: "😠",
    surprised: "😲",
    neutral: "😐",
  };

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
      script.onerror = () => reject();
      document.body.appendChild(script);
    });
  }

  async function loadCascade() {
    const cv = cvRef.current;
    const res = await fetch("/opencv/haarcascade_frontalface_default.xml");
    const data = new Uint8Array(await res.arrayBuffer());
    cv.FS_createDataFile("/", "face.xml", data, true, false, false);
    const classifier = new cv.CascadeClassifier();
    classifier.load("face.xml");
    faceCascadeRef.current = classifier;
  }

  async function loadModel() {
    sessionRef.current = await ort.InferenceSession.create(
      "/models/emotion_yolo11n_cls.onnx",
      { executionProviders: ["wasm"] }
    );
    classesRef.current = await (await fetch("/models/classes.json")).json();
  }

  async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current!.srcObject = stream;
    await videoRef.current!.play();
    setIsRunning(true);
    setStatus("🟢 Real-time Inference");
    requestAnimationFrame(loop);
  }

  function preprocess(face: HTMLCanvasElement) {
    const s = 64;
    const c = document.createElement("canvas");
    c.width = s;
    c.height = s;
    c.getContext("2d")!.drawImage(face, 0, 0, s, s);
    const d = c.getContext("2d")!.getImageData(0, 0, s, s).data;
    const f = new Float32Array(1 * 3 * s * s);
    let k = 0;
    for (let ch = 0; ch < 3; ch++)
      for (let i = 0; i < s * s; i++)
        f[k++] = d[i * 4 + ch] / 255;
    return new ort.Tensor("float32", f, [1, 3, s, s]);
  }

  function softmax(a: Float32Array) {
    const m = Math.max(...a);
    const e = a.map(v => Math.exp(v - m));
    const s = e.reduce((x, y) => x + y, 0);
    return e.map(v => v / s);
  }

  async function loop() {
    const cv = cvRef.current;
    const faceCascade = faceCascadeRef.current;
    const session = sessionRef.current;
    const classes = classesRef.current;
    const video = videoRef.current!;
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const src = cv.imread(canvas);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    const faces = new cv.RectVector();
    faceCascade.detectMultiScale(gray, faces);

    if (faces.size() > 0) {
      const r = faces.get(0);
      ctx.strokeStyle = "#22c55e";
      ctx.lineWidth = 3;
      ctx.shadowColor = "#22c55e";
      ctx.shadowBlur = 12;
      ctx.strokeRect(r.x, r.y, r.width, r.height);
      ctx.shadowBlur = 0;

      const fc = document.createElement("canvas");
      fc.width = r.width;
      fc.height = r.height;
      fc.getContext("2d")!.drawImage(canvas, r.x, r.y, r.width, r.height, 0, 0, r.width, r.height);

      const input = preprocess(fc);
      const out = await session!.run({ [session!.inputNames[0]]: input });
      const probs = softmax(out[session!.outputNames[0]].data as Float32Array);
      const idx = probs.indexOf(Math.max(...probs));
      setEmotion(classes![idx]);
      setConf(probs[idx]);
    }

    src.delete(); gray.delete(); faces.delete();
    requestAnimationFrame(loop);
  }

  useEffect(() => {
    (async () => {
      await loadOpenCV();
      await loadCascade();
      await loadModel();
      setIsReady(true);
      setStatus("Ready");
    })();
  }, []);

  return (
    <main className="min-h-screen bg-neutral-950 text-white flex items-center justify-center p-6">
      <div className="max-w-6xl w-full grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 relative rounded-2xl overflow-hidden bg-neutral-900/70 backdrop-blur-xl border border-white/10 shadow-[0_0_60px_rgba(99,102,241,0.15)]">
          <video ref={videoRef} className="hidden" />
          <canvas ref={canvasRef} className="w-full h-full" />
          {!isRunning && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
              <button
                disabled={!isReady}
                onClick={startCamera}
                className="px-10 py-5 rounded-full font-bold text-lg
                bg-gradient-to-r from-indigo-600 to-purple-600
                hover:shadow-[0_0_40px_rgba(99,102,241,0.9)]
                transition-all">
                Start Camera
              </button>
            </div>
          )}
          <div className="absolute top-4 left-4 flex items-center gap-2 bg-black/60 px-3 py-1 rounded-full text-xs">
            <span className={`h-2 w-2 rounded-full animate-pulse ${isRunning ? "bg-green-400" : "bg-yellow-400"}`} />
            {status}
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="bg-neutral-900 rounded-2xl p-6 text-center border border-white/10">
            <div className="text-5xl mb-2">{emotionEmoji[emotion] ?? "🧠"}</div>
            <div className="text-4xl font-black capitalize">{emotion}</div>
            <div className="mt-4">
              <div className="flex justify-between text-xs mb-1">
                <span>Confidence</span>
                <span>{(conf * 100).toFixed(1)}%</span>
              </div>
              <div className="h-3 bg-neutral-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 transition-all"
                  style={{ width: `${conf * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
