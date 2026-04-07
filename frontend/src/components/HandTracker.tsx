"use client";

import React, { useEffect, useRef, useState } from "react";
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import axios from "axios";
import { Activity, CheckCircle2 } from "lucide-react";
import { motion } from "framer-motion";

const API_BASE_URL = "http://127.0.0.1:8000";
const BACKEND_URL = `${API_BASE_URL}/predict`;

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];

const HandTracker = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [prediction, setPrediction] = useState<string>("Waiting...");
  const [confidence, setConfidence] = useState<number>(0);
  const [isCameraStarted, setIsCameraStarted] = useState(false);
  const [backendStatus, setBackendStatus] = useState<"connecting" | "online" | "offline">("connecting");
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const lastPredictTime = useRef<number>(0);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const lastPredictionErrorRef = useRef<string | null>(null);

  useEffect(() => {
    // Check backend status
    const checkBackend = async () => {
      try {
        const response = await axios.get(API_BASE_URL);
        const isModelLoaded = Boolean(response.data.model_loaded);

        setBackendStatus(isModelLoaded ? "online" : "offline");
        setPredictionError(isModelLoaded ? null : "Backend is running, but the gesture model is not loaded.");
      } catch {
        setBackendStatus("offline");
        setPredictionError(null); // Silent failure for health check
      }
    };
    checkBackend();

    let animationFrameId: number;
    let cameraStream: MediaStream | null = null;
    let lastVideoTime = -1;

    const initializeMediaPipe = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1
        });
        startCamera();
      } catch (err) {
        console.error("Mediapipe initialization failed:", err);
        setPredictionError("Failed to initialize AI model. Please check your internet connection.");
      }
    };

    const startCamera = async () => {
      try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = cameraStream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            setIsCameraStarted(true);
            setCameraError(null);
            detectFrame();
          };
        }
      } catch (error) {
        console.error("Error accessing camera:", error);
        setCameraError(
          error instanceof Error && error.name === "NotAllowedError"
            ? "Camera access denied. Please enable camera permissions in your browser settings."
            : "Could not access camera. Please ensure it is connected and not in use by another app."
        );
      }
    };

    const detectFrame = async () => {
      if (!videoRef.current || !canvasRef.current || !handLandmarkerRef.current) return;

      const startTimeMs = performance.now();
      
      if (videoRef.current.currentTime !== lastVideoTime) {
        lastVideoTime = videoRef.current.currentTime;
        const results = handLandmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);

        const canvasCtx = canvasRef.current.getContext("2d");
        if (canvasCtx) {
          canvasCtx.save();
          canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          
          canvasCtx.translate(canvasRef.current.width, 0);
          canvasCtx.scale(-1, 1);
          
          canvasCtx.drawImage(
            videoRef.current,
            0,
            0,
            canvasRef.current.width,
            canvasRef.current.height
          );

          if (results.landmarks && results.landmarks.length > 0) {
            const landmarks = results.landmarks[0];

            // Render skeleton manually
            canvasCtx.strokeStyle = "#00FF41";
            canvasCtx.lineWidth = 2;
            HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
                const p1 = landmarks[startIdx];
                const p2 = landmarks[endIdx];
                canvasCtx.beginPath();
                canvasCtx.moveTo(p1.x * canvasRef.current!.width, p1.y * canvasRef.current!.height);
                canvasCtx.lineTo(p2.x * canvasRef.current!.width, p2.y * canvasRef.current!.height);
                canvasCtx.stroke();
            });

            canvasCtx.fillStyle = "#FF0000";
            landmarks.forEach((lm) => {
                canvasCtx.beginPath();
                canvasCtx.arc(lm.x * canvasRef.current!.width, lm.y * canvasRef.current!.height, 3, 0, 2 * Math.PI);
                canvasCtx.fill();
            });

            // Model Prediction
            const base_x = landmarks[0].x;
            const base_y = landmarks[0].y;
            const base_z = landmarks[0].z;

            // MediaPipe tasks-vision uses a slightly different coordinate scale,
            // but it should still be relatively similar. If the accuracy drops, 
            // you might need to retrain, but it's identical mathematically.
            const flatFeatures: number[] = [];
            landmarks.forEach((lm) => {
              flatFeatures.push(lm.x - base_x, lm.y - base_y, lm.z - base_z);
            });

            const now = Date.now();
            if (now - lastPredictTime.current > 200) {
              lastPredictTime.current = now;
              axios.post(BACKEND_URL, { landmarks: flatFeatures })
                 .then(response => {
                   if (response.data.status === "success") {
                     setBackendStatus("online");
                     setPredictionError(null);
                     lastPredictionErrorRef.current = null;
                     setPrediction(response.data.prediction);
                     setConfidence(response.data.confidence);
                   }
                 })
                 .catch((error) => {
                   const detail = axios.isAxiosError(error)
                     ? error.response?.data?.detail ?? error.message
                     : "Unknown prediction error";
                   const message = `Prediction failed: ${detail}`;

                   setBackendStatus("offline");
                   setPrediction(message);
                   setConfidence(0);
                   setPredictionError(message);

                   if (lastPredictionErrorRef.current !== message) {
                     console.error("Prediction error:", detail);
                     lastPredictionErrorRef.current = message;
                   }
                 });
            }
          } else {
             setPredictionError(null);
             lastPredictionErrorRef.current = null;
             setPrediction("No hand detected");
             setConfidence(0);
          }
          canvasCtx.restore();
        }
      }
      animationFrameId = requestAnimationFrame(detectFrame);
    };

    initializeMediaPipe();

    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
      if (handLandmarkerRef.current) {
        handLandmarkerRef.current.close();
      }
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-[#0a0a0a] text-white p-6 space-y-8 tracking-wide">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-5xl font-extrabold tracking-tighter bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent">
          VISION2SENSE
        </h1>
        <p className="text-gray-400 font-medium">Real-time AI Gesture Recognition</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 w-full max-w-6xl">
        {/* Main Feed Container */}
        <div className="lg:col-span-8 relative group">
          <div className="relative rounded-3xl overflow-hidden border-2 border-green-500/20 shadow-[0_0_50px_-12px_rgba(0,255,65,0.2)] bg-black aspect-video">
            <video
              ref={videoRef}
              className="hidden"
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              width={640}
              height={480}
              className="w-full h-full object-cover"
            />
            
            {!isCameraStarted && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/80 backdrop-blur-sm">
                <div className="flex flex-col items-center space-y-4 px-8 text-center text-white">
                  {cameraError ? (
                    <>
                      <div className="h-12 w-12 bg-red-500/20 rounded-full flex items-center justify-center border border-red-500/50">
                        <Activity className="h-6 w-6 text-red-500" />
                      </div>
                      <p className="text-red-400 font-medium max-w-sm">{cameraError}</p>
                      <button 
                        onClick={() => window.location.reload()}
                        className="mt-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-xl text-sm font-bold transition-all border border-white/10"
                      >
                        Try Again
                      </button>
                    </>
                  ) : (
                    <>
                      <div className="animate-spin text-green-500 h-10 w-10 border-4 border-t-transparent rounded-full" />
                      <p className="text-green-500 font-semibold">Initializing Camera & AI...</p>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Float Overlay */}
            <div className="absolute top-4 left-4 flex gap-2">
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full backdrop-blur-md border ${
                backendStatus === "online" ? "bg-green-500/10 border-green-500/50 text-green-400" : "bg-red-500/10 border-red-500/50 text-red-400"
              }`}>
                <div className={`h-2 w-2 rounded-full ${backendStatus === "online" ? "bg-green-500 animate-pulse" : "bg-red-500"}`} />
                <span className="text-xs font-bold uppercase tracking-widest">Backend: {backendStatus}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Analytics Section */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          <div className="bg-neutral-900/50 backdrop-blur-xl border border-white/5 rounded-3xl p-6 shadow-2xl space-y-6">
            <div className="flex items-center gap-3">
              <Activity className="text-green-500 h-5 w-5" />
              <h2 className="text-lg font-bold">Live AI Analysis</h2>
            </div>
            
            <div className="space-y-1">
              <p className="text-gray-500 text-xs font-bold uppercase tracking-widest">Detected Gesture</p>
              <div className="text-4xl font-black text-white truncate h-12">
                {prediction}
              </div>
              {predictionError && (
                <p className="text-sm text-red-400 mt-2">{predictionError}</p>
              )}
            </div>

            <div className="space-y-4">
               <div className="flex justify-between items-end">
                 <p className="text-gray-500 text-xs font-bold uppercase tracking-widest">AI Confidence</p>
                 <span className="text-green-500 font-mono font-bold">{(confidence * 100).toFixed(1)}%</span>
               </div>
               <div className="h-3 bg-neutral-800 rounded-full overflow-hidden">
                 <motion.div 
                   className="h-full bg-gradient-to-r from-green-500 to-emerald-400"
                   initial={{ width: 0 }}
                   animate={{ width: `${confidence * 100}%` }}
                 />
               </div>
            </div>
          </div>

          <div className="bg-neutral-900/50 backdrop-blur-xl border border-white/5 rounded-3xl p-6 shadow-2xl">
             <h3 className="text-sm font-bold text-gray-400 mb-4 uppercase tracking-widest">Supported Gestures</h3>
             <div className="grid grid-cols-2 gap-2">
                {["call", "victory", "good luck", "ok", "rockon", "loser"].map((gesture) => (
                  <div key={gesture} className="flex items-center gap-2 bg-black/40 px-3 py-2 rounded-xl border border-white/5">
                    <CheckCircle2 className="h-4 w-4 text-green-500/50" />
                    <span className="text-xs text-gray-300 font-medium uppercase">{gesture}</span>
                  </div>
                ))}
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HandTracker;
