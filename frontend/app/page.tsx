"use client";

import { useState } from "react";
import { ImageUpload } from "@/components/image-upload";
import { PredictionResult } from "@/components/prediction-result";
import { predictEmotion, PredictionResponse } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelect = async (file: File | null) => {
    if (!file) {
      setResult(null);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    // Initial check (optional, but good for immediate feedback)
    if (!file.type.startsWith("image/")) {
      setError("Please upload a valid image file.");
      setLoading(false);
      return;
    }

    try {
      const data = await predictEmotion(file);
      if (!data.success) {
        throw new Error(data.error || "Failed to analyze image");
      }
      setResult(data);
    } catch (err) {
      console.error("Prediction error:", err);
      setError(err instanceof Error ? err.message : "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center space-y-8 text-center pt-10 pb-20">
      <div className="space-y-4">
        <h1 className="text-4xl font-extrabold tracking-tight lg:text-5xl bg-clip-text text-transparent bg-gradient-to-r from-primary to-blue-600">
          Emotion Detection
        </h1>
        <p className="text-xl text-muted-foreground w-full max-w-2xl mx-auto">
          Upload an image to detect the emotion of the person in the photo using our AI model.
        </p>
      </div>

      <div className="w-full max-w-4xl px-4 flex flex-col items-center gap-8">
        <ImageUpload onImageSelect={handleImageSelect} isLoading={loading} />

        {error && (
          <Alert variant="destructive" className="max-w-md mx-auto text-left animate-in fade-in slide-in-from-top-2">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {loading && (
          <div className="w-full max-w-md space-y-4 animate-in fade-in">
            <div className="space-y-2">
              <div className="h-2 bg-secondary rounded-full overflow-hidden">
                <div className="h-full bg-primary animate-progress origin-left w-full"></div>
              </div>
              <p className="text-sm text-muted-foreground animate-pulse font-medium">Analyzing facial expressions...</p>
            </div>
          </div>
        )}

        {result && (
          <PredictionResult
            emotion={result.emotion}
            confidence={result.confidence}
            probabilities={result.probabilities}
            isUncertain={result.is_uncertain}
            message={result.message}
          />
        )}
      </div>

      {/* Sample Images Section */}
      <div className="w-full max-w-4xl px-4 mt-8">
        <h3 className="text-lg font-semibold mb-4 text-center">Don&apos;t have a photo? Try one of ours:</h3>
        <div className="flex justify-center gap-4">
          <button
            onClick={async () => {
              const response = await fetch("/samples/happy.png");
              const blob = await response.blob();
              const file = new File([blob], "happy_sample.png", { type: "image/png" });
              handleImageSelect(file);
            }}
            disabled={loading}
            className="group relative overflow-hidden rounded-lg border-2 border-transparent hover:border-primary transition-all focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="/samples/happy.png"
              alt="Happy Sample"
              className="w-24 h-24 object-cover group-hover:scale-110 transition-transform duration-300"
            />
            <div className="absolute inset-0 bg-black/20 group-hover:bg-transparent transition-colors" />
          </button>
        </div>
      </div>

      {/* How it Works Section */}
      <div className="w-full max-w-5xl px-6 mt-16 grid md:grid-cols-3 gap-8 text-left">
        <div className="bg-card border rounded-xl p-6 shadow-sm">
          <div className="h-10 w-10 bg-primary/10 text-primary rounded-full flex items-center justify-center mb-4 text-xl">📸</div>
          <h3 className="font-semibold text-lg mb-2">1. Upload Photo</h3>
          <p className="text-muted-foreground">Select a clear photo of a face. We support JPEG and PNG formats.</p>
        </div>
        <div className="bg-card border rounded-xl p-6 shadow-sm">
          <div className="h-10 w-10 bg-primary/10 text-primary rounded-full flex items-center justify-center mb-4 text-xl">🧠</div>
          <h3 className="font-semibold text-lg mb-2">2. AI Analysis</h3>
          <p className="text-muted-foreground">Our computer vision model (EfficientNet) detects the face and analyzes micro-expressions.</p>
        </div>
        <div className="bg-card border rounded-xl p-6 shadow-sm">
          <div className="h-10 w-10 bg-primary/10 text-primary rounded-full flex items-center justify-center mb-4 text-xl">📊</div>
          <h3 className="font-semibold text-lg mb-2">3. View Results</h3>
          <p className="text-muted-foreground">Get instant feedback on the detected emotion with detailed confidence scores.</p>
        </div>
      </div>

      {/* Privacy Note */}
      <div className="mt-12 text-sm text-muted-foreground max-w-md mx-auto bg-muted/30 p-4 rounded-lg border border-border/50">
        <p className="flex items-center justify-center gap-2">
          <span className="text-lg">🔒</span>
          <span><strong>Privacy First:</strong> Your images are processed in real-time and deleted immediately. We do not store any photos.</span>
        </p>
      </div>

    </div>
  );
}
