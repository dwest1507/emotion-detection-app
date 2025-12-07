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
          <Alert variant="destructive" className="max-w-md mx-auto text-left">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {loading && (
          <div className="w-full max-w-md space-y-2">
            <div className="h-2 bg-secondary rounded overflow-hidden">
              <div className="h-full bg-primary animate-pulse w-full origin-left-right"></div>
            </div>
            {/* Simple indeterminate progress bar effect since we don't have CSS keyframes defined for 'progress' yet, using pulse */}
            <p className="text-sm text-muted-foreground animate-pulse">Analyzing image...</p>
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
    </div>
  );
}
