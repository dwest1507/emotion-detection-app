"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface Probability {
    [key: string]: number;
}

interface PredictionResultProps {
    emotion: string;
    confidence: number;
    probabilities: Probability;
    isUncertain: boolean;
    message?: string;
}

export function PredictionResult({
    emotion,
    confidence,
    probabilities,
    isUncertain,
    message,
}: PredictionResultProps) {
    // Sort probabilities by value descending
    const sortedProbabilities = Object.entries(probabilities)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5); // Show top 5

    return (
        <Card className="w-full max-w-md mx-auto mt-6">
            <CardHeader>
                <CardTitle className="text-center">Detection Result</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
                <div className="text-center space-y-2">
                    <div className="inline-block px-4 py-2 rounded-full bg-primary/10">
                        <h2 className="text-2xl font-bold capitalize text-primary">{emotion}</h2>
                    </div>
                    <p className="text-muted-foreground text-sm">
                        Confidence: {(confidence * 100).toFixed(1)}%
                    </p>
                    {isUncertain && (
                        <div className="p-3 bg-yellow-500/15 text-yellow-600 rounded-md text-sm border border-yellow-500/20">
                            {message || "The model is uncertain about this prediction."}
                        </div>
                    )}
                    {message && !isUncertain && (
                        <div className="p-3 bg-red-500/15 text-red-600 rounded-md text-sm border border-red-500/20">
                            {message}
                        </div>
                    )}
                </div>

                <div className="space-y-3">
                    <h3 className="text-sm font-medium text-muted-foreground">Probability Distribution</h3>
                    {sortedProbabilities.map(([emo, score]) => (
                        <div key={emo} className="space-y-1">
                            <div className="flex justify-between text-xs">
                                <span className="capitalize font-medium">{emo}</span>
                                <span className="text-muted-foreground">{(score * 100).toFixed(1)}%</span>
                            </div>
                            <Progress value={score * 100} className="h-2" />
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    );
}
