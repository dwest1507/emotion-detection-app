"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress"; // Assuming shadcn/ui progress exists, or we use standard HTML progress
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis, Tooltip, Cell } from "recharts";

interface PredictionResultProps {
    emotion?: string;
    confidence?: number;
    probabilities?: Record<string, number>;
    isUncertain?: boolean;
    message?: string;
}

const EMOTION_COLORS: Record<string, string> = {
    angry: "#ef4444", // red-500
    disgust: "#84cc16", // lime-500
    fear: "#a855f7", // purple-500
    happy: "#eab308", // yellow-500
    sad: "#3b82f6", // blue-500
    surprise: "#f97316", // orange-500
    neutral: "#78716c", // stone-500
};

export function PredictionResult({
    emotion,
    confidence,
    probabilities,
    isUncertain,
    message,
}: PredictionResultProps) {
    if (!emotion || !probabilities) return null;

    // Transform probabilities object to array for Recharts
    const data = Object.entries(probabilities)
        .map(([name, value]) => ({
            name: name.charAt(0).toUpperCase() + name.slice(1),
            value: (value * 100).toFixed(1),
            raw: value,
            fill: EMOTION_COLORS[name.toLowerCase()] || "#8884d8",
        }))
        .sort((a, b) => b.raw - a.raw); // Sort by highest probability

    const primaryColor = EMOTION_COLORS[emotion.toLowerCase()] || "#8884d8";

    return (
        <div className="w-full max-w-2xl animate-in fade-in slide-in-from-bottom-4 duration-700">
            <Card className="border-2 overflow-hidden bg-card/50 backdrop-blur-sm" style={{ borderColor: isUncertain ? "#f59e0b" : primaryColor }}>
                <CardHeader className="bg-muted/50 pb-8">
                    <CardTitle className="text-center space-y-2">
                        <span className="text-sm font-normal text-muted-foreground uppercase tracking-widest">Detected Emotion</span>
                        <div className="flex items-center justify-center gap-3">
                            <h2 className="text-4xl md:text-5xl font-black tracking-tighter capitalize" style={{ color: primaryColor }}>
                                {emotion}
                            </h2>
                            <span className="text-4xl">
                                {emotion === "happy" && "😊"}
                                {emotion === "sad" && "😢"}
                                {emotion === "angry" && "😠"}
                                {emotion === "surprise" && "😮"}
                                {emotion === "fear" && "😱"}
                                {emotion === "disgust" && "🤢"}
                                {emotion === "neutral" && "😐"}
                            </span>
                        </div>
                        {confidence && (
                            <div className="flex items-center justify-center gap-2 mt-2">
                                <span className={`text-sm font-semibold px-3 py-1 rounded-full ${isUncertain
                                        ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400"
                                        : "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                                    }`}>
                                    {(confidence * 100).toFixed(1)}% Confidence
                                </span>
                            </div>
                        )}
                        {message && (
                            <p className="text-sm text-muted-foreground mt-2 font-medium">
                                {message}
                            </p>
                        )}
                    </CardTitle>
                </CardHeader>
                <CardContent className="pt-6">
                    <div className="h-[300px] w-full mt-4">
                        <h3 className="text-sm font-semibold mb-4 text-center text-muted-foreground">Probability Distribution</h3>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                                <XAxis type="number" domain={[0, 100]} hide />
                                <YAxis
                                    dataKey="name"
                                    type="category"
                                    width={80}
                                    tick={{ fontSize: 12 }}
                                    axisLine={false}
                                    tickLine={false}
                                />
                                <Tooltip
                                    cursor={{ fill: 'transparent' }}
                                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                                    formatter={(value: any) => [`${value}%`, 'Probability']}
                                />
                                <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20} animationDuration={1000}>
                                    {data.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.fill} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
