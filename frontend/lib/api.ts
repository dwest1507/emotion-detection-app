import axios from "axios";

// Default to localhost for development, or use environment variable
const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export interface PredictionResponse {
    success: boolean;
    emotion: string;
    confidence: number;
    is_uncertain: boolean;
    probabilities: { [key: string]: number };
    inference_time_ms: number;
    message?: string;
    error?: string;
}

export async function predictEmotion(file: File): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await axios.post(`${API_URL}/predict`, formData, {
            headers: {
                "Content-Type": "multipart/form-data",
            },
        });
        return response.data;
    } catch (error) {
        if (axios.isAxiosError(error) && error.response) {
            // Backend returned an error response
            throw new Error(error.response.data.message || error.response.data.error || "Failed to analyze image");
        }
        throw new Error("Network error or server unavailable");
    }
}
