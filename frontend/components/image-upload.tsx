"use client";

import { useState, useRef } from "react";
import Image from "next/image";
import { Upload, X, FileImage } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface ImageUploadProps {
    onImageSelect: (file: File | null) => void;
    isLoading?: boolean;
}

export function ImageUpload({ onImageSelect, isLoading = false }: ImageUploadProps) {
    const [dragActive, setDragActive] = useState(false);
    const [preview, setPreview] = useState<string | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    };

    const handleFile = (file: File) => {
        if (file.type.split("/")[0] !== "image") {
            alert("Please upload an image file");
            return;
        }

        // Create preview
        const objectUrl = URL.createObjectURL(file);
        setPreview(objectUrl);
        onImageSelect(file);
    };

    const clearImage = (e: React.MouseEvent) => {
        e.stopPropagation();
        setPreview(null);
        onImageSelect(null);
        if (inputRef.current) {
            inputRef.current.value = "";
        }
    };

    return (
        <Card className="w-full max-w-md mx-auto">
            <CardContent className="p-6">
                <form
                    className="relative flex flex-col items-center justify-center w-full"
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => inputRef.current?.click()}
                >
                    <input
                        ref={inputRef}
                        className="hidden"
                        type="file"
                        accept="image/*"
                        onChange={handleChange}
                        disabled={isLoading}
                    />

                    {preview ? (
                        <div className="relative w-full aspect-square md:aspect-[4/3] rounded-lg overflow-hidden bg-muted">
                            <Image
                                src={preview}
                                alt="Selected image"
                                fill
                                className="object-contain"
                            />
                            {!isLoading && (
                                <Button
                                    variant="destructive"
                                    size="icon"
                                    className="absolute top-2 right-2 rounded-full h-8 w-8"
                                    onClick={clearImage}
                                >
                                    <X className="h-4 w-4" />
                                </Button>
                            )}
                        </div>
                    ) : (
                        <div
                            className={cn(
                                "flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer transition-colors",
                                dragActive
                                    ? "border-primary bg-primary/10"
                                    : "border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/50",
                                isLoading && "opacity-50 cursor-not-allowed"
                            )}
                        >
                            <div className="flex flex-col items-center justify-center pt-5 pb-6">
                                <div className="bg-primary/10 p-4 rounded-full mb-4">
                                    <Upload className="w-8 h-8 text-primary" />
                                </div>
                                <p className="mb-2 text-sm font-medium text-foreground">
                                    <span className="font-semibold">Click to upload</span> or drag and drop
                                </p>
                                <p className="text-xs text-muted-foreground">
                                    JPG, PNG or WEBP (MAX. 5MB)
                                </p>
                            </div>
                        </div>
                    )}
                </form>
            </CardContent>
        </Card>
    );
}
