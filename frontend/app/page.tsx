export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      <div className="z-10 w-full max-w-5xl items-center justify-center font-mono text-sm">
        <div className="text-center space-y-6">
          <h1 className="text-4xl font-bold tracking-tight sm:text-6xl">
            🎭 Emotion Detection
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            AI-powered facial emotion recognition using deep learning. 
            Upload an image to detect emotions with confidence scores and probability distributions.
          </p>
          
          <div className="mt-12 p-8 border-2 border-dashed border-border rounded-lg bg-muted/50">
            <p className="text-muted-foreground">
              Upload zone will be implemented here
            </p>
          </div>
        </div>
      </div>
    </main>
  )
}

