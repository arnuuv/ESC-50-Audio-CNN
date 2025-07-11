import Link from "next/link";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-stone-50 p-8">
      <div className = 'mx-auto max-w-[60%] bg-red-400'>
        <div className ="mb-12 text-center">
          <h1 className = "mb-4 text-4xl font-light tracking-tight text-stone-900">CNN Audio Visualizer</h1>
          <p className="text-md mb-8 text-stone-600">
            Upload WAV file to visualize the audio through a CNN.
          </p>
          <div className = "flex flex-col items-center">
            <div className = "relative inline-block">
              <input type = "file" accept=".wav" id="file-upload" className ="absolute inset-0 w-full cursor-pointer opacity-0"
            </div>
          </div>
        </div>
      </div>
      
    </main>
  );
}
