'use client'
import { useState } from "react";
import Image from "next/image";
import React from "react";

interface ButtonProps {
  children: React.ReactNode;
  onClick: () => void;
  disabled: boolean;
}

const Button = ({ children, onClick, disabled }: ButtonProps) => {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="px-4 py-2 font-semibold text-white bg-blue-500 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-75"
    >
      {children}
    </button>
  );
};

interface Prediction {
  class: string;
  confidence: number;
}

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files[0]) {
      const file = files[0];
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setPredictions([]);
    }
  };

  const handleUpload = async () => {
    if (!image) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch("http://localhost:8000/predict/", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setPredictions(data.predictions);
    } catch (error) {
      console.error("Error uploading image:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-2xl font-bold mb-4">Image Classification</h1>
      <input type="file" accept="image/*" onChange={handleImageChange} className="mb-4" />
      {preview && <Image src={preview} alt="Preview" width={200} height={200} className="rounded-lg mb-4" />}
      <Button onClick={handleUpload} disabled={!image || loading}>
        {loading ? "Processing..." : "Classify Image"}
      </Button>
      {predictions.length > 0 && (
        <div className="mt-4 p-4 border rounded-lg w-full max-w-md">
          <h2 className="font-semibold mb-2">Predictions:</h2>
          <ul className="list-disc list-inside">
            {predictions.map((prediction, index) => (
              <li key={index} className="text-gray-700">
                {prediction.class} - {Math.round(prediction.confidence * 100)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
