// page.jsx
"use client";

import { ChatInterface } from "@/components/ChatInterface";
import Navbar from "@/components/Navbar";
import Placeholder from "@/components/Placeholder";
import { TypingBox } from "@/components/TypingBox";
import { useGlobalState } from "@/context/GlobalContext";
import { useState, useEffect } from "react";

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false); // Renamed
  const { selectedAvatar } = useGlobalState();

  useEffect(() => {
    setIsGenerating(loading); // Sync isGenerating with loading state
  }, [loading]);

  return (
    <div className="h-screen w-screen relative bg-[#232A34] flex items-center justify-center overflow-hidden px-2">
      <Navbar />
      <div
        className={`transition-opacity duration-500 ease-in-out ${
          messages.length > 0 ? "hidden" : "block"
        }`}
      >
        <Placeholder />
      </div>
      <div
        className={`transition-opacity duration-500 ease-in-out ${
          messages.length > 0 ? "block" : "hidden"
        }`}
      >
        <ChatInterface
          messages={messages}
          selectedAvatar={selectedAvatar}
          isGenerating={isGenerating}
        />
      </div>
      <TypingBox
        setMessages={setMessages}
        loading={loading}
        setLoading={setLoading}
      />

      <div className="absolute bottom-1 max-md:hidden text-slate-500 tracking-wide">
        <h1>
          finwise.ai may contain errors. We recommend checking important
          information
        </h1>
      </div>
    </div>
  );
}
