import { useRef, useState } from "react";
import {
  Send,
  Mic,
  LoaderPinwheel,
  FileUp,
  X,
  File as FileIcon,
} from "lucide-react";

export const TypingBox = ({ setMessages, loading, setLoading }) => {
  const [question, setQuestion] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);
  const recognitionRef = useRef(null);

  const initializeSpeechRecognition = () => {
    if (!("webkitSpeechRecognition" in window)) {
      alert("Speech recognition is not supported in this browser.");
      return;
    }

    try {
      const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();

      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = "en-US";

      recognition.onstart = () => setIsListening(true);
      recognition.onend = () => setIsListening(false);

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setQuestion(transcript);
        inputRef.current?.focus();
      };

      recognitionRef.current = recognition;
    } catch (error) {
      console.error("Error initializing speech recognition:", error);
      alert("Could not initialize speech recognition");
    }
  };

  const handleMicClick = () => {
    if (!recognitionRef.current) {
      initializeSpeechRecognition();
    }

    if (recognitionRef.current) {
      try {
        if (isListening) {
          recognitionRef.current.stop();
        } else {
          recognitionRef.current.start();
        }
      } catch (error) {
        console.error("Error with speech recognition:", error);
        setIsListening(false);
      }
    }
  };

  const handleSendMessage = async () => {
    if ((!question.trim() && !selectedFile) || loading) return;

    const userMessage = {
      sender: "user",
      type: selectedFile ? "pdf" : "text",
      text: question.trim(),
      ...(selectedFile && {
        pdfName: selectedFile.name,
      }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setLoading(true); // Start loading before API call

    const formData = new FormData();
    formData.append("user_input", question.trim());

    if (selectedFile) {
      formData.append("pdf_files", selectedFile);
    }

    try {
      const response = await fetch("http://127.0.0.1:8000/rag-chatbot", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { sender: "bot", type: "text", text: data.response },
      ]);
    } catch (error) {
      console.error("Error sending message:", error);
      alert("Failed to send message. Please try again.");
    } finally {
      setQuestion("");
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      inputRef.current?.focus();
      setLoading(false); // End loading
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      const file = e.target.files[0];
      if (file.type === "application/pdf") {
        setSelectedFile(file);
      } else {
        alert("Please select a PDF file");
        e.target.value = null;
      }
    }
  };

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  const removeSelectedFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div
      className="z-10 max-w-[700px] w-full max-md:bottom-4 absolute bottom-10 flex flex-col
    bg-gradient-to-tr from-purple-200/40 via-purple-300/50 to-purple-400/50
    p-4 backdrop-blur-md rounded-xl border border-purple-200/90 shadow-lg"
    >
      {selectedFile && (
        <div className="mb-4 p-2 px-4 bg-purple-200/20 rounded-lg flex items-center justify-between border">
          <div className="flex items-center">
            <FileIcon className="w-5 h-5 mr-3 text-white" />
            <span className="text-white text-md truncate max-w-[450px]">
              {selectedFile.name}
            </span>
          </div>
          <button
            onClick={removeSelectedFile}
            className="text-white hover:text-red-200 p-1"
            aria-label="Remove file"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      <div className="flex items-center gap-3 w-full">
        <input
          type="file"
          accept="application/pdf"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="hidden"
        />

        <button
          onClick={triggerFileUpload}
          className="bg-slate-100/20 hover:bg-slate-100/40 p-2 rounded-full text-white transition-all"
          aria-label="Upload PDF"
          disabled={loading}
        >
          <FileUp className="w-5 h-5" />
        </button>

        <input
          ref={inputRef}
          className="w-full flex-grow outline-none bg-transparent py-2 rounded-full text-white placeholder:text-white/80 placeholder:font-semibold transition-all duration-300"
          placeholder="Write your message here..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === "Enter") handleSendMessage();
          }}
          disabled={loading}
        />

        <button
          onClick={handleSendMessage}
          className={`bg-slate-100/20 hover:bg-slate-100/40 p-2 px-4 rounded-full text-white transition-all flex items-center gap-2 ${
            loading || (!question.trim() && !selectedFile)
              ? "cursor-not-allowed opacity-50"
              : ""
          }`}
          disabled={loading || (!question.trim() && !selectedFile)}
        >
          {loading ? (
            <>
              <LoaderPinwheel className="w-4 h-4 animate-spin" />
              Sending...
            </>
          ) : (
            <>
              <Send className="w-4 h-4" />
              Send
            </>
          )}
        </button>

        <button
          className={`p-2 rounded-full transition-all ${
            isListening
              ? "bg-green-500"
              : "bg-slate-100/20 hover:bg-slate-100/40"
          } text-white`}
          aria-label="Use microphone"
          disabled={loading}
          onClick={handleMicClick}
        >
          <Mic className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
};
