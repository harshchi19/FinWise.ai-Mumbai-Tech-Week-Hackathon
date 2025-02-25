import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { File as FileIcon, Volume2, VolumeX } from "lucide-react";

const parseLinksAndBoldText = (text) => {
  const linkRegex = /\[(.*?)\]\((.*?)\)/g;
  const boldRegex = /\*\*(.*?)\*\*/g;
  let match;
  const parts = [];
  let lastIndex = 0;

  while ((match = boldRegex.exec(text)) !== null) {
    const [fullMatch, boldText] = match;
    const startIndex = match.index;

    if (startIndex > lastIndex) {
      parts.push(text.substring(lastIndex, startIndex));
    }

    parts.push({ type: "bold", text: boldText });
    lastIndex = startIndex + fullMatch.length;
  }

  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex));
  }

  const finalParts = [];
  parts.forEach((part) => {
    if (typeof part === "string") {
      let linkMatch;
      let linkLastIndex = 0;
      while ((linkMatch = linkRegex.exec(part)) !== null) {
        const [fullLinkMatch, linkText, linkUrl] = linkMatch;
        const linkStartIndex = linkMatch.index;

        if (linkStartIndex > linkLastIndex) {
          finalParts.push(part.substring(linkLastIndex, linkStartIndex));
        }

        finalParts.push({ type: "link", text: linkText, url: linkUrl });
        linkLastIndex = linkStartIndex + fullLinkMatch.length;
      }

      if (linkLastIndex < part.length) {
        finalParts.push(part.substring(linkLastIndex));
      }
    } else {
      finalParts.push(part);
    }
  });

  return finalParts;
};

export const ChatInterface = ({ messages, selectedAvatar, isGenerating }) => {
  const [typedMessages, setTypedMessages] = useState([]);
  const messagesEndRef = useRef(null);
  const [typingText, setTypingText] = useState("");
  const [currentMessageIndex, setCurrentMessageIndex] = useState(-1);
  const [currentSpeakingIndex, setCurrentSpeakingIndex] = useState(null);
  const synthRef = useRef(null);
  const utterThisRef = useRef(null);

  // Initialize speech synthesis on component mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      synthRef.current = window.speechSynthesis;
    }

    // Clean up on unmount
    return () => {
      if (synthRef.current && utterThisRef.current) {
        synthRef.current.cancel();
      }
    };
  }, []);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  useEffect(() => {
    if (messages.length > 0) {
      setTypedMessages([...messages]);
    }
  }, [messages]);

  useEffect(() => {
    if (messages.length > 0 && currentMessageIndex < messages.length - 1) {
      setCurrentMessageIndex(messages.length - 1);
      setTypingText("");
    }
  }, [messages, currentMessageIndex]);

  useEffect(() => {
    let typingInterval;

    if (currentMessageIndex !== -1 && currentMessageIndex < messages.length) {
      const message = messages[currentMessageIndex];
      const fullText = message.text;

      let currentIndex = 0;
      typingInterval = setInterval(() => {
        if (currentIndex <= fullText.length) {
          setTypingText(fullText.substring(0, currentIndex));
          currentIndex++;
          scrollToBottom();
        } else {
          clearInterval(typingInterval);
        }
      }, 0);
    }

    return () => clearInterval(typingInterval);
  }, [currentMessageIndex, messages, scrollToBottom]);

  const speak = useCallback((text, messageIndex) => {
    // Ensure speech synthesis is available
    if (!synthRef.current) {
      console.error("Speech synthesis not available");
      return;
    }

    // Cancel any ongoing speech
    if (synthRef.current.speaking) {
      synthRef.current.cancel();
    }

    utterThisRef.current = new SpeechSynthesisUtterance(text);

    utterThisRef.current.onstart = () => {
      setCurrentSpeakingIndex(messageIndex);
    };

    utterThisRef.current.onend = () => {
      setCurrentSpeakingIndex(null);
    };

    utterThisRef.current.onerror = (event) => {
      console.error("Speech synthesis error:", event);
      setCurrentSpeakingIndex(null);
    };

    // Get available voices and set a better one if possible
    const voices = synthRef.current.getVoices();
    if (voices.length > 0) {
      // Try to find a good English voice
      const preferredVoice = voices.find(
        (voice) =>
          voice.lang.includes("en") &&
          (voice.name.includes("Google") || voice.name.includes("Premium"))
      );
      if (preferredVoice) {
        utterThisRef.current.voice = preferredVoice;
      }
    }

    // Adjust speech parameters for better clarity
    utterThisRef.current.rate = 1.0; // Normal speed
    utterThisRef.current.pitch = 1.0; // Normal pitch
    utterThisRef.current.volume = 1.0; // Full volume

    synthRef.current.speak(utterThisRef.current);
  }, []);

  const stopSpeaking = useCallback(() => {
    if (synthRef.current) {
      synthRef.current.cancel();
      setCurrentSpeakingIndex(null);
    }
  }, []);

  const renderMessageContent = (msg) => {
    if (msg.type === "pdf" && msg.pdfData) {
      return (
        <div className="flex flex-col">
          <div className="flex items-center mb-2">
            <FileIcon className="mr-2 text-white" size={20} />
            <span className="font-medium">{msg.pdfName || "PDF Document"}</span>
          </div>
          <div className="border border-white/30 rounded-md overflow-hidden bg-gray-700/50">
            <iframe
              src={msg.pdfData}
              className="w-full h-64"
              title={msg.pdfName || "PDF Document"}
            />
          </div>
          <a
            href={msg.pdfData}
            download={msg.pdfName || "document.pdf"}
            className="mt-2 text-blue-200 hover:text-blue-100 underline text-sm self-end"
            target="_blank"
            rel="noopener noreferrer"
          >
            Open PDF
          </a>
        </div>
      );
    }

    const parts = parseLinksAndBoldText(msg.text);

    return (
      <>
        {parts.map((part, index) => {
          if (typeof part === "string") {
            return part;
          } else if (part.type === "link") {
            return (
              <a
                key={`link-${index}`}
                href={part.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-300 hover:text-blue-100 underline"
              >
                {part.text}
              </a>
            );
          } else if (part.type === "bold") {
            return <strong key={`bold-${index}`}>{part.text}</strong>;
          }
          return null;
        })}
      </>
    );
  };

  const getMessageContent = (msg, index) => {
    if (msg.sender === "user") {
      return renderMessageContent(msg);
    } else {
      if (index === currentMessageIndex) {
        return parseLinksAndBoldText(typingText).map((part, idx) => {
          if (typeof part === "string") {
            return part;
          } else if (part.type === "link") {
            return (
              <a
                key={`link-${idx}`}
                href={part.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-300 hover:text-blue-100 underline"
              >
                {part.text}
              </a>
            );
          } else if (part.type === "bold") {
            return <strong key={`bold-${idx}`}>{part.text}</strong>;
          }
          return null;
        });
      } else {
        return renderMessageContent(msg);
      }
    }
  };

  return (
    <div className="z-0 w-[700px] h-[70vh] overflow-y-auto flex flex-col gap-4 p-4 max-md:w-full">
      {messages.length === 0 ? (
        <div className="text-white/50 text-center mt-20">
          <p>No messages yet. Start the conversation!</p>
        </div>
      ) : (
        <>
          {typedMessages.map((msg, index) => (
            <motion.div
              key={`msg-${index}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className={`p-3 rounded-lg max-w-[75%] break-words shadow-md ${
                msg.sender === "user"
                  ? "bg-gray-500 self-end text-white border border-purple-100/50"
                  : "bg-purple-500/70 self-start text-white border border-purple-200/50"
              } relative`}
            >
              <div className="pr-8">
                {" "}
                {/* Add padding to make room for speech button */}
                {getMessageContent(msg, index)}
              </div>

              {msg.sender !== "user" && (
                <div className="absolute bottom-2 right-2">
                  {currentSpeakingIndex === index ? (
                    <button
                      onClick={stopSpeaking}
                      className="p-1 rounded-full bg-white/20 hover:bg-white/30 transition-colors text-white"
                      aria-label="Stop speaking"
                      title="Stop speaking"
                    >
                      <VolumeX size={16} />
                    </button>
                  ) : (
                    <button
                      onClick={() => speak(msg.text, index)}
                      className="p-1 rounded-full bg-white/10 hover:bg-white/20 transition-colors text-white"
                      aria-label="Speak message"
                      title="Speak message"
                    >
                      <Volume2 size={16} />
                    </button>
                  )}
                </div>
              )}
            </motion.div>
          ))}

          <AnimatePresence>
            {isGenerating && (
              <motion.div
                key="thinking-indicator"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.3 }}
                className="p-3 rounded-lg max-w-[75%] break-words self-start text-white flex items-center bg-purple-500/70"
              >
                <div className="flex items-center">
                  <span className="font-medium">Thinking</span>
                  <div className="ml-1 flex">
                    {[0, 1, 2].map((i) => (
                      <motion.span
                        key={`dot-${i}`}
                        initial={{ opacity: 0.3 }}
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{
                          duration: 1.2,
                          repeat: Infinity,
                          delay: i * 0.2,
                          ease: "easeInOut",
                        }}
                        className="text-lg mx-[1px]"
                      >
                        .
                      </motion.span>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}
      <div ref={messagesEndRef} />
    </div>
  );
};
