"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FaUserCircle } from "react-icons/fa";
import { useUser } from "@clerk/nextjs";
import { Input } from "../ui/input";
import { IoIosSend } from "react-icons/io";
import Image from "next/image";
import { Logo } from "@/public/images";
import { BsThreeDotsVertical, BsSearch } from "react-icons/bs";
import { Badge } from "@/components/ui/badge";

const ChatComponent = ({ messages, onSendMessage }) => {
  const [newMessage, setNewMessage] = useState("");
  const messagesEndRef = useRef(null);
  const { user } = useUser();

  const currentUserId = user?.publicMetadata?.userId;
  const currentUserName = user?.username;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = () => {
    if (!newMessage.trim()) return;
    onSendMessage(newMessage); // Pass the new message to the parent
    setNewMessage("");
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      sendMessage();
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className="flex flex-col h-full bg-white border rounded-xl shadow-xl overflow-hidden"
    >
      <div className="px-4 py-3 flex justify-between items-center font-bold text-lg text-center border-b border-purple-100">
        <Image src={Logo} alt="Logo" className="h-8 w-auto" />

        <span className="flex-center gap-x-5">
          <BsSearch className="h-5 w-auto hover:cursor-pointer" />
          <BsThreeDotsVertical className="h-5 w-auto hover:cursor-pointer" />
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
        <AnimatePresence>
          {messages.map((msg, index) => (
            <motion.div
              key={index} // Use index as key since we are passing down the keys
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={`flex items-start space-x-2 ${
                msg.type === "user" ? "flex-row-reverse space-x-reverse" : ""
              }`}
            >
              <FaUserCircle
                className={`h-8 w-8 ${
                  msg.type === "user" ? "text-limeGreen-600" : "text-purple-600"
                }`}
              />
              <motion.div
                whileHover={{ scale: 1.02 }}
                className={`max-w-[80%] p-3 rounded-xl shadow-md border ${
                  msg.type === "user"
                    ? "bg-white text-limeGreen-900 border-limeGreen-100"
                    : "bg-purple-100 text-purple-900 border-purple-100"
                }`}
              >
                <div className="flex justify-between items-center">
                  <div className="font-bold text-md mb-1">
                    {msg.type === "user" ? currentUserName : "Chatbot"}
                  </div>
                  {msg.type === "chatbot" && (
                    <Badge variant="outline">Chatbot</Badge> // Unique badge for identification
                  )}
                </div>
                <h6 className="text-sm font-medium">{msg.text}</h6>
              </motion.div>
            </motion.div>
          ))}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 bg-white border-t border-purple-100 flex items-center space-x-2">
        <Input
          type="text"
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          className="h-10 flex-1 p-3 bg-white text-purple-900 border border-purple-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-300"
          placeholder="Type a message..."
        />
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={sendMessage}
          className="p-1.5 border border-purple-200 text-purple-500 hover:text-white rounded-lg hover:bg-purple-500 transition-colors"
        >
          <IoIosSend className="h-6 w-auto" />
        </motion.button>
      </div>
    </motion.div>
  );
};

export default ChatComponent;
