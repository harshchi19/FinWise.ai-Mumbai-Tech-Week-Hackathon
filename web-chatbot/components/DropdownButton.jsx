"use client";

import React, { useState, useRef, useEffect } from "react";
import Image from "next/image";
import { motion, AnimatePresence } from "framer-motion";
import { useGlobalState } from "@/context/GlobalContext";
import { Gemini } from "@/assets/images";

const avatars = [
  {
    id: 1,
    name: "Gemini Pro",
    image: Gemini,
    color: "bg-gradient-to-br from-indigo-700 to-purple-900",
    hoverColor:
      "hover:bg-gradient-to-br hover:from-indigo-800 hover:to-purple-950",
  },
  {
    id: 2,
    name: "Gemini 1.5 Flash",
    image: Gemini,
    color: "bg-gradient-to-br from-violet-600 to-fuchsia-800",
    hoverColor:
      "hover:bg-gradient-to-br hover:from-violet-700 hover:to-fuchsia-900",
  },
  {
    id: 3,
    name: "Gemini 2.0 Flash",
    image: Gemini,
    color: "bg-gradient-to-br from-blue-800 to-black",
    hoverColor: "hover:bg-gradient-to-br hover:from-blue-900 hover:to-gray-900",
  },
];

export const DropdownButton = () => {
  const { selectedAvatar, setSelectedAvatar } = useGlobalState();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  const selectedAvatarObject = avatars.find(
    (avatar) => avatar.id === selectedAvatar
  );

  const toggleDropdown = () => setIsDropdownOpen(!isDropdownOpen);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleAvatarSelect = (avatarId) => {
    setSelectedAvatar(avatarId);
    setIsDropdownOpen(false);
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <motion.div
        onClick={toggleDropdown}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        className={`w-12 h-12 rounded-full flex items-center justify-center cursor-pointer 
        ${selectedAvatarObject?.color} 
        shadow-lg transition-all duration-300 ease-in-out ring-2 ring-white/20`}
      >
        <Image
          src={selectedAvatarObject?.image}
          alt={selectedAvatarObject?.name}
          className="w-10 h-10 rounded-full object-cover border border-white/30"
        />
      </motion.div>

      <AnimatePresence>
        {isDropdownOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: -10 }}
            transition={{ duration: 0.25 }}
            className="absolute top-full mt-3 right-0 w-56 bg-gradient-to-br from-gray-900 to-black 
            backdrop-blur-lg border border-gray-700/50 rounded-xl shadow-2xl z-50 overflow-hidden 
            ring-2 ring-blue-500/30"
          >
            {avatars.map((avatar) => (
              <motion.div
                key={avatar.id}
                onClick={() => handleAvatarSelect(avatar.id)}
                whileHover={{
                  backgroundColor: "rgba(255, 255, 255, 0.1)",
                  scale: 1.02,
                }}
                className={`flex items-center space-x-4 px-4 py-3 cursor-pointer 
                transition-colors duration-200 
                ${avatar.hoverColor}`}
              >
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center 
                  ${avatar.color} shadow-md ring-1 ring-white/10`}
                >
                  <Image
                    src={avatar.image}
                    alt={avatar.name}
                    className="w-auto h-10 rounded-full object-cover border border-white/30"
                  />
                </div>
                <span className="text-white font-semibold text-md tracking-wide">
                  {avatar.name}
                </span>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
