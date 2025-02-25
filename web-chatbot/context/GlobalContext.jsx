"use client";

import React, { createContext, useContext, useState } from "react";

const GlobalContext = createContext();

export const GlobalProvider = ({ children }) => {
  const [selectedAvatar, setSelectedAvatar] = useState(1);

  return (
    <GlobalContext.Provider value={{ selectedAvatar, setSelectedAvatar }}>
      {children}
    </GlobalContext.Provider>
  );
};

export const useGlobalState = () => useContext(GlobalContext);
