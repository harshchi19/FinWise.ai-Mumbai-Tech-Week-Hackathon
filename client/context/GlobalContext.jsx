"use client";

import React, { createContext, useContext, useState } from "react";

const GlobalContext = createContext(undefined);

export const GlobalProvider = ({ children }) => {
  const [sidebarState, setSidebarState] = useState(true);
  const [loading, setLoading] = useState(false);

  return (
    <GlobalContext.Provider
      value={{ sidebarState, setSidebarState, loading, setLoading }}
    >
      {children}
    </GlobalContext.Provider>
  );
};

export const useGlobalState = () => {
  const context = useContext(GlobalContext);

  if (context === undefined) {
    throw new Error("useGlobalState must be used within a GlobalProvider");
  }

  return context;
};
