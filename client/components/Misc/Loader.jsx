"use client";
import React from "react";
import { MultiStepLoader } from "../ui/multi-step-loader";

const loadingStates = [
  {
    text: "Loading vital components",
  },
  {
    text: "Configuring UI",
  },
  {
    text: "Fetching initial data",
  },
  {
    text: "Preparing user interface",
  },
  {
    text: "Finalizing setup",
  },
];

export function Loader() {
  return (
    <div className="w-full h-screen flex items-center justify-center bg-gray-100 dark:bg-gray-900">
      <MultiStepLoader
        loadingStates={loadingStates}
        loading={true}
        duration={1000}
        loop={true}
      />
    </div>
  );
}
