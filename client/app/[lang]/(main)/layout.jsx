"use client";

import Header from "@/components/Misc/Header";
import Sidebar from "@/components/Misc/Sidebar";
import { Toaster } from "@/components/ui/sonner";
import { NextStepProvider, NextStep } from "nextstepjs";
import { steps } from "@/lib/step";
import { useLanguage } from "@/context/LanguageContext";
import { Loader } from "@/components/Misc/Loader";
import { useGlobalState } from "@/context/GlobalContext";
import { useEffect, useState } from "react";

export default function MainLayout({ children }) {
  const { dict } = useLanguage();
  const { loading, setLoading } = useGlobalState();
  const [showContent, setShowContent] = useState(false);

  useEffect(() => {
    if (!dict) {
      const timer = setTimeout(() => {
        setShowContent(true);
      }, 3000);

      return () => clearTimeout(timer);
    } else {
      setShowContent(true);
    }
  }, [dict]);

  if (!dict && !showContent) {
    return <Loader />;
  }

  return (
    <NextStepProvider>
      <NextStep steps={steps}>
        <div className="h-screen flex">
          <Sidebar />

          <main className="h-screen w-full p-5 flex flex-col justify-start bg-gray-50">
            <Header />
            {children}
          </main>

          <Toaster />
        </div>
      </NextStep>
    </NextStepProvider>
  );
}
