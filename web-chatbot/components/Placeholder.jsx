"use client";

import Image from "next/image";
import { Logo } from "@/assets/images";
import { useGlobalState } from "@/context/GlobalContext";

const Placeholder = () => {
  return (
    <div className="flex flex-col justify-center items-center fira-code">
      <Image
        src={Logo}
        alt="Logo"
        className="h-24 w-auto rounded-3xl mb-4 max-md:h-16 max-md:w-16"
      />

      <div className="text-center text-3xl font-bold max-md:text-xl">
        <h1 className="text-[#868FAA]">Hi, Sameer Gupta</h1>
        <h1 className="text-gray-500">Can I help you with anything?</h1>
      </div>

      <div className="mt-5 text-[#868FAA]/90 font-bold text-center text-md max-md:text-sm">
        <p className="max-md:hidden">
          Ready to assist you with whatever you need!
        </p>
        <p>
          From answering your toughest questions to delivering impactful
          results,
        </p>
        <p>we’re here to make things easy and amazing for you.</p>
      </div>
    </div>
  );
};

export default Placeholder;
