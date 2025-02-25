"use client";

import React from "react";
import Header from "./Header";
import { people } from "@/constants/navbarData";
import { AnimatedTooltip } from "../ui/animated-tooltip";
import { Button } from "../ui/button";
import { MoveRight } from "lucide-react";
import { ContainerScroll } from "../ui/container-scroll-animation";
import Image from "next/image";
import { AppleCardsCarouselDemo } from "./AppleCardCarousel";
import { Testimonials } from "./Testimonials";

const LandingPage = () => {
  return (
    <div className="min-h-screen w-full px-72 py-24 flex flex-col items-center">
      <Header />

      <div className="mt-28 flex flex-col justify-center items-center">
        <h1 className="text-10xl font-bold text-center">
          Transform your <br /> Marketing with FinWise
        </h1>
        <h3 className="text-xl font-bold text-center mt-14">
          Automate Campaigns, Engage Audiences, and Boost <br />
          Lead Generation with Our All-in-One Marketing Solution
        </h3>
      </div>

      <div className="flex flex-col justify-center items-center">
        <div className="flex flex-row items-center justify-center w-full mt-12">
          <AnimatedTooltip items={people} />
          <h1 className="ml-6">⭐ ⭐ ⭐ ⭐ ⭐</h1>
        </div>
        <h1 className="mt-2 font-medium text-sm">Trusted by 12000+ people</h1>
      </div>

      <Button className="w-fit mt-16 bg-primary-400 text-white font-semibold text-lg py-6 hover:bg-primary-500">
        <h1>Watch Video</h1>
        <MoveRight />
      </Button>

      <div className="flex flex-col overflow-hidden">
        <ContainerScroll>
          <Image
            src={`https://ui.aceternity.com/_next/image?url=%2Flinear.webp&w=1920&q=75`}
            alt="hero"
            height={720}
            width={1400}
            className="mx-auto rounded-2xl object-cover h-full object-left-top"
            draggable={false}
          />
        </ContainerScroll>
      </div>

      <AppleCardsCarouselDemo />

      <div className="w-full mt-10">
        <h2 className="max-w-7xl pl-4 mx-auto text-xl md:text-5xl font-bold text-neutral-800 dark:text-neutral-200 font-sans">
          Testimonials
        </h2>
        <Testimonials />
      </div>
    </div>
  );
};

export default LandingPage;
