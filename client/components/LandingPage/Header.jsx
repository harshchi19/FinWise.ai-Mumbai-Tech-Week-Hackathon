import React from "react";
import { motion, useScroll, useSpring } from "framer-motion";
import { useState, useEffect } from "react";
import { Logo } from "@/public/images";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { useLanguage } from "@/context/LanguageContext";
import { useUser } from "@clerk/nextjs";

const Header = () => {
  const [scrolled, setScrolled] = useState(false);
  const { scrollY } = useScroll();
  const router = useRouter();
  const { currentLang } = useLanguage();
  const { isSignedIn } = useUser();

  useEffect(() => {
    const unsubscribe = scrollY.onChange((latest) => {
      setScrolled(latest > 50);
    });
    return () => unsubscribe();
  }, [scrollY]);

  const headerVariants = {
    initial: {
      y: -100,
      opacity: 0,
    },
    animate: {
      y: 0,
      opacity: 1,
    },
  };

  const navItemVariants = {
    hover: {
      scale: 1.05,
      color: "#06b6d4",
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 15,
      },
    },
  };

  return (
    <div className="fixed top-0 left-0 right-0 z-50 flex justify-center p-4">
      <motion.header
        className={`h-14 flex items-center justify-between transition-all duration-500 ease-in-out rounded-lg px-8
          ${
            scrolled
              ? "w-[60%] bg-black/90 backdrop-blur-sm shadow-lg"
              : "w-[70%] bg-white/50 backdrop-blur-sm"
          }
        `}
        variants={headerVariants}
        initial="initial"
        animate="animate"
        transition={{
          duration: 0.7,
          ease: "easeInOut",
        }}
      >
        <div className="flex items-center gap-x-12">
          <motion.div
            className="flex items-center gap-x-4"
            whileHover={{ scale: 1.02 }}
            transition={{ type: "spring", stiffness: 400, damping: 10 }}
          >
            <Image src={Logo} alt="Logo" className="h-8 w-auto" priority />
            <h1
              className={`text-xl font-bold transition-colors duration-300
                ${scrolled ? "text-white" : "text-gray-800"}
              `}
            >
              Demo Template
            </h1>
          </motion.div>

          <nav className="flex items-center space-x-8">
            {["Features", "Pricing", "Blog", "Contact"].map((item) => (
              <motion.a
                key={item}
                href="#"
                className={`text-sm font-medium transition-colors duration-300
                  ${
                    scrolled
                      ? "text-gray-300 hover:text-white"
                      : "text-gray-800 hover:text-gray-900"
                  }
                `}
                variants={navItemVariants}
                whileHover="hover"
              >
                {item}
              </motion.a>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-x-4">
          {isSignedIn ? (
            <motion.button
              className={`px-4 py-2 rounded-md text-sm font-medium border transition-all duration-300
              ${
                scrolled
                  ? "text-white border-primary-400 hover:bg-primary-400/30"
                  : "text-gray-800 border-primary-500 hover:bg-primary-500/20"
              }
                  `}
              whileHover={{ scale: 1.05 }}
              transition={{ type: "spring", stiffness: 400, damping: 10 }}
              onClick={() => router.push(`/${currentLang}/overview`)}
            >
              Go to Dashboard
            </motion.button>
          ) : (
            <motion.button
              className={`px-4 py-2 rounded-md text-sm font-medium border transition-all duration-300
              ${
                scrolled
                  ? "text-white border-primary-400 hover:bg-primary-400/30"
                  : "text-gray-800 border-primary-500 hover:bg-primary-500/20"
              }
                  `}
              whileHover={{ scale: 1.05 }}
              transition={{ type: "spring", stiffness: 400, damping: 10 }}
              onClick={() => router.push(`/${currentLang}/sign-up`)}
            >
              Register
            </motion.button>
          )}
          <motion.button
            className="bg-primary-400 hover:bg-primary-500 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors duration-300"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 400, damping: 10 }}
          >
            Book a demo
          </motion.button>
        </div>
      </motion.header>
    </div>
  );
};

export default Header;
