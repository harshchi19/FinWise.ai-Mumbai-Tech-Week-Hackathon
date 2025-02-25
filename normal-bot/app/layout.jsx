import { Geist, Geist_Mono } from "next/font/google";
import localFont from "next/font/local";
import "./globals.css";
import { GlobalProvider } from "@/context/GlobalContext";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const myFont = localFont({
  src: "./fonts/mandalorian.otf",
  variable: "--font-mandalorian",
});

export const metadata = {
  title: "finwise.ai",
  description: "Created by Mumbai Tech Week Hackathon",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${myFont.variable} antialiased`}
      >
        <GlobalProvider>{children}</GlobalProvider>
      </body>
    </html>
  );
}
