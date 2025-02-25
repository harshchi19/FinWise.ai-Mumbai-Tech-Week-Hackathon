import {
  CameraControls,
  Environment,
  Gltf,
  Html,
  useProgress,
} from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import Chatbot from "./ChatbotUser";
import { degToRad } from "three/src/math/MathUtils";
import { Suspense, useState, useRef, useEffect } from "react";
import { TypingBox } from "./TypingBox";
import MessageBox from "./MessageBox";
import { useLanguage } from "@/context/LanguageContext";
import { Thinking } from "@/public/images";
import Image from "next/image";
import ChatComponent from "../Chats/ChatComponent";

const CameraManager = () => {
  return (
    <CameraControls
      minZoom={1}
      maxZoom={2}
      polarRotateSpeed={-0.3}
      azimuthRotateSpeed={-0.3}
      mouseButtons={{
        left: 1,
        wheel: 16,
      }}
      touches={{
        one: 32,
        two: 512,
      }}
      minAzimuthAngle={degToRad(-10)}
      maxAzimuthAngle={degToRad(10)}
      minPolarAngle={degToRad(90)}
      maxPolarAngle={degToRad(100)}
    />
  );
};

const Loader = ({ progress }) => {
  const { dict } = useLanguage();

  return (
    <Html center>
      <div className="flex flex-col items-center justify-center p-4 space-y-4 w-72">
        <p className="text-xl font-semibold">{dict?.chatbot?.loading}</p>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div
            className="bg-gradient-to-r from-green-400 to-green-600 h-2.5 rounded-full"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
        <p className="text-sm text-gray-500">{Math.round(progress)}%</p>
      </div>
    </Html>
  );
};

const Experience = () => {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [animationNumber, setAnimationNumber] = useState(2);
  const { progress } = useProgress();
  const [showInterface, setShowInterface] = useState(false);
  const [chatbotResponse, setChatbotResponse] = useState("");

  useEffect(() => {
    if (Math.round(progress) === 100) {
      setTimeout(() => setShowInterface(true), 500);
    } else {
      setShowInterface(false);
    }
  }, [progress]);

  const chatbotRef = useRef();
  const [chatbotColor, setChatbotColor] = useState("#FFFFFF");

  const handleSendMessage = async (newMessage) => {
    setLoading(true);

    setTimeout(() => {
      const fakeChatbotResponse = `Chatbot says: ${newMessage}`;
      setChatbotResponse(fakeChatbotResponse);

      setMessages((prevMessages) => [
        ...prevMessages,
        { type: "user", text: newMessage },
        { type: "chatbot", text: fakeChatbotResponse },
      ]);
      setLoading(false);
    }, 1500);
  };

  return (
    <div className="h-full w-full relative overflow-hidden flex gap-x-2">
      <div className="w-full rounded-xl relative overflow-hidden">
        <div
          className={`z-10 md:justify-center absolute bottom-4 left-4 right-4 flex gap-3 flex-wrap justify-stretch transition-opacity duration-500 ${
            showInterface ? "opacity-100" : "opacity-0 hidden"
          }`}
        >
          <TypingBox
            setMessage={(text) => {
              handleSendMessage(text);
            }}
            loading={loading}
            setLoading={setLoading}
            setAnimationNumber={setAnimationNumber}
          />
        </div>
        <div
          className={`z-10 md:justify-center absolute top-10 right-12 transition-opacity duration-500 ${
            showInterface ? "opacity-100" : "opacity-0 hidden"
          }`}
        >
          <MessageBox message={chatbotResponse} />
        </div>
        <div
          className={`z-10 md:justify-center absolute top-3 left-48 flex flex-wrap justify-stretch transition-opacity duration-500 ${
            showInterface ? "opacity-100" : "opacity-0 hidden"
          }`}
        >
          {loading && (
            <Image src={Thinking} alt="thinking" className="h-20 w-auto" />
          )}
        </div>
        <Canvas
          camera={{
            position: [0, 0, 3],
            fov: 50,
          }}
          style={{ width: "100%", height: "100%" }}
        >
          <Environment preset="sunset" background={true} blur={0.5} />
          <ambientLight intensity={0.5} />
          <directionalLight position={[5, 5, 5]} intensity={0.8} />
          <Suspense fallback={<Loader progress={progress} />}>
            <Chatbot
              ref={chatbotRef}
              position={[-1.4, -1.2, -0.6]}
              scale={1.25}
              rotation-x={degToRad(5)}
              rotation-y={degToRad(35)}
              rotation-z={degToRad(-1)}
              animationNumber={animationNumber}
              name="chatbot"
              color={chatbotColor}
            />
          </Suspense>
          <CameraManager />
        </Canvas>
      </div>

      <div className="w-[40%]">
        <ChatComponent messages={messages} onSendMessage={handleSendMessage} />
      </div>
    </div>
  );
};

export default Experience;
