import { Logo } from "@/assets/images";
import Image from "next/image";
import { DropdownButton } from "./DropdownButton";

const Navbar = () => {
  return (
    <div className="h-20 absolute top-0 w-full flex justify-between items-center px-5">
      <div className="flex items-center justify-center gap-x-2">
        <Image src={Logo} alt="Logo" className="h-8 w-auto rounded-lg" />
        <h1 className="text-white text-3xl font-mandalorian tracking-wider">
          FINWISE.ai
        </h1>
      </div>

      <DropdownButton />
    </div>
  );
};

export default Navbar;
