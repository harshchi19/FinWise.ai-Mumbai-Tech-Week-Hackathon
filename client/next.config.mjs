/** @type {import('next').NextConfig} */
const nextConfig = {
    images: {
        remotePatterns: [
            {
                protocol: "https",
                hostname: "images.unsplash.com",
            },
            {
                protocol: "https",
                hostname: "ui.aceternity.com"
            },
            {
                protocol: "https",
                hostname: "clunyfarm.co.za"
            }
        ],
    },
    async redirects() {
        return [
            {
                source: "/",
                destination: "/en",
                permanent: true,
            },
        ];
    },
};

export default nextConfig;
