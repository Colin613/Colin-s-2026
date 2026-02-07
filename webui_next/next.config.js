/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:7860/api/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
