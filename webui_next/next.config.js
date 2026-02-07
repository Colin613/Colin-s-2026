/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  eslint: {
    // 在构建时禁用 ESLint
    ignoreDuringBuilds: true,
  },
  typescript: {
    // 在构建时禁用类型检查
    ignoreBuildErrors: true,
  },
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
