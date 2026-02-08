/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  eslint: {
    // 完全禁用构建时的 ESLint
    ignoreDuringBuilds: true,
  },
  typescript: {
    // 完全禁用构建时的类型检查
    ignoreBuildErrors: true,
  },
  experimental: {
    // 禁用构建时的 linting
    esmExternals: false,
  },
  webpack: (config, { dev }) => {
    // 完全禁用 ESLint 插件
    if (!dev) {
      config.plugins = config.plugins.filter(plugin => {
        return plugin.constructor.name !== 'ESLintWebpackPlugin';
      });
    }
    return config;
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
