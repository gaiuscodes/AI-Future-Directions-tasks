/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  images: {
    unoptimized: true, // Fix for Netlify image optimization
    remotePatterns: [
      { protocol: 'https', hostname: '*.s3.amazonaws.com' }
    ],
  },
};
module.exports = nextConfig;