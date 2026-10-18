import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The Arena preview is served from a sandbox-specific e2b.app host.
  allowedDevOrigins: ["*.e2b.app"],
};

export default nextConfig;
