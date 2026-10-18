import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ProjectForge — Project workspace",
  description: "Plan, build, review, and package project work with human approval gates.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
