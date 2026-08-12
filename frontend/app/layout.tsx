import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title:
    "เว็บแอปพลิเคชันเพื่อจำแนกโรคเท้าแบนจากภาพเอ็กซเรย์ด้วยตัวแบบการเรียนรู้ด้วยเครื่องและการเรียนรู้เชิงลึก",
  description: "เว็บแอปพลิเคชันสำหรับทำนายโรคเท้าแบนโดยใช้ AI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
