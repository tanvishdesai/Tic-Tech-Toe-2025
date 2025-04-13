import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Predictive Maintenance Dashboard',
  description: 'Smart manufacturing dashboard for real-time equipment monitoring and failure prediction',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-slate-100 min-h-screen antialiased`}>
        {children}
      </body>
    </html>
  )
}
