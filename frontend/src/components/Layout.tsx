import { ReactNode } from "react";
import { Link } from "react-router-dom";

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-900">
      <header className="border-b border-gray-800 bg-gray-950">
        <div className="mx-auto max-w-6xl px-4 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3 text-xl font-semibold text-brand-500 hover:text-brand-400">
            <span className="text-2xl">🪷</span>
            Bharatanatyam Analyzer
          </Link>
          <nav className="flex gap-4 text-sm text-gray-400">
            <Link to="/" className="hover:text-white transition-colors">
              Dashboard
            </Link>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
    </div>
  );
}
