import Link from "next/link";
import { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import Sidebar from "@/components/docs/Sidebar";

interface DocsLayoutProps {
    children: ReactNode;
}

export default function DocsLayout({ children }: DocsLayoutProps) {
    return (
        <div className="min-h-screen bg-slate-950 text-slate-100 scroll-smooth">
            <div className="sticky top-0 z-30 border-b border-slate-800/60 bg-slate-950/80 backdrop-blur">
                <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
                    <div className="flex items-center gap-3">
                        <div className="h-8 w-8 rounded-lg bg-emerald-400/20 ring-1 ring-emerald-400/40" />
                        <div>
                            <p className="text-sm font-semibold text-white">Deep Research Agent Docs</p>
                            <p className="text-xs text-slate-400">Documentation</p>
                        </div>
                    </div>
                    <div className="hidden items-center gap-3 md:flex">
                        <div className="relative">
                            <Input
                                placeholder="Search docs"
                                className="w-64 border-slate-800 bg-slate-900/60 text-sm text-slate-200 placeholder:text-slate-500"
                            />
                        </div>
                        <Link href="https://github.com" target="_blank" rel="noreferrer">
                            <Button variant="outline" className="border-slate-700 text-slate-200 hover:bg-slate-900">
                                GitHub
                            </Button>
                        </Link>
                        <Button variant="outline" className="border-slate-700 text-slate-400 hover:bg-slate-900">
                            Theme
                        </Button>
                    </div>
                </div>
            </div>

            <div className="mx-auto flex max-w-6xl gap-6 px-6 pb-12 pt-6">
                <aside className="hidden w-64 shrink-0 md:block">
                    <div className="sticky top-24 rounded-xl border border-slate-800/60 bg-slate-900/40 p-4">
                        <ScrollArea className="h-[calc(100vh-8rem)] pr-2">
                            <Sidebar />
                        </ScrollArea>
                    </div>
                </aside>

                <main className="min-w-0 flex-1">
                    <div className="mx-auto w-full max-w-4xl">{children}</div>
                </main>

                <aside className="hidden w-64 shrink-0 xl:block">
                    <div className="sticky top-24 rounded-xl border border-slate-800/60 bg-slate-900/40 p-4">
                        <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">On this page</p>
                        <ul className="mt-4 space-y-2 text-sm text-slate-300">
                            <li>
                                <a href="#overview" className="hover:text-white">Overview</a>
                            </li>
                            <li>
                                <a href="#features" className="hover:text-white">Core capabilities</a>
                            </li>
                            <li>
                                <a href="#architecture" className="hover:text-white">Architecture</a>
                            </li>
                            <li>
                                <a href="#quick-start" className="hover:text-white">Quick start</a>
                            </li>
                            <li>
                                <a href="#api" className="hover:text-white">API endpoints</a>
                            </li>
                            <li>
                                <a href="#env" className="hover:text-white">Environment</a>
                            </li>
                        </ul>
                    </div>
                </aside>
            </div>
        </div>
    );
}
