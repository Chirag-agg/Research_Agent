"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const NAV_SECTIONS = [
    {
        title: "GET STARTED",
        items: [
            { label: "Overview", href: "/docs#overview" },
            { label: "Quick Start", href: "/docs#quick-start" },
            { label: "Installation", href: "/docs#installation" },
            { label: "Architecture", href: "/docs#architecture" },
        ],
    },
    {
        title: "CORE SYSTEM",
        items: [
            { label: "Dual Research Modes", href: "/docs#dual-research-modes" },
            { label: "Task Graph & Planning", href: "/docs#task-graph" },
            { label: "Evidence Graph", href: "/docs#evidence-graph" },
            { label: "Reflexion & Replanning", href: "/docs#reflexion" },
            { label: "Model Routing", href: "/docs#model-routing" },
        ],
    },
    {
        title: "MEMORY & STORAGE",
        items: [
            { label: "Supabase Integration", href: "/docs#supabase" },
            { label: "Qdrant Vector Store", href: "/docs#qdrant" },
            { label: "Persistent Memory", href: "/docs#persistent-memory" },
        ],
    },
    {
        title: "API & DEPLOYMENT",
        items: [
            { label: "API Endpoints", href: "/docs#api" },
            { label: "Environment Variables", href: "/docs#env" },
            { label: "Docker Deployment", href: "/docs#docker" },
        ],
    },
];

export default function Sidebar() {
    const pathname = usePathname();
    const [openSections, setOpenSections] = useState<Record<string, boolean>>({
        "GET STARTED": true,
        "CORE SYSTEM": true,
        "MEMORY & STORAGE": true,
        "API & DEPLOYMENT": true,
    });
    const [activeHash, setActiveHash] = useState<string>("#overview");

    useEffect(() => {
        const updateHash = () => {
            const hash = window.location.hash || "#overview";
            setActiveHash(hash);
        };

        updateHash();
        window.addEventListener("hashchange", updateHash);
        return () => window.removeEventListener("hashchange", updateHash);
    }, []);

    const activeHref = useMemo(() => {
        if (pathname === "/docs") {
            return `/docs${activeHash}`;
        }
        return pathname;
    }, [pathname, activeHash]);

    return (
        <nav className="space-y-5 text-sm">
            {NAV_SECTIONS.map((section) => {
                const isOpen = openSections[section.title];
                return (
                    <div key={section.title} className="space-y-2">
                        <button
                            type="button"
                            onClick={() =>
                                setOpenSections((prev) => ({
                                    ...prev,
                                    [section.title]: !prev[section.title],
                                }))
                            }
                            className="flex w-full items-center justify-between text-xs font-semibold tracking-widest text-slate-400 hover:text-slate-200"
                        >
                            <span>{section.title}</span>
                            <span className="text-slate-500">{isOpen ? "-" : "+"}</span>
                        </button>
                        {isOpen && (
                            <div className="space-y-1">
                                {section.items.map((item) => {
                                    const isActive = activeHref === item.href;
                                    return (
                                        <Link
                                            key={item.href}
                                            href={item.href}
                                            className={cn(
                                                "block rounded-md px-2 py-1 text-slate-300 transition hover:bg-slate-800/60 hover:text-white",
                                                isActive && "bg-slate-800 text-white"
                                            )}
                                        >
                                            {item.label}
                                        </Link>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                );
            })}
        </nav>
    );
}
