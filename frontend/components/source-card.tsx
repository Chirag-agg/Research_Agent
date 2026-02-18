import { Globe } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface Source {
    url: string;
    reliability?: number;
    agent?: string;
    title?: string;
    description?: string;
    content?: string;
}

interface SourceCardProps {
    source: Source;
    index?: number;
    className?: string;
}

export function SourceCard({ source, index, className }: SourceCardProps) {
    let hostname = "Unknown";
    let displayTitle = source.title;

    try {
        if (source.url) {
            const urlObj = new URL(source.url);
            hostname = urlObj.hostname.replace("www.", "");

            // If no title is provided, generate a readable one from the path
            if (!displayTitle) {
                const pathSegments = urlObj.pathname.split('/').filter(p => p.length > 0);
                if (pathSegments.length > 0) {
                    const lastSegment = pathSegments[pathSegments.length - 1];
                    const cleanSegment = lastSegment
                        .replace(/\.html?$/, '')
                        .replace(/\.php$/, '')
                        .replace(/[-_]/g, ' ');
                    displayTitle = cleanSegment.charAt(0).toUpperCase() + cleanSegment.slice(1);
                } else {
                    displayTitle = hostname;
                }
            }
        }
    } catch (e) { }

    const faviconUrl = `https://www.google.com/s2/favicons?domain=${hostname}&sz=32`;

    // Use description or fallback to content (truncated)
    const snippet = source.description || (source.content ? source.content.slice(0, 150) + (source.content.length > 150 ? "..." : "") : undefined);

    return (
        <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className={cn("block group", className)}
        >
            <Card className="h-full bg-[#18181b] border-zinc-800/50 hover:bg-zinc-900 transition-all duration-200 shadow-none hover:shadow-md rounded-xl overflow-hidden group-hover:border-zinc-700/50">
                <CardContent className="p-4 flex flex-col h-full gap-3">
                    {/* Header: Icon + Site Name */}
                    <div className="flex items-center gap-2">
                        <div className="h-4 w-4 rounded-full bg-black/40 flex items-center justify-center overflow-hidden shrink-0 border border-white/5">
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                                src={faviconUrl}
                                alt=""
                                className="h-2.5 w-2.5 object-cover opacity-80 group-hover:opacity-100 transition-opacity"
                                onError={(e) => {
                                    (e.target as HTMLImageElement).style.display = 'none';
                                    (e.target as HTMLImageElement).parentElement!.classList.add('fallback-icon');
                                }}
                            />
                            <Globe className="h-2.5 w-2.5 text-zinc-500 hidden group-[.fallback-icon]:block" />
                        </div>
                        <span className="text-[11px] font-medium text-zinc-400 truncate max-w-[120px]">
                            {hostname}
                        </span>

                        {/* Index if valid */}
                        {(index !== undefined) && (
                            <span className="ml-auto text-[10px] font-mono text-zinc-600 group-hover:text-zinc-500 transition-colors">
                                #{index + 1}
                            </span>
                        )}
                    </div>

                    {/* Main Content */}
                    <div className="space-y-1">
                        <h4 className="text-[13px] font-semibold text-zinc-200 leading-snug line-clamp-2 group-hover:text-blue-400 transition-colors">
                            {displayTitle || "Research Source"}
                        </h4>

                        {/* Snippet / Description Placeholer */}
                        {snippet && (
                            <p className="text-[11px] text-zinc-500 leading-normal line-clamp-3 group-hover:text-zinc-400 transition-colors">
                                {snippet}
                            </p>
                        )}
                        {/* If no snippet, we can show a truncated URL path as fallback */}
                        {!snippet && (
                            <p className="text-[10px] text-zinc-600 truncate opacity-60">
                                {source.url}
                            </p>
                        )}
                    </div>
                </CardContent>
            </Card>
        </a>
    );
}
