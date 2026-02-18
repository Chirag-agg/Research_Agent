import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { cn } from '@/lib/utils';
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { SourceCard } from "@/components/source-card";
import { Button } from "@/components/ui/button";

interface Source {
    url: string;
    reliability?: number;
    agent?: string;
    title?: string;
    content?: string;
    description?: string;
}

interface MarkdownProps {
    content: string;
    className?: string;
    sources?: Source[];
}

const CitationGroup = ({ indices, sources }: { indices: number[], sources: Source[] }) => {
    // Filter out invalid indices
    const validSources = indices
        .map(i => ({ source: sources[i], index: i }))
        .filter(item => item.source !== undefined);

    if (validSources.length === 0) return null;

    const firstSource = validSources[0].source;
    let mainLabel = "Source";
    try {
        if (firstSource.url) {
            // Get domain name without www. and extension
            const hostname = new URL(firstSource.url).hostname.replace(/^www\./, '');
            mainLabel = hostname.split('.')[0];
            // Capitalize first letter
            mainLabel = mainLabel.charAt(0).toUpperCase() + mainLabel.slice(1);
        }
    } catch (e) { }

    const count = validSources.length;
    // Perplexity style: "TechCrunch +2"
    const label = count > 1 ? `${mainLabel} +${count - 1}` : mainLabel;

    return (
        <HoverCard openDelay={200}>
            <HoverCardTrigger asChild>
                <span className="inline-flex items-center gap-1.5 px-2 py-0.5 mx-1 align-middle text-[11px] font-medium rounded-full bg-cyan-950/50 text-cyan-400 border border-cyan-900/50 cursor-pointer hover:bg-cyan-900/70 transition-colors select-none">
                    {label}
                </span>
            </HoverCardTrigger>
            <HoverCardContent className="w-[450px] p-0 border-zinc-800 bg-[#09090b] shadow-2xl rounded-xl z-50" align="start" sideOffset={4}>
                <div className="px-3 py-2 border-b border-zinc-900/50 flex items-center justify-between">
                    <p className="text-[11px] font-medium text-zinc-500">Sources · {count}</p>
                </div>
                <div className="p-1.5 space-y-0.5 max-h-[400px] overflow-y-auto">
                    {validSources.map(({ source, index }, i) => {
                        let hostname = "";
                        try {
                            hostname = new URL(source.url).hostname.replace("www.", "");
                        } catch (e) { }

                        const faviconUrl = `https://www.google.com/s2/favicons?domain=${hostname}&sz=32`;
                        const snippet = source.description || (source.content ? source.content.slice(0, 120) + (source.content.length > 120 ? "..." : "") : undefined);

                        return (
                            <a
                                key={index}
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-start gap-3 p-2.5 rounded-lg hover:bg-zinc-900 transition-colors group"
                            >
                                <div className="h-4 w-4 rounded-full bg-zinc-800 flex items-center justify-center overflow-hidden shrink-0 border border-zinc-700/50 mt-0.5">
                                    {/* eslint-disable-next-line @next/next/no-img-element */}
                                    <img
                                        src={faviconUrl}
                                        alt=""
                                        className="h-2.5 w-2.5 object-cover opacity-70 group-hover:opacity-100 transition-opacity"
                                    />
                                </div>
                                <div className="min-w-0 flex-1 space-y-1">
                                    <div className="flex items-center justify-between gap-2">
                                        <p className="text-[12px] font-semibold text-zinc-200 leading-snug line-clamp-1 group-hover:text-blue-400 transition-colors truncate">
                                            {source.title || hostname || "Source"}
                                        </p>
                                        <span className="text-[9px] text-zinc-600 font-mono shrink-0">
                                            #{index + 1}
                                        </span>
                                    </div>
                                    {snippet && (
                                        <p className="text-[11px] text-zinc-500 leading-relaxed line-clamp-2">
                                            {snippet}
                                        </p>
                                    )}
                                </div>
                            </a>
                        )
                    })}
                </div>
            </HoverCardContent>
        </HoverCard>
    );
};

const replaceCitations = (children: React.ReactNode, sources: Source[] = []) => {
    if (!sources || sources.length === 0) return children;

    return React.Children.map(children, (child) => {
        if (typeof child === 'string') {
            // Regex to match (Source 1), [Source 1], (Source 1, 2) etc.
            const regex = /([(\[]Source\s+[\d,\s]+(?:,.*?)?[)\]])/gi;
            const parts = child.split(regex);

            return parts.map((part, i) => {
                const isCitation = part.match(/^[(\[]Source\s+/i);
                if (isCitation) {
                    // Extract all numbers
                    const indices = [...part.matchAll(/(\d+)/g)]
                        .map(m => parseInt(m[1]) - 1);

                    if (indices.length > 0) {
                        return <CitationGroup key={i} indices={indices} sources={sources} />;
                    }
                }
                return part;
            });
        }
        return child;
    });
};

export function Markdown({ content, className, sources }: MarkdownProps) {
    // Strip markdown code blocks if present (LLM often wraps output in ```markdown ... ```)
    const cleanedContent = content.replace(/^```markdown\n?/, '').replace(/^```\n?/, '').replace(/\n?```$/, '');

    return (
        <div className={cn("prose prose-invert max-w-none break-words", className)}>
            <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                    h1: ({ className, ...props }) => (
                        <h1 className={cn("mt-2 scroll-m-20 text-3xl font-bold tracking-tight text-white/90", className)} {...props} />
                    ),
                    h2: ({ className, ...props }) => (
                        <h2 className={cn("mt-10 scroll-m-20 border-b border-zinc-800 pb-2 text-2xl font-semibold tracking-tight first:mt-0 text-zinc-100", className)} {...props} />
                    ),
                    h3: ({ className, ...props }) => (
                        <h3 className={cn("mt-8 scroll-m-20 text-xl font-semibold tracking-tight text-zinc-200", className)} {...props} />
                    ),
                    h4: ({ className, ...props }) => (
                        <h4 className={cn("mt-8 scroll-m-20 text-lg font-semibold tracking-tight text-zinc-300", className)} {...props} />
                    ),
                    h5: ({ className, ...props }) => (
                        <h5 className={cn("mt-8 scroll-m-20 text-base font-semibold tracking-tight text-zinc-300", className)} {...props} />
                    ),
                    h6: ({ className, ...props }) => (
                        <h6 className={cn("mt-8 scroll-m-20 text-base font-semibold tracking-tight text-zinc-400", className)} {...props} />
                    ),
                    a: ({ className, href, children, ...props }) => {
                        // Check anchor links that might be citations
                        const sourceIndex = sources?.findIndex(s => s.url === href);
                        if (sourceIndex !== undefined && sourceIndex !== -1 && sources) {
                            return (
                                <a
                                    href={href}
                                    className={cn("font-medium underline underline-offset-4 text-blue-400 hover:text-blue-300 transition-colors", className)}
                                    {...props}
                                >
                                    {children}
                                </a>
                            );
                        }
                        return (
                            <a className={cn("font-medium underline underline-offset-4 text-blue-400 hover:text-blue-300 transition-colors", className)} {...props}>
                                {children}
                            </a>
                        )
                    },
                    // Render paragraphs as divs to avoid invalid nesting when
                    // citation components inject block-level elements (HoverCardContent)
                    p: ({ className, children, ...props }) => (
                        <div role="group" className={cn("leading-7 [&:not(:first-child)]:mt-6 text-zinc-300", className)} {...props}>
                            {replaceCitations(children, sources)}
                        </div>
                    ),
                    li: ({ className, children, ...props }) => (
                        <li className={cn("mt-2", className)} {...props}>
                            {replaceCitations(children, sources)}
                        </li>
                    ),
                    blockquote: ({ className, ...props }) => (
                        <blockquote className={cn("mt-6 border-l-2 border-zinc-700 pl-6 italic text-zinc-400", className)} {...props} />
                    ),
                    img: ({ className, alt, ...props }) => (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img className={cn("rounded-md border border-zinc-800 bg-zinc-900 my-4", className)} alt={alt} {...props} />
                    ),
                    hr: ({ ...props }) => <hr className="my-8 border-zinc-800" {...props} />,
                    table: ({ className, ...props }) => (
                        <div className="my-6 w-full overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-900/30">
                            <table className={cn("w-full caption-bottom text-sm", className)} {...props} />
                        </div>
                    ),
                    tr: ({ className, ...props }) => (
                        <tr className={cn("m-0 border-b border-zinc-800 p-0 even:bg-zinc-900/50 hover:bg-zinc-800/50 transition-colors", className)} {...props} />
                    ),
                    th: ({ className, ...props }) => (
                        <th
                            className={cn(
                                "border-zinc-800 px-4 py-3 text-left font-bold text-zinc-100 [&[align=center]]:text-center [&[align=right]]:text-right bg-zinc-900/80",
                                className
                            )}
                            {...props}
                        />
                    ),
                    td: ({ className, ...props }) => (
                        <td
                            className={cn(
                                "border-zinc-800 px-4 py-3 text-left [&[align=center]]:text-center [&[align=right]]:text-right text-zinc-300",
                                className
                            )}
                            {...props}
                        />
                    ),
                    pre: ({ className, ...props }) => (
                        <pre
                            className={cn(
                                "mb-4 mt-6 overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950 py-4 px-4 dark:bg-zinc-950 text-zinc-200",
                                className
                            )}
                            {...props}
                        />
                    ),
                    code: ({ className, ...props }) => (
                        <code
                            className={cn(
                                "relative rounded bg-zinc-900 px-[0.3rem] py-[0.2rem] font-mono text-sm font-semibold text-zinc-200 border border-zinc-800",
                                className
                            )}
                            {...props}
                        />
                    ),
                }}
            >
                {cleanedContent}
            </ReactMarkdown>
        </div>
    );
}
