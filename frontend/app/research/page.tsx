"use client";
// Force revalidation

import { useState, useRef, useEffect, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import {
  ArrowUp,
  Search,
  Sparkles,
  Terminal,
  FileText,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Clock,
  Globe
} from "lucide-react";
import { Markdown } from "@/components/markdown";
import { SourceCard } from "@/components/source-card";
import { ResearchPlan } from "@/components/research-plan";
import { cn } from "@/lib/utils";
import Aurora from "@/components/Aurora";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";

interface LogEntry {
  timestamp: string;
  message: string;
}

interface ResearchResult {
  report: string;
  sources: { url: string; reliability: number; agent: string, title?: string, content?: string }[];
  metadata: {
    duration_seconds?: number;
    iterations?: number;
    validated_findings?: any;
    task_graph?: any;
    evidence_graph?: any;
    [key: string]: any;
  };
}

interface HistoryItem {
  id: string;
  query: string;
  status: string;
  created_at: string;
  has_result: boolean;
}

function ResearchPageContent() {
  const [query, setQuery] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [continueQuery, setContinueQuery] = useState("");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [mode, setMode] = useState<"quick" | "deep">("deep");
  const scrollRef = useRef<HTMLDivElement>(null);
  const metrics = result?.metadata?.metrics as
    | {
      mode?: "quick" | "deep";
      latency?: number;
      prompt_tokens?: number;
      completion_tokens?: number;
      cost_estimate?: number;
      models_used?: Record<string, number>;
      task_graph?: {
        total_nodes?: number;
        max_depth?: number;
      };
    }
    | undefined;
  const reflexionTriggered = Boolean(result?.metadata?.reflexion?.triggered);
  const metricsMode = metrics?.mode || "deep";
  const isQuickMode = metricsMode === "quick";

  const searchParams = useSearchParams();
  const router = useRouter();

  // Check for API URL from environment
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // Load history on mount
  useEffect(() => {
    fetchHistory();
  }, []); // Only on mount

  // Refresh history when session ID changes (new session created)
  useEffect(() => {
    if (sessionId) {
      fetchHistory();
    }
  }, [sessionId]);

  const fetchHistory = async () => {
    try {
      console.log("Fetching history from:", `${API_URL}/api/history`);
      const res = await fetch(`${API_URL}/api/history`);
      if (res.ok) {
        const data = await res.json();
        console.log("History fetched:", data);
        setHistory(data);
      } else {
        console.error("History fetch failed with status:", res.status);
      }
    } catch (e) {
      console.error("Failed to fetch history:", e);
    }
  };

  // Check URL param
  useEffect(() => {
    const id = searchParams.get("session_id");
    if (id && id !== sessionId) {
      setSessionId(id);
    }
  }, [searchParams]);

  useEffect(() => {
    console.log("Session ID updated:", sessionId);
  }, [sessionId]);

  // Load session data when ID changes
  useEffect(() => {
    if (!sessionId) return;

    const fetchSession = async () => {
      try {
        const res = await fetch(`${API_URL}/api/research/${sessionId}`);
        if (res.ok) {
          const data = await res.json();
          setLogs(data.logs || []);

          if (data.status === "completed") {
            setResult(data.result);
            setIsProcessing(false);
          } else if (data.status === "running" || data.status === "pending") {
            setIsProcessing(true);
            setResult(null); // Clear result while processing new query
            // Result might be partial or null
          } else if (data.status === "failed") {
            setIsProcessing(false);
            setResult(null); // Clear result on failure
          }
          // If loaded from history, query might not be set in UI
          if (data.query) setQuery(data.query);
        }
      } catch (e) {
        console.error("Error fetching session", e);
      }
    };

    fetchSession();
  }, [sessionId]);


  // Auto-scroll logs
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  // Poll for status
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (sessionId && isProcessing) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_URL}/api/research/${sessionId}`);
          if (res.ok) {
            const data = await res.json();
            setLogs(data.logs || []);

            if (data.status === "completed") {
              setResult(data.result);
              setIsProcessing(false);
              clearInterval(interval);
              fetchHistory(); // Refresh sidebar
            } else if (data.status === "failed") {
              setIsProcessing(false);
              setResult(null); // Clear result on failure
              clearInterval(interval);
              fetchHistory(); // Also refresh history on failure
              // Handle error visually
            }
          }
        } catch (error) {
          console.error("Polling error", error);
        }
      }, 2000);
    }

    return () => clearInterval(interval);
  }, [sessionId, isProcessing]);

  const handleSubmit = async (e?: React.FormEvent, overrideQuery?: string) => {
    e?.preventDefault();
    const submittedQuery = (overrideQuery ?? query).trim();
    if (!submittedQuery) return;
    if (overrideQuery) {
      setQuery(submittedQuery);
    }

    setIsProcessing(true);
    setLogs([]);
    setResult(null); // Clear previous result immediately

    const res = await fetch(`${API_URL}/api/research`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: submittedQuery,
        mode,
        session_id: sessionId || undefined,
      }),
    });

    const data = await res.json();
    setSessionId(data.session_id);
    // Update URL without reload
    router.push(`/research?session_id=${data.session_id}`);
  };

  const handleContinue = async () => {
    if (!continueQuery.trim()) return;
    await handleSubmit(undefined, continueQuery);
    setContinueQuery("");
  };

  const loadSession = (id: string) => {
    setSessionId(id);
    router.push(`/research?session_id=${id}`);
    setResult(null); // Clear previous result to show loading/logs
    fetchHistory(); // Refresh history
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex h-screen bg-black text-zinc-100 font-sans selection:bg-zinc-800 relative">
      {/* Aurora Background */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <Aurora
          colorStops={["#7cff67", "#B19EEF", "#5227FF"]}
          blend={0.5}
          amplitude={1.0}
          speed={1}
        />
      </div>

      {/* Sidebar */}
      <div className="w-[260px] hidden md:flex flex-col border-r border-zinc-900 bg-black/95 backdrop-blur-md p-4 space-y-4 relative z-10">
        <div className="flex items-center gap-2 px-2 py-1">
          <div className="h-6 w-6 rounded-md bg-zinc-800 flex items-center justify-center">
            <div className="h-3 w-3 rounded-full bg-white/90" />
          </div>
          <span className="font-semibold text-sm tracking-tight text-white/90">Deep Research</span>
        </div>

        <Button
          variant="outline"
          className="w-full justify-start gap-2 bg-zinc-900 border-zinc-800 hover:bg-zinc-800 text-zinc-300"
          onClick={() => {
            setSessionId(null);
            setResult(null);
            setQuery("");
            setLogs([]);
            router.push("/research");
            fetchHistory();
          }}
        >
          <Sparkles className="h-4 w-4" />
          New Research
        </Button>

        <div className="space-y-1 flex-1 overflow-y-auto pr-1 scrollbar-hide">
          <p className="text-xs font-medium text-zinc-400 px-2 py-2">History</p>
          {history.map((item) => (
            <Button
              key={item.id}
              variant="ghost"
              className={cn(
                "w-full justify-start h-8 px-2 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900/70",
                sessionId === item.id && "bg-zinc-900 text-zinc-100"
              )}
              onClick={() => loadSession(item.id)}
            >
              <Clock className="mr-2 h-3 w-3 opacity-70" />
              <span className="truncate text-xs text-left w-full">{item.query || "Untitled Research"}</span>
            </Button>
          ))}
          {history.length === 0 && (
            <p className="text-xs text-zinc-500 px-2">No history yet</p>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col relative w-full h-full overflow-hidden z-10">

        {/* Header / Top Bar */}
        <div className="absolute top-0 right-0 p-4 flex gap-2 z-20">
          <Link href="/docs" className="hidden sm:inline-flex">
            <Button variant="outline" className="border-zinc-700 text-zinc-200 hover:bg-zinc-900">
              Docs
            </Button>
          </Link>
          <Button variant="ghost" className="text-zinc-400 hover:text-white">Feedback</Button>
          <Button variant="ghost" className="text-zinc-400 hover:text-white">History</Button>
        </div>

        <div className="flex-1 overflow-auto">
          {!result && !isProcessing && logs.length === 0 ? (
            // Hero State
            <div className="h-full flex flex-col items-center justify-center p-8 max-w-3xl mx-auto w-full">
              <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-8 text-transparent bg-clip-text bg-gradient-to-b from-white to-zinc-500 text-center">
                What do you want to research?
              </h1>

              <div className="w-full relative group">
                <div className="absolute -inset-1 rounded-xl bg-gradient-to-r from-zinc-700 via-zinc-600 to-zinc-700 opacity-20 blur group-hover:opacity-40 transition duration-500" />
                <div className="relative bg-black rounded-xl border border-zinc-800 p-2 shadow-2xl">
                  <Textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask Deep Research to investigate..."
                    className="min-h-[60px] w-full resize-none border-0 bg-transparent text-lg placeholder:text-zinc-600 focus-visible:ring-0 px-4 py-3"
                  />
                  <div className="flex flex-col gap-2 px-2 py-2 mt-2 md:flex-row md:items-center md:justify-between">
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        variant={mode === "quick" ? "default" : "outline"}
                        className={cn("rounded-lg", mode === "quick" ? "bg-zinc-100 text-black hover:bg-white" : "border-zinc-800 text-zinc-200")}
                        onClick={() => setMode("quick")}
                      >
                        Quick Research
                      </Button>
                      <Button
                        size="sm"
                        variant={mode === "deep" ? "default" : "outline"}
                        className={cn("rounded-lg", mode === "deep" ? "bg-zinc-100 text-black hover:bg-white" : "border-zinc-800 text-zinc-200")}
                        onClick={() => setMode("deep")}
                      >
                        Deep Research
                      </Button>
                    </div>
                    <div className="flex gap-2 items-center justify-end">
                      <Button size="icon" variant="ghost" className="h-8 w-8 text-zinc-500 hover:text-zinc-300 rounded-lg">
                        <Search className="h-4 w-4" />
                      </Button>
                      <Button
                        onClick={() => handleSubmit()}
                        disabled={!query.trim()}
                        className="bg-zinc-100 hover:bg-white text-black h-8 w-8 p-0 rounded-lg transition-all"
                      >
                        <ArrowUp className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-8 flex gap-3 text-zinc-500">
                <span className="text-xs bg-zinc-900/50 border border-zinc-800 px-3 py-1 rounded-full hover:border-zinc-700 cursor-pointer transition-colors">Latest AI architectures</span>
                <span className="text-xs bg-zinc-900/50 border border-zinc-800 px-3 py-1 rounded-full hover:border-zinc-700 cursor-pointer transition-colors">Quantum computing trends</span>
                <span className="text-xs bg-zinc-900/50 border border-zinc-800 px-3 py-1 rounded-full hover:border-zinc-700 cursor-pointer transition-colors">CRISPR advancements</span>
              </div>
            </div>
          ) : (
            // Results State
            <div className="max-w-4xl mx-auto p-4 md:p-8 min-h-full pb-20">
              {/* Active Task / Status area */}
              <div className="mb-6">
                <h2 className="text-2xl font-bold text-zinc-100 mb-6">{query}</h2>

                {/* Live Logs & Plan */}
                <div className="space-y-4">

                  {/* Research Plan Visualization */}
                  {result?.metadata?.task_graph && (
                    <div className="mb-6 animate-in fade-in slide-in-from-bottom-2 duration-500">
                      <ResearchPlan plan={result.metadata.task_graph} />
                    </div>
                  )}

                  {/* Logs Accordion/Terminal */}
                  <div className="bg-zinc-950 border border-zinc-800 rounded-lg overflow-hidden">
                    <div className="flex items-center justify-between px-4 py-2 bg-zinc-900/50 border-b border-zinc-800">
                      <span className="text-xs font-mono text-zinc-400 flex items-center gap-2">
                        <Terminal className="h-3 w-3" />
                        AGENT TERMINAL
                      </span>
                      {isProcessing && <Loader2 className="h-3 w-3 animate-spin text-zinc-500" />}
                    </div>
                    {/* Only show logs if processing or if no result yet, or collapsed if done */}
                    <div ref={scrollRef} className={cn("overflow-y-auto p-4 font-mono text-xs text-zinc-400 space-y-1 transition-all duration-300", result ? "h-32" : "h-64")}>
                      {logs.map((log, i) => (
                        <div key={i} className="flex gap-2">
                          <span className="text-zinc-600 shrink-0">{log.timestamp.split('T')[1].split('.')[0]}</span>
                          <span className={log.message.includes("Error") ? "text-red-400" : ""}>{log.message}</span>
                        </div>
                      ))}
                      {logs.length === 0 && <span className="text-zinc-600">Initializing agent...</span>}
                    </div>
                  </div>
                </div>
              </div>

              {/* Final Report */}
              {result && (
                <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-150">
                  <div className="flex items-center gap-2 text-green-400">
                    <CheckCircle2 className="h-5 w-5" />
                    <span className="font-medium">Research Complete</span>
                  </div>

                  {metrics && (
                    <Card
                      className={cn(
                        "border shadow-lg",
                        isQuickMode
                          ? "bg-zinc-900/40 border-zinc-800"
                          : "bg-gradient-to-br from-zinc-900/70 via-zinc-950 to-zinc-900/70 border-zinc-700",
                        reflexionTriggered && "animate-[pulse_1.2s_ease-in-out_1]"
                      )}
                    >
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-semibold text-zinc-200">Performance Metrics</CardTitle>
                        <CardDescription className="text-xs text-zinc-500 flex flex-wrap items-center gap-2">
                          <span>
                            Mode: <span className="font-medium text-zinc-300">{metricsMode}</span>
                          </span>
                          {reflexionTriggered && (
                            <Badge className="border border-emerald-500/40 bg-emerald-500/10 text-emerald-300 shadow-[0_0_12px_rgba(16,185,129,0.25)]">
                              Reflexion Activated
                            </Badge>
                          )}
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                        <div className="rounded-lg border border-zinc-800/60 bg-black/30 p-3 sm:col-span-2 lg:col-span-3">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <div className="text-[11px] uppercase tracking-wide text-zinc-500">Planning Depth</div>
                            <Badge
                              variant="outline"
                              className={cn(
                                "text-[10px] uppercase tracking-wide",
                                isQuickMode
                                  ? "border-zinc-700 text-zinc-400"
                                  : "border-zinc-600 text-zinc-200"
                              )}
                            >
                              {isQuickMode ? "Shallow Planning" : "Hierarchical Planning"}
                            </Badge>
                          </div>
                          <div className="mt-2 flex flex-wrap gap-3">
                            <span className="text-xs px-2 py-1 rounded-full border border-zinc-800 text-zinc-300">
                              Nodes: {metrics.task_graph?.total_nodes ?? 0}
                            </span>
                            <span className="text-xs px-2 py-1 rounded-full border border-zinc-800 text-zinc-300">
                              Depth: {metrics.task_graph?.max_depth ?? 0}
                            </span>
                          </div>
                        </div>
                        <div className="rounded-lg border border-zinc-800/60 bg-black/30 p-3">
                          <div className="text-[11px] uppercase tracking-wide text-zinc-500">Latency</div>
                          <div className="text-lg font-semibold text-zinc-100">
                            {(metrics.latency ?? 0).toFixed(2)}s
                          </div>
                        </div>
                        <div className="rounded-lg border border-zinc-800/60 bg-black/30 p-3">
                          <div className="text-[11px] uppercase tracking-wide text-zinc-500">Prompt Tokens</div>
                          <div className="text-lg font-semibold text-zinc-100">
                            {metrics.prompt_tokens ?? 0}
                          </div>
                        </div>
                        <div className="rounded-lg border border-zinc-800/60 bg-black/30 p-3">
                          <div className="text-[11px] uppercase tracking-wide text-zinc-500">Completion Tokens</div>
                          <div className="text-lg font-semibold text-zinc-100">
                            {metrics.completion_tokens ?? 0}
                          </div>
                        </div>
                        <div className="rounded-lg border border-zinc-800/60 bg-black/30 p-3">
                          <div className="text-[11px] uppercase tracking-wide text-zinc-500">Estimated Cost</div>
                          <div className="text-lg font-semibold text-zinc-100">
                            ${Number(metrics.cost_estimate ?? 0).toFixed(4)}
                          </div>
                        </div>
                        <div className="rounded-lg border border-zinc-800/60 bg-black/30 p-3 sm:col-span-2 lg:col-span-2">
                          <div className="text-[11px] uppercase tracking-wide text-zinc-500">Models Used</div>
                          <div className="mt-2 flex flex-wrap gap-2">
                            {metrics.models_used && Object.keys(metrics.models_used).length > 0 ? (
                              Object.entries(metrics.models_used).map(([provider, tokens]) => (
                                <span
                                  key={provider}
                                  className={cn(
                                    "text-xs px-2 py-1 rounded-full border",
                                    isQuickMode
                                      ? "border-zinc-700 text-zinc-400"
                                      : "border-zinc-600 text-zinc-200"
                                  )}
                                >
                                  {provider}: {tokens}
                                </span>
                              ))
                            ) : (
                              <span className="text-xs text-zinc-500">No model usage recorded</span>
                            )}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {sessionId && !isProcessing && (
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                      <Input
                        value={continueQuery}
                        onChange={(e) => setContinueQuery(e.target.value)}
                        placeholder="Continue this research..."
                        className="bg-black/30 border-zinc-800 text-zinc-200 placeholder:text-zinc-500"
                      />
                      <Button
                        onClick={handleContinue}
                        disabled={!continueQuery.trim()}
                        className="bg-zinc-100 hover:bg-white text-black"
                      >
                        Continue
                      </Button>
                    </div>
                  )}

                  <Card className="bg-zinc-950 border-zinc-800 text-zinc-300 shadow-xl">
                    <CardContent className="prose prose-invert max-w-none pt-8 px-8 pb-8">
                      <div className="markdown-body">
                        <Markdown content={result.report} sources={result.sources} />
                      </div>
                    </CardContent>
                  </Card>

                  {/* Sources Section */}
                  {result.sources && result.sources.length > 0 && (
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <h3 className="text-lg font-semibold text-zinc-100">Sources</h3>
                        {/* Source Pill Badge */}
                        <div className="bg-[#1e1e22] text-zinc-400 px-2.5 py-1 rounded-full text-xs font-medium flex items-center gap-1">
                          {(() => {
                            try {
                              return new URL(result.sources[0].url).hostname.replace('www.', '').substring(0, 15) + '...';
                            } catch { return 'Source'; }
                          })()}
                          <span className="text-zinc-500 ml-1">+{result.sources.length - 1}</span>
                        </div>
                      </div>

                      <div className="flex overflow-x-auto pb-4 gap-3 -mx-4 px-4 md:mx-0 md:px-0 scrollbar-hide">
                        {/* Render first 5-6 sources directly */}
                        {result.sources.slice(0, 10).map((source, i) => (
                          <SourceCard key={i} source={source} index={i} />
                        ))}

                        {/* Show All Card */}
                        <Sheet>
                          <SheetTrigger asChild>
                            <div className="min-w-[100px] flex-shrink-0 cursor-pointer group">
                              <Card className="h-full bg-[#1e1e22] border-none hover:bg-[#27272a] transition-all duration-200 shadow-none rounded-xl flex items-center justify-center">
                                <CardContent className="p-4 flex flex-col items-center gap-2 text-zinc-400 group-hover:text-zinc-200">
                                  <Globe className="h-5 w-5 mb-1" />
                                  <span className="text-xs font-semibold whitespace-nowrap">Show all</span>
                                </CardContent>
                              </Card>
                            </div>
                          </SheetTrigger>
                          <SheetContent className="bg-[#09090b] border-l border-zinc-800 w-[400px] sm:w-[600px] lg:w-[800px] sm:max-w-[80vw]">
                            <SheetHeader className="mb-8 px-4">
                              <SheetTitle className="text-zinc-100 flex items-center gap-3 text-xl font-medium">
                                <div className="h-8 w-8 rounded-full bg-zinc-800 flex items-center justify-center">
                                  <Globe className="h-4 w-4 text-zinc-400" />
                                </div>
                                {result.sources.length} Sources
                              </SheetTitle>
                            </SheetHeader>
                            <ScrollArea className="h-[calc(100vh-100px)] pr-0">
                              <div className="max-w-3xl mx-auto px-4 pb-10 grid grid-cols-1 gap-3">
                                {result.sources.map((source, i) => (
                                  // For the sheet view, we probably want a wider list-like card, 
                                  // but reusing SourceCard with width override is okay for now.
                                  <SourceCard key={i} source={source} index={i} className="min-w-full max-w-none" />
                                ))}
                              </div>
                            </ScrollArea>
                          </SheetContent>
                        </Sheet>
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card className="bg-zinc-900/20 border-zinc-800">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-zinc-400">Sources Analyzed</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold text-zinc-100">{result.sources?.length || result.metadata?.validated_findings || 0}</div>
                      </CardContent>
                    </Card>
                    <Card className="bg-zinc-900/20 border-zinc-800">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-zinc-400">Duration</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold text-zinc-100">{result.metadata?.duration_seconds || 60}s</div>
                      </CardContent>
                    </Card>
                    <Card className="bg-zinc-900/20 border-zinc-800">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-zinc-400">Claims Verified</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold text-zinc-100">{result.metadata?.evidence_graph?.claims?.length || 0}</div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function ResearchPage() {
  return (
    <Suspense fallback={<div className="flex h-screen items-center justify-center bg-black text-zinc-500">Loading Deep Research...</div>}>
      <ResearchPageContent />
    </Suspense>
  );
}
