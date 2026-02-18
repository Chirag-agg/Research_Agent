import { CheckCircle2, Circle, Clock, ArrowRight, BrainCircuit } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface TaskNode {
    id: string;
    type: string;
    status: "pending" | "running" | "completed" | "failed";
    task: string;
    dependencies: string[];
    result?: any;
    metadata?: any;
}

interface TaskGraph {
    nodes: Record<string, TaskNode>;
    adjacency: Record<string, string[]>;
}

interface ResearchPlanProps {
    plan: any; // TaskGraph or similar structure
    className?: string;
}

export function ResearchPlan({ plan, className }: ResearchPlanProps) {
    if (!plan || !plan.nodes) return null;

    const nodes = Object.values(plan.nodes) as TaskNode[];
    // Sort roughly by ID or dependency logic if possible, but basic list is fine for now
    // Assuming numeric IDs or order in dict matches creation

    // Group by status? Or just list them.
    // Creating a "Thought Process" visualization

    return (
        <Card className={cn("bg-zinc-900/30 border-zinc-800", className)}>
            <CardHeader className="pb-3 border-b border-zinc-800/50">
                <CardTitle className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                    <BrainCircuit className="h-4 w-4 text-purple-400" />
                    Research Strategy & Execution Plan
                </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
                <div className="relative">
                    {/* Vertical line connecting steps */}
                    <div className="absolute left-6 top-4 bottom-4 w-px bg-zinc-800" />

                    <div className="space-y-0">
                        {nodes.map((node, index) => (
                            <div key={node.id} className="relative flex items-start gap-4 p-4 hover:bg-zinc-900/50 transition-colors group">
                                {/* Status Icon */}
                                <div className="relative z-10 flex h-5 w-5 items-center justify-center bg-black ring-4 ring-black rounded-full shrink-0 mt-0.5">
                                    {node.status === "completed" ? (
                                        <CheckCircle2 className="h-4 w-4 text-green-500" />
                                    ) : node.status === "running" ? (
                                        <Clock className="h-4 w-4 text-blue-500 animate-pulse" />
                                    ) : node.status === "failed" ? (
                                        <Circle className="h-4 w-4 text-red-500" />
                                    ) : (
                                        <Circle className="h-4 w-4 text-zinc-700" />
                                    )}
                                </div>

                                <div className="flex-1 min-w-0 space-y-1.5">
                                    <div className="flex items-center justify-between gap-2">
                                        <span className="text-xs font-semibold text-zinc-200">
                                            {node.metadata?.action || node.type}
                                        </span>
                                        <Badge variant="outline" className="text-[10px] h-5 px-1.5 border-zinc-700 text-zinc-500">
                                            {node.id}
                                        </Badge>
                                    </div>

                                    <p className="text-sm text-zinc-400 leading-relaxed">
                                        {node.task}
                                    </p>

                                    {node.result && (
                                        <p className="text-[11px] text-zinc-500 bg-zinc-950/50 p-2 rounded border border-zinc-800/50 mt-2 font-mono truncate">
                                            Result: {typeof node.result === 'string' ? node.result.slice(0, 80) : JSON.stringify(node.result).slice(0, 80)}...
                                        </p>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}
