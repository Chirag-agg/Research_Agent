import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

interface Claim {
    text: string;
    confidence: number;
    sources: Array<{ id: string; url?: string; title?: string }>;
}

interface Cluster {
    topic: string;
    summary: string;
    claims: Claim[];
}

interface AuditData {
    query: string;
    generated_at: string;
    clusters: Cluster[];
    metadata: {
        total_claims: number;
        total_sources: number;
        topic_count: number;
    };
}

interface AuditDisplayProps {
    data: AuditData;
}

export function AuditDisplay({ data }: AuditDisplayProps) {
    if (!data || !data.clusters) return null;

    const getConfidenceColor = (score: number) => {
        if (score >= 0.8) return "bg-green-500";
        if (score >= 0.5) return "bg-yellow-500";
        return "bg-red-500";
    };

    const getConfidenceLabel = (score: number) => {
        if (score >= 0.8) return "High Confidence";
        if (score >= 0.5) return "Medium Confidence";
        return "Low Confidence";
    };

    return (
        <div className="space-y-6 w-full max-w-4xl mx-auto p-4">
            <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold tracking-tight">Research Audit Trail</h2>
                <Badge variant="outline" className="text-sm">
                    {new Date(data.generated_at).toLocaleDateString()}
                </Badge>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium">Total Claims</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{data.metadata.total_claims}</div>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium">Sources Used</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{data.metadata.total_sources}</div>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium">Topics Covered</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{data.metadata.topic_count}</div>
                    </CardContent>
                </Card>
            </div>

            <Tabs defaultValue={data.clusters[0]?.topic} className="w-full">
                <ScrollArea className="w-full whitespace-nowrap rounded-md border">
                    <TabsList className="w-full justify-start p-1 h-auto">
                        {data.clusters.map((cluster, idx) => (
                            <TabsTrigger
                                key={idx}
                                value={cluster.topic}
                                className="px-4 py-2"
                            >
                                {cluster.topic}
                            </TabsTrigger>
                        ))}
                    </TabsList>
                </ScrollArea>

                {data.clusters.map((cluster, idx) => (
                    <TabsContent key={idx} value={cluster.topic} className="mt-4 space-y-4">
                        <Alert>
                            <AlertTitle>Topic Summary</AlertTitle>
                            <AlertDescription>{cluster.summary}</AlertDescription>
                        </Alert>

                        <div className="space-y-4">
                            {cluster.claims.map((claim, cIdx) => (
                                <Card key={cIdx}>
                                    <CardContent className="pt-6">
                                        <div className="flex items-start justify-between gap-4">
                                            <p className="text-sm leading-relaxed flex-1">
                                                {claim.text}
                                            </p>
                                            <div className="flex flex-col items-end gap-2 min-w-[120px]">
                                                <Badge
                                                    className={`${getConfidenceColor(claim.confidence)} text-white hover:${getConfidenceColor(claim.confidence)}`}
                                                >
                                                    {getConfidenceLabel(claim.confidence)}
                                                </Badge>
                                                <span className="text-xs text-muted-foreground">
                                                    Score: {(claim.confidence * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                        </div>

                                        {claim.sources && claim.sources.length > 0 && (
                                            <div className="mt-4 pt-4 border-t">
                                                <p className="text-xs font-semibold mb-2">Supported by:</p>
                                                <div className="flex flex-wrap gap-2">
                                                    {claim.sources.map((source, sIdx) => (
                                                        <Badge key={sIdx} variant="secondary" className="text-xs font-normal">
                                                            {source.title || "Source " + (sIdx + 1)}
                                                        </Badge>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </TabsContent>
                ))}
            </Tabs>
        </div>
    );
}
