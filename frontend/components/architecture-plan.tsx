"use client";

import React, { useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Tabs } from "@/components/ui/tabs";
import { Alert } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";

interface ArchitecturePlan {
    metadata: {
        system_name: string;
        dau: number;
        compliance_requirements: string[];
        confidence_score?: number;
    };
    executive_summary: string;
    system_diagram: {
        format: string;
        diagram: string;
    };
    components: Array<{
        name: string;
        purpose: string;
        technology: string;
        sla: Record<string, string>;
    }>;
    technology_stack: Array<{
        component: string;
        technology: string;
        reasoning: string;
        pros: string[];
        cons: string[];
        cost_monthly_usd: number;
    }>;
    cost_model: {
        total_monthly_cost: {
            total_usd: number;
            llm_cost_usd: number;
            infrastructure_cost_usd: number;
        };
    };
    risk_mitigation: Array<{
        risk: string;
        probability: string;
        impact: string;
        mitigation: string[];
        rto: string;
    }>;
    deployment_architecture: any;
    scalability_strategy: any;
    observability_plan: any;
    security_compliance: any;
    future_evolution: any;
}

interface ArchitecturePlanProps {
    architecture: ArchitecturePlan | null;
    loading?: boolean;
    error?: string | null;
    onGenerateRunbook?: (targetCloud: string) => void;
}

export function ArchitecturePlanDisplay({
    architecture,
    loading = false,
    error = null,
    onGenerateRunbook,
}: ArchitecturePlanProps) {
    const [expandedRisk, setExpandedRisk] = useState<number | null>(null);
    const [selectedCloud, setSelectedCloud] = useState<string>("gcp");

    if (loading) {
        return (
            <div className="space-y-4 p-6">
                <Skeleton className="h-8 w-80" />
                <Skeleton className="h-64 w-full" />
                <Skeleton className="h-32 w-full" />
            </div>
        );
    }

    if (error) {
        return (
            <Alert className="bg-red-50 border-red-200">
                <p className="text-red-800">Architecture generation failed: {error}</p>
            </Alert>
        );
    }

    if (!architecture) {
        return (
            <Alert className="bg-blue-50 border-blue-200">
                <p className="text-blue-800">
                    No architecture plan generated yet. Generate one from your research findings above.
                </p>
            </Alert>
        );
    }

    const totalMonthlyCost = (
        architecture.cost_model.total_monthly_cost.total_usd || 0
    ).toFixed(2);
    const llmCost = (
        architecture.cost_model.total_monthly_cost.llm_cost_usd || 0
    ).toFixed(2);
    const infraCost = (
        architecture.cost_model.total_monthly_cost.infrastructure_cost_usd || 0
    ).toFixed(2);

    return (
        <div className="space-y-6 p-6 bg-slate-900 rounded-lg border border-slate-700">
            {/* Header */}
            <div>
                <h2 className="text-2xl font-bold text-white mb-2">
                    {architecture.metadata.system_name} - Production Architecture
                </h2>
                <div className="flex items-center gap-4 text-sm text-slate-300">
                    <span>📊 {architecture.metadata.dau.toLocaleString()} DAU</span>
                    <span>🔐 {architecture.metadata.compliance_requirements.join(", ")}</span>
                    {architecture.metadata.confidence_score && (
                        <span>✓ {(architecture.metadata.confidence_score * 100).toFixed(0)}% confidence</span>
                    )}
                </div>
            </div>

            {/* Tabs for different sections */}
            <Tabs defaultValue="summary" className="w-full">
                <div className="flex gap-2 border-b border-slate-700 overflow-x-auto">
                    <button
                        onClick={() => { }}
                        className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white border-b-2 border-transparent hover:border-blue-500"
                        data-value="summary"
                    >
                        Summary
                    </button>
                    <button
                        onClick={() => { }}
                        className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white border-b-2 border-transparent hover:border-blue-500"
                        data-value="diagram"
                    >
                        System Diagram
                    </button>
                    <button
                        onClick={() => { }}
                        className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white border-b-2 border-transparent hover:border-blue-500"
                        data-value="tech-stack"
                    >
                        Tech Stack
                    </button>
                    <button
                        onClick={() => { }}
                        className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white border-b-2 border-transparent hover:border-blue-500"
                        data-value="costs"
                    >
                        Costs
                    </button>
                    <button
                        onClick={() => { }}
                        className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white border-b-2 border-transparent hover:border-blue-500"
                        data-value="risks"
                    >
                        Risks
                    </button>
                </div>

                {/* Summary Tab */}
                <div className="mt-6">
                    <div className="prose prose-invert max-w-none">
                        <div className="bg-slate-800 p-4 rounded border border-slate-700 text-slate-100 text-sm leading-relaxed">
                            <p>{architecture.executive_summary}</p>
                        </div>
                    </div>
                </div>
            </Tabs>

            {/* Tech Stack Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="bg-slate-800 border-slate-700">
                    <div className="p-4">
                        <h3 className="font-bold text-white mb-3">Key Technologies</h3>
                        <div className="space-y-2">
                            {architecture.technology_stack.slice(0, 5).map((tech, idx) => (
                                <div key={idx} className="text-sm text-slate-300">
                                    <p className="font-semibold text-blue-400">{tech.component}</p>
                                    <p className="text-xs text-slate-400">{tech.technology}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </Card>

                <Card className="bg-slate-800 border-slate-700">
                    <div className="p-4">
                        <h3 className="font-bold text-white mb-3">Monthly Costs</h3>
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm text-slate-300">
                                <span>LLM & Models</span>
                                <span className="text-green-400 font-semibold">${llmCost}</span>
                            </div>
                            <div className="flex justify-between text-sm text-slate-300">
                                <span>Infrastructure</span>
                                <span className="text-green-400 font-semibold">${infraCost}</span>
                            </div>
                            <div className="h-px bg-slate-700 my-2" />
                            <div className="flex justify-between text-base font-bold text-white">
                                <span>Total</span>
                                <span className="text-green-400">${totalMonthlyCost}</span>
                            </div>
                        </div>
                    </div>
                </Card>
            </div>

            {/* System Diagram */}
            <div>
                <h3 className="text-lg font-bold text-white mb-3">System Architecture</h3>
                <Card className="bg-slate-800 border-slate-700 p-4">
                    <pre className="text-xs text-slate-300 overflow-x-auto max-h-96">
                        {architecture.system_diagram.diagram}
                    </pre>
                </Card>
            </div>

            {/* Components */}
            <div>
                <h3 className="text-lg font-bold text-white mb-3">Core Components</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {architecture.components.slice(0, 6).map((comp, idx) => (
                        <Card key={idx} className="bg-slate-800 border-slate-700 p-3">
                            <h4 className="font-bold text-blue-400 text-sm mb-1">{comp.name}</h4>
                            <p className="text-xs text-slate-300 mb-2">{comp.purpose}</p>
                            <p className="text-xs text-slate-400">
                                <strong>Tech:</strong> {comp.technology}
                            </p>
                            <p className="text-xs text-slate-400">
                                <strong>SLA:</strong> {comp.sla.latency_p99 || comp.sla.availability}
                            </p>
                        </Card>
                    ))}
                </div>
            </div>

            {/* Risk Mitigation */}
            <div>
                <h3 className="text-lg font-bold text-white mb-3">Production Risks & Mitigation</h3>
                <div className="space-y-2">
                    {architecture.risk_mitigation.slice(0, 4).map((risk, idx) => (
                        <Card
                            key={idx}
                            className="bg-slate-800 border-slate-700 p-3 cursor-pointer hover:bg-slate-750"
                            onClick={() => setExpandedRisk(expandedRisk === idx ? null : idx)}
                        >
                            <div className="flex items-start justify-between">
                                <div>
                                    <h4 className="font-bold text-red-400 text-sm">{risk.risk}</h4>
                                    <p className="text-xs text-slate-400 mt-1">
                                        <strong>Probability:</strong> {risk.probability} | <strong>RTO:</strong> {risk.rto}
                                    </p>
                                </div>
                                <span className="text-xl text-slate-500">
                                    {expandedRisk === idx ? "−" : "+"}
                                </span>
                            </div>
                            {expandedRisk === idx && (
                                <div className="mt-3 pt-3 border-t border-slate-700">
                                    <p className="text-xs text-yellow-300 mb-2">
                                        <strong>Impact:</strong> {risk.impact}
                                    </p>
                                    <p className="text-xs text-slate-300">
                                        <strong>Mitigation:</strong>
                                    </p>
                                    <ul className="list-disc list-inside text-xs text-slate-400 mt-1">
                                        {risk.mitigation.map((m, midx) => (
                                            <li key={midx}>{m}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </Card>
                    ))}
                </div>
            </div>

            {/* Deployment Runbook */}
            <div className="bg-slate-800 border border-slate-700 p-4 rounded">
                <h3 className="text-lg font-bold text-white mb-3">Deployment Guide</h3>
                <div className="space-y-3">
                    <p className="text-sm text-slate-300">
                        Generate a deployment runbook for your target cloud platform:
                    </p>
                    <div className="flex gap-2">
                        {["gcp", "aws", "azure"].map((cloud) => (
                            <Button
                                key={cloud}
                                variant={selectedCloud === cloud ? "default" : "outline"}
                                size="sm"
                                onClick={() => setSelectedCloud(cloud)}
                                className="uppercase"
                            >
                                {cloud}
                            </Button>
                        ))}
                    </div>
                    <Button
                        onClick={() => onGenerateRunbook?.(selectedCloud)}
                        className="w-full"
                    >
                        Generate {selectedCloud.toUpperCase()} Runbook
                    </Button>
                </div>
            </div>

            {/* Scalability Roadmap */}
            <div>
                <h3 className="text-lg font-bold text-white mb-3">Scalability Roadmap</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    {Object.entries(architecture.scalability_strategy)
                        .filter(([key]) => key.includes("phase"))
                        .map(([phase, details]: [string, any], idx) => (
                            <Card key={idx} className="bg-slate-800 border-slate-700 p-3">
                                <h4 className="font-bold text-blue-400 text-sm mb-2 capitalize">
                                    {phase.replace(/_/g, " ")}
                                </h4>
                                <div className="text-xs text-slate-300 space-y-1">
                                    {typeof details === "object" &&
                                        Object.entries(details)
                                            .slice(0, 3)
                                            .map(([key, val]: [string, any], vidx) => (
                                                <div key={vidx}>
                                                    <strong className="text-slate-400">{key.replace(/_/g, " ")}:</strong>
                                                    <p className="text-slate-500 line-clamp-1">
                                                        {typeof val === "string" ? val : JSON.stringify(val).substring(0, 50)}
                                                    </p>
                                                </div>
                                            ))}
                                </div>
                            </Card>
                        ))}
                </div>
            </div>

            {/* Call to Action */}
            <div className="bg-blue-900 border border-blue-700 p-4 rounded text-center">
                <p className="text-sm text-blue-100 mb-3">
                    Ready to deploy? Generate a detailed runbook and start building production infrastructure.
                </p>
                <div className="flex gap-2 justify-center">
                    <Button variant="default">Download Architecture PDF</Button>
                    <Button variant="outline">Share with Team</Button>
                </div>
            </div>
        </div>
    );
}

export default ArchitecturePlanDisplay;
