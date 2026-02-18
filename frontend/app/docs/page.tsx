import Link from "next/link";
import { Button } from "@/components/ui/button";

const FEATURES = [
    {
        title: "Dual Research Modes",
        description: "Quick mode optimizes for speed, deep mode for coverage with reflexion and validation.",
    },
    {
        title: "Hierarchical Task Graph",
        description: "LLM-driven planning builds a DAG of research tasks with budgets and dependencies.",
    },
    {
        title: "Evidence Graph",
        description: "Claims, sources, and relations are linked for verifiable provenance.",
    },
    {
        title: "Reflexion Engine",
        description: "Self-critique loop can trigger replanning when gaps are detected.",
    },
    {
        title: "Adaptive Model Routing",
        description: "Model tiers route tasks to fast or large models based on complexity.",
    },
    {
        title: "Production Metrics",
        description: "Latency, tokens, cost, and task-graph stats emitted per run.",
    },
];

const BULLETS = [
    "Multi-hop research orchestration",
    "Claim-level provenance links",
    "Model escalation logic",
    "Execution telemetry",
    "Graceful degradation on missing services",
];

export default function DocsPage() {
    return (
        <div className="space-y-16 pb-20">
            <section id="overview" className="rounded-3xl border border-slate-800/60 bg-gradient-to-br from-slate-900/80 via-slate-950 to-slate-900/60 px-10 py-12">
                <div className="space-y-6">
                    <p className="text-xs uppercase tracking-widest text-emerald-300">Documentation</p>
                    <h1 className="text-4xl font-semibold text-white md:text-5xl">Deep Research Agent Documentation</h1>
                    <p className="max-w-2xl text-base text-slate-300">
                        Autonomous hierarchical research system with verifiable provenance.
                    </p>
                    <div className="flex flex-wrap gap-3">
                        <Link href="#quick-start">
                            <Button className="bg-emerald-400 text-slate-950 hover:bg-emerald-300">Get Started</Button>
                        </Link>
                        <Link href="https://github.com" target="_blank" rel="noreferrer">
                            <Button variant="outline" className="border-slate-700 text-slate-200 hover:bg-slate-900">
                                View GitHub
                            </Button>
                        </Link>
                    </div>
                </div>
            </section>

            <section id="features" className="space-y-8">
                <div>
                    <h2 className="text-2xl font-semibold text-white">Core capabilities</h2>
                    <p className="text-sm text-slate-400">Aligned to the current V2 implementation.</p>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                    {FEATURES.map((feature) => (
                        <div
                            key={feature.title}
                            className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5 transition hover:border-emerald-400/40 hover:bg-slate-900/70"
                        >
                            <h3 className="text-lg font-semibold text-white">{feature.title}</h3>
                            <p className="mt-2 text-sm text-slate-300">{feature.description}</p>
                        </div>
                    ))}
                </div>
            </section>

            <section id="architecture" className="space-y-6">
                <div>
                    <h2 className="text-2xl font-semibold text-white">Architecture</h2>
                    <p className="text-sm text-slate-400">
                        The V2 orchestrator coordinates planning, execution, evidence, and memory services.
                    </p>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                    <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5">
                        <h3 className="text-lg font-semibold text-white">Planning</h3>
                        <p className="mt-2 text-sm text-slate-300">
                            HierarchicalPlannerAgent decomposes the query into goals and task nodes with dependencies and budgets.
                        </p>
                    </div>
                    <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5">
                        <h3 className="text-lg font-semibold text-white">Execution</h3>
                        <p className="mt-2 text-sm text-slate-300">
                            TaskExecutor routes tasks to agents (search, validation, extraction, synthesis) and runs them in parallel.
                        </p>
                    </div>
                    <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5">
                        <h3 className="text-lg font-semibold text-white">Evidence</h3>
                        <p className="mt-2 text-sm text-slate-300">
                            EvidenceGraph links claims to sources with supports/contradicts/mentions relations.
                        </p>
                    </div>
                    <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5">
                        <h3 className="text-lg font-semibold text-white">Memory</h3>
                        <p className="mt-2 text-sm text-slate-300">
                            Supabase stores sessions, sources, claims, and snapshots, while Qdrant holds semantic vectors.
                        </p>
                    </div>
                </div>
            </section>

            <section id="installation" className="space-y-6">
                <div>
                    <h2 className="text-2xl font-semibold text-white">Installation</h2>
                    <p className="text-sm text-slate-400">Python 3.11+ required.</p>
                </div>
                <div className="grid gap-4 lg:grid-cols-2">
                    <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5">
                        <h3 className="text-lg font-semibold text-white">Backend</h3>
                        <div className="mt-3 space-y-3 text-sm text-slate-200">
                            <div className="rounded-xl border border-slate-800/80 bg-slate-950 px-4 py-3">
                                <code>pip install -r requirements.txt</code>
                            </div>
                            <div className="rounded-xl border border-slate-800/80 bg-slate-950 px-4 py-3">
                                <code>cp .env.example .env</code>
                            </div>
                        </div>
                    </div>
                    <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5">
                        <h3 className="text-lg font-semibold text-white">Frontend</h3>
                        <div className="mt-3 space-y-3 text-sm text-slate-200">
                            <div className="rounded-xl border border-slate-800/80 bg-slate-950 px-4 py-3">
                                <code>cd frontend</code>
                            </div>
                            <div className="rounded-xl border border-slate-800/80 bg-slate-950 px-4 py-3">
                                <code>npm install</code>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            <section id="quick-start" className="grid gap-8 lg:grid-cols-2">
                <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-6">
                    <h2 className="text-xl font-semibold text-white">Quick Start</h2>
                    <div className="mt-4 space-y-4 text-sm text-slate-200">
                        <div className="rounded-xl border border-slate-800/80 bg-slate-950 px-4 py-3">
                            <code>python server.py</code>
                        </div>
                        <div className="rounded-xl border border-slate-800/80 bg-slate-950 px-4 py-3">
                            <code>cd frontend && npm run dev</code>
                        </div>
                        <div className="text-xs text-slate-400">
                            API runs at http://localhost:8000, frontend at http://localhost:3000.
                        </div>
                    </div>
                </div>
                <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-6">
                    <h2 className="text-xl font-semibold text-white">Why Deep Research Agent?</h2>
                    <ul className="mt-4 space-y-2 text-sm text-slate-300">
                        {BULLETS.map((bullet) => (
                            <li key={bullet} className="flex items-start gap-2">
                                <span className="mt-1 h-2 w-2 rounded-full bg-emerald-400" />
                                {bullet}
                            </li>
                        ))}
                    </ul>
                </div>
            </section>

            <section id="dual-research-modes" className="space-y-6">
                <div>
                    <h2 className="text-2xl font-semibold text-white">Dual research modes</h2>
                    <p className="text-sm text-slate-400">Mode affects budgets, planning depth, and validation.</p>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                    <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5">
                        <h3 className="text-lg font-semibold text-white">Quick mode</h3>
                        <ul className="mt-3 space-y-2 text-sm text-slate-300">
                            <li>max_time_ms: 60,000</li>
                            <li>max_iterations: 1</li>
                            <li>min_sources: 3</li>
                            <li>top_k: 3</li>
                            <li>Skips validation and reflexion tasks</li>
                            <li>Biases model tiers to fast model</li>
                        </ul>
                    </div>
                    <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5">
                        <h3 className="text-lg font-semibold text-white">Deep mode</h3>
                        <ul className="mt-3 space-y-2 text-sm text-slate-300">
                            <li>max_time_ms: 480,000</li>
                            <li>max_iterations: 3</li>
                            <li>min_sources: 6</li>
                            <li>top_k: 8</li>
                            <li>Includes validation + reflexion</li>
                            <li>Uses full model tier routing</li>
                        </ul>
                    </div>
                </div>
            </section>

            <section id="task-graph" className="space-y-6">
                <div>
                    <h2 className="text-2xl font-semibold text-white">Task graph & planning</h2>
                    <p className="text-sm text-slate-400">HierarchicalPlannerAgent builds a DAG of tasks.</p>
                </div>
                <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5 text-sm text-slate-300">
                    <ul className="space-y-2">
                        <li>Search tasks: SEARCH_WEB, SEARCH_ACADEMIC, SEARCH_TECHNICAL, SEARCH_CITATION</li>
                        <li>Extraction: EXTRACT_CLAIMS (depends on search results)</li>
                        <li>Validation: VALIDATE_CLAIMS (deep mode only)</li>
                        <li>Merge: MERGE_EVIDENCE (internal aggregation)</li>
                        <li>Reflexion: REFLEXION (deep mode only)</li>
                        <li>Synthesis: SYNTHESIZE_REPORT (final report)</li>
                    </ul>
                </div>
            </section>

            <section id="evidence-graph" className="space-y-6">
                <div>
                    <h2 className="text-2xl font-semibold text-white">Evidence graph</h2>
                    <p className="text-sm text-slate-400">Claims and sources are linked with explicit relations.</p>
                </div>
                <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5 text-sm text-slate-300">
                    <ul className="space-y-2">
                        <li>Claim: text, normalized_text, confidence, provenance</li>
                        <li>Source: url, title, reliability_score, excerpt, metadata</li>
                        <li>EvidenceEdge: supports, contradicts, mentions</li>
                    </ul>
                </div>
            </section>

            <section id="reflexion" className="space-y-6">
                <div>
                    <h2 className="text-2xl font-semibold text-white">Reflexion & replanning</h2>
                    <p className="text-sm text-slate-400">Critique loop can trigger task graph replanning.</p>
                </div>
                <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5 text-sm text-slate-300">
                    <ul className="space-y-2">
                        <li>Critique outputs: quality, coverage, gaps, refined_queries</li>
                        <li>Replan triggers when gaps &gt;= 2 or quality &lt; 0.5</li>
                        <li>Disabled in quick mode</li>
                    </ul>
                </div>
            </section>

            <section id="model-routing" className="space-y-6">
                <div>
                    <h2 className="text-2xl font-semibold text-white">Model routing</h2>
                    <p className="text-sm text-slate-400">LLMClient routes tasks via model tiers and env config.</p>
                </div>
                <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5 text-sm text-slate-300">
                    <ul className="space-y-2">
                        <li>DEFAULT_MODEL and FAST_MODEL define baseline tiers</li>
                        <li>Quick mode forces all tiers to fast model</li>
                        <li>Providers supported via env keys (Gemini, Together, OpenRouter, Cerebras)</li>
                    </ul>
                </div>
            </section>

            <section id="supabase" className="space-y-6">
                <div>
                    <h2 className="text-2xl font-semibold text-white">Supabase integration</h2>
                    <p className="text-sm text-slate-400">SupabaseStorage persists sessions, sources, claims, and edges.</p>
                </div>
                <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5 text-sm text-slate-300">
                    <ul className="space-y-2">
                        <li>Sessions include query_text, iterations, metadata JSONB</li>
                        <li>Sources and claims are stored with embeddings and reliability scores</li>
                        <li>Evidence edges link claims to sources</li>
                    </ul>
                </div>
            </section>

            <section id="qdrant" className="space-y-6">
                <div>
                    <h2 className="text-2xl font-semibold text-white">Qdrant vector store</h2>
                    <p className="text-sm text-slate-400">Semantic memory uses Qdrant for similarity search.</p>
                </div>
                <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5 text-sm text-slate-300">
                    <ul className="space-y-2">
                        <li>QDRANT_URL and QDRANT_API_KEY for cloud, else local at http://localhost:6333</li>
                        <li>Collection name: research_memory</li>
                        <li>Vector size controlled by EMBEDDING_DIM (default 1536)</li>
                    </ul>
                </div>
            </section>

            <section id="persistent-memory" className="space-y-6">
                <div>
                    <h2 className="text-2xl font-semibold text-white">Persistent memory</h2>
                    <p className="text-sm text-slate-400">Summary snapshots compress long sessions for recall.</p>
                </div>
                <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5 text-sm text-slate-300">
                    <ul className="space-y-2">
                        <li>SummarySnapshot stores compressed_text, embeddings, and claim IDs</li>
                        <li>MemoryAPI coordinates Supabase + Qdrant with in-memory fallbacks</li>
                    </ul>
                </div>
            </section>

            <section id="api" className="space-y-6">
                <div>
                    <h2 className="text-2xl font-semibold text-white">API endpoints</h2>
                    <p className="text-sm text-slate-400">FastAPI server in server.py.</p>
                </div>
                <div className="space-y-3 text-sm text-slate-200">
                    <div className="rounded-xl border border-slate-800/80 bg-slate-950 px-4 py-3">
                        <code>POST /api/research</code>
                    </div>
                    <div className="rounded-xl border border-slate-800/80 bg-slate-950 px-4 py-3">
                        <code>GET /api/research/&lt;session_id&gt;</code>
                    </div>
                    <div className="rounded-xl border border-slate-800/80 bg-slate-950 px-4 py-3">
                        <code>GET /api/history</code>
                    </div>
                    <div className="rounded-xl border border-slate-800/80 bg-slate-950 px-4 py-3">
                        <code>GET /api/health</code>
                    </div>
                </div>
            </section>

            <section id="env" className="space-y-6">
                <div>
                    <h2 className="text-2xl font-semibold text-white">Environment variables</h2>
                    <p className="text-sm text-slate-400">Configure providers and storage in .env.</p>
                </div>
                <div className="grid gap-4 lg:grid-cols-2">
                    <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5 text-sm text-slate-300">
                        <p className="text-xs uppercase tracking-widest text-slate-400">LLM Providers</p>
                        <ul className="mt-3 space-y-2">
                            <li>OPENROUTER_API_KEY</li>
                            <li>GEMINI_API_KEY</li>
                            <li>TOGETHER_API_KEY</li>
                            <li>CEREBRAS_API_KEY</li>
                        </ul>
                    </div>
                    <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5 text-sm text-slate-300">
                        <p className="text-xs uppercase tracking-widest text-slate-400">Search</p>
                        <ul className="mt-3 space-y-2">
                            <li>EXA_API_KEY</li>
                            <li>TAVILY_API_KEY</li>
                            <li>FIRECRAWL_API_KEY</li>
                        </ul>
                    </div>
                    <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5 text-sm text-slate-300">
                        <p className="text-xs uppercase tracking-widest text-slate-400">Model routing</p>
                        <ul className="mt-3 space-y-2">
                            <li>DEFAULT_MODEL</li>
                            <li>FAST_MODEL</li>
                        </ul>
                    </div>
                    <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5 text-sm text-slate-300">
                        <p className="text-xs uppercase tracking-widest text-slate-400">Storage</p>
                        <ul className="mt-3 space-y-2">
                            <li>SUPABASE_URL</li>
                            <li>SUPABASE_KEY</li>
                            <li>QDRANT_URL (optional)</li>
                            <li>QDRANT_API_KEY (optional)</li>
                            <li>EMBEDDING_DIM (optional)</li>
                        </ul>
                    </div>
                </div>
            </section>

            <section id="docker" className="space-y-6">
                <div>
                    <h2 className="text-2xl font-semibold text-white">Docker deployment</h2>
                    <p className="text-sm text-slate-400">Kestra workflow engine is provided via docker-compose.yaml.</p>
                </div>
                <div className="rounded-2xl border border-slate-800/60 bg-slate-900/40 p-5 text-sm text-slate-300">
                    <div className="rounded-xl border border-slate-800/80 bg-slate-950 px-4 py-3">
                        <code>docker compose up -d</code>
                    </div>
                    <p className="mt-3 text-xs text-slate-400">
                        This spins up Kestra plus Postgres for local orchestration flows in /kestra.
                    </p>
                </div>
            </section>
        </div>
    );
}
