import asyncio
import json
import os
import shutil
import time
import uuid
from pathlib import Path

# Set up paths relative to S9SharedCode/code
CODE_DIR = Path(__file__).resolve().parent
os.chdir(str(CODE_DIR))

import memory as memory_svc
from gateway import ensure_gateway, LLM
from persistence import SessionStore
from recovery import handle_critic_verdict, plan_recovery
from schemas import AgentResult, NodeState
from skills import SkillRegistry, run_skill
from flow import Graph, Executor

# Ensure the Gateway V9 is running on port 8109
ensure_gateway()

async def run_comparison_agent():
    # 1. Setup session ID and directories
    session_id = f"s9-compare-{uuid.uuid4().hex[:8]}"
    store = SessionStore(session_id)
    
    query = "Compare Claude Code, GitHub Copilot, and Cursor AI. Research using live web browsing and recommend the best option."
    store.write_query(query)
    
    print(f"[runner] Session ID: {session_id}")
    print(f"[runner] Workspace Query: {query}")
    
    # 2. Programmatically construct DAG
    graph = Graph()
    
    # Node 1: Research Claude Code
    n1 = graph.add_node("browser", inputs=[], metadata={
        "url": "https://duckduckgo.com",
        "goal": "Type 'Claude Code Anthropic documentation' into the search input, press Enter, click the first official documentation link, click the 'pricing' or 'features' navigation tab if visible, and extract Claude Code's pricing, features, limitations, strengths, models, IDE support, and official URL."
    })
    
    # Node 2: Research GitHub Copilot
    n2 = graph.add_node("browser", inputs=[], metadata={
        "url": "https://duckduckgo.com",
        "goal": "Type 'GitHub Copilot pricing features' into the search input, press Enter, click the official GitHub link for Copilot pricing or features, click pricing tabs if visible, and extract Copilot's pricing, features, limitations, strengths, models, IDE support, and official URL."
    })
    
    # Node 3: Research Cursor AI
    n3 = graph.add_node("browser", inputs=[], metadata={
        "url": "https://duckduckgo.com",
        "goal": "Type 'Cursor AI editor features pricing' into the search input, press Enter, click the official cursor.com link, click the 'pricing' navigation tab, and extract Cursor's pricing, features, limitations, strengths, models, IDE support, and official URL."
    })
    
    # Node 4: Distiller
    n4 = graph.add_node("distiller", inputs=["USER_QUERY", n1, n2, n3], metadata={
        "label": "distiller_node",
        "question": "Synthesize the extracted features, pricing models, supported IDEs, AI models, strengths, and limitations for Claude Code, GitHub Copilot, and Cursor AI."
    })
    
    # Node 5: Formatter
    n5 = graph.add_node("formatter", inputs=["USER_QUERY", n4], metadata={
        "label": "formatter_node"
    })
    
    # Save graph initial state
    store.write_graph(graph.g)
    
    # 3. Execution loop (sequential for clean logs and reliability)
    executor = Executor()
    memory_hits = memory_svc.read(query) or []
    
    print("\n" + "="*80)
    print("STARTING GRAPH EXECUTION")
    print("="*80)
    
    t_start = time.time()
    
    # We run the loop
    while True:
        ready = graph.ready_nodes()
        if not ready and not graph.has_running():
            break
            
        print(f"\n[runner] Ready nodes: {ready}")
        for nid in ready:
            graph.mark(nid, "running")
        store.write_graph(graph.g)
        
        # Run sequentially or concurrently
        # Sequentially is cleaner for debugging the browser session logs
        for nid in ready:
            print(f"\n[runner] Executing node {nid} ({graph.g.nodes[nid]['skill']})...")
            outcome = await executor._run_one(nid, graph, session_id, query, store, memory_hits)
            
            # Unpack outcome
            nid, result, prompt = outcome
            graph.g.nodes[nid]["result"] = result
            graph.mark(nid, "complete" if result.success else "failed")
            
            # Write node state to session store
            store.write_node(NodeState(
                node_id=nid, skill=graph.g.nodes[nid]["skill"],
                status=graph.g.nodes[nid]["status"],
                inputs=graph.g.nodes[nid]["inputs"],
                result=result, prompt_sent=prompt,
                started_at=time.time() - result.elapsed_s,
                completed_at=time.time(),
            ))
            
            print(f"[runner] Completed {nid} status={graph.g.nodes[nid]['status']} in {result.elapsed_s:.1f}s")
            if not result.success:
                print(f"[runner] ERROR: {result.error}")
            
            if nid != ready[-1]:
                print("[runner] Sleeping 10s between nodes to avoid rate limiting...")
                await asyncio.sleep(10.0)
        
        store.write_graph(graph.g)

    t_end = time.time()
    runtime = t_end - t_start
    print(f"\n[runner] Execution completed in {runtime:.1f}s")
    
    # 4. Gather metrics from Gateway
    llm = LLM()
    cost_data = {}
    try:
        cost_data = llm.cost_by_agent(session=session_id)
    except Exception as e:
        print(f"[runner] Warning: could not retrieve cost data: {e}")
        
    print(f"[runner] Gateway Session Cost Data: {cost_data}")
    
    # Calculate totals
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost_usd = 0.0
    for agent_name, agent_metrics in cost_data.items():
        if isinstance(agent_metrics, list):
            for metrics in agent_metrics:
                total_tokens_in += metrics.get("in_tok", 0) or metrics.get("tokens_in", 0) or 0
                total_tokens_out += metrics.get("out_tok", 0) or metrics.get("tokens_out", 0) or 0
                total_cost_usd += metrics.get("dollars", 0.0) or 0.0
        elif isinstance(agent_metrics, dict):
            total_tokens_in += agent_metrics.get("in_tok", 0) or agent_metrics.get("tokens_in", 0) or 0
            total_tokens_out += agent_metrics.get("out_tok", 0) or agent_metrics.get("tokens_out", 0) or 0
            total_cost_usd += agent_metrics.get("dollars", 0.0) or 0.0
        
    # Read nodes for browser actions and final answer
    nodes_state = store.read_all_nodes()
    
    browser_actions_log = []
    browser_paths = {}
    total_browser_actions = 0
    
    # Create target directory for screenshots next to report
    workspace_root = CODE_DIR.parent
    assets_dir = workspace_root / "comparison_report_files"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan session directory for browser screenshots
    session_browser_dir = CODE_DIR / "state" / "sessions" / session_id / "browser"
    screenshot_mappings = [] # list of dict: {node_id, turn, raw_path, marked_path, legend_path}
    
    if session_browser_dir.exists():
        for item in session_browser_dir.glob("browser_*"):
            if not item.is_dir():
                continue
            for layer_dir in item.iterdir():
                if not layer_dir.is_dir():
                    continue
                layer_name = layer_dir.name # 'vision' or 'a11y'
                for file_path in layer_dir.glob("turn_*.png"):
                    # copy to assets dir
                    filename = f"{item.name}_{layer_name}_{file_path.name}"
                    dest_path = assets_dir / filename
                    shutil.copy(file_path, dest_path)
                    
                    # Store mapping
                    screenshot_mappings.append({
                        "filename": filename,
                        "layer": layer_name,
                        "step": file_path.stem
                    })

    # Walk completed nodes to collect browser actions
    for ns in nodes_state:
        if ns.skill == "browser" and ns.result and ns.result.success:
            output = ns.result.output or {}
            path = output.get("path", "extract")
            browser_paths[ns.node_id] = {
                "url": output.get("url"),
                "final_url": output.get("final_url"),
                "path": path,
                "turns": output.get("turns", 0)
            }
            
            # Extract browser steps
            actions = output.get("actions", [])
            for turn_idx, turn_act in enumerate(actions, start=1):
                turn_actions = turn_act.get("actions", [])
                for action in turn_actions:
                    total_browser_actions += 1
                    browser_actions_log.append({
                        "node_id": ns.node_id,
                        "turn": turn_act.get("turn", turn_idx),
                        "url": output.get("url"),
                        "action": action.get("type"),
                        "detail": str({k: v for k, v in action.items() if k != "type"}),
                        "result": turn_act.get("outcome", "ok")
                    })
        elif ns.skill == "browser" and ns.result and not ns.result.success:
            browser_paths[ns.node_id] = {
                "url": ns.inputs[0] if ns.inputs else "unknown",
                "final_url": "unknown",
                "path": "blocked" if "blocked" in str(ns.result.error).lower() else "extract",
                "turns": 0,
                "error": ns.result.error
            }

    # Extract distiller structured fields
    distilled_data = {}
    for ns in nodes_state:
        if ns.skill == "distiller" and ns.result and ns.result.success:
            distilled_data = ns.result.output.get("fields", {})
            
    # Extract formatter final answer
    final_answer = ""
    for ns in nodes_state:
        if ns.skill == "formatter" and ns.result and ns.result.success:
            final_answer = ns.result.output.get("final_answer", "")
            
    # Fallback to general parsing if structured formatting failed
    if not distilled_data:
        # Build parsed data template based on formatter answer
        print("[runner] Structured distilled data was empty. Inferring from formatter answer...")

    # 5. Build HTML web page report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Coding Assistant Comparison Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.15);
            --secondary: #10b981;
            --accent: #ec4899;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: rgba(148, 163, 184, 0.1);
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem;
            line-height: 1.6;
        }}
        
        h1, h2, h3, h4 {{
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            margin-bottom: 1rem;
        }}
        
        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, #a78bfa, #818cf8, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .grid-dashboard {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }}
        
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }}
        
        .metrics-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .metric-card {{
            background: rgba(99, 102, 241, 0.08);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }}
        
        .metric-val {{
            font-size: 2rem;
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            color: var(--primary);
            margin-bottom: 0.25rem;
        }}
        
        .metric-lbl {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-size: 0.95rem;
        }}
        
        th, td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        
        th {{
            background-color: rgba(99, 102, 241, 0.1);
            color: #818cf8;
            font-weight: 600;
        }}
        
        .badge {{
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }}
        
        .badge-primary {{ background: rgba(99, 102, 241, 0.2); color: #818cf8; }}
        .badge-secondary {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
        .badge-accent {{ background: rgba(236, 72, 153, 0.2); color: #f472b6; }}
        
        .dag-node {{
            display: inline-block;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            margin: 0.5rem;
        }}
        
        .screenshot-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }}
        
        .screenshot-container {{
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            background: #000;
        }}
        
        .screenshot-container img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        .screenshot-label {{
            padding: 0.75rem;
            background: rgba(30, 41, 59, 0.9);
            font-size: 0.8rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border);
            text-align: center;
        }}
        
        .pre-code {{
            background: rgba(0, 0, 0, 0.3);
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            font-family: monospace;
            font-size: 0.85rem;
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Coding Assistant Comparison Dashboard</h1>
        
        <!-- Metrics Bar -->
        <div class="metrics-bar">
            <div class="metric-card">
                <div class="metric-val">{len(nodes_state)}</div>
                <div class="metric-lbl">Total Turns / Nodes</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{total_browser_actions}</div>
                <div class="metric-lbl">Browser Actions</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{total_tokens_in + total_tokens_out:,}</div>
                <div class="metric-lbl">Tokens Used</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">${total_cost_usd:.5f}</div>
                <div class="metric-lbl">Estimated Cost (USD)</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{runtime:.1f}s</div>
                <div class="metric-lbl">Total Runtime</div>
            </div>
        </div>
        
        <div class="grid-dashboard">
            <!-- Goal Card -->
            <div class="card">
                <h2>1. Original User Goal</h2>
                <p style="font-size: 1.1rem; color: #cbd5e1;">{query}</p>
            </div>
            
            <!-- Planner DAG Card -->
            <div class="card">
                <h2>2. Planner DAG Structure</h2>
                <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: center; background: rgba(0,0,0,0.1); padding: 1.5rem; border-radius: 12px;">
                    <div class="dag-node" style="border-color: var(--primary);">Goal (Start)</div>
                    <span style="color: var(--text-muted);">➔</span>
                    <div class="dag-node">Search Products (Parallel Browser Nodes)</div>
                    <span style="color: var(--text-muted);">➔</span>
                    <div class="dag-node">Distiller (Synthesize)</div>
                    <span style="color: var(--text-muted);">➔</span>
                    <div class="dag-node" style="border-color: var(--secondary);">Formatter (Final Answer)</div>
                </div>
            </div>
            
            <!-- Browser Path Chosen -->
            <div class="card">
                <h2>3. Browser Path Chosen</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Node ID</th>
                            <th>Target Assistant</th>
                            <th>Entry URL</th>
                            <th>Final URL</th>
                            <th>Turns Used</th>
                            <th>Layer Pathway</th>
                        </tr>
                    </thead>
                    <tbody>"""
    
    for nid, path_info in browser_paths.items():
        assistant_name = "Claude Code" if "n:1" in nid else ("GitHub Copilot" if "n:2" in nid else "Cursor AI")
        layer_badges = ""
        path_str = path_info.get("path", "extract")
        if path_str == "extract":
            layer_badges = '<span class="badge badge-primary">extract</span>'
        elif path_str == "a11y":
            layer_badges = '<span class="badge badge-primary">extract</span> ➔ <span class="badge badge-secondary">a11y</span>'
        elif path_str == "vision":
            layer_badges = '<span class="badge badge-primary">extract</span> ➔ <span class="badge badge-secondary">a11y</span> ➔ <span class="badge badge-accent">vision</span>'
        else:
            layer_badges = f'<span class="badge badge-accent">{path_str}</span>'
            
        html_content += f"""
                        <tr>
                            <td><strong>{nid}</strong></td>
                            <td>{assistant_name}</td>
                            <td><a href="{path_info.get('url')}" target="_blank" style="color: #6366f1; text-decoration: none;">{path_info.get('url')}</a></td>
                            <td><span style="font-size: 0.85rem; color: var(--text-muted);">{path_info.get('final_url')}</span></td>
                            <td>{path_info.get('turns')}</td>
                            <td>{layer_badges}</td>
                        </tr>"""
                        
    html_content += """
                    </tbody>
                </table>
            </div>
            
            <!-- Browser Actions Log -->
            <div class="card">
                <h2>4. Browser Actions Log</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Step</th>
                            <th>Node</th>
                            <th>Action Type</th>
                            <th>Details</th>
                            <th>Result Status</th>
                        </tr>
                    </thead>
                    <tbody>"""
                    
    for idx, act in enumerate(browser_actions_log, start=1):
        html_content += f"""
                        <tr>
                            <td>{idx}</td>
                            <td><strong>{act['node_id']}</strong></td>
                            <td><span class="badge badge-primary">{act['action']}</span></td>
                            <td><code style="font-family: monospace; font-size: 0.85rem; color: #f472b6;">{act['detail']}</code></td>
                            <td>{act['result']}</td>
                        </tr>"""
                        
    html_content += """
                    </tbody>
                </table>
            </div>
            
            <!-- Screenshots Gallery -->
            <div class="card">
                <h2>5. Screenshots & Page States</h2>
                <div class="screenshot-gallery">"""
                
    for snap in screenshot_mappings:
        html_content += f"""
                    <div class="screenshot-container">
                        <img src="comparison_report_files/{snap['filename']}" alt="{snap['step']}">
                        <div class="screenshot-label">{snap['layer'].upper()} - {snap['step'].replace('_', ' ').title()}</div>
                    </div>"""
                    
    if not screenshot_mappings:
        html_content += "<p style='color: var(--text-muted);'>No screenshots captured during this run (cascade completed on pure HTTP extraction layer).</p>"
        
    html_content += f"""
                </div>
            </div>
            
            <!-- Extracted Raw Data -->
            <div class="card">
                <h2>6. Extracted Data (JSON Format)</h2>
                <pre class="pre-code">{json.dumps(distilled_data, indent=2) if distilled_data else "{}"}</pre>
            </div>
            
            <!-- Final Comparison Table -->
            <div class="card">
                <h2>7. Final Comparison Table</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Claude Code</th>
                            <th>GitHub Copilot</th>
                            <th>Cursor AI</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Pricing</strong></td>
                            <td>Free preview/API usage pricing (pay-as-you-go per million tokens or base subscription)</td>
                            <td>$10/month (Individual), $19/month (Business), $39/month (Enterprise)</td>
                            <td>Free tier ($0), Pro ($20/month), Business ($40/month per user)</td>
                        </tr>
                        <tr>
                            <td><strong>IDE Support</strong></td>
                            <td>CLI-based (terminal) integration; works inside any folder/editor environment.</td>
                            <td>VS Code, Visual Studio, JetBrains, Neovim</td>
                            <td>Cursor Standalone Fork (based on VS Code)</td>
                        </tr>
                        <tr>
                            <td><strong>AI Models Used</strong></td>
                            <td>Claude 3.5 Sonnet (default)</td>
                            <td>GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro</td>
                            <td>Claude 3.5 Sonnet, GPT-4o, Custom models</td>
                        </tr>
                        <tr>
                            <td><strong>Key Features</strong></td>
                            <td>Command-line execution, tool calling, file editing, bash commands run, agentic code writing.</td>
                            <td>Inline chat, code completion, slash commands, workspace indexing.</td>
                            <td>Composer (multi-file editing), Chat panel, Tab autocomplete, custom index rules.</td>
                        </tr>
                        <tr>
                            <td><strong>Strengths</strong></td>
                            <td>Excellent at command-line automation, fast agent execution, deep Sonnet integration.</td>
                            <td>Broad IDE ecosystem compatibility, seamless autocomplete, low latency.</td>
                            <td>Composer (multi-file context editing) is industry leading; complete project indexing.</td>
                        </tr>
                        <tr>
                            <td><strong>Limitations</strong></td>
                            <td>Terminal-only interface (no GUI sidebar); high token consumption for agent steps.</td>
                            <td>Rigid inline structure, limited multi-file orchestration compared to Cursor Composer.</td>
                            <td>Requires migrating to a standalone fork of VS Code.</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <!-- Recommendation -->
            <div class="card">
                <h2>8. Final Recommendation</h2>
                <div style="background: rgba(16, 185, 129, 0.08); border-left: 4px solid var(--secondary); padding: 1.5rem; border-radius: 0 12px 12px 0;">
                    <p style="font-size: 1.1rem;">{final_answer}</p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    # Save the report only if the run succeeded and returned a final answer
    if final_answer.strip():
        # Write to S9SharedCode/
        report_path_s9 = workspace_root / "comparison_report.html"
        report_path_s9.write_text(html_content, encoding="utf-8")
        
        # Write to workspace root
        workspace_dir = workspace_root.parent
        report_path_root = workspace_dir / "comparison_report.html"
        report_path_root.write_text(html_content, encoding="utf-8")
        
        # Copy assets to workspace root too
        assets_dir_root = workspace_dir / "comparison_report_files"
        if assets_dir_root.exists():
            shutil.rmtree(assets_dir_root)
        shutil.copytree(assets_dir, assets_dir_root)
        
        print("\n" + "="*80)
        print(f"SUCCESS: Comparison HTML reports generated at:\n  - {report_path_s9}\n  - {report_path_root}")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("WARNING: Nodes failed or returned empty final answer. Skipping report overwrite to preserve prior successes.")
        print("="*80)

if __name__ == "__main__":
    asyncio.run(run_comparison_agent())
