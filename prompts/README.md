# Agent System Prompts

This directory contains the system prompts for each agent in the Deep Research Agent system.

## Prompt Files

| File | Agent | Purpose |
|------|-------|---------|
| `master_planner_prompt.txt` | Master Planning Agent | Query decomposition, synthesis |
| `web_search_prompt.txt` | Web Search Agent | General web content retrieval |
| `academic_search_prompt.txt` | Academic Search Agent | Scholarly papers and research |
| `technical_search_prompt.txt` | Technical Search Agent | Code, docs, repositories |
| `source_validator_prompt.txt` | Source Validation Agent | Reliability scoring |
| `reflexion_prompt.txt` | Reflexion Agent | Self-correction loop |

## How Prompts Are Loaded

The Python agents load prompts at runtime from this directory:

```python
def load_prompt(prompt_name: str) -> str:
    """Load a system prompt from the prompts directory."""
    prompts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'prompts')
    prompt_path = os.path.join(prompts_dir, f'{prompt_name}.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()
```

## Customization

Edit these `.txt` files to:
- Adjust agent behavior
- Add domain-specific instructions
- Modify output formats
- Fine-tune scoring criteria

No code changes needed - prompts are loaded dynamically!
