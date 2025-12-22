import os
import pytest
from pathlib import Path
from codur.config import load_config
from codur.graph.nodes.coding import coding_node
from langchain_core.messages import HumanMessage

# Skip if GROQ_API_KEY is not set
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_codur_coding_agent_hello_world(tmp_path, monkeypatch):
    # Change CWD to tmp_path so the agent writes main.py there
    monkeypatch.chdir(tmp_path)

    # Ensure main.py exists so coding_node can update it
    main_py = tmp_path / "main.py"
    main_py.write_text("# placeholder\n", encoding="utf-8")
    
    # Load config
    root_dir = Path(__file__).parent.parent.parent
    config_path = root_dir / "codur.yaml"
    if not config_path.exists():
        pytest.fail(f"Config file not found at {config_path}")
        
    config = load_config(config_path)
    
    # Ensure allow_outside_workspace is True just in case, though tmp_path should be fine
    config.runtime.allow_outside_workspace = True
    
    # Setup state
    state = {
        "messages": [HumanMessage(content="Write a python script that prints 'Hello, Integration!'")],
        "iterations": 0,
        "verbose": True
    }
    
    # Run the node
    print("\nRunning coding_node with Groq...")
    result = coding_node(state, config)
    
    # Assertions
    outcome = result.get("agent_outcome", {})
    assert outcome.get("status") == "success", f"Agent failed: {outcome.get('result')}"
    
    main_py = tmp_path / "main.py"
    assert main_py.exists(), "main.py was not created"
