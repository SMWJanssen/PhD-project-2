from types import SimpleNamespace
from model_definitions import initializeLLM
from main import initialize_DAG

print("Starting smoke test...", flush=True)

args = SimpleNamespace()
args.llm = "ollama"
args.topic = "diabetes mellitus"
args.dimensions = ["tasks"]

# keep it tiny
args.init_levels = 0
args.max_depth = 1
args.max_density = 999999

print("Initializing LLM...", flush=True)
args = initializeLLM(args)
print("Building initial DAG (this will call Ollama once)...", flush=True)

roots, id2node, label2node = initialize_DAG(args)

print("Smoke test OK", flush=True)
print("Root dimensions:", list(roots.keys()), flush=True)
print("Root label:", roots["tasks"].label, flush=True)
print("Children count:", len(roots["tasks"].children), flush=True)