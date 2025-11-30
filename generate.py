
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel

# Base model and adapter settings
BASE_MODEL = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "outputs/llama2-dapt")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
use_4bit = torch.cuda.is_available() and os.environ.get("LOAD_IN_4BIT", "true").lower() == "true"

# Quantization config
bnb_config = None
if use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

# Load base model and tokenizer
print(f"\n🔹 Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
)

# Load fine-tuned adapter
print(f"🔹 Loading fine-tuned adapter from: {ADAPTER_DIR}")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

# Generation configuration
gen_cfg = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    repetition_penalty=1.1,
)

print("\n✅ Model is ready! Type your question (type 'exit' to quit)\n")

# Continuous user interaction loop
while True:
    user_input = input("🧠 Enter your question: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("\n👋 Exiting... Goodbye!")
        break

    # Build the prompt dynamically
    prompt = f"""You are an assistant specialized in cricket, medical, and education topics.
Q: {user_input}
A:"""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        output = model.generate(**inputs, generation_config=gen_cfg)

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n🧩 Generated Answer:\n")
    print(answer)
    print("\n" + "=" * 70 + "\n")
