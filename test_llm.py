import argparse
import os

# os.environ.setdefault("HF_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

from app.services.qa import LLMService
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

def main():
    parser = argparse.ArgumentParser(description="Quick test for LLMService.generate")
    parser.add_argument("--prompt", "-p", default="Translate to plain English: The QFT of a harmonic oscillator.", help="Prompt to send to the model")
    parser.add_argument("--max-tokens", "-m", type=int, default=64, help="Max tokens for generation")
    args = parser.parse_args()

    print(f"Instantiating LLMService with model={os.environ['HF_MODEL_NAME']}...")
    llm = LLMService()
    print("LLM available:", llm.available)

    if not llm.available:
        print("LLM generation is not available. See logs for load errors. Exiting.")
        return

    print("Running generate()...")
    try:
        out = llm.generate(args.prompt, max_tokens=args.max_tokens)
        print("\n=== Generated Output ===\n")
        print(out)
        print("\n========================\n")
    except Exception as e:
        print("Generation failed:", e)


if __name__ == "__main__":
    main()