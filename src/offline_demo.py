import sglang

# sglang.Engine launches its scheduler with the multiprocessing "spawn" start
# method, which re-imports this module in the child process. Everything must
# therefore live under a __main__ guard, or the child recursively re-creates
# the engine and crashes with "An attempt has been made to start a new process
# before the current process has finished its bootstrapping phase".
if __name__ == "__main__":
    llm = sglang.Engine(model_path="qwen/qwen2.5-0.5b-instruct")

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}\n")

    llm.shutdown()