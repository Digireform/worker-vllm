from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", trust_remote_code=True)

def handler(event):
    messages = event['input'].get('messages', [])
    prompt = ""

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role and content:
            prompt += f"{role}: {content}\n"
    prompt += "assistant:"

    sampling_params = SamplingParams(temperature=0.7, max_tokens=512, top_p=0.9)

    result = llm.generate(prompt, sampling_params)[0]
    answer = result.outputs[0].text.strip()

    return {
        "id": event['id'],
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ]
    }
