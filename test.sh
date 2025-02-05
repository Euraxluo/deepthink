curl -i -X POST \
   -H "Content-Type:application/json" \
   -H "X-DeepSeek-API-Token:ollama" \
   -H "X-OpenAI-API-Token:ollama" \
   -H "X-Target-Model:openai" \
   -H "X-DeepSeek-Endpoint-URL:http://127.0.0.1:11434/v1/chat/completions" \
   -H "X-OpenAI-Endpoint-URL:http://127.0.0.1:11434/v1/chat/completions" \
   -d \
'{
    "messages": [
        {
            "role": "user",
            "content": "我是谁啊,小草莓?"
        }
    ],
    "stream":false,
    "deepseek_config": {
        "headers": {},
        "body": {
            "model":"deepseek-r1:14b"
        }
    },
    "openai_config": {
        "headers": {},
        "body": {
            "model":"qwen2.5:14b"
        }
    }
}' \
 'http://127.0.0.1:3000/'