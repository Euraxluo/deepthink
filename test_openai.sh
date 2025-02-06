curl -i -X POST \
   -H "Content-Type: application/json" \
   -H "Authorization: Bearer ollama" \
   -d '{
    "messages": [
        {
            "role": "system",
            "content": "\nCurrent model: gpt-4\nCurrent date: 2025-02-06T02:35:45.877Z\n\nYou are a helpful assistant. You can help me by answering my questions. You can also ask me questions."
        },
        {
            "role": "user",
            "content": "你好啊,小草莓"
        }
    ],
    "model": "gpt-4",
    "temperature": 0.82,
    "top_p": 1,
    "stream": false
}' \
 'http://127.0.0.1:3000/v1/chat/completions' 