<div align="center">
<h1>DeepThink 🐬🧠</h1>

受 DeepClaude 项目启发，基于 Ollama 实现的本地化多模型推理框架，通过组合小型模型实现强大的推理和对话能力。

[![GitHub license](https://img.shields.io/github/license/euraxluo/deepthink)](https://github.com/euraxluo/deepthink/blob/main/LICENSE.md)
[![Rust](https://img.shields.io/badge/rust-v1.75%2B-orange)](https://www.rust-lang.org/)
[![Ollama](https://img.shields.io/badge/ollama-latest-blue)](https://ollama.ai)

</div>

## Overview

DeepThink 是一个高性能的本地化 LLM 推理框架，基于 Ollama 实现。它的核心理念是通过组合多个开源小型模型，实现类似大模型的推理和对话能力。特别是利用 DeepSeek-R1 小模型优秀的思维链（Chain of Thought）能力，结合其他模型（如 Qwen）的特长，在保持轻量级部署的同时提供强大的 AI 能力。

## Why DeepThink?

DeepThink 提供了一个创新的方案来实现高质量的 AI 对话：

- **本地化部署** - 基于 Ollama 实现完全的本地化部署，无需云端 API
- **小型模型组合** - 通过组合多个专精不同任务的小型模型，实现接近大模型的效果
- **DeepSeek-R1 思维链** - 利用 DeepSeek-R1 (14B) 模型优秀的推理和思维链能力
- **灵活模型选择** - 支持 Qwen、DeepSeek 等多个开源模型，可根据需求自由组合
- **高性能流式响应** - Rust 实现的高性能服务，支持流式输出
- **完全控制** - 本地部署意味着完全的数据隐私和系统控制

## Getting Started

### Prerequisites

- Rust 1.75 或更高版本
- [Ollama](https://ollama.ai) 最新版本
- 支持的模型：
  - deepseek-r1:14b
  - qwen2.5:14b
  - 其他 Ollama 支持的模型

### Configuration

创建一个 `config.toml` 文件在项目根目录：

```toml
[server]
host = "127.0.0.1"  # 服务器监听地址
port = 3000         # 服务器监听端口

[endpoints]
deepseek = "http://localhost:11434/v1/chat/completions"  # Ollama API 端点
anthropic = "http://localhost:11434/v1/chat/completions"  # Ollama API 端点
openai = "http://localhost:11434/v1/chat/completions"     # Ollama API 端点
```

### Basic Example

```python
import requests

response = requests.post(
    "http://127.0.0.1:3000/",
    headers={
        "X-DeepSeek-API-Token": "ollama",
        "X-OpenAI-API-Token": "ollama",
        "X-Target-Model": "openai",
        "X-DeepSeek-Endpoint-URL": "http://127.0.0.1:11434/v1/chat/completions",
        "X-OpenAI-Endpoint-URL": "http://localhost:11434/v1/chat/completions"
    },
    json={
        "messages": [
            {"role": "user", "content": "分析下面这段代码的复杂度"}
        ],
        "stream": False,
        "verbose": False,
        "system": "你是一个代码分析专家",
        "openai_config": {
            "headers": {},
            "body": {
                "model": "qwen2.5:14b"
            }
        },
        "deepseek_config": {
            "headers": {},
            "body": {
                "model": "deepseek-r1:14b"
            }
        }
    }
)

print(response.json())
```

### Streaming Example

```python
import asyncio
import json
import httpx

async def stream_response():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://127.0.0.1:3000/",
            headers={
                "X-DeepSeek-API-Token": "ollama",
                "X-OpenAI-API-Token": "ollama",
                "X-Target-Model": "openai",
                "X-DeepSeek-Endpoint-URL": "http://127.0.0.1:11434/v1/chat/completions",
                "X-OpenAI-Endpoint-URL": "http://localhost:11434/v1/chat/completions"
            },
            json={
                "stream": True,
                "messages": [
                    {"role": "user", "content": "分析下面这段代码的复杂度"}
                ],
                "openai_config": {
                    "headers": {},
                    "body": {
                        "model": "qwen2.5:14b"
                    }
                },
                "deepseek_config": {
                    "headers": {},
                    "body": {
                        "model": "deepseek-r1:14b"
                    }
                }
            }
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    if line.startswith('data: '):
                        data = line[6:]
                        try:
                            parsed_data = json.loads(data)
                            if 'content' in parsed_data:
                                content = parsed_data.get('content', '')[0]['text']
                                print(content, end='', flush=True)
                            else:
                                print(data, flush=True)
                        except json.JSONDecodeError:
                            pass

if __name__ == "__main__":
    asyncio.run(stream_response())
```

## Configuration Options

API 支持通过请求体进行广泛的配置：

```json
{
    "messages": [
        {
            "role": "user",
            "content": "你是谁?"
        }
    ],
    "system":"你是角色扮演助手,你可以完成角色扮演,但是你需要和用户对话,收集你进行角色班线需要的信息",
    "stream": true,
    "verbose": true,
    "openai_config": {
        "headers": {},
        "body": {
            "model": "qwen2.5:14b"
        }
    },
    "deepseek_config": {
        "headers": {},
        "body": {
            "model": "deepseek-r1:14b"
        }
    }
}
```

### 支持的请求头

- `X-DeepSeek-API-Token`: Ollama 认证令牌（默认为 "ollama"）
- `X-OpenAI-API-Token`: Ollama 认证令牌（默认为 "ollama"）
- `X-Target-Model`: 目标模型类型（"openai" 或 "deepseek"）
- `X-DeepSeek-Endpoint-URL`: DeepSeek 模型的 Ollama 端点
- `X-OpenAI-Endpoint-URL`: OpenAI 兼容模型的 Ollama 端点

## Self-Hosting

DeepThink 可以在您自己的基础设施上部署。按照以下步骤操作：

### 1. 安装 Ollama

首先需要安装 Ollama 并下载所需模型：

```bash
# MacOS
curl -fsSL https://ollama.com/install.sh | sh

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# 下载所需模型
ollama pull deepseek-r1:14b
ollama pull qwen2.5:14b
```

### 2. 部署 DeepThink

1. 克隆代码库：
```bash
git clone https://github.com/euraxluo/deepthink.git
cd deepthink
```

2. 配置环境：
   - 创建 `config.toml` 文件并配置服务器和 Ollama 端点
   - 确保 Ollama 服务正在运行（默认端口 11434）

3. 构建项目：
```bash
cargo build --release
```

4. 运行服务：
```bash
./target/release/deepthink
```

## 安全

- 完全本地化部署，数据不会离开您的基础设施
- 所有模型和服务都在本地运行
- 支持自定义 Ollama 认证
- 定期安全审计和更新

## 许可证

本项目采用 Apache 2.0 License 许可证 - 详见 [LICENSE](LICENSE.md) 文件。

## 致谢

DeepThink 是一个受 DeepClaude 启发，由 [Euraxluo](https://github.com/euraxluo) 开发的免费开源项目。特别感谢：

- [DeepClaude](https://github.com/getasterisk/deepclaude) 项目的启发
- [Ollama](https://ollama.ai) 提供的本地模型部署方案
- [DeepSeek](https://github.com/deepseek-ai) 开源的优秀模型
- [Qwen](https://github.com/QwenLM/Qwen2.5) 开源的优秀模型
- 开源社区的持续支持

---

<div align="center">
<a href="https://github.com/euraxluo">Euraxluo</a> built with ❤️
</div>