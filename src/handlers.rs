//! Request handlers for the API endpoints.
//!
//! This module contains the main request handlers and supporting functions
//! for processing chat requests, including both streaming and non-streaming
//! responses. It coordinates between different AI models and handles
//! usage tracking and cost calculations.

use crate::{
    clients::{
        AnthropicClient, DeepSeekClient, OpenAIClient,
        DEEPSEEK_ENDPOINT_URL_HEADER, OPENAI_ENDPOINT_URL_HEADER, ANTHROPIC_ENDPOINT_URL_HEADER,
    },
    config::Config,
    error::{ApiError, Result, SseResponse},
    models::{
        ApiRequest, ApiResponse, ContentBlock,
        ExternalApiResponse, Message, Role, StreamEvent,
    },
};

// 添加 AssistantMessage 导入
use crate::clients::deepseek::AssistantMessage;

use axum::{
    extract::State,
    response::{sse::Event, IntoResponse},
    Json,
};
use chrono::Utc;
use futures::StreamExt;
use std::{sync::Arc, collections::HashMap};
use tokio_stream::wrappers::ReceiverStream;

/// Application state shared across request handlers.
///
/// Contains configuration that needs to be accessible
/// to all request handlers.
pub struct AppState {
    pub config: Config,
}

/// Extracts API tokens from request headers.
///
/// # Arguments
///
/// * `headers` - The HTTP headers containing the API tokens
///
/// # Returns
///
/// * `Result<(String, String)>` - A tuple of (DeepSeek token, Anthropic token)
///
/// # Errors
///
/// Returns `ApiError::MissingHeader` if either token is missing
/// Returns `ApiError::BadRequest` if tokens are malformed
fn extract_api_tokens(
    headers: &axum::http::HeaderMap,
) -> Result<(String, String)> {
    let deepseek_token = headers
        .get("X-DeepSeek-API-Token")
        .ok_or_else(|| ApiError::MissingHeader { 
            header: "X-DeepSeek-API-Token".to_string() 
        })?
        .to_str()
        .map_err(|_| ApiError::BadRequest { 
            message: "Invalid DeepSeek API token".to_string() 
        })?
        .to_string();

    let anthropic_token = headers
        .get("X-Anthropic-API-Token")
        .ok_or_else(|| ApiError::MissingHeader { 
            header: "X-Anthropic-API-Token".to_string() 
        })?
        .to_str()
        .map_err(|_| ApiError::BadRequest { 
            message: "Invalid Anthropic API token".to_string() 
        })?
        .to_string();

    Ok((deepseek_token, anthropic_token))
}

/// Main handler for chat requests.
///
/// Routes requests to either streaming or non-streaming handlers
/// based on the request configuration.
///
/// # Arguments
///
/// * `state` - Application state containing configuration
/// * `headers` - HTTP request headers
/// * `request` - The parsed chat request
///
/// # Returns
///
/// * `Result<Response>` - The API response or an error
pub async fn handle_chat(
    state: State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(request): Json<ApiRequest>,
) -> Result<axum::response::Response> {
    tracing::info!("Handling chat request");
    tracing::info!("{:#?}", request);
    if request.stream {
        let stream_response = chat_stream(state, headers, Json(request)).await?;
        Ok(stream_response.into_response())
    } else {
        let json_response = chat(state, headers, Json(request)).await?;
        Ok(json_response.into_response())
    }
}

/// Handler for non-streaming chat requests.
///
/// Processes the request through both AI models sequentially,
/// combining their responses and tracking usage.
///
/// # Arguments
///
/// * `state` - Application state containing configuration
/// * `headers` - HTTP request headers
/// * `request` - The parsed chat request
///
/// # Returns
///
/// * `Result<Json<ApiResponse>>` - The combined API response or an error
pub(crate) async fn chat(
    State(_state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(request): Json<ApiRequest>,
) -> Result<Json<ApiResponse>> {
    // Validate system prompt
    if !request.validate_system_prompt() {
        return Err(ApiError::InvalidSystemPrompt);
    }

    // Extract API tokens
    let deepseek_token = headers
        .get("X-DeepSeek-API-Token")
        .ok_or_else(|| ApiError::MissingHeader { 
            header: "X-DeepSeek-API-Token".to_string() 
        })?
        .to_str()
        .map_err(|_| ApiError::BadRequest { 
            message: "Invalid DeepSeek API token".to_string() 
        })?
        .to_string();

    let (target_model, target_token) = get_target_client(&headers)?;

    // Initialize clients with custom base URLs if provided
    let deepseek_client = match headers.get(DEEPSEEK_ENDPOINT_URL_HEADER).and_then(|h| h.to_str().ok()) {
        Some(base_url) => DeepSeekClient::new_with_base_url(deepseek_token, base_url.to_string()),
        None => DeepSeekClient::new(deepseek_token),
    };

    let messages = request.get_messages_with_system();

    // Call DeepSeek API
    let deepseek_response = deepseek_client.chat(messages.clone(), &request.deepseek_config).await?;

    // Extract reasoning content and wrap in thinking tags
    let reasoning_content = deepseek_response
        .choices
        .first()
        .and_then(|c| c.message.reasoning_content.as_ref())
        .map(|content| content.trim())
        .ok_or_else(|| ApiError::DeepSeekError { 
            message: "No reasoning content in response".to_string(),
            type_: "missing_content".to_string(),
            param: None,
            code: None
        })?;

    // 只保留推理内容,不添加额外的标记
    let thinking_content = if reasoning_content.starts_with("<think>") && reasoning_content.ends_with("</think>") {
        reasoning_content.to_string()
    } else {
        format!("<think>\n{}\n</think>", reasoning_content)
    };

    // Add thinking content to messages for target model
    let mut target_messages = messages;
    
    // 移除可能存在的系统消息
    target_messages.retain(|msg| msg.role != Role::System);
    
    // 添加推理内容
    target_messages.push(Message {
        role: Role::Assistant,
        content: thinking_content.clone(),
    });

    // Call target model API
    let (target_response, target_status, target_headers) = match target_model.as_str() {
        "openai" => {
            let openai_client = match headers.get(OPENAI_ENDPOINT_URL_HEADER).and_then(|h| h.to_str().ok()) {
                Some(base_url) => OpenAIClient::new_with_base_url(target_token, base_url.to_string()),
                None => OpenAIClient::new(target_token),
            };
            tracing::info!("Calling OpenAI client");
            tracing::info!("{:#?}", request);
            tracing::info!("Target messages: {:?}", target_messages);
            tracing::info!("OpenAI config: {:?}", request.openai_config);
            let response = openai_client.chat(target_messages, &request.openai_config).await?;
            (serde_json::to_value(&response)?, 200, HashMap::<String, String>::new())
        }
        _ => {
            let anthropic_client = match headers.get(ANTHROPIC_ENDPOINT_URL_HEADER).and_then(|h| h.to_str().ok()) {
                Some(base_url) => AnthropicClient::new_with_base_url(target_token, base_url.to_string()),
                None => AnthropicClient::new(target_token),
            };
            let response = anthropic_client.chat(
                target_messages,
                request.get_system_prompt().map(String::from),
                &request.anthropic_config
            ).await?;
            (serde_json::to_value(&response)?, 200, HashMap::new())
        }
    };

    // Combine thinking content with target model's response
    let mut content = Vec::new();
    content.push(ContentBlock::text(thinking_content));

    // Add target model's response blocks
    match target_model.as_str() {
        "openai" => {
            if let Some(choice) = target_response.get("choices").and_then(|c| c.as_array()).and_then(|c| c.first()) {
                if let Some(message) = choice.get("message") {
                    if let Some(content_str) = message.get("content").and_then(|c| c.as_str()) {
                        content.push(ContentBlock::text(content_str.to_string()));
                    }
                }
            }
        }
        _ => {
            if let Some(content_array) = target_response.get("content").and_then(|c| c.as_array()) {
                content.extend(content_array.iter().filter_map(|block| {
                    Some(ContentBlock {
                        content_type: block.get("type")?.as_str()?.to_string(),
                        text: block.get("text")?.as_str()?.to_string(),
                    })
                }));
            }
        }
    }

    // Build response
    let response = ApiResponse {
        created: Utc::now(),
        content,
        // deepseek_response: request.verbose.then(|| ExternalApiResponse {
        //     status: deepseek_status,
        //     headers: deepseek_headers,
        //     body: serde_json::to_value(&deepseek_response).unwrap_or_default(),
        // }),
        // anthropic_response: request.verbose.then(|| ExternalApiResponse {
        //     status: target_status,
        //     headers: target_headers,
        //     body: target_response.clone(),
        // }),
    };

    Ok(Json(response))
}

/// Handler for streaming chat requests.
///
/// Processes the request through both AI models sequentially,
/// streaming their responses as Server-Sent Events.
///
/// # Arguments
///
/// * `state` - Application state containing configuration
/// * `headers` - HTTP request headers
/// * `request` - The parsed chat request
///
/// # Returns
///
/// * `Result<SseResponse>` - A stream of Server-Sent Events or an error
pub(crate) async fn chat_stream(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(request): Json<ApiRequest>,
) -> Result<SseResponse> {
    // Validate system prompt
    if !request.validate_system_prompt() {
        return Err(ApiError::InvalidSystemPrompt);
    }

    // Extract API tokens
    let deepseek_token = headers
        .get("X-DeepSeek-API-Token")
        .ok_or_else(|| ApiError::MissingHeader { 
            header: "X-DeepSeek-API-Token".to_string() 
        })?
        .to_str()
        .map_err(|_| ApiError::BadRequest { 
            message: "Invalid DeepSeek API token".to_string() 
        })?
        .to_string();

    let (target_model, target_token) = get_target_client(&headers)?;

    // Initialize clients with custom base URLs if provided
    let deepseek_client = match headers.get(DEEPSEEK_ENDPOINT_URL_HEADER).and_then(|h| h.to_str().ok()) {
        Some(base_url) => DeepSeekClient::new_with_base_url(deepseek_token, base_url.to_string()),
        None => DeepSeekClient::new(deepseek_token),
    };

    let messages = request.get_messages_with_system();

    // Create channel for stream events
    let (tx, rx) = tokio::sync::mpsc::channel(100);
    let tx = Arc::new(tx);

    // Spawn task to handle streaming
    let config = state.config.clone();
    let request_clone = request.clone();
    tokio::spawn(async move {
        let tx = tx.clone();

        // Start event
        let _ = tx
            .send(Ok(Event::default().event("start").data(
                serde_json::to_string(&StreamEvent::Start {
                    created: Utc::now(),
                })
                .unwrap_or_default(),
            )))
            .await;

        // Stream from DeepSeek
        let mut complete_reasoning = String::new();
        let mut current_chunk = String::new();
        let mut deepseek_stream = deepseek_client.chat_stream(messages.clone(), &request_clone.deepseek_config);
        
        // Send initial thinking tag
        let _ = tx
            .send(Ok(Event::default().event("content").data(
                serde_json::to_string(&StreamEvent::Content {
                    content: vec![ContentBlock {
                        content_type: "text".to_string(),
                        text: "<thinking>\n".to_string(),
                    }],
                })
                .unwrap_or_default(),
            )))
            .await;
        
        while let Some(chunk) = deepseek_stream.next().await {
            match chunk {
                Ok(response) => {
                    if let Some(choice) = response.choices.first() {
                        tracing::info!("Stream Response: {:?}", response);
                        
                        // 处理 delta 如果存在
                        if let Some(delta) = &choice.delta {
                            // 处理 content
                            if let Some(content) = &delta.content {
                                tracing::info!("Found delta content: {}", content);
                                if response.system_fingerprint == "fp_ollama" {
                                    // 直接发送 content 作为流式输出
                                    if !content.is_empty() {
                           
                                    }
                                    tracing::info!("Processing ollama delta content");
                                    current_chunk.push_str(content);
                                    tracing::info!("Updated current_chunk: {}", current_chunk);
                                    if current_chunk.contains("<think>") && !current_chunk.contains("</think>"){
                                        if content != "<think>" {
                                        let _ = tx
                                        .send(Ok(Event::default().event("content").data(
                                            serde_json::to_string(&StreamEvent::Content {
                                                content: vec![ContentBlock {
                                                    content_type: "text_delta".to_string(),
                                                    text: content.to_string(),
                                                }],
                                            })
                                            .unwrap_or_default(),
                                        )))
                                        .await;
                                        }
                                    }
                                    if current_chunk.contains("<think>") && current_chunk.contains("</think>") {
                                        tracing::info!("Found complete think tags in delta");
                                        if let Some((reasoning, _)) = AssistantMessage::extract_think_content(&current_chunk) {
                                            tracing::info!("Extracted reasoning from delta: {}", reasoning);
                                            complete_reasoning.push_str(&reasoning);
                                            tracing::info!("Updated complete_reasoning from delta think tags: {}", complete_reasoning);
                                            current_chunk.clear();
                                        }
                                    }
                                }
                            }

                            // 处理 reasoning_content
                            if let Some(reasoning) = &delta.reasoning_content {
                                tracing::info!("Found delta reasoning_content: {}", reasoning);
                                if !reasoning.is_empty() {
                                    // 直接发送 reasoning 作为流式输出
                                    let _ = tx
                                        .send(Ok(Event::default().event("content").data(
                                            serde_json::to_string(&StreamEvent::Content {
                                                content: vec![ContentBlock {
                                                    content_type: "text_delta".to_string(),
                                                    text: reasoning.to_string(),
                                                }],
                                            })
                                            .unwrap_or_default(),
                                        )))
                                        .await;

                                    complete_reasoning.push_str(reasoning);
                                    tracing::info!("Updated complete_reasoning from delta: {}", complete_reasoning);
                                }
                            }
                        }
                        
                        // 处理 message 如果存在
                        if let Some(message) = &choice.message {
                            if let Some(content) = &message.content {
                                if response.system_fingerprint == "fp_ollama" {
                                    tracing::info!("Processing ollama message content");
                                    if let Some((reasoning, _)) = AssistantMessage::extract_think_content(content) {
                                        complete_reasoning.push_str(&reasoning);
                                        tracing::info!("Updated complete_reasoning from message think tags: {}", complete_reasoning);
                                    }
                                }
                            }

                            if let Some(reasoning) = &message.reasoning_content {
                                tracing::info!("Found message reasoning_content: {}", reasoning);
                                if !reasoning.is_empty() {
                                    complete_reasoning.push_str(reasoning);
                                    tracing::info!("Updated complete_reasoning from message: {}", complete_reasoning);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = tx
                        .send(Ok(Event::default().event("error").data(
                            serde_json::to_string(&StreamEvent::Error {
                                message: e.to_string(),
                                code: 500,
                            })
                            .unwrap_or_default(),
                        )))
                        .await;
                    return;
                }
            }
        }
        
        // Send closing thinking tag
        let _ = tx
            .send(Ok(Event::default().event("content").data(
                serde_json::to_string(&StreamEvent::Content {
                    content: vec![ContentBlock {
                        content_type: "text".to_string(),
                        text: "\n</thinking>".to_string(),
                    }],
                })
                .unwrap_or_default(),
            )))
            .await;

        tracing::info!("Stream completed. Final complete_reasoning: {}", complete_reasoning);
        // Add complete thinking content to messages for target model
        let mut target_messages = messages;
        target_messages.push(Message {
            role: Role::Assistant,
            content: format!("<thinking>\n{}\n</thinking>", complete_reasoning),
        });

        // Stream from target model
        match target_model.as_str() {
            "openai" => {
                tracing::info!("Starting OpenAI stream");
                let openai_client = match headers.get(OPENAI_ENDPOINT_URL_HEADER).and_then(|h| h.to_str().ok()) {
                    Some(base_url) => OpenAIClient::new_with_base_url(target_token, base_url.to_string()),
                    None => OpenAIClient::new(target_token),
                };
                let mut openai_stream = openai_client.chat_stream(target_messages.clone(), &request_clone.openai_config);
                tracing::info!("OpenAI messages: {:?}", target_messages);

                while let Some(chunk) = openai_stream.next().await {
                    match chunk {
                        Ok(response) => {
                            tracing::info!("OpenAI response chunk: {:?}", response);
                            if let Some(choice) = response.choices.first() {
                                if let Some(content) = &choice.delta.content {
                                    if !content.is_empty() {
                                        tracing::info!("OpenAI content chunk: {}", content);
                                        let _ = tx
                                            .send(Ok(Event::default().event("content").data(
                                                serde_json::to_string(&StreamEvent::Content {
                                                    content: vec![ContentBlock {
                                                        content_type: "text_delta".to_string(),
                                                        text: content.to_string(),
                                                    }],
                                                })
                                                .unwrap_or_default(),
                                            )))
                                            .await;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("OpenAI stream error: {}", e);
                            let _ = tx
                                .send(Ok(Event::default().event("error").data(
                                    serde_json::to_string(&StreamEvent::Error {
                                        message: e.to_string(),
                                        code: 500,
                                    })
                                    .unwrap_or_default(),
                                )))
                                .await;
                            return;
                        }
                    }
                }
                tracing::info!("OpenAI stream completed");
            }
            _ => {
                tracing::info!("Starting Anthropic stream");
                let anthropic_client = match headers.get(ANTHROPIC_ENDPOINT_URL_HEADER).and_then(|h| h.to_str().ok()) {
                    Some(base_url) => AnthropicClient::new_with_base_url(target_token, base_url.to_string()),
                    None => AnthropicClient::new(target_token),
                };
                tracing::info!("Anthropic messages: {:?}", target_messages);
                let mut anthropic_stream = anthropic_client.chat_stream(
                    target_messages.clone(),
                    request_clone.get_system_prompt().map(String::from),
                    &request_clone.anthropic_config,
                );

                while let Some(chunk) = anthropic_stream.next().await {
                    match chunk {
                        Ok(event) => {
                            tracing::info!("Anthropic event: {:?}", event);
                            match event {
                                crate::clients::anthropic::StreamEvent::MessageStart { message } => {
                                    tracing::info!("Anthropic message start: {:?}", message);
                                    // Only send content event if there's actual content to send
                                    if !message.content.is_empty() {
                                        let _ = tx
                                            .send(Ok(Event::default().event("content").data(
                                                serde_json::to_string(&StreamEvent::Content { 
                                                    content: message.content.into_iter()
                                                        .map(ContentBlock::from_anthropic)
                                                        .collect()
                                                })
                                                .unwrap_or_default(),
                                            )))
                                            .await;
                                    }
                                }
                                crate::clients::anthropic::StreamEvent::ContentBlockDelta { delta, .. } => {
                                    tracing::info!("Anthropic content delta: {:?}", delta);
                                    // Send content update
                                    let _ = tx
                                        .send(Ok(Event::default().event("content").data(
                                            serde_json::to_string(&StreamEvent::Content {
                                                content: vec![ContentBlock {
                                                    content_type: delta.delta_type,
                                                    text: delta.text,
                                                }],
                                            })
                                            .unwrap_or_default(),
                                        )))
                                        .await;
                                }
                                _ => {
                                    tracing::info!("Anthropic other event: {:?}", event);
                                }
                            }
                        },
                        Err(e) => {
                            tracing::error!("Anthropic stream error: {}", e);
                            let _ = tx
                                .send(Ok(Event::default().event("error").data(
                                    serde_json::to_string(&StreamEvent::Error {
                                        message: e.to_string(),
                                        code: 500,
                                    })
                                    .unwrap_or_default(),
                                )))
                                .await;
                            return;
                        }
                    }
                }
                tracing::info!("Anthropic stream completed");
            }
        }

        // Send done event
        let _ = tx
            .send(Ok(Event::default().event("done").data(
                serde_json::to_string(&StreamEvent::Done)
                    .unwrap_or_default(),
            )))
            .await;
    });

    // Convert receiver into stream
    let stream = ReceiverStream::new(rx);
    Ok(SseResponse::new(stream))
}

/// 获取目标模型的客户端
fn get_target_client(headers: &axum::http::HeaderMap) -> Result<(String, String)> {
    let target_model = headers
        .get("X-Target-Model")
        .map(|h| h.to_str().unwrap_or("anthropic"))
        .unwrap_or("anthropic");

    match target_model {
        "openai" => {
            let openai_token = headers
                .get("X-OpenAI-API-Token")
                .ok_or_else(|| ApiError::MissingHeader { 
                    header: "X-OpenAI-API-Token".to_string() 
                })?
                .to_str()
                .map_err(|_| ApiError::BadRequest { 
                    message: "Invalid OpenAI API token".to_string() 
                })?
                .to_string();
            Ok(("openai".to_string(), openai_token))
        }
        _ => {
            let anthropic_token = headers
                .get("X-Anthropic-API-Token")
                .ok_or_else(|| ApiError::MissingHeader { 
                    header: "X-Anthropic-API-Token".to_string() 
                })?
                .to_str()
                .map_err(|_| ApiError::BadRequest { 
                    message: "Invalid Anthropic API token".to_string() 
                })?
                .to_string();
            Ok(("anthropic".to_string(), anthropic_token))
        }
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(err: serde_json::Error) -> Self {
        ApiError::Internal {
            message: format!("JSON error: {}", err),
        }
    }
}
