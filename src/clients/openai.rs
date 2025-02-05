use crate::{
    error::{ApiError, Result},
    models::{ApiConfig, Message},
};
use futures::Stream;
use reqwest::{header::HeaderMap, Client};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, pin::Pin};
use futures::StreamExt;
use serde_json;

pub(crate) const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";
const DEFAULT_MODEL: &str = "gpt-3.5-turbo";

/// Client for interacting with OpenAI-compatible API models.
///
/// This client handles authentication, request construction, and response parsing
/// for both streaming and non-streaming interactions with OpenAI-compatible APIs.
#[derive(Debug)]
pub struct OpenAIClient {
    pub(crate) client: Client,
    api_token: String,
    base_url: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OpenAIResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Choice {
    pub index: i32,
    pub message: AssistantMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AssistantMessage {
    pub role: String,
    pub content: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct StreamChoice {
    pub index: i32,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct StreamDelta {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct StreamResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OpenAIRequest {
    messages: Vec<Message>,
    stream: bool,
    #[serde(flatten)]
    additional_params: serde_json::Value,
}

impl OpenAIClient {
    pub fn new(api_token: String) -> Self {
        Self {
            client: Client::new(),
            api_token,
            base_url: OPENAI_API_URL.to_string(),
        }
    }

    pub fn new_with_base_url(api_token: String, base_url: String) -> Self {
        Self {
            client: Client::new(),
            api_token,
            base_url,
        }
    }

    pub(crate) fn get_base_url(&self, custom_headers: Option<&HashMap<String, String>>) -> String {
        if let Some(headers) = custom_headers {
            if let Some(endpoint_url) = headers.get(super::OPENAI_ENDPOINT_URL_HEADER) {
                return endpoint_url.clone();
            }
        }
        self.base_url.clone()
    }

    pub(crate) fn build_headers(&self, custom_headers: Option<&HashMap<String, String>>) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "Authorization",
            format!("Bearer {}", self.api_token)
                .parse()
                .map_err(|e| ApiError::Internal { 
                    message: format!("Invalid API token: {}", e) 
                })?,
        );
        headers.insert(
            "Content-Type",
            "application/json"
                .parse()
                .map_err(|e| ApiError::Internal { 
                    message: format!("Invalid content type: {}", e) 
                })?,
        );
        headers.insert(
            "Accept",
            "application/json"
                .parse()
                .map_err(|e| ApiError::Internal { 
                    message: format!("Invalid accept header: {}", e) 
                })?,
        );

        if let Some(custom) = custom_headers {
            headers.extend(super::build_headers(custom)?);
        }

        Ok(headers)
    }

    pub(crate) fn build_request(&self, messages: Vec<Message>, stream: bool, config: &ApiConfig) -> OpenAIRequest {
        let mut request_value = serde_json::json!({
            "messages": messages,
            "stream": stream,
            "model": config.body.get("model").unwrap_or(&serde_json::json!(DEFAULT_MODEL)),
            "max_tokens": config.body.get("max_tokens").unwrap_or(&serde_json::json!(4096)),
            "temperature": config.body.get("temperature").unwrap_or(&serde_json::json!(1.0)),
        });

        if let serde_json::Value::Object(mut map) = request_value {
            if let serde_json::Value::Object(mut body) = serde_json::to_value(&config.body).unwrap_or_default() {
                body.remove("stream");
                body.remove("messages");
                
                for (key, value) in body {
                    map.insert(key, value);
                }
            }
            request_value = serde_json::Value::Object(map);
        }

        serde_json::from_value(request_value).unwrap_or_else(|_| OpenAIRequest {
            messages,
            stream,
            additional_params: config.body.clone(),
        })
    }

    pub async fn chat(
        &self,
        messages: Vec<Message>,
        config: &ApiConfig,
    ) -> Result<OpenAIResponse> {
        tracing::info!("Building headers");
        let headers = self.build_headers(Some(&config.headers))?;
        let request = self.build_request(messages, false, config);
        let base_url = self.get_base_url(Some(&config.headers));


        // 打印详细的请求信息用于调试
        tracing::info!("OpenAI Request Debug Info:");
        tracing::info!("URL: {}", base_url);
        tracing::info!("Headers: {:#?}", headers);
        tracing::info!("Body: {}", serde_json::to_string_pretty(&request).unwrap_or_default());

        
        let response = self
            .client
            .post(&base_url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| ApiError::OpenAIError { 
                message: format!("Request failed: {}", e),
                type_: "request_failed".to_string(),
                param: None,
                code: None
            })?;

        if !response.status().is_success() {
            let error = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            tracing::error!("OpenAI API error response: {}", error); // 添加错误日志
            return Err(ApiError::OpenAIError { 
                message: error,
                type_: "api_error".to_string(),
                param: None,
                code: None
            });
        }

        response
            .json::<OpenAIResponse>()
            .await
            .map_err(|e| ApiError::OpenAIError { 
                message: format!("Failed to parse response: {}", e),
                type_: "parse_error".to_string(),
                param: None,
                code: None
            })
    }

    pub fn chat_stream(
        &self,
        messages: Vec<Message>,
        config: &ApiConfig,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamResponse>> + Send>> {
        let headers = match self.build_headers(Some(&config.headers)) {
            Ok(h) => h,
            Err(e) => return Box::pin(futures::stream::once(async move { Err(e) })),
        };

        let request = self.build_request(messages, true, config);
        let client = self.client.clone();
        let base_url = self.get_base_url(Some(&config.headers));

        Box::pin(async_stream::try_stream! {
            let mut stream = client
                .post(&base_url)
                .headers(headers)
                .json(&request)
                .send()
                .await
                .map_err(|e| ApiError::OpenAIError { 
                    message: format!("Request failed: {}", e),
                    type_: "request_failed".to_string(),
                    param: None,
                    code: None
                })?
                .bytes_stream();

            let mut data = String::new();
            
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| ApiError::OpenAIError { 
                    message: format!("Stream error: {}", e),
                    type_: "stream_error".to_string(),
                    param: None,
                    code: None
                })?;
                data.push_str(&String::from_utf8_lossy(&chunk));

                let mut start = 0;
                while let Some(end) = data[start..].find("\n\n") {
                    let end = start + end;
                    let line = &data[start..end].trim();
                    start = end + 2;
                    
                    if line.starts_with("data: ") {
                        let json_data = &line["data: ".len()..];
                        if let Ok(response) = serde_json::from_str::<StreamResponse>(json_data) {
                            yield response;
                        }
                    }
                }

                if start > 0 {
                    data = data[start..].to_string();
                }
            }
        })
    }
} 