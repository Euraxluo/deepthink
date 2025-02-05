//! Response models for the API endpoints.
//!
//! This module defines the structures used to represent API responses,
//! including chat completions and usage statistics.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Primary response structure for chat API endpoints.
///
/// Contains the complete response from both AI models, including
/// content blocks, usage statistics, and optional raw API responses.
#[derive(Debug, Serialize, Clone)]
pub struct ApiResponse {
    pub created: DateTime<Utc>,
    pub content: Vec<ContentBlock>,
}

/// A block of content in a response.
///
/// Represents a single piece of content in the response,
/// with its type and actual text content.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

/// Raw response from an external API.
///
/// Contains the complete response details from an external API
/// call, including status code, headers, and response body.
#[derive(Debug, Serialize, Clone)]
pub struct ExternalApiResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: serde_json::Value,
}



// Streaming event types
/// Events emitted during streaming responses.
///
/// Represents different types of events that can occur
/// during a streaming response, including content updates
/// and usage statistics.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "start")]
    Start {
        created: DateTime<Utc>,
    },
    #[serde(rename = "content")]
    Content {
        content: Vec<ContentBlock>,
    },
    #[serde(rename = "error")]
    Error {
        message: String,
        code: i32,
    },
    #[serde(rename = "done")]
    Done,
}

impl Default for StreamEvent {
    fn default() -> Self {
        StreamEvent::Done
    }
}

impl ContentBlock {
    /// Creates a new text content block.
    ///
    /// # Arguments
    ///
    /// * `text` - The text content to include in the block
    ///
    /// # Returns
    ///
    /// A new `ContentBlock` with the type set to "text"
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content_type: "text".to_string(),
            text: text.into(),
        }
    }

    /// Converts an Anthropic content block to a generic content block.
    ///
    /// # Arguments
    ///
    /// * `block` - The Anthropic-specific content block to convert
    ///
    /// # Returns
    ///
    /// A new `ContentBlock` with the same content type and text
    pub fn from_anthropic(block: crate::clients::anthropic::ContentBlock) -> Self {
        Self {
            content_type: block.content_type,
            text: block.text,
        }
    }
}

impl ApiResponse {
    /// Creates a new API response with simple text content.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content for the response
    ///
    /// # Returns
    ///
    /// A new `ApiResponse` with default values and the provided content
    #[allow(dead_code)]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            created: Utc::now(),
            content: vec![ContentBlock::text(content)],
            // deepseek_response: None,
            // anthropic_response: None,
        }
    }
}

