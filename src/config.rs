//! Configuration management for the application.
//!
//! This module handles loading and managing configuration settings from files
//! and environment variables. It includes endpoint configurations for different
//! AI model providers and server settings.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::collections::HashMap;

/// Root configuration structure containing all application settings.
///
/// This structure is typically loaded from a TOML configuration file
/// and provides access to all configurable aspects of the application.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub endpoints: EndpointConfig,
    pub models: ModelConfig,
    pub auth: AuthConfig,
}

/// Server-specific configuration settings.
///
/// Contains settings related to the HTTP server, such as the
/// host address and port number to bind to.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

/// Endpoint configuration for all supported AI models.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct EndpointConfig {
    pub deepseek: String,
    pub anthropic: String,
    pub openai: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ModelConfig {
    pub default_deepseek: String,
    pub default_openai: String,
    pub default_anthropic: String,
    pub model_mappings: HashMap<String, ModelMapping>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ModelMapping {
    pub deepseek_model: String,
    pub target_model: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AuthConfig {
    pub default_tokens: TokenConfig,
    pub token_mappings: HashMap<String, TokenConfig>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TokenConfig {
    pub deepseek_token: String,
    pub openai_token: String,
    pub anthropic_token: String,
}

impl Config {
    /// Loads configuration from the default config file.
    ///
    /// Attempts to load and parse the configuration from 'config.toml'.
    /// Falls back to default values if the file cannot be loaded or parsed.
    ///
    /// # Returns
    ///
    /// * `anyhow::Result<Self>` - The loaded configuration or an error if loading fails
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The config file cannot be read
    /// - The TOML content cannot be parsed
    /// - The parsed content doesn't match the expected structure
    pub fn load() -> anyhow::Result<Self> {
        let config_path = Path::new("./config.toml");
        let config = config::Config::builder()
            .add_source(config::File::from(config_path))
            .build()?;

        Ok(config.try_deserialize()?)
    }
}

/// Provides default configuration values.
///
/// These defaults are used when a configuration file is not present
/// or when specific values are not provided in the config file.
impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 3000,
            },
            endpoints: EndpointConfig {
                deepseek: "https://api.deepseek.com/v1/chat/completions".to_string(),
                anthropic: "https://api.anthropic.com/v1/messages".to_string(),
                openai: "https://api.openai.com/v1/chat/completions".to_string(),
            },
            models: ModelConfig {
                default_deepseek: "deepseek-r1:14b".to_string(),
                default_openai: "qwen2.5:14b".to_string(),
                default_anthropic: "claude-3-sonnet-20240229".to_string(),
                model_mappings: HashMap::new(),
            },
            auth: AuthConfig {
                default_tokens: TokenConfig {
                    deepseek_token: "ollama".to_string(),
                    openai_token: "ollama".to_string(),
                    anthropic_token: "ollama".to_string(),
                },
                token_mappings: HashMap::new(),
            },
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            default_deepseek: "deepseek-r1:14b".to_string(),
            default_openai: "qwen2.5:14b".to_string(),
            default_anthropic: "claude-3-sonnet-20240229".to_string(),
            model_mappings: HashMap::new(),
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            default_tokens: TokenConfig {
                deepseek_token: "ollama".to_string(),
                openai_token: "ollama".to_string(),
                anthropic_token: "ollama".to_string(),
            },
            token_mappings: HashMap::new(),
        }
    }
}
