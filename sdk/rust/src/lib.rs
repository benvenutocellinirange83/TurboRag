// TurboRag Rust SDK
// =================
// Async HTTP client for the TurboRag REST API.
//
// Add to Cargo.toml:
//   [dependencies]
//   turborag-sdk = { path = "." }    # local
//   reqwest = { version = "0.12", features = ["json"] }
//   serde = { version = "1", features = ["derive"] }
//   tokio = { version = "1", features = ["full"] }
//
// Usage:
//   let client = TurboRagClient::new("http://127.0.0.1:8000");
//   client.index("Paris is the capital of France.", None).await?;
//   let results = client.search("capital", 5, None).await?;
//   let resp = client.ask("What is the capital?", 5, None).await?;
//   println!("{}", resp.answer);

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AskResponse {
    pub answer: String,
    pub sources: Vec<SearchResult>,
    pub question: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub query: String,
    pub count: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IndexResponse {
    pub id: String,
    pub status: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbedResponse {
    pub embedding: Vec<f32>,
    pub dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StatsResponse {
    pub doc_count: usize,
    pub dim: Option<usize>,
    pub bit_width: u8,
    pub index_path: String,
    pub embed_model: String,
    pub llm_model: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum TurboRagError {
    #[error("HTTP error {status}: {body}")]
    Http { status: u16, body: String },
    #[error("Request error: {0}")]
    Request(#[from] reqwest::Error),
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct TurboRagClient {
    base_url: String,
    client: reqwest::Client,
    api_key: Option<String>,
}

impl TurboRagClient {
    /// Create a new client.
    ///
    /// # Arguments
    /// * `base_url` - e.g. `"http://127.0.0.1:8000"`
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::with_key(base_url, None)
    }

    pub fn with_key(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client,
            api_key,
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    pub async fn health(&self) -> bool {
        self.get::<serde_json::Value>("/health")
            .await
            .map(|v| v.get("status").and_then(|s| s.as_str()) == Some("ok"))
            .unwrap_or(false)
    }

    pub async fn stats(&self) -> Result<StatsResponse, TurboRagError> {
        self.get("/stats").await
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, TurboRagError> {
        #[derive(Serialize)]
        struct Req<'a> { text: &'a str }
        let resp: EmbedResponse = self.post("/embed", &Req { text }).await?;
        Ok(resp.embedding)
    }

    pub async fn index(
        &self,
        text: &str,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<String, TurboRagError> {
        #[derive(Serialize)]
        struct Req<'a> {
            text: &'a str,
            metadata: HashMap<String, serde_json::Value>,
            chunk: bool,
        }
        let resp: IndexResponse = self
            .post(
                "/index",
                &Req {
                    text,
                    metadata: metadata.unwrap_or_default(),
                    chunk: false,
                },
            )
            .await?;
        Ok(resp.id)
    }

    pub async fn index_batch(
        &self,
        texts: Vec<String>,
        metadatas: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<Vec<String>, TurboRagError> {
        #[derive(Serialize)]
        struct Req {
            texts: Vec<String>,
            metadatas: Option<Vec<HashMap<String, serde_json::Value>>>,
        }
        let body: serde_json::Value = self
            .post("/index/batch", &Req { texts, metadatas })
            .await?;
        Ok(body["ids"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect())
    }

    pub async fn search(
        &self,
        query: &str,
        k: usize,
        filter_ids: Option<Vec<String>>,
    ) -> Result<Vec<SearchResult>, TurboRagError> {
        #[derive(Serialize)]
        struct Req<'a> {
            query: &'a str,
            k: usize,
            #[serde(skip_serializing_if = "Option::is_none")]
            filter_ids: Option<Vec<String>>,
        }
        let resp: SearchResponse = self
            .post("/search", &Req { query, k, filter_ids })
            .await?;
        Ok(resp.results)
    }

    pub async fn ask(
        &self,
        question: &str,
        k: usize,
        system: Option<&str>,
    ) -> Result<AskResponse, TurboRagError> {
        #[derive(Serialize)]
        struct Req<'a> {
            question: &'a str,
            k: usize,
            #[serde(skip_serializing_if = "Option::is_none")]
            system: Option<&'a str>,
        }
        self.post("/ask", &Req { question, k, system }).await
    }

    pub async fn delete(&self, doc_id: &str) -> Result<bool, TurboRagError> {
        let url = format!("{}/document/{}", self.base_url, doc_id);
        let mut req = self.client.delete(&url);
        if let Some(key) = &self.api_key {
            req = req.header("X-API-Key", key);
        }
        let resp = req.send().await?;
        Ok(resp.status().is_success())
    }

    // ------------------------------------------------------------------
    // Internal
    // ------------------------------------------------------------------

    async fn get<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
    ) -> Result<T, TurboRagError> {
        let url = format!("{}{}", self.base_url, path);
        let mut req = self.client.get(&url);
        if let Some(key) = &self.api_key {
            req = req.header("X-API-Key", key);
        }
        let resp = req.send().await?;
        self.handle(resp).await
    }

    async fn post<B: Serialize, T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, TurboRagError> {
        let url = format!("{}{}", self.base_url, path);
        let mut req = self.client.post(&url).json(body);
        if let Some(key) = &self.api_key {
            req = req.header("X-API-Key", key);
        }
        let resp = req.send().await?;
        self.handle(resp).await
    }

    async fn handle<T: for<'de> Deserialize<'de>>(
        &self,
        resp: reqwest::Response,
    ) -> Result<T, TurboRagError> {
        let status = resp.status().as_u16();
        if status >= 400 {
            let body = resp.text().await.unwrap_or_default();
            return Err(TurboRagError::Http { status, body });
        }
        Ok(resp.json::<T>().await?)
    }
}

// ---------------------------------------------------------------------------
// Example (cargo test -- --nocapture)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health() {
        let client = TurboRagClient::new("http://127.0.0.1:8000");
        // Just checks that the call doesn't panic; server may not be running in CI
        let _ = client.health().await;
    }
}
