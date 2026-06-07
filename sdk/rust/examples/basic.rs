//! Basic TurboRag Rust SDK example
//! Run: cargo run --example basic

use turborag_sdk::TurboRagClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = TurboRagClient::new("http://127.0.0.1:8000");

    // Health check
    println!("Server alive: {}", client.health().await);

    // Index a document
    let id = client.index("Paris is the capital of France.", None).await?;
    println!("Indexed document: {}", id);

    // Search
    let results = client.search("capital of France", 3, None).await?;
    for r in &results {
        println!("  [{:.3}] {}", r.score, r.text);
    }

    // Ask
    let resp = client.ask("What is the capital of France?", 5, None).await?;
    println!("Answer: {}", resp.answer);

    Ok(())
}
