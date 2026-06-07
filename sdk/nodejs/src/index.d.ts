/**
 * TurboRag Node.js SDK — TypeScript declarations
 */

export interface SearchResult {
  id: string;
  text: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface AskResponse {
  answer: string;
  sources: SearchResult[];
  question: string;
}

export interface StatsResponse {
  doc_count: number;
  dim: number | null;
  bit_width: number;
  index_path: string;
  embed_model: string;
  llm_model: string | null;
}

export declare class TurboRagError extends Error {
  statusCode: number;
}

export declare class TurboRagClient {
  constructor(baseUrl?: string, apiKey?: string | null, timeout?: number);

  health(): Promise<boolean>;
  stats(): Promise<StatsResponse>;
  embed(text: string): Promise<number[]>;
  index(text: string, metadata?: Record<string, unknown>, chunk?: boolean): Promise<string>;
  indexBatch(texts: string[], metadatas?: Record<string, unknown>[] | null): Promise<string[]>;
  search(query: string, k?: number, filterIds?: string[] | null): Promise<SearchResult[]>;
  ask(question: string, k?: number, system?: string): Promise<AskResponse>;
  delete(docId: string): Promise<boolean>;
}
