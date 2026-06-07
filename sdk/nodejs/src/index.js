/**
 * TurboRag Node.js SDK
 * ====================
 * Zero-dependency HTTP client for the TurboRag REST API.
 *
 * Installation:
 *   npm install turborag-sdk   (or copy this file into your project)
 *
 * Usage:
 *   const { TurboRagClient } = require('./turborag');
 *
 *   const client = new TurboRagClient('http://127.0.0.1:8000');
 *
 *   await client.index('Paris is the capital of France.');
 *   const results = await client.search('capital of France', 3);
 *   const { answer } = await client.ask('What is the capital of France?');
 *   console.log(answer);
 */

'use strict';

const http = require('http');
const https = require('https');
const { URL } = require('url');

class TurboRagError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.name = 'TurboRagError';
    this.statusCode = statusCode;
  }
}

class TurboRagClient {
  /**
   * @param {string} baseUrl   Base URL of TurboRag API, e.g. "http://127.0.0.1:8000"
   * @param {string|null} apiKey  Optional API key for X-API-Key header
   * @param {number} timeout   Request timeout in ms (default 60000)
   */
  constructor(baseUrl = 'http://127.0.0.1:8000', apiKey = null, timeout = 60000) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = apiKey;
    this.timeout = timeout;
  }

  // ----------------------------------------------------------------
  // Public API
  // ----------------------------------------------------------------

  /** @returns {Promise<boolean>} true if server is alive */
  async health() {
    try {
      const resp = await this._get('/health');
      return resp.status === 'ok';
    } catch (_) {
      return false;
    }
  }

  /** @returns {Promise<Object>} index statistics */
  async stats() {
    return this._get('/stats');
  }

  /**
   * Get the embedding vector for text.
   * @param {string} text
   * @returns {Promise<number[]>}
   */
  async embed(text) {
    const resp = await this._post('/embed', { text });
    return resp.embedding;
  }

  /**
   * Index a document.
   * @param {string} text
   * @param {Object} [metadata]
   * @param {boolean} [chunk]  Auto-chunk long text
   * @returns {Promise<string>}  document ID
   */
  async index(text, metadata = {}, chunk = false) {
    const resp = await this._post('/index', { text, metadata, chunk });
    return resp.id;
  }

  /**
   * Index multiple documents.
   * @param {string[]} texts
   * @param {Object[]} [metadatas]
   * @returns {Promise<string[]>}  list of document IDs
   */
  async indexBatch(texts, metadatas = null) {
    const resp = await this._post('/index/batch', { texts, metadatas });
    return resp.ids;
  }

  /**
   * Semantic search.
   * @param {string} query
   * @param {number} [k]
   * @param {string[]} [filterIds]
   * @returns {Promise<Array<{id, text, score, metadata}>>}
   */
  async search(query, k = 5, filterIds = null) {
    const body = { query, k };
    if (filterIds) body.filter_ids = filterIds;
    const resp = await this._post('/search', body);
    return resp.results;
  }

  /**
   * Full RAG: retrieve + generate.
   * @param {string} question
   * @param {number} [k]
   * @param {string} [system]
   * @returns {Promise<{answer: string, sources: Array, question: string}>}
   */
  async ask(question, k = 5, system = '') {
    const body = { question, k };
    if (system) body.system = system;
    return this._post('/ask', body);
  }

  /**
   * Delete a document by ID.
   * @param {string} docId
   * @returns {Promise<boolean>}
   */
  async delete(docId) {
    try {
      await this._delete(`/document/${encodeURIComponent(docId)}`);
      return true;
    } catch (_) {
      return false;
    }
  }

  // ----------------------------------------------------------------
  // HTTP helpers
  // ----------------------------------------------------------------

  _headers() {
    const h = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };
    if (this.apiKey) h['X-API-Key'] = this.apiKey;
    return h;
  }

  _get(path) {
    return this._request('GET', path, null);
  }

  _post(path, body) {
    return this._request('POST', path, body);
  }

  _delete(path) {
    return this._request('DELETE', path, null);
  }

  _request(method, path, body) {
    return new Promise((resolve, reject) => {
      const url = new URL(this.baseUrl + path);
      const lib = url.protocol === 'https:' ? https : http;
      const payload = body ? JSON.stringify(body) : null;

      const options = {
        hostname: url.hostname,
        port: url.port || (url.protocol === 'https:' ? 443 : 80),
        path: url.pathname + url.search,
        method,
        headers: {
          ...this._headers(),
          ...(payload ? { 'Content-Length': Buffer.byteLength(payload) } : {}),
        },
        timeout: this.timeout,
      };

      const req = lib.request(options, (res) => {
        let data = '';
        res.on('data', (chunk) => (data += chunk));
        res.on('end', () => {
          try {
            const parsed = JSON.parse(data);
            if (res.statusCode >= 400) {
              reject(new TurboRagError(parsed.detail || data, res.statusCode));
            } else {
              resolve(parsed);
            }
          } catch (e) {
            reject(new TurboRagError(`Invalid JSON: ${data}`, res.statusCode));
          }
        });
      });

      req.on('timeout', () => {
        req.destroy();
        reject(new TurboRagError('Request timed out', 0));
      });

      req.on('error', (err) => {
        reject(new TurboRagError(err.message, 0));
      });

      if (payload) req.write(payload);
      req.end();
    });
  }
}

module.exports = { TurboRagClient, TurboRagError };
