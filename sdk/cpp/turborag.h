/**
 * TurboRag C/C++ SDK
 * ==================
 * Thin libcurl wrapper around the TurboRag REST API.
 *
 * Build:
 *   gcc -o example turborag.c -lcurl -ljson-c
 *   # or with CMake: target_link_libraries(myapp turborag curl json-c)
 *
 * Usage:
 *   turborag_client_t *c = turborag_create("http://127.0.0.1:8000", NULL);
 *   turborag_index(c, "Paris is the capital of France.", NULL);
 *   turborag_result_t res = turborag_ask(c, "What is the capital?", 5);
 *   printf("%s\n", res.answer);
 *   turborag_free_result(&res);
 *   turborag_destroy(c);
 */

#ifndef TURBORAG_H
#define TURBORAG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* -----------------------------------------------------------------------
 * Opaque client handle
 * --------------------------------------------------------------------- */
typedef struct turborag_client turborag_client_t;

/* -----------------------------------------------------------------------
 * Result types
 * --------------------------------------------------------------------- */

typedef struct {
    char  *id;
    char  *text;
    float  score;
    char  *metadata_json;  /* raw JSON string */
} turborag_hit_t;

typedef struct {
    turborag_hit_t *hits;
    size_t          count;
    char           *error;   /* NULL on success */
} turborag_search_result_t;

typedef struct {
    char  *answer;
    turborag_search_result_t sources;
    char  *error;            /* NULL on success */
} turborag_ask_result_t;

/* -----------------------------------------------------------------------
 * API
 * --------------------------------------------------------------------- */

/**
 * Create a new TurboRag client.
 * @param base_url  e.g. "http://127.0.0.1:8000"
 * @param api_key   Optional API key; pass NULL if not required.
 * @return          Handle; must be freed with turborag_destroy().
 */
turborag_client_t *turborag_create(const char *base_url, const char *api_key);

/** Destroy a client handle and release resources. */
void turborag_destroy(turborag_client_t *client);

/**
 * Check server liveness.
 * @return 1 if alive, 0 otherwise.
 */
int turborag_health(turborag_client_t *client);

/**
 * Index a document.
 * @param text          Document text.
 * @param metadata_json Optional metadata as JSON string, e.g. "{\"source\":\"wiki\"}".
 * @return              Allocated document ID string (caller must free()), or NULL on error.
 */
char *turborag_index(
    turborag_client_t *client,
    const char        *text,
    const char        *metadata_json
);

/**
 * Semantic search.
 * @param query  Search query text.
 * @param k      Number of results to return.
 * @return       Result struct; call turborag_free_search_result() when done.
 */
turborag_search_result_t turborag_search(
    turborag_client_t *client,
    const char        *query,
    int                k
);

/**
 * Full RAG: retrieve + generate.
 * @param question  The question to answer.
 * @param k         Number of context chunks to retrieve.
 * @return          Result struct; call turborag_free_ask_result() when done.
 */
turborag_ask_result_t turborag_ask(
    turborag_client_t *client,
    const char        *question,
    int                k
);

/**
 * Delete a document by ID.
 * @return 1 if deleted, 0 otherwise.
 */
int turborag_delete(turborag_client_t *client, const char *doc_id);

/** Free memory inside a search result (does NOT free the struct itself). */
void turborag_free_search_result(turborag_search_result_t *result);

/** Free memory inside an ask result. */
void turborag_free_ask_result(turborag_ask_result_t *result);

#ifdef __cplusplus
}
#endif

#endif /* TURBORAG_H */
