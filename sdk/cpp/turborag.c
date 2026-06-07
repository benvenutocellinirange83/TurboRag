/**
 * turborag.c — TurboRag C SDK implementation
 * Requires: libcurl, json-c
 * Build: gcc -shared -fPIC -o libturborag.so turborag.c -lcurl -ljson-c
 */

#include "turborag.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <curl/curl.h>
#include <json-c/json.h>

/* -----------------------------------------------------------------------
 * Internal types
 * --------------------------------------------------------------------- */

struct turborag_client {
    char *base_url;
    char *api_key;   /* may be NULL */
};

typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} _buf_t;

/* -----------------------------------------------------------------------
 * Write callback for libcurl
 * --------------------------------------------------------------------- */
static size_t _write_cb(void *ptr, size_t size, size_t nmemb, void *userdata) {
    _buf_t *buf = (struct _buf_t *)userdata;
    size_t need = size * nmemb;
    if (buf->len + need + 1 > buf->cap) {
        buf->cap = (buf->len + need + 1) * 2;
        buf->data = realloc(buf->data, buf->cap);
    }
    memcpy(buf->data + buf->len, ptr, need);
    buf->len += need;
    buf->data[buf->len] = '\0';
    return need;
}

/* -----------------------------------------------------------------------
 * HTTP helpers
 * --------------------------------------------------------------------- */

static char *_request(
    turborag_client_t *c,
    const char        *method,
    const char        *path,
    const char        *body   /* NULL for GET/DELETE */
) {
    CURL *curl = curl_easy_init();
    if (!curl) return NULL;

    char url[4096];
    snprintf(url, sizeof(url), "%s%s", c->base_url, path);

    _buf_t buf = { .data = malloc(256), .len = 0, .cap = 256 };
    buf.data[0] = '\0';

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: application/json");

    if (c->api_key) {
        char hdr[512];
        snprintf(hdr, sizeof(hdr), "X-API-Key: %s", c->api_key);
        headers = curl_slist_append(headers, hdr);
    }

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, _write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);

    if (strcmp(method, "POST") == 0 && body) {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(body));
    } else if (strcmp(method, "DELETE") == 0) {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    }

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        free(buf.data);
        return NULL;
    }
    return buf.data;  /* caller must free() */
}

/* -----------------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------------- */

turborag_client_t *turborag_create(const char *base_url, const char *api_key) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    turborag_client_t *c = calloc(1, sizeof(*c));
    c->base_url = strdup(base_url);
    c->api_key  = api_key ? strdup(api_key) : NULL;
    return c;
}

void turborag_destroy(turborag_client_t *c) {
    if (!c) return;
    free(c->base_url);
    free(c->api_key);
    free(c);
    curl_global_cleanup();
}

int turborag_health(turborag_client_t *c) {
    char *resp = _request(c, "GET", "/health", NULL);
    if (!resp) return 0;
    int ok = strstr(resp, "\"ok\"") != NULL;
    free(resp);
    return ok;
}

char *turborag_index(turborag_client_t *c, const char *text, const char *meta_json) {
    /* Build JSON body */
    json_object *obj = json_object_new_object();
    json_object_object_add(obj, "text", json_object_new_string(text));
    if (meta_json) {
        json_object *meta = json_tokener_parse(meta_json);
        json_object_object_add(obj, "metadata", meta ? meta : json_object_new_object());
    } else {
        json_object_object_add(obj, "metadata", json_object_new_object());
    }
    const char *body = json_object_to_json_string(obj);

    char *resp = _request(c, "POST", "/index", body);
    json_object_put(obj);

    if (!resp) return NULL;

    json_object *jresp = json_tokener_parse(resp);
    free(resp);
    if (!jresp) return NULL;

    json_object *jid;
    char *id = NULL;
    if (json_object_object_get_ex(jresp, "id", &jid)) {
        id = strdup(json_object_get_string(jid));
    }
    json_object_put(jresp);
    return id;  /* caller must free() */
}

turborag_search_result_t turborag_search(turborag_client_t *c, const char *query, int k) {
    turborag_search_result_t out = {0};

    json_object *obj = json_object_new_object();
    json_object_object_add(obj, "query", json_object_new_string(query));
    json_object_object_add(obj, "k",     json_object_new_int(k));
    const char *body = json_object_to_json_string(obj);

    char *resp = _request(c, "POST", "/search", body);
    json_object_put(obj);

    if (!resp) { out.error = strdup("Request failed"); return out; }

    json_object *jresp = json_tokener_parse(resp);
    free(resp);
    if (!jresp) { out.error = strdup("Invalid JSON"); return out; }

    json_object *jresults;
    if (!json_object_object_get_ex(jresp, "results", &jresults)) {
        json_object_put(jresp);
        out.error = strdup("Missing 'results' field");
        return out;
    }

    int n = json_object_array_length(jresults);
    out.hits  = calloc(n, sizeof(turborag_hit_t));
    out.count = (size_t)n;

    for (int i = 0; i < n; i++) {
        json_object *item = json_object_array_get_idx(jresults, i);
        json_object *jf;
        if (json_object_object_get_ex(item, "id",   &jf)) out.hits[i].id   = strdup(json_object_get_string(jf));
        if (json_object_object_get_ex(item, "text", &jf)) out.hits[i].text = strdup(json_object_get_string(jf));
        if (json_object_object_get_ex(item, "score",&jf)) out.hits[i].score = (float)json_object_get_double(jf);
        if (json_object_object_get_ex(item, "metadata", &jf))
            out.hits[i].metadata_json = strdup(json_object_to_json_string(jf));
    }

    json_object_put(jresp);
    return out;
}

turborag_ask_result_t turborag_ask(turborag_client_t *c, const char *question, int k) {
    turborag_ask_result_t out = {0};

    json_object *obj = json_object_new_object();
    json_object_object_add(obj, "question", json_object_new_string(question));
    json_object_object_add(obj, "k",        json_object_new_int(k));
    const char *body = json_object_to_json_string(obj);

    char *resp = _request(c, "POST", "/ask", body);
    json_object_put(obj);

    if (!resp) { out.error = strdup("Request failed"); return out; }

    json_object *jresp = json_tokener_parse(resp);
    free(resp);
    if (!jresp) { out.error = strdup("Invalid JSON"); return out; }

    json_object *jf;
    if (json_object_object_get_ex(jresp, "answer", &jf))
        out.answer = strdup(json_object_get_string(jf));

    json_object_put(jresp);
    return out;
}

int turborag_delete(turborag_client_t *c, const char *doc_id) {
    char path[512];
    snprintf(path, sizeof(path), "/document/%s", doc_id);
    char *resp = _request(c, "DELETE", path, NULL);
    if (!resp) return 0;
    int ok = strstr(resp, "deleted") != NULL;
    free(resp);
    return ok;
}

void turborag_free_search_result(turborag_search_result_t *r) {
    if (!r) return;
    for (size_t i = 0; i < r->count; i++) {
        free(r->hits[i].id);
        free(r->hits[i].text);
        free(r->hits[i].metadata_json);
    }
    free(r->hits);
    free(r->error);
    r->hits  = NULL;
    r->count = 0;
    r->error = NULL;
}

void turborag_free_ask_result(turborag_ask_result_t *r) {
    if (!r) return;
    free(r->answer);
    turborag_free_search_result(&r->sources);
    free(r->error);
    r->answer = NULL;
    r->error  = NULL;
}
