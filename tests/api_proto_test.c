// Host tests for chat/api_proto.c (the OpenAI-compatible API's protocol part).
// Build and run: gcc -std=c99 -Wall -Wextra -g -fsanitize=address,undefined -I chat tests/api_proto_test.c chat/api_proto.c -o build/api_proto_test && build/api_proto_test
#include <stdio.h>
#include <string.h>

#include "api_proto.h"

static int fails, checks;
#define CHECK(c)                                                              \
    do {                                                                      \
        checks++;                                                             \
        if (!(c)) {                                                           \
            fails++;                                                          \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);               \
        }                                                                     \
    } while (0)

static void test_http(void) {
    HttpRequest r;
    const char* post = "POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: ps4\r\nContent-Type: application/json\r\n"
                       "content-length: 13\r\nAuthorization: Bearer abc\r\n\r\n{\"a\":\"b\"}    ";
    CHECK(api_http_parse(post, strlen(post), &r) == 1);
    CHECK(!strcmp(r.method, "POST"));
    CHECK(!strcmp(r.path, "/v1/chat/completions"));
    CHECK(r.content_length == 13);
    CHECK(r.body_len == 13 && !memcmp(r.body, "{\"a\":\"b\"}    ", 13));
    CHECK(!strcmp(r.auth, "Bearer abc"));

    const char* partial = "POST /x HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc";
    CHECK(api_http_parse(partial, strlen(partial), &r) == 0);
    const char* head_only = "GET /v1/models HTTP/1.1\r\nHost: a\r\n";
    CHECK(api_http_parse(head_only, strlen(head_only), &r) == 0);
    const char* get = "GET /v1/models HTTP/1.1\r\nHost: a\r\n\r\n";
    CHECK(api_http_parse(get, strlen(get), &r) == 1);
    CHECK(!strcmp(r.path, "/v1/models") && r.content_length == 0);
    const char* bad = "GET\r\n\r\n";
    CHECK(api_http_parse(bad, strlen(bad), &r) == -1);
    const char* badcl = "POST / HTTP/1.1\r\nContent-Length: nope\r\n\r\n";
    CHECK(api_http_parse(badcl, strlen(badcl), &r) == -1);
    const char* bigcl = "POST / HTTP/1.1\r\nContent-Length: 99999999\r\n\r\n";
    CHECK(api_http_parse(bigcl, strlen(bigcl), &r) == -1);
}

static void test_json(void) {
    JNode pool[64];
    const char* doc = "{ \"a\" : [1, 2.5, -3e2, true, false, null], \"s\": \"x\\\"y\\\\\\n\\u00e9\\ud83d\\ude00\" }";
    int root = json_parse(doc, strlen(doc), pool, 64);
    CHECK(root >= 0);
    int a = json_get(pool, root, "a");
    CHECK(a >= 0 && pool[a].type == J_ARR && pool[a].count == 6);
    int c = pool[a].first;
    CHECK(pool[c].type == J_NUM && pool[c].num == 1);
    c = pool[c].next;
    CHECK(pool[c].num == 2.5);
    c = pool[c].next;
    CHECK(pool[c].num == -300);
    c = pool[c].next;
    CHECK(pool[c].type == J_BOOL && pool[c].b == 1);
    int s = json_get(pool, root, "s");
    Sb out;
    sb_init(&out);
    json_str_sb(pool, s, &out);
    CHECK(out.n == strlen("x\"y\\\n\xc3\xa9\xf0\x9f\x98\x80"));
    CHECK(out.p && !memcmp(out.p, "x\"y\\\n\xc3\xa9\xf0\x9f\x98\x80", out.n));
    sb_free(&out);

    CHECK(json_parse("{\"a\":}", 6, pool, 64) == -1);
    CHECK(json_parse("[1,2", 4, pool, 64) == -1);
    CHECK(json_parse("{} x", 4, pool, 64) == -1);
    CHECK(json_parse("[1,2]", 5, pool, 2) == -1);   // pool too small
    CHECK(json_get(pool, root, "missing") == -1);
}

static void test_chat(void) {
    static JNode pool[API_MAX_NODES];
    ChatRequest req;
    const char* err = NULL;
    const char* body =
        "{\"model\":\"x\",\"messages\":[{\"role\":\"system\",\"content\":\"Be brief.\"},"
        "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hi \"},{\"type\":\"text\",\"text\":\"there\"}]}],"
        "\"temperature\":0.2,\"max_tokens\":64,\"stream\":true,\"seed\":7}";
    CHECK(api_chat_parse(body, strlen(body), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.n_messages == 2 && req.stream == 1 && req.max_tokens == 64 && req.has_seed && req.seed == 7);
    CHECK(req.temperature > 0.19 && req.temperature < 0.21);
    char role[16];
    Sb content;
    sb_init(&content);
    int m = pool[req.messages].first;
    CHECK(api_message(pool, m, role, sizeof(role), &content) == 0 && !strcmp(role, "system"));
    CHECK(content.n == 9 && !memcmp(content.p, "Be brief.", 9));
    sb_free(&content);
    sb_init(&content);
    m = pool[m].next;
    CHECK(api_message(pool, m, role, sizeof(role), &content) == 0 && !strcmp(role, "user"));
    CHECK(content.n == 8 && !memcmp(content.p, "Hi there", 8));
    sb_free(&content);

    const char* defaults = "{\"messages\":[{\"role\":\"user\",\"content\":\"yo\"}]}";
    CHECK(api_chat_parse(defaults, strlen(defaults), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.max_tokens == 512 && req.stream == 0 && !req.has_seed);

    const char* bad[] = {
        "[]",
        "{\"messages\":[]}",
        "{\"messages\":[1]}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"n\":2}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"temperature\":3}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"max_tokens\":0}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"stream\":\"yes\"}",
        "{\"messages\":[{\"role\":\"user\"",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        CHECK(api_chat_parse(bad[i], strlen(bad[i]), pool, API_MAX_NODES, &req, &err) == -1);
        CHECK(err != NULL);
    }
}

// The request's reasoning fields: the chat template's switch first, then the plainer spellings; none = -1.
static void test_think_fields(void) {
    static JNode pool[API_MAX_NODES];
    ChatRequest req;
    const char* err = NULL;
    static const struct {
        const char* extra;
        int think;
    } cases[] = {
        {"", -1},
        {",\"chat_template_kwargs\":{\"enable_thinking\":true}", 1},
        {",\"chat_template_kwargs\":{\"enable_thinking\":false}", 0},
        {",\"enable_thinking\":true", 1},
        {",\"think\":false", 0},
        {",\"reasoning_effort\":\"high\"", 1},
        {",\"reasoning_effort\":\"low\"", 1},
        {",\"reasoning_effort\":\"none\"", 0},
        {",\"reasoning_effort\":\"minimal\"", 0},
        {",\"chat_template_kwargs\":{\"enable_thinking\":false},\"reasoning_effort\":\"high\"", 0},
        {",\"chat_template_kwargs\":{},\"think\":true", 1},
        {",\"enable_thinking\":\"yes\"", -1},   // (not a boolean: ignored, as unknown fields are)
    };
    char body[256];
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        snprintf(body, sizeof(body), "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}]%s}", cases[i].extra);
        CHECK(api_chat_parse(body, strlen(body), pool, API_MAX_NODES, &req, &err) == 0);
        CHECK(req.think == cases[i].think);
    }
}

static int switch_of(const char* in, const char* left, int want) {
    Sb s;
    sb_init(&s);
    sb_str(&s, in);
    int r = api_think_switch(&s);
    int ok = r == want && s.n == strlen(left) && (!s.n || !memcmp(s.p, left, s.n));
    sb_free(&s);
    return ok;
}

static int drop_of(const char* in, const char* left) {
    Sb s;
    sb_init(&s);
    sb_str(&s, in);
    api_drop_think(&s, NULL);
    int ok = s.n == strlen(left) && (!s.n || !strcmp(s.p, left));
    sb_free(&s);
    return ok;
}

static void test_tool_fields(void) {
    static JNode pool[API_MAX_NODES];
    ChatRequest req;
    const char* err = NULL;
#define MSG "\"messages\":[{\"role\":\"user\",\"content\":\"a\"}]"
#define TOOL "{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}}"
    const char* ok = "{" MSG ",\"tools\":[" TOOL "]}";
    CHECK(api_chat_parse(ok, strlen(ok), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.n_tools == 1 && req.tools >= 0 && req.tool_choice == TOOL_AUTO);
    const char* none = "{" MSG ",\"tools\":[" TOOL "],\"tool_choice\":\"none\"}";
    CHECK(api_chat_parse(none, strlen(none), pool, API_MAX_NODES, &req, &err) == 0 && req.tool_choice == TOOL_NONE);
    const char* named = "{" MSG ",\"tools\":[" TOOL "],\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"f\"}}}";
    CHECK(api_chat_parse(named, strlen(named), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.tool_choice == TOOL_NAMED && !strcmp(req.tool_name, "f"));
    const char* plain = "{" MSG "}";
    CHECK(api_chat_parse(plain, strlen(plain), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.n_tools == 0 && req.tools == -1 && req.tool_choice == TOOL_NONE);
    const char* bad[] = {
        "{" MSG ",\"tools\":{}}",
        "{" MSG ",\"tools\":[{\"type\":\"function\",\"function\":{}}]}",
        "{" MSG ",\"tool_choice\":\"required\"}",
        "{" MSG ",\"tools\":[" TOOL "],\"tool_choice\":\"sometimes\"}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        CHECK(api_chat_parse(bad[i], strlen(bad[i]), pool, API_MAX_NODES, &req, &err) == -1);
        CHECK(err != NULL);
    }
#undef MSG
#undef TOOL
}

// A call written as JSON (the form of earlier Qwen models) is read too.
static void test_tool_json_call(void) {
    ApiToolCall calls[API_MAX_TOOL_CALLS];
    Sb before;
    sb_init(&before);
    const char* text = "Sure.\n<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>";
    int n = api_parse_tool_calls(NULL, NULL, text, strlen(text), &before, calls, API_MAX_TOOL_CALLS);
    CHECK(n == 1 && !strcmp(calls[0].name, "f") && calls[0].arguments.p && !strcmp(calls[0].arguments.p, "{\"a\":1}"));
    CHECK(before.n == 5 && !memcmp(before.p, "Sure.", 5));
    api_free_tool_calls(calls, n);
    sb_free(&before);
}

static void test_think_text(void) {
    CHECK(switch_of("Why is the sky blue? /think", "Why is the sky blue?", 1));
    CHECK(switch_of("Hi /no_think\n", "Hi", 0));
    CHECK(switch_of("Hi\n/nothink", "Hi", 0));
    CHECK(switch_of("/think", "", 1));
    CHECK(switch_of("Hello", "Hello", -1));
    CHECK(switch_of("look in /usr/think", "look in /usr/think", -1));   // (a path, not the switch)
    CHECK(switch_of("/think about it", "/think about it", -1));         // (only at the end)
    CHECK(drop_of("<think>\nfirst this\n</think>\n\nThe answer.", "The answer."));
    CHECK(drop_of("reasoning without its start</think>\nAnswer", "Answer"));
    CHECK(drop_of("a</think>b</think>\n\nc", "c"));
    CHECK(drop_of("No reasoning here.", "No reasoning here."));
}

static void test_writers(void) {
    Sb s;
    sb_init(&s);
    api_json_completion(&s, "chatcmpl-1", "Qwen3.5-9B", 1700000000, "", 0, "Hola \"mundo\"\n", 13, NULL, 0, "stop", 5, 4, 0);
    CHECK(s.p && strstr(s.p, "\"content\":\"Hola \\\"mundo\\\"\\n\"") != NULL);
    CHECK(strstr(s.p, "\"total_tokens\":9") != NULL);
    CHECK(strstr(s.p, "\"finish_reason\":\"stop\"") != NULL);
    CHECK(strstr(s.p, "reasoning") == NULL);   // (no reasoning: the same JSON as before reasoning existed)
    sb_free(&s);

    sb_init(&s);
    api_json_completion(&s, "c", "m", 1, "Let me see.", 11, "Yes.", 4, NULL, 0, "stop", 5, 30, 26);
    CHECK(strstr(s.p, "\"message\":{\"role\":\"assistant\",\"content\":\"Yes.\",\"reasoning_content\":\"Let me see.\"}"));
    CHECK(strstr(s.p, "\"completion_tokens\":30,\"total_tokens\":35,"
                      "\"completion_tokens_details\":{\"reasoning_tokens\":26}}") != NULL);
    sb_free(&s);

    sb_init(&s);
    api_json_chunk(&s, "c", "m", 1, "assistant", "", 0, "", 0, NULL);
    CHECK(!strcmp(s.p, "{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\","
                       "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}"));
    sb_free(&s);

    sb_init(&s);
    api_json_chunk(&s, "c", "m", 1, NULL, "Hmm", 3, "", 0, NULL);
    CHECK(strstr(s.p, "\"delta\":{\"reasoning_content\":\"Hmm\"},\"finish_reason\":null") != NULL);
    sb_free(&s);

    sb_init(&s);
    api_json_error(&s, "bad \x01 input", "invalid_request_error", "bad_body");
    CHECK(!strcmp(s.p, "{\"error\":{\"message\":\"bad \\u0001 input\",\"type\":\"invalid_request_error\","
                       "\"code\":\"bad_body\"}}"));
    sb_free(&s);

    sb_init(&s);
    api_json_models(&s, "Qwen3.5-9B", 0);
    CHECK(strstr(s.p, "\"data\":[{\"id\":\"Qwen3.5-9B\",\"object\":\"model\"") != NULL);
    sb_free(&s);
}

int main(void) {
    test_http();
    test_json();
    test_chat();
    test_think_fields();
    test_tool_fields();
    test_tool_json_call();
    test_think_text();
    test_writers();
    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
