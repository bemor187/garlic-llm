// OpenAI-compatible API jobs (api.c queues them; included in main.c, so it sees the engine and the conversation).
// A job runs on the model between the chat's turns: the request's messages are the whole conversation, so the
// conversation on screen is cleared, and the reply is written to the connection (whole, or as server-sent events).
// The reply is generated as the chat generates its own: several tokens per step with the MTP head where the app has
// it (GARLIC_SPEC), and Circle on the controller stops it. What the screen shows meanwhile is main.c's part
// (api_chat_begin / _reply / _step / _end).
//
// Reasoning works as in the chat (main.c, think_one): the turn opens <think> and the model reasons at THINK_T for up
// to think_cap tokens, then answers; if it has not closed </think> by then, Qwen's closing sentence and </think> are
// fed. The reasoning goes to the client apart from the answer (reasoning_content), and max_tokens counts the answer
// only. Whether a reply reasons: the last user message's /think or /no_think, else the request's own field
// (api_chat_parse), else the console's setting (R3).

#define API_IDS 16384             // tokens a job's prompt may have: the longest context of the app
#define API_TAIL 32               // kept free for the assistant's turn after the messages (at most 17 tokens)
#define API_THINK_MIN 64          // the least reasoning worth starting; with less room the reply comes without
static uint32_t api_ids[API_IDS];   // the prompt of a job

// How many bytes at the start of p (n) are whole UTF-8 characters (the rest waits for the next token).
static size_t api_utf8_whole(const uint8_t* p, size_t n) {
    size_t i = n;
    for (int back = 1; back <= 4 && i > 0; back++) {
        i--;
        uint8_t c = p[i];
        if ((c & 0xC0) == 0x80) continue;   // a continuation byte: keep looking for its lead byte
        size_t need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        return back < (int)need ? i : n;
    }
    return n;
}

// one server-sent event; -1 if the client went away
static int api_send_event(int fd, const char* json) {
    Sb s;
    sb_init(&s);
    sb_str(&s, "data: ");
    sb_str(&s, json);
    sb_str(&s, "\n\n");
    int r = s.p && !s.oom ? api_send(fd, s.p, s.n) : -1;
    sb_free(&s);
    return r;
}

// The whole characters of pend as one event (reasoning or answer), taken out of pend; -1 if the client went away.
static int api_stream_piece(int fd, const char* id, const char* model, long long created, Sb* pend, int reasoning) {
    if (!pend->n) return 0;
    size_t whole = api_utf8_whole((const uint8_t*)pend->p, pend->n);
    if (!whole) return 0;
    Sb ev;
    sb_init(&ev);
    if (reasoning) api_json_chunk(&ev, id, model, created, NULL, pend->p, whole, "", 0, NULL);
    else api_json_chunk(&ev, id, model, created, NULL, "", 0, pend->p, whole, NULL);
    int r = ev.p && !ev.oom ? api_send_event(fd, ev.p) : -1;
    sb_free(&ev);
    memmove(pend->p, pend->p + whole, pend->n - whole);
    pend->n -= whole;
    pend->p[pend->n] = 0;
    return r;
}

// The role a message is read with: system, assistant and tool as they are; any other role is the user's.
static int api_is_user(const JNode* pool, int m) {
    int r = json_get(pool, m, "role");
    if (r < 0 || pool[r].type != J_STR) return 0;
    static const char* const other[] = {"system", "assistant", "tool"};
    for (int i = 0; i < 3; i++)
        if (pool[r].slen == strlen(other[i]) && !memcmp(pool[r].s, other[i], pool[r].slen)) return 0;
    return 1;
}

// The messages of the prompt: each as <|im_start|>role\ncontent<|im_end|>\n. The last user message's reasoning switch
// is taken out of its text (into *think_switch: 1, 0 or -1, see api_think_switch), and reasoning that a client sent
// back in an earlier assistant message is left out (api_drop_think). Returns the number of ids, -1 if a message is
// malformed, or -2 if the messages do not fit API_IDS. The limit counts tokens, not bytes: a long conversation may
// use the whole context. (Every token stands for one byte or more, so the turn after the messages fits API_TAIL.)
static int api_prompt(const JNode* pool, const ChatRequest* cr, int* think_switch) {
    const int cap = API_IDS - API_TAIL;   // where the messages must end
    int n = 0, i = 0, last_user = -1;
    *think_switch = -1;
    for (int m = pool[cr->messages].first; m >= 0; m = pool[m].next, i++)
        if (api_is_user(pool, m)) last_user = i;
    i = 0;
    for (int m = pool[cr->messages].first; m >= 0; m = pool[m].next, i++) {
        Sb content;
        sb_init(&content);
        char role[16] = "user";
        int ok = api_message(pool, m, role, sizeof(role), &content) == 0;
        if (!ok || content.oom) {
            sb_free(&content);
            return -1;
        }
        if (strcmp(role, "system") && strcmp(role, "assistant") && strcmp(role, "user") && strcmp(role, "tool"))
            strcpy(role, "user");
        if (!strcmp(role, "assistant")) api_drop_think(&content);
        if (i == last_user) *think_switch = api_think_switch(&content);
        if (n >= cap) {
            sb_free(&content);
            return -2;
        }
        api_ids[n++] = IM_START;
        n = encode_into(api_ids, n, cap, role);
        n = encode_into(api_ids, n, cap, "\n");
        if (content.p) n = encode_into(api_ids, n, cap, content.p);
        sb_free(&content);
        if (n + 2 > cap) return -2;   // the text filled what was left (it may have been cut): too long
        api_ids[n++] = IM_END;
        n = encode_into(api_ids, n, cap, "\n");
    }
    return n;
}

// The assistant's turn after the n ids of the messages: <think>\n to reason, or the empty thought block (as the chat
// does, enable_thinking=false). Returns the number of ids.
static int api_turn(int n, int think) {
    api_ids[n++] = IM_START;
    n = encode_into(api_ids, n, API_IDS, "assistant\n");
    if (eng.qwen35) {
        api_ids[n++] = THINK;
        if (think) return encode_into(api_ids, n, API_IDS, "\n");
        n = encode_into(api_ids, n, API_IDS, "\n\n");
        api_ids[n++] = THINK_END;
        n = encode_into(api_ids, n, API_IDS, "\n\n");
    }
    return n;
}

static void api_error(int fd, const char* status_line, const char* message, const char* type, const char* code) {
    Sb s;
    sb_init(&s);
    api_json_error(&s, message, type, code);
    Sb h;
    sb_init(&h);
    sb_str(&h, status_line);
    sb_str(&h, "\r\nContent-Type: application/json\r\nContent-Length: ");
    sb_int(&h, (long long)s.n);
    sb_str(&h, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
    if (h.p && s.p && api_send(fd, h.p, h.n) == 0) api_send(fd, s.p, s.n);
    sb_free(&h);
    sb_free(&s);
}

static int api_is_blank(const uint8_t* p, int n) {
    for (int i = 0; i < n; i++)
        if (p[i] != '\n' && p[i] != '\r' && p[i] != ' ') return 0;
    return 1;
}

static void api_job(ApiJob* job) {
    static unsigned api_count;
    if (api_client_gone(job->fd)) {   // (it waited in line, and its client has given up meanwhile)
        say("api: the client of a waiting request left; nothing to answer");
        return;
    }
    JNode* pool = malloc(sizeof(JNode) * API_MAX_NODES);
    ChatRequest cr;
    const char* err = NULL;
    if (!pool || api_chat_parse(job->body, job->body_len, pool, API_MAX_NODES, &cr, &err)) {
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", err ? err : "Out of memory.", "invalid_request_error",
                  "invalid_body");
        return;
    }
    int think_switch;
    int n = api_prompt(pool, &cr, &think_switch);
    if (n == -1) {
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", "A message is malformed: it needs a 'role', and 'content' must "
                  "be a string or an array of parts.", "invalid_request_error", "invalid_body");
        return;
    }
    if (n < 0) {
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", "The messages do not fit the context window.",
                  "invalid_request_error", "context_length_exceeded");
        return;
    }
    free(pool);
    // Reasoning, when asked for and when the context has room for it besides the answer and the closing sentence.
    int want = think_switch >= 0 ? think_switch : cr.think >= 0 ? cr.think : think_on;
    int think_budget = 0;
    if (want && eng.qwen35) {
        think_budget = (int)eng.ctx_limit - (n + API_TAIL) - cr.max_tokens - 64;
        if (think_budget > think_cap) think_budget = think_cap;
        if (think_budget < API_THINK_MIN) think_budget = 0;
    }
    int think = think_budget > 0;
    n = api_turn(n, think);
    if (n + cr.max_tokens + 2 > (int)eng.ctx_limit) {
        api_error(job->fd, "HTTP/1.1 400 Bad Request", "The messages and max_tokens do not fit the context window.",
                  "invalid_request_error", "context_length_exceeded");
        return;
    }
    // the model answers this request: the conversation on screen starts again from nothing (a request refused
    // above leaves it as it was)
    pos = 0;
    need_close = 0;
    hist_n = hist_done = 0;
    thinking = think_turn = 0;
    sys_pre = 0;
    view.scroll = 0;
    api_chat_begin();   // (clears the conversation: a note says that a request is being answered)
    if (cr.has_seed) seed = cr.seed;
    char id[32];
    snprintf(id, sizeof(id), "chatcmpl-%u", ++api_count);
    const char* model = api_model();
    long long created = (long long)time(NULL);
    say("api: prompt of %d tokens, up to %d new, T %.2f%s%s", n, cr.max_tokens, cr.temperature,
        think ? ", reasoning" : "", cr.stream ? ", streamed" : "");
    engine_reset(&eng);
    if (engine_run(&eng, api_ids, n, 0, think ? THINK_T : (float)cr.temperature, seed++, &next_token)) {
        api_error(job->fd, "HTTP/1.1 500 Internal Server Error", eng.error, "server_error", NULL);
        pos = 0;
        api_chat_end();
        return;
    }
    pos = n;

    int fd = job->fd, streamed = 0, client_gone = 0, error = 0;
    if (cr.stream) {
        const char* h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
        if (api_send(fd, h, strlen(h))) client_gone = 1;
        streamed = 1;
        Sb first;
        sb_init(&first);
        api_json_chunk(&first, id, model, created, "assistant", "", 0, "", 0, NULL);
        if (!client_gone && first.p && api_send_event(fd, first.p)) client_gone = 1;
        sb_free(&first);
    }

    // The answer (all of it) and the reasoning before it, each with the bytes of its last tokens that are not yet a
    // whole character.
    Sb text, pend, rtext, rpend;
    sb_init(&text);
    sb_init(&pend);
    sb_init(&rtext);
    sb_init(&rpend);
    int produced = 0, think_n = 0, stopped = 0, cut = 0, in_think = think;
    const char* finish = "length";
    // The tokens to write: the one a step drew, after those it kept from the MTP head's draft (the model has read
    // these already). Without the head a step is one token.
    uint32_t q[9];
    int nq = 1, iq = 0;
    q[0] = next_token;
    double t0 = now();
    api_chat_reply();
    while (produced < cr.max_tokens) {
        uint32_t t = q[iq++];
        const uint8_t* bytes = NULL;
        if (in_think) {
            int natural = t == THINK_END;
            if (natural || t == IM_END || t == eng.eos2 || t == eng.eos || think_n >= think_budget) {
                in_think = 0;   // the answer begins (the bytes of a cut character are left out of the reasoning)
                rpend.n = 0;
                if (!natural) {
                    // Qwen's closing sentence, </think> and the blank line; the answer is drawn after them
                    uint32_t ids[64];
                    int k = encode_into(ids, 0, 56, THINK_CLOSE);
                    ids[k++] = THINK_END;
                    k = encode_into(ids, k, 64, "\n\n");
                    cut = 1;
                    if (engine_run(&eng, ids, k, pos, (float)cr.temperature, seed++, &next_token)) {
                        error = 1;
                        break;
                    }
                    pos += k;
                    q[0] = next_token;
                    nq = 1, iq = 0;
                    continue;
                }
                // (</think> as the model wrote it goes in below like any token, and the answer is drawn after it)
            } else {
                int nb = tok_piece(&tok, t, &bytes);
                think_n++;
                if (nb > 0) {
                    sb_add(&rtext, (const char*)bytes, nb);
                    sb_add(&rpend, (const char*)bytes, nb);
                }
                if (streamed && api_stream_piece(fd, id, model, created, &rpend, 1)) client_gone = 1;
            }
        } else {
            if (t == IM_END || t == eng.eos2 || t == eng.eos) {
                finish = "stop";
                break;
            }
            int nb = tok_piece(&tok, t, &bytes);
            // (after reasoning the answer begins with a blank line: not part of it, and not counted)
            int lead = think && !text.n && nb > 0 && api_is_blank(bytes, nb);
            if (!lead) produced++;
            if (nb > 0 && !lead) {
                sb_add(&text, (const char*)bytes, nb);
                sb_add(&pend, (const char*)bytes, nb);
            }
            if (streamed && api_stream_piece(fd, id, model, created, &pend, 0)) client_gone = 1;
            if (produced == cr.max_tokens) break;
        }
        if (client_gone) break;
        if (iq < nq) continue;   // (the next token is one the last step kept)
        if (api_chat_step(produced + think_n)) {   // Circle: the reply ends here, cut short as by max_tokens
            stopped = 1;
            break;
        }
        float temp = in_think ? THINK_T : (float)cr.temperature;
        int ne = 0;   // tokens of the draft this step keeps
#ifdef GARLIC_SPEC
        if (spec_active(&eng)) {
            int left = in_think ? think_budget - think_n : cr.max_tokens - produced;   // (a step never keeps more)
            if (engine_spec_step(&eng, t, pos, temp, seed++, left > 0 ? left + 1 : 1, q, &ne, &next_token)) {
                error = 1;
                break;
            }
        } else
#endif
        if (engine_run(&eng, &t, 1, pos, temp, seed++, &next_token)) {
            error = 1;
            break;
        }
        pos += 1 + ne;
        q[ne] = next_token;
        nq = ne + 1, iq = 0;
    }
    double dt = now() - t0;
    if (think)
        say("api: reasoning %d tokens%s", think_n, cut ? " (cut: closing sentence fed)" : in_think ? " (unfinished)" : "");
    say("api: %d tokens in %.1f s (%.1f tokens/s), %s", produced + think_n, dt, dt > 0 ? (produced + think_n) / dt : 0.0,
        error ? "engine error" : stopped ? "stopped on the console" : finish);

    if (streamed) {
        if (!client_gone && !error) {
            Sb last;
            sb_init(&last);
            api_json_chunk(&last, id, model, created, NULL, "", 0, "", 0, finish);
            if (last.p) api_send_event(fd, last.p);
            sb_free(&last);
            api_send(fd, "data: [DONE]\n\n", 14);
        }
        if (error && !client_gone) {   // the headers are out: the error goes in the stream
            Sb e;
            sb_init(&e);
            api_json_error(&e, eng.error, "server_error", NULL);
            if (e.p) api_send_event(fd, e.p);
            sb_free(&e);
        }
    } else if (error) {
        api_error(fd, "HTTP/1.1 500 Internal Server Error", eng.error, "server_error", NULL);
    } else {
        Sb body, head;
        sb_init(&body);
        sb_init(&head);
        // a reply cut short (max_tokens, Circle) may end inside a character: those bytes are left out, as in a stream,
        // so the JSON stays valid UTF-8; the reasoning's last line break goes too
        size_t tn = text.p ? api_utf8_whole((const uint8_t*)text.p, text.n) : 0;
        size_t rn = rtext.p ? api_utf8_whole((const uint8_t*)rtext.p, rtext.n) : 0;
        while (rn && (rtext.p[rn - 1] == '\n' || rtext.p[rn - 1] == ' ')) rn--;
        api_json_completion(&body, id, model, created, rtext.p ? rtext.p : "", rn, text.p ? text.p : "", tn, finish, n,
                            produced + think_n, think_n);
        sb_str(&head, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ");
        sb_int(&head, (long long)body.n);
        sb_str(&head, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        if (head.p && body.p && api_send(fd, head.p, head.n) == 0) api_send(fd, body.p, body.n);
        sb_free(&body);
        sb_free(&head);
    }
    sb_free(&text);
    sb_free(&pend);
    sb_free(&rtext);
    sb_free(&rpend);
    // the engine's state is this request's; the conversation on screen starts again at position 0
    pos = 0;
    need_close = 0;
    api_chat_end();
}
