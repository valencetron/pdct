---
date: 2026-02-20
concepts: [rate-limit-incident, exponential-backoff, api-reliability, incident-postmortem]
tags: [distillation, example]
gist: Postmortem of the February rate-limit incident; fixed with jittered exponential backoff
---
# API Rate Limit Incident Postmortem

The nightly sync job hammered the partner API after a pagination bug caused infinite retries, tripping their rate limiter and getting the key temporarily banned. Fix: jittered exponential backoff (base 2s, cap 5min), a circuit breaker after 5 consecutive failures, and an alert on 429 density.

## Key Points
- Root cause: pagination bug caused unbounded retry loop
- Fix: jittered exponential backoff, circuit breaker at 5 failures
- New alert: more than 10 HTTP 429s in 5 minutes pages the on-call
