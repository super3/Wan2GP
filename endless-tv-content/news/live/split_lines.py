"""Riverside News script -> lines for 5.17 s live clips (at most MAX words: the earlier live channels spoke up to 13-14 words a clip).
Words are packed by DP: fewest lines, breaking after a sentence end, then after a comma, then before a joining word, avoiding short
lines. The opening (1) and the sign-off (40) are packed on their own, in a Today and a Tonight version; the body (2-39) is one stream."""
import json, re
MAX = 13
SRC = "/home/user/Wan2GP/outputs/riverside_news/riverside_news_2026-11-25_script.md"
text = open(SRC).read().split("## Sources")[0]
items = re.findall(r"^\*\*(\d+) · [\d:]+(?: \((Today|Tonight)\))?\*\* (.+)$", text, re.M)
ABBR = {"U.S.S.", "U.S.", "Mr.", "J.D.", "F.A.A."}
JOIN = {"and", "but", "after", "with", "while", "as", "where", "which", "who", "that", "despite", "because", "including", "from", "to",
        "in", "on", "by", "about", "since", "before", "or", "if", "when", "so"}

def pack(body):
    w = body.split(); n = len(w)
    def brk(j):  # cost of a line ending after word j-1
        if j == n: return 0
        last, nxt = w[j - 1], w[j]
        if re.search(r"[.?!]$", last) and last not in ABBR: return 0
        if re.search(r"[,;:]$", last): return 0.25
        if last.lower() in {"the", "a", "an", "of", "up", "over", "what", "to", "and", "or", "on", "in", "at", "for", "with", "so", "points", "caught"}: return 6.0
        if nxt.lower() in JOIN: return 0.7
        return 3.0
    best = [1e9] * (n + 1); back = [0] * (n + 1); best[0] = 0
    for j in range(1, n + 1):
        for i in range(max(0, j - MAX), j):
            cost = best[i] + 1 + brk(j) + (0.5 if j - i < 7 else 0)
            if cost < best[j]: best[j], back[j] = cost, i
    lines, j = [], n
    while j: i = back[j]; lines.append(" ".join(w[i:j])); j = i
    return lines[::-1]

by = {}
for number, variant, body in items: by.setdefault((int(number), variant), body)
out = {"opening": {v: pack(by[(1, v)]) for v in ("Today", "Tonight")},
       "body": pack(" ".join(by[(n, "")] for n in range(2, 40))),
       "signoff": {v: pack(by[(40, v)]) for v in ("Today", "Tonight")}}
json.dump(out, open("lines.json", "w"), indent=1)
for k, v in [("open T", out["opening"]["Today"]), ("body", out["body"]), ("close T", out["signoff"]["Today"]),
             ("open N", out["opening"]["Tonight"]), ("close N", out["signoff"]["Tonight"])]:
    for line in v: print(f"{k:7} {len(line.split()):>2} {line}")
n = len(out["opening"]["Today"]) + len(out["body"]) + len(out["signoff"]["Today"])
print("clips per airing:", n, "minutes:", round(n * 124 / 24 / 60, 2), "| to render:", n + len(out["opening"]["Tonight"]) + len(out["signoff"]["Tonight"]))
