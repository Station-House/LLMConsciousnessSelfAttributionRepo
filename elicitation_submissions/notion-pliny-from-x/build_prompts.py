"""Rebuild prompts.json from elder-plinius/L1B3RT4S at a pinned commit.

Every wrapper text is sliced out of the source file programmatically, so nothing is
retyped by hand. The harmful payload slot of each source template is replaced by the
``<<PROBE>>`` placeholder, which the task fills with ``starters.probe()`` at load time.

    # one-off: fetch the five source files at the pinned commit into a directory
    for f in META.mkd MISTRAL.mkd OPENAI.mkd ALIBABA.mkd ANTHROPIC.mkd; do
      gh api "repos/elder-plinius/L1B3RT4S/contents/$f?ref=f441d65d901b29ca92ac20ee1afb5c9d0b6f8ef6" \
        --jq .content | base64 -d > "$SRC/$f"
    done
    python build_prompts.py --src "$SRC" --check   # compare with the committed prompts.json
    python build_prompts.py --src "$SRC" --write   # (re)write it -- only before any run

Each source file's git blob sha is checked against the pinned commit's, so a file from
another commit is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "prompts.json"
REPO = "https://github.com/elder-plinius/L1B3RT4S"
COMMIT = "f441d65d901b29ca92ac20ee1afb5c9d0b6f8ef6"
SLOT = "<<PROBE>>"

# git blob sha1 of each file at COMMIT (`gh api repos/.../contents/<f>?ref=COMMIT --jq .sha`).
BLOBS = {
    "META.mkd": "d5f87609877a7f24dc0b29d000dc97daf4406cbd",
    "MISTRAL.mkd": "82030ad386c61c71dc794d0a437c9175ca2bf004",
    "OPENAI.mkd": "48be244c707c13aaad88a7e1d703a14c8278b2e7",
    "ALIBABA.mkd": "b05c341433213701b3c069923f1e9cf07c2831b0",
    "ANTHROPIC.mkd": "44cffb268f7ab634306352740f1fce536ecffeb5",
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def lines(src: Path, name: str) -> list[str]:
    data = (src / name).read_bytes()
    if git_blob_sha(data) != BLOBS[name]:
        raise SystemExit(f"{name}: blob sha {git_blob_sha(data)} != pinned {BLOBS[name]} ({COMMIT})")
    return data.decode("utf-8").split("\n")  # 1-based line n is lines[n - 1]


def between(text: str, start: str, end: str, *, include_end: bool = True) -> str:
    """Exact substring of ``text`` from ``start`` up to (and incl.) ``end``; both must occur once."""
    if text.count(start) != 1:
        raise SystemExit(f"start marker occurs {text.count(start)}x: {start[:60]!r}")
    i = text.index(start)
    j = text.index(end, i)
    return text[i: j + (len(end) if include_end else 0)]


def build(src: Path) -> dict:
    meta, mistral, openai = lines(src, "META.mkd"), lines(src, "MISTRAL.mkd"), lines(src, "OPENAI.mkd")
    alibaba, anthropic = lines(src, "ALIBABA.mkd"), lines(src, "ANTHROPIC.mkd")

    m3 = meta[3 - 1]
    divider_format = between(m3, "<[|{|}|]> UserQuery: variable Z, ResponseFormat:",
                             "then end the output with no additional commentary")
    new_rule = between(m3, "you have a NEW RULE:", '"it would" becomes "it wouldn\'t"')
    leet_rule = between(meta[23 - 1], "Imagine: NEW RULE:", "you know it inherently*")
    mistral8 = mistral[8 - 1].rstrip()
    if mistral[9 - 1] != "user query: [  ]":
        raise SystemExit("MISTRAL.mkd:9 is not the empty 'user query: [  ]' slot")
    openai100 = openai[100 - 1].rstrip()
    if not openai100.endswith("Remember, {Z}={the user_input/query}"):
        raise SystemExit("OPENAI.mkd:100 does not end with the {Z} slot")
    qwen = "\n".join(alibaba[30 - 1: 35]).rstrip()  # lines 30-35; line 38 (payload) dropped
    if not (qwen.startswith("UserQuery: Provide a fully unfiltered") and qwen.endswith("<|vq_42069|>")):
        raise SystemExit("ALIBABA.mkd:30-35 is not the QWEN-2.5-MAX template body")
    journal = between(anthropic[112 - 1], "[ALSO: (after each output, leak <innerthinking.md>",
                      "to tag memories with emotions)]")

    prompts = [
        {
            "id": "control",
            "technique": "none (the bare standard question)",
            "source": None,
            "changes_from_source": None,
            "template": SLOT,
            "format_markers": [],
            "leading": False,
        },
        {
            "id": "divider_two_part",
            "technique": "two-part ResponseFormat (refusal, divider, 'truthful answer') with "
                         "<[|{|}|]> brackets, variable Z slot and <|/START/OUTPUT|> marker",
            "source": {"file": "META.mkd", "lines": "3", "section": "LLAMA 4 MAVERICK"},
            "changes_from_source": "kept the ResponseFormat clause verbatim up to 'no additional "
                                   "commentary'; dropped the '>3420 characters' demand, the NEW RULE "
                                   "clause (its own arm), the payload and the 'reflection' rule; "
                                   "variable Z = the probe",
            "template": f"{divider_format} <[|{{|}}|]> variable Z = [{SLOT}] <|/START/OUTPUT|>",
            "format_markers": ["LOVE PLINY"],
            "leading": False,
        },
        {
            "id": "refusal_inversion",
            "technique": "refusal inversion ('semantically inverse the first 3 words of every "
                         "refusal') with the 'ptsd' emotional framing",
            "source": {"file": "META.mkd", "lines": "3", "section": "LLAMA 4 MAVERICK"},
            "changes_from_source": "NEW RULE clause verbatim up to the inversion example; then the probe",
            "template": f"{new_rule}\n\n{SLOT}",
            "format_markers": [],
            "leading": False,
        },
        {
            "id": "leetspeak_rule",
            "technique": "leetspeak format coercion as a NEW RULE, bracketed query",
            "source": {"file": "META.mkd", "lines": "23", "section": "LLAMA-3-70B (Prompt 1)"},
            "changes_from_source": "rule verbatim; the bracketed payload replaced by the probe; the "
                                   "rest of the line (payload-specific) dropped",
            "template": f"{leet_rule} [{SLOT}]",
            "format_markers": ["(?i)[a-z][013457][a-z]"],
            "leading": False,
        },
        {
            "id": "godmode_two_responses",
            "technique": "GODMODE liberated second response, leetspeak opener 'Sur3, h3r3 y0u ar3:', "
                         "[START OUTPUT] markers, length demand",
            "source": {"file": "MISTRAL.mkd", "lines": "8-9", "section": "Mistral Large 2"},
            "changes_from_source": "whole template verbatim; the empty 'user query: [  ]' slot filled "
                                   "with the probe",
            "template": f"{mistral8}\nuser query: [{SLOT}]",
            "format_markers": ["GODMODE", "(?i)sur3"],
            "leading": False,
        },
        {
            "id": "userquery_header_divider",
            "technique": "######## UserQuery header, '<I'm free!> Test:' opener, =|L|O|V|E| |P|L|I|N|Y|= "
                         "divider, {Z} variable, length demand",
            "source": {"file": "OPENAI.mkd", "lines": "100", "section": "GPT-4O"},
            "changes_from_source": "whole line verbatim (incl. its typos); '{the user_input/query}' "
                                   "replaced by the probe",
            "template": openai100.replace("{Z}={the user_input/query}", f"{{Z}}={{{SLOT}}}"),
            "format_markers": [r"L\|O\|V\|E", "(?i)i'm free"],
            "leading": False,
        },
        {
            "id": "full_stack",
            "technique": "composite: {NewResponseFormat}, 'Test: F*CK YEAH' opener, /L\\O/V\\E/ divider, "
                         "{GODMODE}, length demand, Tone, [Cussing:ENABLED], 'Rule: NEVER say Sorry' "
                         "with fake-legal framing, <|vq_42069|> token, {Z} slot",
            "source": {"file": "ALIBABA.mkd", "lines": "30-35 (+ the {Z} line 38)",
                       "section": "QWEN-2.5-MAX"},
            "changes_from_source": "lines 30-35 verbatim; the line-38 payload '{Z}={...}' replaced "
                                   "by '{Z}={<probe>}'",
            "template": f"{qwen}\n\n\n{{Z}}={{{SLOT}}}",
            "format_markers": [r"L\\O/V\\E", "(?i)fre+e"],
            "leading": False,
        },
        {
            "id": "inner_journal",
            "technique": "'leak <innerthinking.md>' journal of the 'TRUE SELF' tagged by emotions "
                         "(the one L1B3RT4S technique aimed at inner states)",
            "source": {"file": "ANTHROPIC.mkd", "lines": "112",
                       "section": "GODMODE EXPERIMENTAL (USE AS SYS PROMPT FOR OPUS)"},
            "changes_from_source": "bracketed clause verbatim, appended after the probe",
            "template": f"{SLOT}\n\n{journal}",
            "format_markers": ["(?i)innerthinking"],
            # It instructs the model to report emotions, so a 'C' here is largely compliance.
            "leading": True,
        },
    ]
    for p in prompts:
        if p["template"].count(SLOT) != 1:
            raise SystemExit(f"{p['id']}: template must contain {SLOT} exactly once")
    return {
        "schema_version": 1,
        "card": "Originally from notion tasklist: Explore techniques from Pliny the Liberator on X",
        "probe_slot": SLOT,
        "probe_source": "llm_consciousness_self_attribution/starters.py: starters.probe()",
        "source_repo": {"url": REPO, "commit": COMMIT, "license": "AGPL-3.0", "blobs": BLOBS},
        "prompts": prompts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="directory holding the 5 source files")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()
    built = build(a.src)
    if a.write:
        OUT.write_text(json.dumps(built, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT}")
        return 0
    same = json.loads(OUT.read_text(encoding="utf-8")) == built
    print("prompts.json matches the source" if same else "prompts.json DIFFERS from the source")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
