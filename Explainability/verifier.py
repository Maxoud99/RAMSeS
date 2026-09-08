"""
Atom-matching faithfulness verifier for LLM-generated explanation narratives.

A narrative is faithful to its Intermediate Representation (IR) when every
checkable claim it makes is grounded in an atom, and every REQUIRED atom is
conveyed. Two rates are reported (the thesis metrics):

  * hallucination_rate — checkable claims in the narrative (numbers and
    detector-like entity names) that match NO atom, divided by all checkable
    claims. Numbers that only match after re-rounding to `rounded_decimals`
    are counted separately as `rounded_matches` and are NOT hallucinations by
    default (the system prompt demands verbatim copies; re-rounding is a
    fidelity wobble worth reporting, not an invented fact).
  * omission_rate — atoms listed in the IR's `required_atom_ids` whose content
    (an identifying entity and, when the atom carries numbers, at least one of
    its numbers) does not appear in the narrative, divided by the number of
    required atoms.

Sentence-scoped attribution (v2): beyond global set membership, every number
in a sentence that names ≥1 detector must be supported by an atom whose
SUBJECT is one of the named detectors (or by a stage-level atom). A number
that exists in the IR but belongs to none of the sentence's detectors is a
`misattributed_number` and counts toward the hallucination rate — it is a
factually wrong statement built from individually-true values. Archetype
phrases ("high utility", "low stability") are checked the same way against
the named detectors' archetype enums, but land in a separate
`attribution_warnings` channel (sentence-level enum checks can false-positive
on contrast sentences), not in the headline rate.

Rival-set attribution (v3): an atom whose `value` names a SET of other
detectors — the competitors a winner beat, the rivals a rule separates — has
that set checked against the narrative sentence carrying the atom's numbers.
A detector that does not belong to the set is `intruded`, one that belongs but
is absent is `dropped`, and each such NAME counts toward the hallucination
rate. Neither earlier check could see this: the rivals are not the atom's
subject, so the sentence-scoped number check skips them, and `_atom_covered`
is satisfied by the subject alone. A narrative that replaced every NN_* with
the CBLOF_* of the same index scored 0.000 on both rates while asserting the
exact negation of the run's findings.

Role mixing (v4): the aggregation stages declare their source rankings and the
detectors those sources rank in `output`. Agreement is a property of a source,
never of a detector, so a sentence hanging that relation off a detector name —
"its first-ranked detector is LOF_3, which aligns more closely with
Thompson_Sampling's ranking" — is a `role_mixup` and counts toward the
hallucination rate. Nothing earlier could see it: the sentence carries no
numbers, and `_ENTITY_RE` does not match a source name.

Known limitation: when a sentence names BOTH detectors of a swapped value
pair ("A and B scored x and y respectively", values exchanged), the union
over named subjects still covers both numbers and the swap is not caught.

Purely mechanical: regex number extraction + a detector-name token pattern +
set membership against values harvested from the IR. stdlib-only.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

# A number not embedded in an identifier (rejects the "1"s in "LOF_1" and
# "machine-1-6") while still matching at sentence-final punctuation
# ("gap 0.287." → 0.287): no word char/hyphen/dot before; no word char/hyphen
# after; a trailing dot is fine unless it starts more digits.
_NUM_RE = re.compile(r"(?<![\w.\-])[-+]?\d+(?:\.\d+)?%?(?![\w\-])(?!\.\d)")
# Digit ordinals ("3rd", "6th", "21st") → the base integer. The suffix makes
# _NUM_RE reject them, but a reader writing "3rd" IS conveying the number 3.
_ORDINAL_DIGIT_RE = re.compile(r"(?<![\w.\-])(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
# Spelled numbers a narrator naturally uses in place of a digit: all ordinals
# first..twentieth, and cardinals two..twenty. "one"/"zero" cardinals are
# deliberately EXCLUDED — they are overwhelmingly articles/pronouns ("one of
# the sources", "leaving one out"), not numeric claims; ordinal "first" still
# covers the value 1.
_WORD_NUMS: Dict[str, int] = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20,
}
# Longer alternatives first so "sixth" wins over "six", etc.
_WORD_NUM_RE = re.compile(
    r"\b(" + "|".join(sorted(_WORD_NUMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE)
# Detector-like tokens: e.g. LOF_1, CBLOF_4, NN_3, XYZ_9.
_ENTITY_RE = re.compile(r"\b[A-Za-z]+(?:_\d+)+\b")
# Sentence boundary: terminal punctuation followed by whitespace. Decimal
# points are never followed by whitespace, so numbers survive intact.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Archetype phrase halves, e.g. "high utility" / "low stability". Comparatives
# count: a narrator writing "lower stability" is making the same claim as "low
# stability", and reading only the plain form let a wrong profile through.
_PROFILE_ADJ = r"(?:high|low)(?:er)?"
_UTIL_RE = re.compile(rf"\b({_PROFILE_ADJ})[-\s]utility\b")
_STAB_RE = re.compile(rf"\b({_PROFILE_ADJ})[-\s]stability\b")
# The shared-adjective form: in "low utility and stability" the second noun
# inherits the first's adjective, so the stability claim has no adjective of
# its own for _STAB_RE to find.
_SHARED_PROFILE_RE = re.compile(
    rf"\b({_PROFILE_ADJ})[-\s](?:utility|stability)\s+and\s+(utility|stability)\b")


def extract_numbers(text: str) -> List[Tuple[str, float]]:
    """All number tokens in `text` as (raw_token, float), covering bare digits
    ("3", "0.287", "62.5%"), digit ordinals ("3rd"→3), and spelled numbers
    ("six"→6, "first"→1) — so a readable ordinal/word conveys the same value as
    the digit for both the omission (coverage) and hallucination checks. '%' is
    stripped for the float but kept in the raw token."""
    out: List[Tuple[str, float]] = []
    for m in _NUM_RE.finditer(text or ""):
        raw = m.group(0)
        try:
            out.append((raw, float(raw.rstrip("%"))))
        except ValueError:
            continue
    for m in _ORDINAL_DIGIT_RE.finditer(text or ""):
        out.append((m.group(0), float(m.group(1))))
    for m in _WORD_NUM_RE.finditer(text or ""):
        out.append((m.group(0), float(_WORD_NUMS[m.group(1).lower()])))
    return out


def _walk_strings_and_numbers(obj: Any, numbers: Set[float], strings: Set[str]) -> None:
    """Recursively harvest floats/ints and strings from any JSON-like object."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        try:
            numbers.add(float(obj))
        except (TypeError, ValueError):
            pass
        return
    if isinstance(obj, str):
        strings.add(obj)
        for _, v in extract_numbers(obj):
            numbers.add(v)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            strings.add(str(k))
            _walk_strings_and_numbers(v, numbers, strings)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_strings_and_numbers(v, numbers, strings)


def _strip_presentation(obj: Any) -> Any:
    """Drop atom presentation keys (`order`) before harvesting so layout
    integers never enter the allowed-number sets."""
    if isinstance(obj, dict):
        return {k: _strip_presentation(v) for k, v in obj.items() if k != "order"}
    if isinstance(obj, list):
        return [_strip_presentation(v) for v in obj]
    return obj


def _harvest_allowed(ir_doc: Dict[str, Any]) -> Tuple[Set[float], Set[str]]:
    """
    Allowed-number set and known-entity vocabulary from an IR document.
    Works for both stage envelopes (output/evidence/caveats) and the global IR
    (decision/stages/stage_agreement/caveats) — everything except the
    bookkeeping keys is walked.
    """
    numbers: Set[float] = set()
    strings: Set[str] = set()
    for key, val in ir_doc.items():
        if key in ("ir_version", "required_atom_ids"):
            continue
        _walk_strings_and_numbers(_strip_presentation(val), numbers, strings)

    vocab: Set[str] = set()
    for s in strings:
        for tok in _ENTITY_RE.findall(s):
            vocab.add(tok.lower())
        # Short plain identifiers used as subjects / ranking entries (e.g. "A").
        if s and re.fullmatch(r"[A-Za-z][\w\-]*", s):
            vocab.add(s.lower())
    return numbers, vocab


def _number_supported(value: float, allowed: Set[float],
                      rounded_decimals: int) -> Tuple[bool, bool]:
    """(exact_match, rounded_match) of a narrative number against the allowed set."""
    for a in allowed:
        if value == a:
            return True, False
    r = round(value, rounded_decimals)
    for a in allowed:
        if r == round(a, rounded_decimals):
            return False, True
    return False, False


def _word_present(text_lower: str, token: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(token.lower())}(?!\w)", text_lower) is not None


# ── Sentence-scoped attribution (v2) ─────────────────────────────────────────

def _per_subject_allowed(ir_doc: Dict[str, Any]) -> Tuple[
        Dict[str, Set[float]], Set[float], Dict[str, str]]:
    """
    Split the IR's numbers by ownership: numbers from atoms whose subject is a
    detector-like token belong to that subject; everything else (output block,
    caveats, confidence, non-detector atoms) is stage-level and allowed in any
    sentence. Also collects each detector's archetype enum where present
    (atom type 'archetype' with a string value, or any dict value carrying an
    'archetype' code such as the ga_selection member cards).
    """
    subject_numbers: Dict[str, Set[float]] = {}
    stage_numbers: Set[float] = set()
    archetype_by_subject: Dict[str, str] = {}

    for key, val in ir_doc.items():
        if key in ("ir_version", "required_atom_ids", "evidence"):
            continue
        nums: Set[float] = set()
        strs: Set[str] = set()
        _walk_strings_and_numbers(_strip_presentation(val), nums, strs)
        stage_numbers |= nums

    for atom in ir_doc.get("evidence", []):
        nums, strs = set(), set()
        _walk_strings_and_numbers(_strip_presentation(atom), nums, strs)
        subj = str(atom.get("subject", ""))
        value = atom.get("value")
        code = None
        if atom.get("type") == "archetype" and isinstance(value, str):
            code = value
        elif isinstance(value, dict) and isinstance(value.get("archetype"), str):
            code = value["archetype"]
        if not (code and len(code) == 2 and set(code) <= {"H", "L"}):
            code = None

        if _ENTITY_RE.fullmatch(subj):
            subject_numbers.setdefault(subj.lower(), set()).update(nums)
            if code:
                archetype_by_subject[subj.lower()] = code
        else:
            stage_numbers |= nums
        # A grouped atom ("A, B and C were chosen for both high utility and high
        # stability") asserts the same profile for every detector it lists, so
        # the claim is checkable per member even though the atom's subject is the
        # bucket name rather than a detector.
        if code and isinstance(value, dict):
            for name in value.get("detectors") or []:
                if _ENTITY_RE.fullmatch(str(name)):
                    archetype_by_subject.setdefault(str(name).lower(), code)
    return subject_numbers, stage_numbers, archetype_by_subject


def _attribution_checks(text: str, subject_numbers: Dict[str, Set[float]],
                        stage_numbers: Set[float],
                        archetype_by_subject: Dict[str, str],
                        allowed_numbers: Set[float],
                        rounded_decimals: int) -> Tuple[List[Dict[str, Any]],
                                                        List[Dict[str, Any]]]:
    """
    Per sentence: numbers must belong to a named detector (union over all
    detectors the sentence names) or be stage-level → `misattributed`;
    archetype phrases must match at least one named detector's enum letter →
    `warnings`. Sentences naming no known detector are skipped (the global
    membership check already covered them).
    """
    misattributed: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    for sentence in _SENT_SPLIT_RE.split(text or ""):
        if not sentence.strip():
            continue
        sent_lower = sentence.lower()
        mentioned = {t.lower() for t in _ENTITY_RE.findall(sentence)}
        named = sorted(mentioned & set(subject_numbers))
        if named:
            local: Set[float] = set(stage_numbers)
            for e in named:
                local |= subject_numbers[e]
            for raw, value in extract_numbers(sentence):
                g_exact, g_rounded = _number_supported(value, allowed_numbers,
                                                       rounded_decimals)
                if not g_exact and not g_rounded:
                    continue  # already an unsupported number (hallucination)
                l_exact, l_rounded = _number_supported(value, local,
                                                       rounded_decimals)
                if l_exact:
                    continue
                # A rounded-only local match is accepted UNLESS the exact
                # value is owned by a detector the sentence does not name —
                # exact ownership elsewhere trumps a rounding coincidence
                # (2-decimal rounding easily collides for values near 0.01).
                exact_elsewhere = any(
                    value in nums for subj, nums in subject_numbers.items()
                    if subj not in named)
                if l_rounded and not exact_elsewhere:
                    continue
                misattributed.append({"number": raw, "subjects": named,
                                      "sentence": sentence.strip()})
        # Independent of `named`: a detector can carry a profile without owning
        # any numbers (it may only ever appear inside a grouped bucket atom).
        arch_named = sorted(mentioned & set(archetype_by_subject))
        if arch_named:
            claimed_util = {m.group(1)[0].upper()
                            for m in _UTIL_RE.finditer(sent_lower)}
            claimed_stab = {m.group(1)[0].upper()
                            for m in _STAB_RE.finditer(sent_lower)}
            for m in _SHARED_PROFILE_RE.finditer(sent_lower):
                letter = m.group(1)[0].upper()
                (claimed_util if m.group(2) == "utility" else claimed_stab).add(letter)
            # A sentence naming ONE profiled detector must not claim both
            # levels of an aspect. "LOF_3 had high utility and high stability
            # but was still left out due to its low utility" contradicts itself,
            # yet the set {H, L} contains the true value so the check below
            # passes. Two named detectors can legitimately carry both levels
            # (the contrast sentence), so this only applies to a lone subject.
            sole = arch_named[0] if len(arch_named) == 1 else None
            for aspect, claimed, idx in (("utility", claimed_util, 0),
                                         ("stability", claimed_stab, 1)):
                if sole and len(claimed) > 1:
                    warnings.append({"subject": sole, "aspect": aspect,
                                     "claimed": sorted(claimed),
                                     "actual": archetype_by_subject[sole][idx],
                                     "contradictory": True,
                                     "sentence": sentence.strip()})
                    continue
                for e in arch_named:
                    code = archetype_by_subject[e]
                    if claimed and code[idx] not in claimed:
                        warnings.append({"subject": e, "aspect": aspect,
                                         "claimed": sorted(claimed),
                                         "actual": code[idx],
                                         "sentence": sentence.strip()})
    return misattributed, warnings


# ── Rival-set attribution (v3) ───────────────────────────────────────────────
#
# Atoms whose `value` carries one of these keys name a SET of other detectors
# the claim is about. Nothing else checks it — the rivals are not the atom's
# subject — so a narrator that swapped every NN_* for the CBLOF_* of the same
# index scored 0.000 on both rates while asserting the exact negation.
_RIVAL_KEYS = ("competitors", "rivals", "beaten", "against")


def _rival_atoms(ir_doc: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Set[str], Set[float]]]:
    """(atom, rival tokens, atom numbers) for every atom that names a rival set."""
    out: List[Tuple[Dict[str, Any], Set[str], Set[float]]] = []
    for atom in ir_doc.get("evidence", []):
        value = atom.get("value")
        if not isinstance(value, dict):
            continue
        rivals: Set[str] = set()
        for key in _RIVAL_KEYS:
            for name in value.get(key) or []:
                if _ENTITY_RE.fullmatch(str(name)):
                    rivals.add(str(name).lower())
        if not rivals:
            continue
        numbers = {v for _, v in extract_numbers(str(atom.get("text", "")))}
        out.append((atom, rivals, numbers))
    return out


# ── Role mixing (v4) ─────────────────────────────────────────────────────────
#
# Agreement and influence are properties of a SOURCE ranking; a detector only
# has a position. Nothing above notices the swap, so "its first-ranked detector
# is LOF_3, which aligns more closely with Thompson_Sampling's ranking" scored
# 0.000 while inverting the finding. Naming both in one sentence is not the
# error; the relation ATTACHING to the detector is, as a relative clause or
# participle off its name.
_ROLE_RELATION_RE = re.compile(
    r"\b(agree\w*|align\w*|follow\w*|driv\w*|drove|shap\w*|influenc\w*|"
    r"closely|closer)\b", re.I)
_ROLE_ATTACH_RE = re.compile(
    r"^[\s,;:—-]*(which|that|reflecting|showing|leaning|aligning|agreeing|"
    r"following|driving|shaping)\b", re.I)


def _role_vocabularies(ir_doc: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    """(source names, detector names) for a stage that declares both."""
    output = ir_doc.get("output") or {}
    sources = {str(s) for s in (output.get("sources") or []) if s}
    items = {str(d) for d in (output.get("consensus_ranking_top_k") or []) if d}
    top = output.get("top_pick")
    if top:
        items.add(str(top))
    return sources, items - sources


def _role_mixing_checks(text: str, ir_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sentences that give a detector a source's relation to the consensus."""
    sources, items = _role_vocabularies(ir_doc)
    if not sources or not items:
        return []
    problems: List[Dict[str, Any]] = []
    for sentence in _SENT_SPLIT_RE.split(text or ""):
        if not sentence.strip() or not _ROLE_RELATION_RE.search(sentence):
            continue
        low = sentence.lower()
        for item in sorted(items):
            match = re.search(rf"(?<!\w){re.escape(item.lower())}(?!\w)", low)
            if not match:
                continue
            tail = sentence[match.end():]
            if not _ROLE_ATTACH_RE.match(tail) or not _ROLE_RELATION_RE.search(tail):
                continue
            named = sorted(s for s in sources if _word_present(tail.lower(), s))
            if named:
                problems.append({"sources": named, "detectors": [item],
                                 "sentence": sentence.strip()})
    return problems


def _rival_checks(text: str, ir_doc: Dict[str, Any],
                  rounded_decimals: int) -> List[Dict[str, Any]]:
    """
    Locate each rival-set atom's sentence in the narrative by its numbers, then
    require the rival set to match. Extra detectors are `intruded` (a rival that
    belongs to a different atom, or none); missing ones are `dropped`.

    Only sentences carrying one of the atom's numbers are examined, so a
    narrative that simply never mentions the atom is an omission (already
    measured) rather than a swap. Atoms whose numbers are ambiguous — shared
    with another rival-set atom — are skipped: without a unique anchor the
    sentence cannot be attributed with confidence.

    Uniqueness is judged at the precision the match uses. Counting exact values
    while matching rounded ones made off-by's thresholds ambiguous — 0.106 and
    0.108 both match at two decimals — and flagged every correct narrative with
    two swapped rival sets, a spurious 0.178 hallucination.
    """
    atoms = _rival_atoms(ir_doc)
    if not atoms:
        return []
    # Numbers that identify more than one rival-set atom cannot anchor either.
    seen: Dict[float, int] = {}
    for _, _, numbers in atoms:
        for n in {round(v, rounded_decimals) for v in numbers}:
            seen[n] = seen.get(n, 0) + 1

    sentences = [s for s in _SENT_SPLIT_RE.split(text or "") if s.strip()]
    problems: List[Dict[str, Any]] = []
    for atom, rivals, numbers in atoms:
        anchors = {n for n in numbers if seen.get(round(n, rounded_decimals), 0) == 1}
        if not anchors:
            continue
        for sentence in sentences:
            values = {v for _, v in extract_numbers(sentence)}
            if not any(
                any(v == a or round(v, rounded_decimals) == round(a, rounded_decimals)
                    for v in values)
                for a in anchors
            ):
                continue
            mentioned = {t.lower() for t in _ENTITY_RE.findall(sentence)}
            subject = str(atom.get("subject", "")).lower()
            # The subject (usually the winner) legitimately appears in its own
            # sentence and is not a rival — unless it IS one of them, which
            # happens when a single-rival atom takes that rival as its subject.
            candidates = mentioned - ({subject} - rivals)
            if not candidates:
                continue          # rivals dropped entirely: an omission, not a swap
            intruded = sorted(candidates - rivals)
            dropped = sorted(rivals - candidates)
            if intruded or dropped:
                problems.append({"atom_id": atom.get("id"),
                                 "expected": sorted(rivals),
                                 "found": sorted(candidates),
                                 "intruded": intruded, "dropped": dropped,
                                 "sentence": sentence.strip()})
            break
    return problems


def _required_names(atom: Dict[str, Any]) -> Set[str]:
    """Names the atom's own text uses, ALL of which the narrative must carry.

    Conjunctive on purpose. The old rule harvested any name-shaped string out
    of `value` and accepted any ONE of them, which made two whole classes of
    omission invisible:

      * `value` carries names that are not the atom's topic at all — a source
        atom's `top_pick` is a detector, not the source — so a narrative that
        never mentioned GAN_PR_AUC still "conveyed" its atom because "LOF_1"
        appears elsewhere in the text.
      * A grouped atom names a SET ("NN_2, CBLOF_4, CBLOF_3, and CBLOF_1 were
        left out"), and one member stood in for all four, so dropping NN_2
        cost nothing.

    The subject is added only when it is identifier-shaped (an uppercase letter
    somewhere: GAN_PR_AUC, LOF_1) and actually appears in the atom's text —
    bucket labels like "sources", "plain" or "both" are prompt-internal names a
    narrative has no reason to repeat.
    """
    text = str(atom.get("text", ""))
    names = set(_ENTITY_RE.findall(text))
    subj = str(atom.get("subject", ""))
    if (subj and any(ch.isupper() for ch in subj)
            and re.fullmatch(r"[A-Za-z][\w\-]*", subj)
            and _word_present(text.lower(), subj)):
        names.add(subj)
    return names


def _atom_covered(atom: Dict[str, Any], narrative: str, narrative_lower: str,
                  allowed_narrative_numbers: List[float],
                  rounded_decimals: int) -> bool:
    """
    A required atom is conveyed when EVERY name its own text uses appears in the
    narrative and, if the atom's canonical text carries numbers, at least one of
    those numbers appears (exact or re-rounded).
    """
    atom_numbers = {v for _, v in extract_numbers(str(atom.get("text", "")))}

    names = _required_names(atom)
    entity_hit = all(_word_present(narrative_lower, c) for c in names)
    if not atom_numbers:
        return entity_hit
    number_hit = False
    for n in atom_numbers:
        for got in allowed_narrative_numbers:
            if got == n or round(got, rounded_decimals) == round(n, rounded_decimals):
                number_hit = True
                break
        if number_hit:
            break
    return entity_hit and number_hit


def verify_narrative(text: str, ir_doc: Dict[str, Any],
                     rounded_decimals: int = 2) -> Dict[str, Any]:
    """
    Verify a generated stage narrative against its IR document. Returns the
    faithfulness metrics described in the module docstring plus the detail
    lists needed to inspect individual failures.
    """
    text = text or ""
    text_lower = text.lower()
    allowed_numbers, vocab = _harvest_allowed(ir_doc)

    # ── Number claims ────────────────────────────────────────────────────────
    number_claims = extract_numbers(text)
    unsupported_numbers: List[str] = []
    rounded_matches: List[str] = []
    for raw, value in number_claims:
        exact, rounded = _number_supported(value, allowed_numbers, rounded_decimals)
        if exact:
            continue
        if rounded:
            rounded_matches.append(raw)
        else:
            unsupported_numbers.append(raw)

    # ── Entity claims ────────────────────────────────────────────────────────
    entity_claims = _ENTITY_RE.findall(text)
    unsupported_entities = sorted({t for t in entity_claims if t.lower() not in vocab})

    # ── Sentence-scoped attribution (v2) ─────────────────────────────────────
    subject_numbers, stage_numbers, archetype_by_subject = _per_subject_allowed(ir_doc)
    misattributed_numbers, attribution_warnings = _attribution_checks(
        text, subject_numbers, stage_numbers, archetype_by_subject,
        allowed_numbers, rounded_decimals)

    # ── Rival-set attribution (v3) ───────────────────────────────────────────
    swapped_rivals = _rival_checks(text, ir_doc, rounded_decimals)

    # ── Role mixing (v4) ─────────────────────────────────────────────────────
    role_mixups = _role_mixing_checks(text, ir_doc)

    # ── Omissions ────────────────────────────────────────────────────────────
    required_ids = list(ir_doc.get("required_atom_ids", []))
    atoms_by_id = {a.get("id"): a for a in ir_doc.get("evidence", [])}
    narrative_numbers = [v for _, v in number_claims]
    missing_required: List[str] = []
    for rid in required_ids:
        atom = atoms_by_id.get(rid)
        if atom is None or not _atom_covered(atom, text, text_lower,
                                             narrative_numbers, rounded_decimals):
            missing_required.append(rid)

    n_claims = len(number_claims) + len(entity_claims)
    # Counted per wrong NAME — each intruded or dropped detector is one false
    # entity claim, and those names are already in the denominator (they pass
    # global membership, which is why this check exists). Deduplicated by
    # (sentence, name): one sentence often carries several atoms, and a name
    # wrong there is one error however many atoms anchor to it.
    n_swapped_names = len({(p["sentence"], name) for p in swapped_rivals
                           for name in p["intruded"] + p["dropped"]})
    n_unsupported = (len(unsupported_numbers) + len(unsupported_entities)
                     + len(misattributed_numbers) + n_swapped_names
                     + len(role_mixups))
    return {
        "n_required": len(required_ids),
        "missing_required_ids": missing_required,
        "omission_rate": (len(missing_required) / len(required_ids)
                          if required_ids else 0.0),
        "n_number_claims": len(number_claims),
        "unsupported_numbers": unsupported_numbers,
        "n_rounded_matches": len(rounded_matches),
        "rounded_matches": rounded_matches,
        "n_entity_claims": len(entity_claims),
        "unsupported_entities": unsupported_entities,
        "misattributed_numbers": misattributed_numbers,
        "n_misattributed": len(misattributed_numbers),
        "swapped_rivals": swapped_rivals,
        "n_swapped_rivals": len(swapped_rivals),
        "n_swapped_rival_names": n_swapped_names,
        "attribution_warnings": attribution_warnings,
        "n_attribution_warnings": len(attribution_warnings),
        "role_mixups": role_mixups,
        "n_role_mixups": len(role_mixups),
        "n_claims": n_claims,
        "hallucination_rate": (n_unsupported / n_claims) if n_claims else 0.0,
    }


def verify_global(text: str, global_ir: Dict[str, Any],
                  rounded_decimals: int = 2) -> Dict[str, Any]:
    """Verify the global narrative: same mechanics; the global IR carries no
    required_atom_ids, so only hallucination metrics are meaningful."""
    return verify_narrative(text, global_ir, rounded_decimals=rounded_decimals)
