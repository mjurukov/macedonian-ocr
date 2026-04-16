#!/usr/bin/env python3
"""
MK-OCR Post-Processing Module v2
==================================
Spell-check, dehyphenation, fragment merging, and frequency-aware correction.

Key improvements:
- Frequency-aware: if confusion candidate is 50x+ more common, override even valid words
- Fragment merging: "соп рутата" -> "сопругата" (detection splitting fix)
- All dash types handled for dehyphenation
- More confusion pairs based on actual V5 error analysis

Usage:
    python3 mk_postprocess.py --build-dict ~/macedonian-llm/books_clean/books_txt
    python3 mk_postprocess.py --correct "Тој ммаше друти планови за мако"
    python3 mk_postprocess.py --test
"""

import json
import os
import re
import sys
import glob
from collections import Counter

DASH_RE = re.compile(r'[\-\u2010\u2011\u2012\u2013\u2014\u2015\u00AD]')
CYRILLIC_WORD_RE = re.compile(r'[а-яА-ЯѓЃќЌљЉњЊџЏѕЅјЈѐЀѝЍ]+')


class MKPostProcessor:

    CONFUSION_PAIRS = [
        ("м", "и"),
        ("и", "м"),
        ("т", "г"),
        ("г", "т"),
        ("н", "и"),
        ("п", "и"),
        ("о", "а"),
        ("а", "о"),
        ("л", "д"),
        ("д", "л"),
        ("у", "п"),
        ("п", "у"),
        ("к", "у"),
        ("у", "к"),
    ]

    # Only override a dictionary-confirmed word if a confusion candidate is
    # overwhelmingly more common AND the current word is itself very rare.
    # 30 was too aggressive — it replaced correct words with high-freq wrong ones.
    FREQ_OVERRIDE_RATIO = 200
    FREQ_OVERRIDE_MAX_ORIG = 20  # only override if original word has freq < this

    def __init__(self, dict_path=None):
        self.word_freq = {}
        self.word_set = set()
        if dict_path and os.path.exists(dict_path):
            self.load_dictionary(dict_path)

    def load_dictionary(self, path):
        with open(path, encoding="utf-8") as f:
            self.word_freq = json.load(f)
        self.word_set = set(self.word_freq.keys())
        print(f"  Loaded dictionary: {len(self.word_set):,} words")

    def save_dictionary(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word_freq, f, ensure_ascii=False, indent=0)
        print(f"  Saved dictionary: {len(self.word_freq):,} words -> {path}")

    @staticmethod
    def build_dictionary(books_dir, min_freq=2, max_words=500000):
        books_dir = os.path.expanduser(books_dir)
        txt_files = glob.glob(os.path.join(books_dir, "**", "*.txt"), recursive=True)
        print(f"  Building dictionary from {len(txt_files)} files...")
        counter = Counter()
        total = 0
        for fp in txt_files:
            try:
                with open(fp, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        words = CYRILLIC_WORD_RE.findall(line)
                        counter.update(w.lower() for w in words)
                        total += len(words)
            except Exception:
                continue
        print(f"  Total words: {total:,}, unique: {len(counter):,}")
        filtered = {w: c for w, c in counter.most_common(max_words) if c >= min_freq}
        print(f"  After filtering (freq >= {min_freq}): {len(filtered):,}")
        return filtered

    # ── Public API ──

    def process(self, lines):
        if not lines:
            return []
        merged = self._merge_hyphenated_lines(lines)
        return [self._correct_line(line) for line in merged]

    def process_text(self, text):
        text = self._dehyphenate(text)
        text = self._merge_fragments(text)
        if self.word_set:
            text = self._correct_words(text)
        return text

    # Legacy
    def correct_text(self, text):
        return self.process_text(text)

    def correct_lines(self, lines):
        return self.process(lines)

    # ── Dehyphenation ──

    def _dehyphenate(self, text):
        def repl(m):
            if m.group(2) and m.group(2)[0].islower():
                return m.group(1) + m.group(2)
            return m.group(0)
        return re.sub(r'(\w+)' + DASH_RE.pattern + r'\s+(\w+)', repl, text)

    def _merge_hyphenated_lines(self, lines):
        result = []
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if line and DASH_RE.search(line[-1:]) and i + 1 < len(lines):
                next_line = lines[i + 1].lstrip()
                m = re.search(r'(\S+)' + DASH_RE.pattern + r'$', line)
                if m and next_line:
                    nm = re.match(r'(\S+)(.*)', next_line)
                    if nm and nm.group(1)[0:1].islower():
                        line = line[:m.start()] + m.group(1) + nm.group(1)
                        rest = nm.group(2).lstrip()
                        if rest:
                            lines[i + 1] = rest
                        else:
                            i += 1
            result.append(line)
            i += 1
        return result

    # ── Fragment merging ──

    def _merge_fragments(self, text):
        if not self.word_set:
            return text
        words = text.split()
        merged = []
        i = 0
        while i < len(words):
            if i + 1 < len(words):
                w1 = words[i]
                w2 = words[i + 1]
                combined_lower = (w1 + w2).lower()
                w1_lower = w1.lower()
                w2_lower = w2.lower()

                w1_in = w1_lower in self.word_set
                w2_in = w2_lower in self.word_set
                comb_in = combined_lower in self.word_set

                should_merge = False

                # Only merge if BOTH parts are unknown words — if both are valid
                # dictionary words, they're almost certainly meant to be separate.
                if comb_in and not w1_in and not w2_in:
                    should_merge = True

                # Frequency-based merge: combined much more common than either part,
                # but only when at least one part is not a valid standalone word.
                if comb_in and (not w1_in or not w2_in):
                    cf = self.word_freq.get(combined_lower, 0)
                    wf = self.word_freq.get(w1_lower, 0) + self.word_freq.get(w2_lower, 0)
                    if cf > wf * 5:
                        should_merge = True

                if should_merge:
                    merged.append(_restore_case(combined_lower, w1 + w2))
                    i += 2
                    continue

                # Try combined + confusion correction
                if len(w1_lower) <= 4 or len(w2_lower) <= 4 or not w1_in or not w2_in:
                    for cand in self._confusion_candidates(combined_lower):
                        if cand in self.word_set and self.word_freq.get(cand, 0) > 50:
                            merged.append(_restore_case(cand, w1 + w2))
                            i += 2
                            should_merge = True
                            break
                    if should_merge:
                        continue

            merged.append(words[i])
            i += 1
        return " ".join(merged)

    # ── Word correction ──

    def _correct_words(self, text):
        def repl(m):
            w, _ = self._correct_word(m.group(0))
            return w
        return CYRILLIC_WORD_RE.sub(repl, text)

    def _correct_word(self, word):
        if not self.word_set:
            return word, False

        lower = word.lower()
        orig_freq = self.word_freq.get(lower, 0)

        # 1. In dictionary — only override if the word is very rare AND candidate
        #    is overwhelmingly more common (avoids replacing correct words)
        if lower in self.word_set:
            if orig_freq < self.FREQ_OVERRIDE_MAX_ORIG:
                best, best_freq = self._best_confusion_candidate(lower)
                if best and best_freq > orig_freq * self.FREQ_OVERRIDE_RATIO:
                    return _restore_case(best, word), True
            return word, False

        # 2. Try splitting merged words
        left, right = self._try_split(lower)
        if left:
            return _restore_case(left, word[:len(left)]) + " " + _restore_case(right, word[len(left):]), True

        # 3. Confusion candidates (up to 2 swaps)
        best, best_freq = self._best_confusion_candidate(lower)
        if best:
            return _restore_case(best, word), True

        # 4. Single-edit candidates
        best, best_freq = self._best_single_edit(lower)
        if best:
            return _restore_case(best, word), True

        return word, False

    def _best_confusion_candidate(self, lower):
        best, best_freq = None, 0
        for cand in self._confusion_candidates(lower):
            f = self.word_freq.get(cand, 0)
            if cand in self.word_set and f > best_freq:
                best, best_freq = cand, f
        return best, best_freq

    def _best_single_edit(self, lower):
        # Minimum frequency: only substitute with fairly common words.
        # Without this, proper nouns and rare-but-correct words get replaced
        # with the highest-frequency word 1 edit away, which is almost always wrong.
        MIN_FREQ = 300
        best, best_freq = None, 0
        alpha = "абвгдѓежзѕијклљмнњопрстќуфхцчџш"
        for i in range(len(lower)):
            for ch in alpha:
                if ch != lower[i]:
                    cand = lower[:i] + ch + lower[i+1:]
                    f = self.word_freq.get(cand, 0)
                    if cand in self.word_set and f > best_freq and f >= MIN_FREQ:
                        best, best_freq = cand, f
        return best, best_freq

    def _confusion_candidates(self, word):
        cands = set()
        pairs = {a: b for a, b in self.CONFUSION_PAIRS}
        for i, ch in enumerate(word):
            if ch in pairs:
                c1 = word[:i] + pairs[ch] + word[i+1:]
                cands.add(c1)
                for j, ch2 in enumerate(c1):
                    if j != i and ch2 in pairs:
                        cands.add(c1[:j] + pairs[ch2] + c1[j+1:])
        return cands

    def _try_split(self, word):
        if len(word) < 6:
            return None, None
        best, best_score = None, 0
        for i in range(3, len(word) - 2):
            left, right = word[:i], word[i:]
            if left in self.word_set and right in self.word_set:
                score = self.word_freq.get(left, 1) * self.word_freq.get(right, 1)
                if score > best_score:
                    best, best_score = (left, right), score
        return best if best else (None, None)

    def _correct_line(self, line):
        line = self._dehyphenate(line)
        line = self._fix_quotes(line)
        line = self._merge_fragments(line)
        if self.word_set:
            line = self._correct_words(line)
        return line

    def _fix_quotes(self, line):
        """Fix common quote character substitutions the model makes.

        The closing Macedonian quote is " (U+201C). The model often outputs
        a straight ASCII " instead, or drops it entirely. We can't fully
        recover dropped quotes, but we can normalise substituted ones.

        Pattern: if a line contains „ (opening) and ends with " or contains
        a " adjacent to a Cyrillic word boundary, normalise to U+201C.
        """
        # ASCII " after a Cyrillic character → likely a closing quote
        line = re.sub(r'([а-яА-ЯѓЃќЌљЉњЊџЏѕЅјЈѐЀѝЍ])"', r'\1' + '\u201c', line)
        # ASCII " before a space or end of string, if „ is present → closing
        if '\u201e' in line:
            line = re.sub(r'"(\s|$)', '\u201c' + r'\1', line)
        return line


def _restore_case(corrected, original):
    if not original:
        return corrected
    if original.isupper():
        return corrected.upper()
    elif original[0].isupper():
        return corrected[0].upper() + corrected[1:]
    return corrected


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 mk_postprocess.py --build-dict ~/macedonian-llm/books_clean/books_txt")
        print("  python3 mk_postprocess.py --correct 'text to correct'")
        print("  python3 mk_postprocess.py --test")
        sys.exit(1)

    if sys.argv[1] == "--build-dict":
        books_dir = sys.argv[2] if len(sys.argv) > 2 else "~/macedonian-llm/books_clean/books_txt"
        output = sys.argv[3] if len(sys.argv) > 3 else "mk_wordfreq.json"
        freq = MKPostProcessor.build_dictionary(books_dir)
        pp = MKPostProcessor()
        pp.word_freq = freq
        pp.save_dictionary(output)

    elif sys.argv[1] == "--correct":
        pp = MKPostProcessor("mk_wordfreq.json")
        text = " ".join(sys.argv[2:])
        print(f"  Input:  {text}")
        print(f"  Output: {pp.process_text(text)}")

    elif sys.argv[1] == "--test":
        pp = MKPostProcessor("mk_wordfreq.json")
        tests = [
            ("Тој ммаше друти планови", "и/м and т/г correction"),
            ("бидејќи мако ја сакам", "мако -> иако (freq override)"),
            ("Вис- тина беше", "dehyphenation with ASCII hyphen"),
            ("нејзинатарезервираност не ме", "split merged word"),
            ("со сопрутата двата", "сопрутата -> сопругата"),
            ("мзгледаше неважна", "мзгледаше -> изгледаше"),
            ("мли подоцна ќе морам", "мли -> или"),
            ("кутам стан не затоа", "кутам -> купам"),
            ("соп рутата напротив", "fragment merge + correction"),
            ("Мол тени би требало", "fragment merge"),
            ("изнајмува чка на соби", "fragment merge"),
        ]
        print(f"\n  Testing with {len(pp.word_set):,} word dictionary\n")
        for text, desc in tests:
            result = pp.process_text(text)
            tag = "OK" if result != text else "SAME"
            print(f"  [{tag:4}] {desc}")
            print(f"         In:  {text}")
            print(f"         Out: {result}")
            print()

    else:
        print(f"Unknown: {sys.argv[1]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
