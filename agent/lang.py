"""English, Urdu and Hindi on the same call.

Callers may speak any of the three and switch mid-sentence. Deepgram's multilingual model writes
Hindi and spoken Urdu mostly in Devanagari, the agent may answer in Urdu script, and callers mix in
English words. The booking checks (booking.py, receptionist.py) are written against English
keywords, so before matching, `gloss()` swaps the Urdu and Hindi words they look for (in Devanagari,
Urdu script and Roman Urdu) for English ones: "शाम पाँच बजे" becomes "evening 5 o'clock".
Only matching uses the glossed text; what the caller said is never changed.

`romanize()` writes a Devanagari name in Latin letters for the business's dashboard ("अहमद" ->
"Ahmad"), and `script_language()` tells the TTS which language a reply is written in.
"""

import re

# Phrases first (longest first), then single words. Values are what booking.py's patterns expect.
_PHRASES = {
    # Closings and "anything else?"
    "अल्लाह हाफ़िज़": "goodbye", "अल्लाह हाफिज": "goodbye", "खुदा हाफ़िज़": "goodbye", "खुदा हाफिज": "goodbye",
    "اللہ حافظ": "goodbye", "خدا حافظ": "goodbye", "allah hafiz": "goodbye", "khuda hafiz": "goodbye",
    "बस शुक्रिया": "that's all", "بس شکریہ": "that's all", "bas shukriya": "that's all",
    "बस इतना ही": "that's all", "بس اتنا ہی": "that's all", "bas itna hi": "that's all",
    "और कुछ नहीं": "nothing else", "اور کچھ نہیں": "nothing else", "aur kuch nahi": "nothing else",
    "कुछ और": "anything else", "और कुछ": "anything else", "کچھ اور": "anything else", "اور کچھ": "anything else",
    "kuch aur": "anything else", "aur kuch": "anything else",
    # Holding on
    "एक मिनट": "one minute", "ایک منٹ": "one minute", "ek minute": "one minute", "ek minat": "one minute",
    # Agreement and correctness
    "ठीक है": "okay", "ٹھیک ہے": "okay", "theek hai": "okay", "thik hai": "okay",
    "सही है": "correct", "صحیح ہے": "correct", "sahi hai": "correct",
    # Line checks and repeats
    "आवाज़ आ रही है": "can you hear me", "आवाज आ रही है": "can you hear me", "آواز آ رہی ہے": "can you hear me",
    "awaz aa rahi hai": "can you hear me", "सुन रहे हैं": "can you hear me", "سن رہے ہیں": "can you hear me",
    "फिर से": "repeat", "پھر سے": "repeat", "phir se": "repeat",
    # Names and bookings
    "मेरा नाम": "my name is", "میرا نام": "my name is", "mera naam": "my name is",
    "बुक हो गई": "is booked", "बुक हो गया": "is booked", "بک ہو گئی": "is booked", "بک ہو گیا": "is booked",
    "कन्फर्म हो गई": "is confirmed", "कन्फर्म हो गया": "is confirmed", "کنفرم ہو گئی": "is confirmed",
    "کنفرم ہو گیا": "is confirmed",
    # "PM" and "AM" as transcripts spell them in Urdu or Hindi script
    "پی ایم": "pm", "اے ایم": "am", "पी एम": "pm", "पीएम": "pm", "ए एम": "am", "एएम": "am",
    # Half hours: "साढ़े पाँच" is half past five; "डेढ़" and "ढाई" are 1:30 and 2:30 on their own.
    "डेढ़ बजे": "half past 1", "ढाई बजे": "half past 2", "ڈیڑھ بجے": "half past 1", "ڈھائی بجے": "half past 2",
}

_WORDS = {
    # Yes / no
    "हाँ": "yes", "हां": "yes", "जी": "yes", "जीहाँ": "yes", "ہاں": "yes", "ہان": "yes", "جی": "yes",
    "haan": "yes", "han": "yes", "ji": "yes", "jee": "yes", "jii": "yes",
    "ज़रूर": "sure", "जरूर": "sure", "ضرور": "sure", "zaroor": "sure", "zarur": "sure",
    "अच्छा": "okay", "اچھا": "okay", "acha": "okay", "achha": "okay", "accha": "okay",
    "नहीं": "no", "नही": "no", "ना": "no", "نہیں": "no", "نہی": "no", "نا": "no", "nahi": "no", "nahin": "no",
    "nai": "no", "na": "no",
    "ग़लत": "wrong", "गलत": "wrong", "غلط": "wrong", "galat": "wrong", "ghalat": "wrong",
    "सही": "correct", "صحیح": "correct", "sahi": "correct", "درست": "correct", "durust": "correct",
    # The details the agent asks for
    "नाम": "name", "نام": "name", "naam": "name",
    "नंबर": "number", "नम्बर": "number", "نمبر": "number", "fon": "phone",
    "फ़ोन": "phone", "फोन": "phone", "فون": "phone", "मोबाइल": "phone", "موبائل": "phone",
    "वक़्त": "time", "वक्त": "time", "समय": "time", "टाइम": "time", "وقت": "time", "ٹائم": "time",
    "waqt": "time", "samay": "time",
    "दिन": "day", "دن": "day", "din": "day",
    "तारीख़": "date", "तारीख": "date", "تاریخ": "date", "tareekh": "date", "tarikh": "date",
    "कब": "when", "کب": "when", "kab": "when",
    "सर्विस": "service", "सेवा": "service", "سروس": "service", "servis": "service",
    "बुक": "book", "بک": "book", "बुकिंग": "booking", "بکنگ": "booking",
    "अपॉइंटमेंट": "appointment", "अपोइंटमेंट": "appointment", "اپوائنٹمنٹ": "appointment",
    # Days
    "आज": "today", "آج": "today", "aaj": "today",
    "कल": "tomorrow", "کل": "tomorrow", "kal": "tomorrow",
    "परसों": "day after tomorrow", "پرسوں": "day after tomorrow", "parson": "day after tomorrow",
    "सोमवार": "monday", "पीर": "monday", "پیر": "monday", "somwar": "monday", "peer": "monday", "pir": "monday",
    "मंगलवार": "tuesday", "मंगल": "tuesday", "منگل": "tuesday", "mangal": "tuesday",
    "बुधवार": "wednesday", "बुध": "wednesday", "بدھ": "wednesday", "budh": "wednesday",
    "गुरुवार": "thursday", "जुमेरात": "thursday", "جمعرات": "thursday", "jumeraat": "thursday",
    "शुक्रवार": "friday", "जुम्मा": "friday", "जुमा": "friday", "جمعہ": "friday", "jumma": "friday", "juma": "friday",
    "शनिवार": "saturday", "हफ़्ता": "saturday", "ہفتہ": "saturday", "hafta": "saturday",
    "रविवार": "sunday", "इतवार": "sunday", "اتوار": "sunday", "itwar": "sunday", "itwaar": "sunday",
    # Times of day
    "बजे": "o'clock", "بجے": "o'clock", "baje": "o'clock",
    "सुबह": "morning", "صبح": "morning", "subah": "morning", "subha": "morning",
    "दोपहर": "afternoon", "دوپہر": "afternoon", "dopahar": "afternoon", "dopehar": "afternoon",
    "शाम": "evening", "شام": "evening", "sham": "evening", "shaam": "evening",
    "रात": "night", "رات": "night", "raat": "night",
    "साढ़े": "half past", "ساڑھے": "half past", "sadhe": "half past", "sarhe": "half past",
    # Greetings, thanks, goodbyes
    "हेलो": "hello", "हैलो": "hello", "ہیلو": "hello", "नमस्ते": "hello", "नमस्कार": "hello", "आदाब": "hello",
    "سلام": "salam", "सलाम": "salam", "آداب": "hello",
    "शुक्रिया": "thanks", "धन्यवाद": "thanks", "شکریہ": "thanks", "shukriya": "thanks", "shukria": "thanks",
    "बाय": "bye", "بائے": "bye",
    "रुको": "wait", "رکو": "wait", "ruko": "wait", "रुकिए": "wait", "رکیں": "wait",
    "दोबारा": "repeat", "دوبارہ": "repeat", "dobara": "repeat",
    # Question words, so a caller's question is not taken as a detail
    "क्या": "what", "کیا": "what", "kya": "what", "कितने": "how", "कितना": "how", "کتنے": "how", "کتنا": "how",
    "kitne": "how", "kitna": "how", "कौन": "which", "کون": "which", "kaun": "which",
    "कहाँ": "where", "کہاں": "where", "kahan": "where", "क्यों": "why", "کیوں": "why", "kyun": "why",
    # English words as an Urdu or Hindi transcript spells them
    "ٹو": "2", "ون": "1", "تھری": "3", "فور": "4", "فائیو": "5", "فائو": "5", "سکس": "6", "سیون": "7", "ایٹ": "8",
    "نائن": "9", "ٹین": "10", "اوہ": "oh",
    "منڈے": "monday", "ٹیوزڈے": "tuesday", "ونزڈے": "wednesday", "تھرزڈے": "thursday", "فرائیڈے": "friday",
    "سیٹرڈے": "saturday", "سنڈے": "sunday", "ٹومارو": "tomorrow", "ٹوڈے": "today", "ایوننگ": "evening", "مارننگ": "morning",
    "یس": "yes", "یپ": "yes", "اوکے": "okay", "کریکٹ": "correct", "کرکٹ": "correct",
    "سوری": "sorry", "پلیز": "please",
    "यस": "yes", "ओके": "okay", "सॉरी": "sorry", "प्लीज़": "please", "प्लीज": "please",
    # Months (resolve_day reads English month names)
    "جنوری": "january", "فروری": "february", "مارچ": "march", "اپریل": "april", "مئی": "may", "جون": "june",
    "جولائی": "july", "اگست": "august", "ستمبر": "september", "اکتوبر": "october", "نومبر": "november", "دسمبر": "december",
    "जनवरी": "january", "फ़रवरी": "february", "फरवरी": "february", "मार्च": "march", "अप्रैल": "april", "मई": "may",
    "जून": "june", "जुलाई": "july", "अगस्त": "august", "सितंबर": "september", "सितम्बर": "september",
    "अक्टूबर": "october", "अक्तूबर": "october", "नवंबर": "november", "नवम्बर": "november", "दिसंबर": "december",
    "दिसम्बर": "december",
    # Filler that only gets in the way of matching ("मेरा नाम अहमद है" -> "my name is अहमद")
    "है": "", "हैं": "", "ہے": "", "ہیں": "", "hai": "", "hain": "",
    # Numbers (times, dates, phone digits). "do" is left out: in Roman Urdu it usually means "give".
    "शून्य": "0", "ज़ीरो": "0", "जीरो": "0", "صفر": "0", "زیرو": "0", "sifar": "0",
    "एक": "1", "ایک": "1", "ek": "1",
    "दो": "2", "دو": "2",
    "तीन": "3", "تین": "3", "teen": "3", "tin": "3",
    "चार": "4", "چار": "4", "char": "4", "chaar": "4",
    "पाँच": "5", "पांच": "5", "پانچ": "5", "panch": "5", "paanch": "5",
    "छह": "6", "छः": "6", "छे": "6", "چھ": "6", "chhe": "6", "chay": "6", "che": "6",
    "सात": "7", "سات": "7", "saat": "7",
    "आठ": "8", "آٹھ": "8", "aath": "8", "ath": "8",
    "नौ": "9", "نو": "9", "nau": "9",
    "दस": "10", "دس": "10", "das": "10",
    "ग्यारह": "11", "گیارہ": "11", "gyarah": "11", "barah": "12", "बारह": "12", "بارہ": "12",
}

# Devanagari, Arabic-Indic and Extended Arabic-Indic digits as ASCII.
_DIGITS = str.maketrans("०१२३४५६७८९٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "0123456789" * 3)
# Urdu's question mark, full stop and comma, and Hindi's full stop (danda), as ASCII.
_PUNCT = str.maketrans({"؟": "?", "۔": ".", "،": ",", "।": "."})

_PHRASE_RE = re.compile("|".join(re.escape(p) for p in sorted(_PHRASES, key=len, reverse=True)))
_WORD_RE = re.compile(r"[^\W\d_]+(?:[ऀ-ॿ؀-ۿ‌‍]*)", re.UNICODE)


def gloss(text: str) -> str:
    """`text` lowercased, with Urdu and Hindi keywords (any script) swapped for English ones."""
    text = text.lower().replace("’", "'").translate(_DIGITS).translate(_PUNCT)
    text = _PHRASE_RE.sub(lambda m: _PHRASES[m[0]], text)

    def word(m: re.Match[str]) -> str:
        w = m[0]
        return _WORDS.get(w, w)

    text = _WORD_RE.sub(word, text)
    return re.sub(r"\s+", " ", text).strip()


# --- Devanagari to Latin, for names on the dashboard ------------------------------------------

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "f", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "w", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l", "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "फ़": "f", "ड़": "r",
    "ढ़": "rh", "य़": "y",
}
_VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ए": "e", "ऐ": "ai",
           "ओ": "o", "औ": "au", "ऑ": "o", "ॲ": "e"}
_MATRAS = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o",
           "ौ": "au", "ॉ": "o", "ॅ": "e"}
_NUKTA, _VIRAMA = "़", "्"
_NASAL = {"ं": "n", "ँ": "n", "ः": "h"}


def _romanize_word(word: str) -> str:
    # Split into (consonant, vowel) units; vowel None means the inherent "a".
    units: list[tuple[str, str | None]] = []
    i = 0
    while i < len(word):
        ch = word[i]
        if i + 1 < len(word) and word[i + 1] == _NUKTA:
            ch, i = ch + _NUKTA, i + 1
        if ch in _CONSONANTS:
            nxt = word[i + 1] if i + 1 < len(word) else ""
            if nxt in _MATRAS:
                units.append((_CONSONANTS[ch], _MATRAS[nxt]))
                i += 2
                continue
            if nxt == _VIRAMA:
                units.append((_CONSONANTS[ch], ""))
                i += 2
                continue
            units.append((_CONSONANTS[ch], None))
        elif ch in _VOWELS:
            units.append(("", _VOWELS[ch]))
        elif ch in _NASAL:
            units.append((_NASAL[ch], ""))
        elif ch not in (_NUKTA, _VIRAMA):
            units.append((ch, ""))
        i += 1

    # Schwa deletion, as in speech: the inherent "a" is silent at the end of a word, and after a vowel
    # when the next consonant is itself voiced ("अहमद" is "ahmad", "ज़ैनब" is "zainab"). Decided from
    # the end of the word backwards, because each choice depends on the one after it.
    vowels: list[str] = [""] * len(units)
    for k in range(len(units) - 1, -1, -1):
        cons, vowel = units[k]
        if vowel is not None:
            vowels[k] = vowel
            continue
        if k == len(units) - 1:
            vowels[k] = "" if k > 0 else "a"
            continue
        prev_has_vowel = k > 0 and units[k - 1][1] != ""
        next_cons, _ = units[k + 1]
        silent = prev_has_vowel and bool(next_cons) and vowels[k + 1] != ""
        vowels[k] = "" if silent else "a"
    return "".join(cons + vowels[k] for k, (cons, _) in enumerate(units))


_DEVANAGARI = re.compile(r"[ऀ-ॿ]+")


def romanize(text: str) -> str:
    """Devanagari words written in Latin letters; anything else is left as it is."""
    return _DEVANAGARI.sub(lambda m: _romanize_word(m[0]), text)


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))


# --- Matching words across scripts ----------------------------------------------------------------

# Urdu letters to rough Latin consonants. Urdu writes few vowels, so matching compares consonant
# skeletons ("ہیئر کٹ" and "Haircut" are both "hrkt") rather than full spellings.
_URDU_LETTERS = {
    "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s", "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ڈ": "d",
    "ذ": "z", "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z",
    "غ": "gh", "ف": "f", "ق": "q", "ک": "k", "ك": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "w",
    "ہ": "h", "ھ": "h", "ه": "h", "ۃ": "h", "ی": "y", "ي": "y", "ے": "e", "ئ": "", "ء": "", "ا": "a", "آ": "a", "ع": "",
}


def _latin_letters(text: str) -> str:
    """Any script in rough Latin letters: Devanagari romanized, Urdu letter by letter."""
    text = romanize(text)
    return "".join(_URDU_LETTERS.get(ch, ch) for ch in text)


def skeleton(text: str) -> str:
    """The consonants of `text` in any script, with sounds spelled several ways made the same."""
    t = _latin_letters(text).lower()
    t = re.sub(r"c(?=[eiy])", "s", t)  # a soft c sounds like s: "facial", "service"
    for a, b in (("ph", "f"), ("ck", "k"), ("c", "k"), ("q", "k"), ("x", "ks"), ("v", "w"), ("z", "s"),
                 ("sh", "s"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("ch", "c")):
        t = t.replace(a, b)
    t = re.sub(r"[^a-z]", "", t)
    t = re.sub(r"[aeiouy]", "", t)
    return re.sub(r"(.)\1+", r"\1", t)  # doubled letters count once


# --- Which language a reply is written in -----------------------------------------------------

_URDU_SCRIPT = re.compile(r"[؀-ۿ]")


def script_language(text: str) -> str | None:
    """'ur' for Urdu script, 'hi' for Devanagari, 'en' for Latin letters, None if no letters yet."""
    if _URDU_SCRIPT.search(text):
        return "ur"
    if _DEVANAGARI.search(text):
        return "hi"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return None
