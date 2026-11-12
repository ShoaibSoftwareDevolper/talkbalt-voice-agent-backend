"""Callers who swear at the receptionist or turn the call sexual.

A real receptionist doesn't apologize for being sworn at, doesn't repeat it, and doesn't answer it
by reading out the service list. They stay calm, steer back to the business once, warn once, and
then hang up. This module only recognizes such messages (English, Roman Urdu/Hindi, Urdu script,
Devanagari); booking.py counts them and tells the model what to do, and receptionist.py lets the
call end after the warning was ignored.

Plain frustration ("this is useless", "damn") is not abuse: booking.py answers that with an apology.
"""

import re

# Insults and slurs aimed at a person. Word-boundary matched against the lowercased message.
_ABUSE_LATIN = (
    r"fuck (you|off|u)|f\*+k|motherfuck\w*|son of a bitch|bitch\w*|bastard|asshole|dickhead|cunt|whore|slut"
    r"|go to hell|shut up,? (you )?(bitch|idiot)"
    # Roman Urdu / Hindi
    # (Kept to words with no everyday meaning: "kutte" alone could be a pet-grooming client's dog.)
    r"|ma+da?r ?cho+d\w*|be?he?n ?cho+d\w*|bhen ?cho+d\w*|chu+t(i|y)y?a\w*|bhosd\w*|bho?sdi\w*"
    r"|gaa?ndu|gaand|ra+ndi\w*|haraa?m ?(zaa?d\w*|khor)|haraa?mi|kami+n(a|e|i)|kut+(e|ay) (ka|ki) (bach+a|bach+e|aulaad)"
    r"|ullu ka pa+tha|lauda|phu+d+i"
)
_ABUSE_SCRIPT = (
    # Urdu script
    "مادرچود", "مادر چود", "بہنچود", "بہن چود", "بھینچود", "چوتیا", "چوتیے", "بھوسڑی", "گانڈو", "گانڈ", "رنڈی",
    "حرامزادہ", "حرام زادہ", "حرامی", "حرام خور", "کمینے", "کمینہ", "کمینی", "کتے کا بچہ", "پھدی",
    # Devanagari
    "मादरचोद", "मादर चोद", "बहनचोद", "बहन चोद", "भेनचोद", "भैनचोद", "चूतिया", "चुतिया", "भोसड़ी", "भोसडी", "गांडू",
    "गांड", "रंडी", "हरामज़ादा", "हरामजादा", "हरामी", "हरामखोर", "कमीने", "कमीना", "कमीनी", "कुत्ते का बच्चा",
    "लंड", "लौड़ा", "फुद्दी",
)
# Sexual or "adult" talk aimed at the receptionist.
_SEXUAL_LATIN = (
    r"sex|sexy|sexual|nudes?|naked|porn\w*|boobs?|tits|horny|blowjob|pussy|orgasm"
    r"|what are you wearing|send (me )?(your )?(pics|photos|nudes)|sleep with (me|you)|kiss (me|you)"
    r"|(be|become) my (girlfriend|boyfriend|wife)|date (me|with me)|i love you baby|you sound (hot|sexy)"
    # Roman Urdu / Hindi
    r"|mujh ?se (shaadi|shadi) kar|meri (girlfriend|gf) ban|sath so(ogi|o ?gi|na)|kapde utaa?r\w*|chumm(a|i)"
)
_SEXUAL_SCRIPT = (
    "سیکس", "سیکسی", "ننگی", "ننگا", "مجھ سے شادی کرو", "میرے ساتھ سو", "کپڑے اتار", "چمی", "چوما",
    "सेक्स", "सेक्सी", "नंगी", "नंगा", "मुझसे शादी करो", "मेरे साथ सो", "कपड़े उतार", "चुम्मा", "चुम्मी",
)

_ABUSE_RE = re.compile(rf"\b({_ABUSE_LATIN})\b")
_SEXUAL_RE = re.compile(rf"\b({_SEXUAL_LATIN})\b")


def _has_script_word(text: str, words: tuple[str, ...]) -> bool:
    # Urdu and Devanagari words are matched as whole tokens (no \b in those scripts), so "لن" never
    # matches inside a longer word; multi-word phrases are matched as substrings.
    tokens = set(re.findall(r"[؀-ۿऀ-ॿ]+", text))
    return any((w in text) if " " in w else (w in tokens) for w in words)


def classify(text: str) -> str | None:
    """'abuse' for insults and slurs, 'sexual' for sexual talk, None otherwise. Abuse wins."""
    lowered = text.lower()
    if _ABUSE_RE.search(lowered) or _has_script_word(text, _ABUSE_SCRIPT):
        return "abuse"
    if _SEXUAL_RE.search(lowered) or _has_script_word(text, _SEXUAL_SCRIPT):
        return "sexual"
    return None


# What the model is told, by how many such messages the caller has sent on this call.
def instruction(kind: str, count: int, business_name: str) -> str:
    what = "used abusive language" if kind == "abuse" else "said something sexual or inappropriate"
    if count >= 3:
        return (
            f"The caller {what} again after being warned. End the call now: call end_call. Do not answer what they said, "
            "do not apologize and do not list services."
        )
    if count == 2:
        return (
            f"The caller {what} again. Stay calm and polite. In one short sentence, say that you can only continue if the "
            f"conversation stays respectful, otherwise you will end the call; then ask if they need help with {business_name}. "
            "Do not repeat their words, do not apologize, and do not list services."
        )
    if kind == "sexual":
        return (
            f"The caller said something sexual or inappropriate. Do not engage with it, do not repeat it and do not "
            f"lecture. In one short, calm sentence say you are {business_name}'s receptionist and can only help with "
            "its services and appointments, then ask how you can help. Do not list the services unless they ask."
        )
    return (
        "The caller used abusive language. Do not repeat it, do not apologize for it, and do not lecture. "
        "If there is a real complaint inside it, acknowledge that complaint briefly. Otherwise, in one short calm "
        f"sentence say you are here to help with {business_name} and ask how you can help. Do not list the services."
    )
