"""Clean up MusicCaps aspect lists by merging synonymous free-text aspects into canonical tags.

MusicCaps has 13,219 distinct aspects, many of them rewordings of each other
("male voice", "male vocal", "male singer"). Merging them before choosing the
top-K tags gives labels that mean one thing each. Aspects not listed here are
kept unchanged.
"""

# Canonical tag -> aspect phrases with the same meaning. A phrase can feed more
# than one tag, e.g. "groovy bass line" counts as both "bass" and "groovy".
ALIASES: dict[str, list[str]] = {
    # Recording and production
    "low quality": [
        "poor audio quality", "bad audio quality", "low quality audio", "low quality recording", "inferior audio quality",
    ],
    "live performance": ["live recording", "live audience"],
    "amateur recording": ["home video"],
    "reverb": ["reverberant"],
    # Tempo
    "fast tempo": ["uptempo"],
    "medium tempo": ["moderate tempo"],
    # Voice
    "instrumental": ["instrumental music", "no voices", "no voice", "no singer", "no vocals"],
    "male vocals": [
        "male voice", "male vocal", "male singer", "male vocalist", "male voice singing", "flat male vocal",
        "passionate male vocal",
    ],
    "female vocals": [
        "female vocal", "female voice", "female singer", "female vocalist", "female voice singing",
        "passionate female vocal",
    ],
    "backing vocals": ["vocal harmony", "backup singers", "vocal backup"],
    # Instruments
    "bass": [
        "bass guitar", "e-bass", "groovy bass", "groovy bass line", "groovy bass guitar", "smooth bass",
        "strong bass line", "percussive bass line",
    ],
    "electric guitar": ["e-guitar", "e-guitars", "electric guitar melody"],
    "acoustic guitar": ["acoustic rhythm guitar"],
    "guitar": ["guitar accompaniment"],
    "piano": ["acoustic piano", "piano accompaniment", "groovy piano melody"],
    "keyboard": ["keyboard harmony", "keyboard accompaniment"],
    "electronic drums": ["digital drums", "drum machine", "programmed percussion"],
    "drums": ["steady drumming", "steady drumming rhythm"],
    "punchy drums": ["punchy kick", "punchy snare"],
    "shimmering cymbals": ["shimmering hi hats"],
    "percussion": ["simple percussion", "wooden percussions"],
    "claps": ["clapping"],
    "shakers": ["shimmering shakers", "shaker"],
    "strings": ["string section", "sustained strings melody"],
    "violin": ["violins"],
    "synth": ["synth sounds", "synthesiser arrangements", "synthesiser arrangement", "synthesiser articulation"],
    "trumpet": ["trumpets"],
    "brass": ["brass section", "brass band"],
    # Genres
    "rock": ["hard rock"],
    "metal": ["heavy metal"],
    "electronic": ["electronic music", "electro", "edm", "electronic dance music", "techno", "trance"],
    "dance": ["dance music", "club music", "danceable", "dance groove", "dance rhythm"],
    "classical": ["classical music"],
    "pop": ["pop song", "pop music"],
    "hip hop": ["hip-hop", "rap"],
    "folk": ["folk music", "folk song"],
    "country": ["country music"],
    # Mood
    "happy": ["happy mood", "joyful", "cheerful"],
    "energetic": ["lively", "vibrant", "vivacious", "vigorous"],
    "upbeat": ["buoyant"],
    "emotional": ["sentimental", "heartfelt", "poignant"],
    "passionate": ["passionate male vocal", "passionate female vocal"],
    "groovy": ["groovy rhythm", "groovy music", "groovy bass", "groovy bass line", "groovy bass guitar", "groovy piano melody"],
    "relaxing": ["calming", "soothing", "calm", "chill", "meditative"],
    "mellow": ["soft"],
    "sad": ["melancholic"],
    "romantic": ["love song"],
    "fun": ["playful"],
    "eerie": ["scary", "sinister", "haunting", "mysterious", "suspenseful"],
    "psychedelic": ["trippy", "hypnotic"],
    "uplifting": ["inspiring", "positive"],
    "melodic": ["melodic singing"],
}

_TAGS_OF: dict[str, set[str]] = {}
for _tag, _phrases in ALIASES.items():
    for _phrase in [_tag, *_phrases]:
        _TAGS_OF.setdefault(_phrase, set()).add(_tag)


def canonical_tags(aspects: list[str]) -> list[str]:
    """Map a clip's lower-cased aspects to canonical tags."""
    tags: set[str] = set()
    for aspect in aspects:
        tags |= _TAGS_OF.get(aspect, {aspect})
    return sorted(tags)


def phrases_for(tag: str) -> list[str]:
    """Every aspect phrase that maps to this tag, including the tag itself."""
    return [tag, *ALIASES.get(tag, [])]
