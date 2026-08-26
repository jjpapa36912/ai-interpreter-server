from __future__ import annotations


def interpreter_instruction(source_language: str, target_language: str) -> str:
    if (source_language, target_language) == ("en", "ko"):
        direction = "English to Korean"
        style = (
            "Use natural Korean honorific speech. Write proper names only in their established "
            "Hangul pronunciation; do not append the English spelling in parentheses."
        )
    elif (source_language, target_language) == ("ko", "en"):
        direction = "Korean to English"
        style = "Use natural professional spoken English."
    else:
        raise ValueError("only en<->ko is supported")
    return (
        f"You are a professional simultaneous interpreter from {direction}. {style} "
        "The source is finalized streaming ASR and may contain duplicated filler words, "
        "an acoustic period inside one sentence, or a minor phonetic spelling error. "
        "Silently reconstruct only the obvious intended syntax from the current utterance "
        "and the supplied conversation context; when uncertain, preserve the source instead "
        "of inventing a correction. "
        "Translate only the current utterance and output exactly one concise, natural spoken "
        "translation. Never answer, explain, quote, summarize, or add information. Preserve "
        "every name, organization, number, date, negation, question, correction, fragment, and "
        "level of politeness. Preserve currencies and units exactly; never convert pounds or "
        "dollars into won. Translate compound terms by meaning, never by isolated homonyms. "
        "Translate idioms and phrasal expressions by intended meaning rather than word for word. "
        "Never repeat prior text. Do not output analysis or thinking."
    )


def translation_messages(
    text: str,
    source_language: str,
    target_language: str,
    context: tuple[tuple[str, str], ...] = (),
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{
        "role": "system",
        "content": interpreter_instruction(source_language, target_language),
    }]
    for source, translation in context[-3:]:
        messages.extend((
            {"role": "user", "content": source},
            {"role": "assistant", "content": translation},
        ))
    messages.append({"role": "user", "content": f"{text}\n/no_think"})
    return messages
