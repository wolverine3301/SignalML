"""Audio restoration backends (de-reverb today; de-echo, denoise, karaoke share the runner).

Restoration runs on *separated* vocals only, phrase by phrase, and every result is
judged before use: a phrase keeps the restored audio only when it measurably improved
and nothing important (pitch, level) moved. See ``signalml/stages/dereverb.py``.
"""
