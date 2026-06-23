# ── Special tokens ────────────────────────────────────────────────────────────
TS_TOKEN = "<ts>"
TE_TOKEN = "<te>"
BC_TOKEN = "<bc>"   # both begin and end of backchannel
PAUSE_TOKEN = "<pause>"
SILENCE_TOKEN = "<silence>"

SPEAKER_TOKENS = {
    "A": ["<speaker_A>", "</speaker_A>"],
    "B": ["<speaker_B>", "</speaker_B>"],
}
STREAMING_CONT = " ..."   # suffix meaning "transcript not yet complete"
IGNORE_INDEX = -100

# ── Default hyper-params (mirror DataArguments style) ────────────────────────
DEFAULT_CHUNK_SECS   = 0.5    # seconds of audio per streaming step
DEFAULT_SAMPLE_RATE  = 16_000
DEFAULT_CONTEXT_LENGTH = 1

MODAL_INDEX_MAP = {
    "<A>": -201,
    "<B>": -202,
    "<audio>": -203
}
