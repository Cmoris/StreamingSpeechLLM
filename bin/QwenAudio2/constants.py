# ── Special tokens ────────────────────────────────────────────────────────────
TS_TOKEN = "<ts>"
TE_TOKEN = "<te>"
BC_TOKEN = "<bc>"
PAUSE_TOKEN = "<pause>"
SILENCE_TOKEN = "<silence>"

SPEAKER_TOKENS = {
    "A": ["<speaker_A>", "</speaker_A>"],
    "B": ["<speaker_B>", "</speaker_B>"],
}
STREAMING_CONT = " ..."   # suffix meaning "transcript not yet complete"

# ── Default hyper-params (mirror DataArguments style) ────────────────────────
DEFAULT_CHUNK_SECS   = 2    # seconds of audio per streaming step
DEFAULT_SAMPLE_RATE  = 16_000

QUERY = """You are a streaming dialogue transcriber.

        Transcribe the speech from Speaker A and Speaker B.

        Use special tokens to represent dialogue events:

        <ts> : turn switch
        <te> : turn end
        <bc> : backchannel
        <pause> : speaker pause
        <silence> : conversation silence

        Output the transcript in chronological order."""
