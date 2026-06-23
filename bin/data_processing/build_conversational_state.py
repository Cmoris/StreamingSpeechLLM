from typing import Optional
from dataclasses import dataclass, field

@dataclass
class Token:
    t_start: float
    t_end:   float
    text:    str
    speaker: str          # "A" or "B"
    audio_path: str
    
@dataclass
class StreamEvent:
    start:   float
    end:     float
    speaker: str          # "A" | "B" | "SYSTEM"
    token:   str          # surface text or special token
    kind:    str          # "asr" | "ts" | "te" | "bc" | "silence" | "yield"

# ---------------------------------------------------------
# Hyper parameters
# ---------------------------------------------------------

SWITCH_GAP = 0.7          # turn switch最大间隔
RESUME_WINDOW = 1.5       # 原speaker多久内恢复算hold floor

PAUSE_THRESHOLD = 0.3     # speaker内部pause
SILENCE_THRESHOLD = 1.5   # conversation silence

BC_MAX_DURATION = 1.0     # backchannel最长时间
BC_MAX_TOKENS = 8         # backchannel最大token数


# ---------------------------------------------------------
# Special tokens
# ---------------------------------------------------------

TS_TOKEN = "<ts>"              # floor transferred
TE_TOKEN = "<te>"              # utterance completed
PAUSE_TOKEN = "<pause>"        # same speaker continues later
SILENCE_TOKEN = "<silence>"    # nobody speaks
BC_TOKEN = "<bc>"              # backchannel


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def utt_start(utt: list[Token]):
    return utt[0].t_start


def utt_end(utt: list[Token]):
    return utt[-1].t_end


def utt_duration(utt: list[Token]):
    return utt_end(utt) - utt_start(utt)


def has_overlap(utt_a: list[Token], utt_b: list[Token]):
    a_start, a_end = utt_a[0].t_start,  utt_a[-1].t_end
    b_start, b_end = utt_b[0].t_start,  utt_b[-1].t_end
    return a_start < b_end and b_start < a_end


def find_next_same_speaker(
    utterances,
    curr_idx,
):
    curr_spk = utterances[curr_idx][0].speaker

    for j in range(curr_idx + 1, len(utterances)):
        if utterances[j][0].speaker == curr_spk:
            return utterances[j]

    return None


def find_next_other_speaker(
    utterances,
    curr_idx,
):
    curr_spk = utterances[curr_idx][0].speaker

    for j in range(curr_idx + 1, len(utterances)):
        if utterances[j][0].speaker != curr_spk:
            return utterances[j]

    return None


# ---------------------------------------------------------
# Backchannel heuristic
# ---------------------------------------------------------

def is_backchannel(
    curr_utt,
    next_utt,
    next_same,
):
    
    if getattr(curr_utt[0], "bc_label", None):
        if getattr(curr_utt[0], "bc_label") != "non_backchannel":
            return True
        else:
            return False

    # speaker必须不同
    if next_utt[0].speaker == curr_utt[0].speaker:
        return False

    # 必须overlap
    if not has_overlap(curr_utt, next_utt):
        return False

    # 很短
    if utt_duration(next_utt) > BC_MAX_DURATION:
        return False

    # token很少
    if len(next_utt) > BC_MAX_TOKENS:
        return False

    # 原speaker继续
    if next_same is None:
        return False

    if utt_start(next_same) - utt_end(curr_utt) > RESUME_WINDOW:
        return False

    return True


# ---------------------------------------------------------
# Turn switch heuristic
# ---------------------------------------------------------

def is_turn_switch(
    curr_utt,
    next_other,
    next_same,
):

    if next_other is None:
        return False

    curr_end = utt_end(curr_utt)

    other_gap = utt_start(next_other) - curr_end

    # 不同speaker快速接话
    if not (other_gap <= SWITCH_GAP):
        return False

    # 原speaker没有resume
    if next_same is not None:

        same_gap = utt_start(next_same) - curr_end

        if same_gap <= RESUME_WINDOW:
            return False

    return True


FINAL_SUFFIXES = (
    "です", "ます", "でした", "ました",
    "ですね", "ですよ", "ですよね", "だよね",
    "ですか", "ますか", "でしょうか",
    "と思います", "ということです",
    "わかりました", "そうです", "はい",
)

NONFINAL_SUFFIXES = (
    "で", "けど", "けども", "から", "ので",
    "が", "を", "に", "と", "は",
    "その", "あの", "えっと", "なんか",
)

def utt_text(utt):
    return "".join(tok.text for tok in utt).strip()

def has_semantic_completion(utt):
    text = utt_text(utt)

    if not text:
        return False

    # 明显未完成：不要标 te
    if text.endswith(NONFINAL_SUFFIXES):
        return False

    # 明显完成：可以标 te
    if text.endswith(FINAL_SUFFIXES):
        return True

    # 短回应一般可以视为完整 TCU
    if len(text) <= 8 and text in {
        "はい", "うん", "ええ", "そう", "そうです",
        "なるほど", "わかりました", "了解です",
    }:
        return True

    # 兜底：不确定时不要强行标 te
    return False


def is_turn_end(
    curr_utt,
    next_same,
    next_other,
):
    curr_end = utt_end(curr_utt)

    # 当前 speaker 很快继续：不是 turn end
    if next_same is not None:
        same_gap = utt_start(next_same) - curr_end
        if same_gap <= RESUME_WINDOW:
            return False

    # 别人快速接管：这是 ts，不是 te
    if next_other is not None:
        other_gap = utt_start(next_other) - curr_end
        if 0 <= other_gap <= SWITCH_GAP:
            return False

    # 语义不完整：不要标 te
    if not has_semantic_completion(curr_utt):
        return False

    return True


# ---------------------------------------------------------
# Pause heuristic
# ---------------------------------------------------------

def is_pause(
    curr_utt,
    next_same,
):

    if next_same is None:
        return False

    gap = utt_start(next_same) - utt_end(curr_utt)

    return (
        PAUSE_THRESHOLD <= gap < SILENCE_THRESHOLD
    )


# ---------------------------------------------------------
# Silence heuristic
# ---------------------------------------------------------

def is_silence(
    curr_utt,
    utterances,
    curr_idx,
):

    curr_end = utt_end(curr_utt)

    if curr_idx == len(utterances) - 1:
        return False

    next_start = utt_start(utterances[curr_idx + 1])

    gap = next_start - curr_end

    return gap >= SILENCE_THRESHOLD


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def build_stream(
    utterances: list[list[Token]],
) -> list[StreamEvent]:

    events: list[StreamEvent] = []
    interrupted_at: list[Optional[float]] = [None] * len(utterances)
    for i in range(len(utterances) - 1):
        curr_utt = utterances[i]
        next_utt = utterances[i + 1]
        if (next_utt[0].speaker != curr_utt[0].speaker and has_overlap(curr_utt, next_utt)):
            # 打断时刻 = 下一speaker开始说话的时刻
            interrupted_at[i] = next_utt[0].t_start
    interrupted_at[i] = None
    for i, utt in enumerate(utterances):
        spk = utt[0].speaker
        curr_end = utt_end(utt)
        interrupted_time = interrupted_at[i]
        # -------------------------------------------------
        # emit ASR tokens
        # -------------------------------------------------

        for tok in utt:
            if interrupted_time is None or tok.t_end <= interrupted_time:
                events.append(
                    StreamEvent(
                        start=tok.t_start,
                        end=tok.t_end,
                        speaker=spk,
                        token=tok.text,
                        kind="asr",
                    )
                )
            elif tok.t_end >= interrupted_time:
                events.append(
                    StreamEvent(
                        start=interrupted_time,
                        end=interrupted_time,
                        speaker=spk,
                        token=tok.text,
                        kind="overlap",
                    )
                )
                
                break

        # -------------------------------------------------
        # future utterances
        # -------------------------------------------------

        next_same = find_next_same_speaker(
            utterances,
            i,
        )

        next_other = find_next_other_speaker(
            utterances,
            i,
        )

        next_uttr = utterances[i+1] if (i+1) < len(utterances) else None
        # -------------------------------------------------
        # event decision
        # priority:
        # bc > ts > pause > silence > te
        # -------------------------------------------------

        boundary_token = None
        boundary_kind = None

        # -----------------------------
        # backchannel
        # -----------------------------
        if (
            next_other is not None
            and is_backchannel(
                utt,
                next_other,
                next_same,
            )
        ):
            
            boundary_token = BC_TOKEN
            boundary_kind = "bc"

        # -----------------------------
        # turn switch
        # -----------------------------

        elif is_turn_switch(
            utt,
            next_other,
            next_same,
        ):

            boundary_token = TS_TOKEN
            boundary_kind = "ts"

        # -----------------------------
        # pause
        # -----------------------------

        elif is_pause(
            utt,
            next_same,
        ):

            boundary_token = PAUSE_TOKEN
            boundary_kind = "pause"

        # -----------------------------
        # silence
        # -----------------------------

        elif is_silence(
            utt,
            utterances,
            i,
        ):

            boundary_token = SILENCE_TOKEN
            boundary_kind = "silence"

        # -----------------------------
        # utterance end
        # -----------------------------

        elif is_turn_end(
            utt,
            next_same,
            next_other,
        ):

            boundary_token = TE_TOKEN
            boundary_kind = "te"

        # -------------------------------------------------
        # emit boundary token
        # -------------------------------------------------
        if boundary_token is not None:
            events.append(
                StreamEvent(
                    start=curr_end,
                    end=curr_end,
                    speaker=spk,
                    token=boundary_token,
                    kind=boundary_kind,
                )
            )

    # -----------------------------------------------------
    # global sort
    # -----------------------------------------------------

    kind_order = {
        "asr": 0,
        "overlap": 0,
        "bc": 1,
        "ts": 1,
        "te": 1,
        "pause": 2,
        "silence": 2,
    }

    events.sort(
        key=lambda e: (
            e.start,
            kind_order[e.kind],
        )
    )

    return events