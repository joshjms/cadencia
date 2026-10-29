from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class Event:
    start_s: float
    lane: int
    source_line: int
    source_type: int
    source_x: int
    id: str
    analysis_eligible: bool = True

    def __post_init__(self):
        if self.lane not in range(4):
            raise ValueError("osu!mania events must use lanes 0 through 3")


@dataclass(frozen=True, kw_only=True)
class Hit(Event):
    kind: str = field(default="hit", init=False)


@dataclass(frozen=True, kw_only=True)
class Hold(Event):
    end_s: float
    kind: str = field(default="hold", init=False)

    def __post_init__(self):
        super().__post_init__()
        if self.end_s <= self.start_s:
            raise ValueError("A hold must end after it starts")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s
