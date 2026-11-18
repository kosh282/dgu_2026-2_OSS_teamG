"""롤링 전사의 겹침을 줄이는 콘솔 데모용 임시 후처리."""
from dataclasses import dataclass


@dataclass
class Word:
    start: float
    end: float
    text: str
    seen: int = 1


class OverlapMerger:
    def __init__(self, stable_passes=2, tolerance=0.8):
        self.stable_passes = stable_passes
        self.tolerance = tolerance
        self.previous = []
        self.committed_end = -1.0
        self.parts = []
        self.recent_committed = []
        self.last_pending = ''

    def update(self, words):
        current = []
        previous_index = 0
        for start, end, text in words:
            duplicate = any(old.text.strip() == text.strip()
                            and abs(old.start - start) <= 0.5
                            and abs(old.end - end) <= 0.5
                            for old in self.recent_committed)
            if not text.strip() or start < self.committed_end - 0.2 or duplicate:
                continue
            word = Word(start, end, text)
            # 같은 단어라도 발화 시각이 다른 경우에는 별도 발화로 취급한다.
            for index in range(previous_index, len(self.previous)):
                old = self.previous[index]
                if (old.text.strip() == text.strip()
                        and abs(old.start - start) <= self.tolerance
                        and abs(old.end - end) <= self.tolerance):
                    word.seen = old.seen + 1
                    previous_index = index + 1
                    break
            current.append(word)

        committed = []
        for word in current:
            if word.seen < self.stable_passes:
                break
            committed.append(word.text)
            self.committed_end = max(self.committed_end, word.end)
            self.recent_committed.append(word)
        if committed:
            self.parts.extend(committed)
            self.recent_committed = [word for word in self.recent_committed
                                     if word.end >= self.committed_end - 10]
        self.previous = current[len(committed):]
        if words:
            self.last_pending = ''.join(word.text for word in self.previous).strip()
        return ''.join(committed).strip()

    def final_text(self):
        return ''.join(self.parts).strip()

    def pending_text(self):
        return self.last_pending
