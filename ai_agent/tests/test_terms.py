import unittest

from ai_agent import Material, MaterialIndex, MaterialPage, lecture_terms


def index(*pages):
    material = Material("doc", "강의.pdf",
                        tuple(MaterialPage("doc", "강의.pdf", number, text) for number, text in enumerate(pages, 1)))
    return MaterialIndex([material])


class LectureTermsTests(unittest.TestCase):
    def test_repeated_terms_are_ranked_and_particles_removed(self):
        terms = lecture_terms(index(
            "인터페이스는 추상 메서드를 가진다. OnClickListener 인터페이스를 구현한다.",
            "버튼의 onClick 메서드는 OnClickListener 인터페이스에 있다. 추상 클래스와 비교한다.",
            "AppCompatActivity를 상속한다. 인터페이스와 추상 클래스를 함께 사용한다.",
        ))
        self.assertEqual(terms[0], "인터페이스")
        for expected in ("추상", "OnClickListener", "메서드", "클래스"):
            self.assertIn(expected, terms)
        # 한 쪽에만 나온 말과 일반 단어·조사 붙은 형태는 넣지 않는다.
        for unexpected in ("AppCompatActivity", "사용", "인터페이스는", "인터페이스를", "비교"):
            self.assertNotIn(unexpected, terms)

    def test_english_spellings_are_merged(self):
        terms = lecture_terms(index("git commit 실행", "Git commit 기록", "git merge"))
        self.assertEqual([t for t in terms if t.lower() == "git"], ["git"])
        self.assertIn("commit", terms)

    def test_single_page_material_and_limits(self):
        self.assertEqual(lecture_terms(index("트랜잭션 격리 수준")), ["트랜잭션", "격리", "수준"])
        long_terms = lecture_terms(index(*[" ".join(f"Term{i}" for i in range(80))] * 2), max_chars=40)
        self.assertLessEqual(len(", ".join(long_terms)), 40)
        self.assertEqual(lecture_terms(MaterialIndex([])), [])

    def test_nominal_endings_and_generic_words_are_skipped(self):
        terms = lecture_terms(index("객체 간의 관계를 나타냄", "객체 간의 메시지를 나타냄"))
        self.assertIn("객체", terms)
        for unexpected in ("간의", "나타냄"):
            self.assertNotIn(unexpected, terms)


if __name__ == "__main__":
    unittest.main()
