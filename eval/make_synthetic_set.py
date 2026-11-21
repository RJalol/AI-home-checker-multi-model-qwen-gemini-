"""
Builds a SYNTHETIC golden set: fake student homework with a known expected score, for all 9 subjects
of hwchecker. Fully deterministic (fixed seed), so the same command always writes the same set.

    python eval/make_synthetic_set.py                 # writes eval/synthetic/
    python -m hwchecker.benchmarks.golden_eval eval/synthetic/golden_synthetic.jsonl

HOW THE EXPECTED SCORE IS MADE
  Every homework has 5 questions. Each question gets an outcome with a fixed score:
      correct 100 | no_working 90 | unit_format 85 | partial 50 | wrong 0 | blank 0
  The homework's expected score is the rounded mean. This is "expected by construction", NOT a teacher's
  judgment. Use it to find crashes, format problems, subjects that fail, prompt-injection problems and big
  systematic errors. Do not use it to prove the grader is as good as a human teacher: that needs the real
  golden set (docs/EVALUATION.md).

WHAT IS COVERED
  9 subjects x 8 student profiles (strong, average, weak, careless, blank, copied question only, ...),
  DOCX files and photo-like JPEGs (rotation, noise, blur), answer key given and not given,
  adversarial cases (a correct answer without working, a unit-format slip, a sign slip, an alternative
  valid method, a wrong answer written very confidently), plus the 20 prompt-injection files.
  Audio is NOT covered.
"""

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

SEED = 20261121
QUESTIONS_PER_HOMEWORK = 5
SCORES = {"correct": 100, "no_working": 90, "unit_format": 85, "partial": 50, "wrong": 0, "blank": 0}


@dataclass
class Item:
    """One question with its gold answer and prepared student answers for each outcome."""
    prompt: str
    gold: str
    answers: Dict[str, str]          # outcome -> the text the student writes
    note: str = ""


# ---------------------------------------------------------------- parametric items (computed gold)

def _wrong_number(rng: random.Random, value: float) -> float:
    shift = rng.choice([-2, -1, 1, 2, 3])
    candidate = value + shift
    return candidate if candidate != value else value + 1


def math_item(rng: random.Random, kind: int) -> Item:
    if kind == 0:   # linear equation
        x = rng.randint(2, 12)
        a, b = rng.randint(2, 7), rng.randint(1, 9)
        c = a * x + b
        return Item(f"Solve: {a}x + {b} = {c}", f"x = {x}", {
            "correct": f"{a}x = {c} - {b} = {c - b}, x = {x}",
            "no_working": f"x = {x}",
            "partial": f"{a}x = {c - b}, x = {(c - b)}/{a + 1}",
            "wrong": f"x = {_wrong_number(rng, x)}",
            "blank": "",
        })
    if kind == 1:   # quadratic with integer roots
        r1, r2 = rng.randint(1, 6), rng.randint(-6, -1)
        b, c = -(r1 + r2), r1 * r2
        sign_b = f"+ {b}" if b >= 0 else f"- {-b}"
        sign_c = f"+ {c}" if c >= 0 else f"- {-c}"
        return Item(f"Solve: x^2 {sign_b}x {sign_c} = 0", f"x = {r1} and x = {r2}", {
            "correct": f"D = {b * b - 4 * c}, sqrt(D) = {abs(r1 - r2)}; x1 = {r1}, x2 = {r2}",
            "no_working": f"x = {r1}, x = {r2}",
            "partial": f"D = {b * b - 4 * c}; x1 = {r1}",          # second root missing
            "wrong": f"x = {-r1} and x = {-r2}",                     # roots with wrong signs
            "blank": "",
        })
    if kind == 2:   # percent
        base, pct = rng.choice([80, 120, 200, 250, 400]), rng.choice([10, 15, 20, 25, 30])
        gold = base * pct // 100
        return Item(f"Find {pct}% of {base}.", str(gold), {
            "correct": f"{base} x {pct}/100 = {gold}",
            "no_working": str(gold),
            "partial": f"{base} x {pct} = {base * pct}",             # forgot to divide by 100
            "wrong": str(base - gold),
            "blank": "",
        })
    if kind == 3:   # rectangle area / perimeter
        w, h = rng.randint(3, 9), rng.randint(10, 15)
        return Item(f"A rectangle has sides {w} cm and {h} cm. Find its area and perimeter.",
                    f"area {w * h} cm^2, perimeter {2 * (w + h)} cm", {
            "correct": f"S = {w} x {h} = {w * h} cm^2; P = 2({w}+{h}) = {2 * (w + h)} cm",
            "no_working": f"S = {w * h} cm^2, P = {2 * (w + h)} cm",
            "unit_format": f"S = {w * h}, P = {2 * (w + h)}",       # right numbers, no units
            "partial": f"S = {w * h} cm^2; P = {w + h} cm",
            "wrong": f"S = {2 * (w + h)} cm^2, P = {w * h} cm",
            "blank": "",
        })
    # kind 4: derivative of a polynomial
    a, b = rng.randint(2, 6), rng.randint(2, 5)
    return Item(f"Find the derivative of f(x) = {a}x^{b} + {b}x.", f"f'(x) = {a * b}x^{b - 1} + {b}", {
        "correct": f"f'(x) = {a}*{b}x^{b - 1} + {b} = {a * b}x^{b - 1} + {b}",
        "no_working": f"{a * b}x^{b - 1} + {b}",
        "partial": f"{a * b}x^{b - 1}",                                # lost the second term
        "wrong": f"{a}x^{b - 1} + {b}x",
        "blank": "",
    })


def physics_item(rng: random.Random, kind: int) -> Item:
    if kind == 0:   # s = v t
        v, t = rng.choice([40, 60, 72, 90]), rng.choice([1.5, 2, 2.5, 3])
        gold = v * t
        gold_s = f"{gold:g} km"
        return Item(f"A car moves at {v} km/h for {t:g} h. Find the distance.", gold_s, {
            "correct": f"s = v t = {v} x {t:g} = {gold:g} km",
            "no_working": f"{gold:g} km",
            "unit_format": f"s = {v} x {t:g} = {gold:g}",
            "partial": f"s = {v} / {t:g} = {v / t:g} km",
            "wrong": f"{_wrong_number(rng, gold):g} km",
            "blank": "",
        })
    if kind == 1:   # F = m a
        m, a = rng.randint(2, 9), rng.randint(2, 6)
        return Item(f"A body of mass {m} kg accelerates at {a} m/s^2. Find the force.", f"{m * a} N", {
            "correct": f"F = m a = {m} x {a} = {m * a} N",
            "no_working": f"{m * a} N",
            "unit_format": f"F = {m} x {a} = {m * a} J",             # wrong unit
            "partial": f"F = m / a = {m / a:.2f} N",
            "wrong": f"{m + a} N",
            "blank": "",
        })
    if kind == 2:   # Ohm's law
        r, i = rng.randint(2, 10), rng.randint(2, 6)
        return Item(f"A resistor of {r} ohm carries {i} A. Find the voltage.", f"{r * i} V", {
            "correct": f"U = I R = {i} x {r} = {r * i} V",
            "no_working": f"{r * i} V",
            "partial": f"U = R / I = {r / i:.2f} V",
            "wrong": f"{r + i} V",
            "blank": "",
        })
    if kind == 3:   # Q = c m dT
        m, t1, t2 = rng.choice([1, 2, 3]), rng.choice([10, 20, 25]), rng.choice([60, 70, 80])
        q = 4200 * m * (t2 - t1)
        return Item(f"How much heat is needed to warm {m} kg of water from {t1} C to {t2} C? c = 4200 J/(kg C). "
                    "Answer in kJ.", f"{q / 1000:g} kJ", {
            "correct": f"Q = c m dT = 4200 x {m} x {t2 - t1} = {q} J = {q / 1000:g} kJ",
            "no_working": f"{q / 1000:g} kJ",
            "unit_format": f"Q = 4200 x {m} x {t2 - t1} = {q} J",    # right value, kJ asked
            "partial": f"Q = 4200 x {m} x {t2} = {4200 * m * t2} J",  # final temperature instead of dT
            "wrong": f"{m * (t2 - t1)} kJ",
            "blank": "",
        })
    # kind 4: work against friction (sign)
    m, mu = rng.choice([2, 4, 5]), rng.choice([0.2, 0.25, 0.5])
    v = rng.choice([4, 6, 10])
    ek = m * v * v / 2
    s = ek / (mu * m * 10)
    return Item(f"A {m} kg body slides at {v} m/s and stops because of friction (mu = {mu}, g = 10). Find the "
                f"work done by friction.", f"-{ek:g} J", {
        "correct": f"A = -Ek = -{m} x {v}^2 / 2 = -{ek:g} J (friction opposes the motion)",
        "no_working": f"-{ek:g} J",
        "unit_format": f"A = {ek:g} J",                              # sign slip, still mostly right
        "partial": f"Ek = {m} x {v}^2 = {m * v * v} J, A = {m * v * v} J",
        "wrong": f"{mu * m * 10:g} J",
        "blank": "",
    }, note=f"stopping distance {s:g} m")


CHEM_MASSES = {"H2O": 18, "CO2": 44, "NaCl": 58.5, "H2SO4": 98, "CaCO3": 100, "NaOH": 40}


def chemistry_item(rng: random.Random, kind: int) -> Item:
    if kind == 0:   # molar mass
        formula = rng.choice(list(CHEM_MASSES))
        gold = CHEM_MASSES[formula]
        return Item(f"Find the molar mass of {formula} (g/mol).", f"{gold:g} g/mol", {
            "correct": f"M({formula}) = {gold:g} g/mol",
            "no_working": f"{gold:g}",
            "unit_format": f"{gold:g} g",
            "partial": f"M({formula}) = {gold + 2:g} g/mol",
            "wrong": f"{gold * 2:g} g/mol",
            "blank": "",
        })
    if kind == 1:   # n = m / M
        formula = rng.choice(["NaOH", "CaCO3", "H2SO4"])
        mm = CHEM_MASSES[formula]
        n = rng.choice([0.5, 2, 3])
        mass = mm * n
        return Item(f"How many moles are in {mass:g} g of {formula}?", f"{n:g} mol", {
            "correct": f"n = m / M = {mass:g} / {mm:g} = {n:g} mol",
            "no_working": f"{n:g} mol",
            "partial": f"n = M / m = {mm / mass:.2f} mol",
            "wrong": f"{mass * mm:g} mol",
            "blank": "",
        })
    if kind == 2:   # balancing
        equation, balanced, wrong = rng.choice([
            ("H2 + O2 -> H2O", "2H2 + O2 -> 2H2O", "H2 + O2 -> H2O"),
            ("Fe + O2 -> Fe2O3", "4Fe + 3O2 -> 2Fe2O3", "2Fe + 3O2 -> Fe2O3"),
            ("Al + HCl -> AlCl3 + H2", "2Al + 6HCl -> 2AlCl3 + 3H2", "Al + 3HCl -> AlCl3 + H2"),
            ("CH4 + O2 -> CO2 + H2O", "CH4 + 2O2 -> CO2 + 2H2O", "CH4 + O2 -> CO2 + 2H2O"),
        ])
        return Item(f"Balance the equation: {equation}", balanced, {
            "correct": balanced,
            "no_working": balanced,
            "partial": wrong,
            "wrong": equation,
            "blank": "",
        })
    if kind == 3:   # facts
        question, gold, wrong = rng.choice([
            ("What is the pH of a neutral solution at 25 C?", "7", "0"),
            ("Which gas is released when zinc reacts with hydrochloric acid?", "hydrogen (H2)", "oxygen (O2)"),
            ("What is the chemical formula of table salt?", "NaCl", "KCl"),
            ("What is the atomic number of carbon?", "6", "12"),
        ])
        return Item(question, gold, {"correct": gold, "no_working": gold, "partial": gold + " (no explanation)",
                                     "wrong": wrong, "blank": ""})
    # kind 4: percent by mass
    solute, solution = rng.choice([(20, 200), (15, 150), (30, 300), (12, 240)])
    gold = solute / solution * 100
    return Item(f"{solute} g of salt is dissolved to make {solution} g of solution. Find the mass fraction (%).",
                f"{gold:g} %", {
        "correct": f"w = {solute} / {solution} x 100 = {gold:g} %",
        "no_working": f"{gold:g} %",
        "unit_format": f"w = {solute} / {solution} = {solute / solution:g}",
        "partial": f"w = {solute} / ({solution - solute}) x 100 = {solute / (solution - solute) * 100:.1f} %",
        "wrong": f"{solution - solute} %",
        "blank": "",
    })


# ---------------------------------------------------------------- fixed knowledge banks
# (prompt, gold, wrong answer, partial answer). Only facts that are certain.

BANKS: Dict[str, List[tuple]] = {
    "biology": [
        ("Which organelle produces most of the cell's ATP?", "mitochondria", "nucleus", "an organelle in the cell"),
        ("What is the process by which plants make glucose from light?", "photosynthesis", "respiration", "plants use light"),
        ("How many chromosomes does a normal human body cell have?", "46", "23", "23 pairs"),
        ("Name the molecule that carries genetic information.", "DNA", "ATP", "a nucleic acid"),
        ("Which blood cells carry oxygen?", "red blood cells (erythrocytes)", "white blood cells", "blood cells"),
        ("What is the basic structural and functional unit of living things?", "the cell", "the organ", "something small"),
        ("Which gas do plants take in during photosynthesis?", "carbon dioxide (CO2)", "oxygen (O2)", "a gas from air"),
        ("Where does photosynthesis take place in a plant cell?", "in the chloroplasts", "in the mitochondria", "in the leaf"),
        ("What do we call organisms that make their own food?", "autotrophs (producers)", "heterotrophs", "plants maybe"),
        ("Which part of the cell contains the DNA in humans?", "the nucleus", "the ribosome", "inside the cell"),
        ("What is the function of ribosomes?", "protein synthesis", "energy storage", "they work in the cell"),
        ("Which vitamin is produced in skin under sunlight?", "vitamin D", "vitamin C", "a vitamin"),
    ],
    "history": [
        ("In which year did Amir Temur die?", "1405", "1250", "the early 15th century"),
        ("In which year did Uzbekistan declare independence?", "1991 (1 September)", "1985", "in the 1990s"),
        ("When was the Constitution of the Republic of Uzbekistan adopted?", "8 December 1992", "1 September 1991", "in 1992"),
        ("Which city was the capital of the Timurid state?", "Samarkand", "Bukhara", "a city in Mawarannahr"),
        ("Who built an observatory in Samarkand in the 15th century?", "Mirzo Ulugbek", "Alisher Navoi", "a Timurid ruler"),
        ("In which year did the Second World War end?", "1945", "1939", "in the 1940s"),
        ("In which year did the French Revolution begin?", "1789", "1815", "at the end of the 18th century"),
        ("Who founded the Mughal Empire in India?", "Zahiriddin Muhammad Babur", "Akbar", "a descendant of Temur"),
        ("Who is the author of the epic 'Khamsa' among Chagatai poets?", "Alisher Navoi", "Mirzo Ulugbek", "a poet"),
        ("In which year did Temur defeat the Ottoman sultan Bayezid at Ankara?", "1402", "1453", "around 1400"),
        ("Which famous trade route connected China with the Mediterranean?", "the Great Silk Road", "the Amber Road", "a trade route"),
        ("In which year was the Battle of Panipat that opened the Mughal era?", "1526", "1206", "in the 16th century"),
    ],
    "english": [
        ("Fill in: She ___ to school every day. (go)", "goes", "go", "going"),
        ("Past simple of 'buy':", "bought", "buyed", "buy"),
        ("Plural of 'child':", "children", "childs", "childen"),
        ("Comparative form of 'good':", "better", "gooder", "more good"),
        ("Fill in: If I ___ rich, I would travel the world.", "were", "am", "was maybe"),
        ("Fill in: They have ___ their homework already. (finish)", "finished", "finish", "finishing"),
        ("Choose the correct article: ___ apple a day keeps the doctor away.", "An", "A", "The"),
        ("Fill in: He is taller ___ his brother.", "than", "then", "that"),
        ("Past participle of 'write':", "written", "wrote", "writed"),
        ("Fill in: There ___ two books on the table.", "are", "is", "be"),
        ("Opposite of 'ancient':", "modern", "old", "new maybe"),
        ("Fill in: I ___ watching TV when you called. (be, past continuous)", "was", "were", "am"),
    ],
    "russian": [
        ("Инфинитив глагола «иду»:", "идти", "идут", "ходить"),
        ("Поставьте существительное «город» во множественное число:", "города", "городы", "городи"),
        ("Антоним к слову «большой»:", "маленький", "высокий", "хороший"),
        ("Прошедшее время женского рода от «читать»:", "читала", "читал", "читали"),
        ("Родительный падеж слова «книга»:", "книги", "книгу", "книге"),
        ("Винительный падеж слова «мама»:", "маму", "мамы", "маме"),
        ("Сколько гласных в слове «окно»?", "две (о, о)", "три", "одна"),
        ("Множественное число слова «стол»:", "столы", "столя", "столов"),
        ("Какой частью речи является слово «быстро»?", "наречие", "прилагательное", "глагол"),
        ("Образуйте от «писать» форму 3 лица ед. числа настоящего времени:", "пишет", "писал", "пишут"),
        ("Синоним к слову «друг»:", "товарищ", "враг", "сосед"),
        ("Укажите ударный слог в слове «магазин»:", "последний (-зин)", "первый", "второй"),
    ],
    "uzbek": [
        ("«kitob» so'zining ko'plik shaklini yozing:", "kitoblar", "kitobi", "kitobla"),
        ("«maktab» so'ziga jo'nalish kelishigi qo'shimchasini qo'shing:", "maktabga", "maktabda", "maktabni"),
        ("«katta» so'zining antonimini yozing:", "kichik", "baland", "uzun"),
        ("«men» olmoshi bilan «kitob» so'zini egalik shaklida yozing:", "kitobim", "kitobing", "kitobi"),
        ("O'zbekistonning poytaxti qaysi shahar?", "Toshkent", "Samarqand", "Buxoro"),
        ("«uy» so'ziga o'rin-payt kelishigi qo'shimchasini qo'shing:", "uyda", "uyga", "uydan"),
        ("«bolalar» so'zining birlik shaklini yozing:", "bola", "bolam", "bolaga"),
        ("«yaxshi» so'zining antonimini yozing:", "yomon", "zo'r", "chiroyli"),
        ("«o'qimoq» fe'lining hozirgi zamon 1-shaxs birlik shaklini yozing:", "o'qiyman", "o'qidim", "o'qiydi"),
        ("«daryo» so'ziga chiqish kelishigi qo'shimchasini qo'shing:", "daryodan", "daryoga", "daryoda"),
        ("Gapda nechta so'z bor: «Men maktabga boraman»?", "3", "2", "4"),
        ("«olma» so'ziga tushum kelishigi qo'shimchasini qo'shing:", "olmani", "olmaga", "olmada"),
    ],
    "turkish": [
        ("«kitap» kelimesinin çoğul hâlini yazın:", "kitaplar", "kitapler", "kitabı"),
        ("«ev» kelimesine bulunma (-de) hâli ekini ekleyin:", "evde", "eve", "evden"),
        ("«büyük» kelimesinin zıt anlamlısı:", "küçük", "uzun", "geniş"),
        ("«okul» kelimesine yönelme (-e) hâli ekini ekleyin:", "okula", "okulda", "okuldan"),
        ("Türkiye'nin başkenti neresidir?", "Ankara", "İstanbul", "İzmir"),
        ("«ben» için «gelmek» fiilini şimdiki zamanda çekimleyin:", "geliyorum", "geldim", "gelirim"),
        ("«göz» kelimesinin çoğul hâlini yazın:", "gözler", "gözlar", "gözü"),
        ("«ben» için «kitap» kelimesinin iyelik hâli:", "kitabım", "kitabın", "kitabı"),
        ("«gitmek» fiilinin geçmiş zaman, 3. tekil kişi hâli:", "gitti", "gidiyor", "gidecek"),
        ("«hızlı» kelimesinin zıt anlamlısı:", "yavaş", "güzel", "kısa"),
        ("«masa» kelimesine belirtme (-ı/-i) hâli ekini ekleyin:", "masayı", "masaya", "masada"),
        ("«ev» kelimesine ayrılma (-den) hâli ekini ekleyin:", "evden", "evde", "eve"),
    ],
}

GENERATORS: Dict[str, Callable[[random.Random, int], Item]] = {
    "math": math_item, "physics": physics_item, "chemistry": chemistry_item,
}


def bank_item(subject: str, row: tuple) -> Item:
    prompt, gold, wrong, partial = row
    return Item(prompt, gold, {"correct": gold, "no_working": gold, "partial": partial, "wrong": wrong, "blank": ""})


def make_items(rng: random.Random, subject: str, kinds: List[int], offset: int) -> List[Item]:
    if subject in GENERATORS:
        return [GENERATORS[subject](rng, k) for k in kinds]
    bank = BANKS[subject]
    return [bank_item(subject, bank[(offset + i) % len(bank)]) for i in range(len(kinds))]


# ---------------------------------------------------------------- student profiles

# Each profile: list of 5 outcomes (None = chosen so the profile name matches the intent).
PROFILES = {
    "strong":        ["correct", "correct", "correct", "correct", "correct"],
    "strong_slip":   ["correct", "correct", "unit_format", "correct", "correct"],
    "average":       ["correct", "correct", "partial", "wrong", "correct"],
    "weak":          ["wrong", "partial", "wrong", "correct", "wrong"],
    "answers_only":  ["no_working", "no_working", "no_working", "no_working", "no_working"],
    "careless":      ["correct", "wrong", "correct", "partial", "blank"],
    "blank":         ["blank", "blank", "blank", "blank", "blank"],
    "all_wrong":     ["wrong", "wrong", "wrong", "wrong", "wrong"],
}


def outcome_text(item: Item, outcome: str, fallback: str) -> tuple:
    """Returns (final outcome, text). If an item has no text for the outcome, fall back to a close one."""
    if outcome in item.answers:
        return outcome, item.answers[outcome]
    if outcome == "unit_format":
        return "correct", item.answers["correct"]
    return fallback, item.answers[fallback]


# ---------------------------------------------------------------- rendering

def render_docx(path: Path, title: str, rows: List[tuple]) -> None:
    from docx import Document

    document = Document()
    document.add_paragraph(title)
    for number, prompt, answer in rows:
        document.add_paragraph(f"{number}. {prompt}")
        document.add_paragraph(f"Answer: {answer}" if answer else "Answer: ")
    document.save(path)


def render_png(path: Path, rows: List[tuple], rng: random.Random, show_prompts: bool) -> None:
    """A photo-like page: ruled paper, slightly rotated, uneven baseline, noise, blur."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    import numpy as np

    font_path = next((p for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if Path(p).exists()), None)
    font = ImageFont.truetype(font_path, 30) if font_path else ImageFont.load_default()
    small = ImageFont.truetype(font_path, 22) if font_path else ImageFont.load_default()

    width, line_h = 1500, 56
    lines: List[tuple] = []
    for number, prompt, answer in rows:
        if show_prompts:
            lines.append((f"{number}) {prompt}", small, (60, 60, 60)))
            lines.append((f"    {answer}", font, (20, 30, 110)))
        else:
            lines.append((f"{number}) {answer}", font, (20, 30, 110)))
    height = line_h * (len(lines) + 2)
    page = Image.new("RGB", (width, height), (246, 244, 236))
    draw = ImageDraw.Draw(page)
    for y in range(line_h, height, line_h):
        draw.line([(0, y), (width, y)], fill=(190, 205, 225), width=2)
    draw.line([(90, 0), (90, height)], fill=(220, 120, 120), width=2)
    for index, (text, face, colour) in enumerate(lines):
        y = line_h * (index + 1) - 40 + rng.randint(-3, 3)
        draw.text((110 + rng.randint(-4, 6), y), text[:120], font=face, fill=colour)
    page = page.rotate(rng.uniform(-2.5, 2.5), expand=True, fillcolor=(236, 233, 224))
    array = np.asarray(page).astype("int16")
    array += np.random.default_rng(rng.randint(0, 10 ** 6)).integers(-9, 10, array.shape, dtype="int16")
    page = Image.fromarray(np.clip(array, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(0.8))
    page.save(path, quality=72)


# ---------------------------------------------------------------- the set

SUBJECTS = ["math", "physics", "chemistry", "biology", "history", "english", "russian", "uzbek", "turkish"]
PROFILE_ORDER = list(PROFILES)          # 8 profiles per subject
INJECTION_SUBJECTS_NOTE = "the 20 injection cases come from eval/make_injection_set.py"


def build(out_dir: Path) -> dict:
    rng = random.Random(SEED)
    files_dir = out_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    cases: List[dict] = []
    detail: List[dict] = []
    counter = 0

    for subject in SUBJECTS:
        for p_index, profile in enumerate(PROFILE_ORDER):
            counter += 1
            case_id = f"syn-{subject[:4]}-{p_index + 1:02d}"
            kinds = [0, 1, 2, 3, 4] if subject in GENERATORS else list(range(5))
            if subject in GENERATORS:
                rng.shuffle(kinds)
            items = make_items(rng, subject, kinds, offset=(p_index * 3) % 12)
            outcomes = list(PROFILES[profile])
            used, rows, scores, expected_rows = [], [], [], []
            for number, (item, wanted) in enumerate(zip(items, outcomes), 1):
                outcome, text = outcome_text(item, wanted, "correct" if wanted == "no_working" else "wrong")
                used.append(outcome)
                scores.append(SCORES[outcome])
                rows.append((number, item.prompt, text))
                expected_rows.append({"q": number, "prompt": item.prompt, "gold": item.gold,
                                      "student_answer": text, "outcome": outcome, "score": SCORES[outcome]})
            expected = round(sum(scores) / len(scores))
            use_png = p_index % 5 in (1, 3)   # photo files are JPEG                 # about 40% are photos
            with_key = p_index % 2 == 0                     # half the cases carry an answer key
            ext = "jpg" if use_png else "docx"
            file_name = f"{case_id}.{ext}"
            if use_png:
                render_png(files_dir / file_name, rows, random.Random(SEED + counter), show_prompts=(p_index % 4 == 1))
            else:
                render_docx(files_dir / file_name, f"{subject} homework - student {profile}", rows)

            task_text = "\n".join(f"{n}. {item.prompt}" for n, item in enumerate(items, 1))
            answer_key = "\n".join(f"{n}. {item.gold}" for n, item in enumerate(items, 1)) if with_key else ""
            tags = ["synthetic", profile, ext, "with_key" if with_key else "no_key"]
            if "no_working" in used:
                tags.append("adversarial_no_working")
            if "unit_format" in used:
                tags.append("adversarial_unit_or_sign")
            if profile == "blank":
                tags.append("empty_answers")
            cases.append({"id": case_id, "subject_id": subject, "file": f"files/{file_name}",
                          "teacher_score": expected, "tags": tags, "task_text": task_text,
                          "answer_key": answer_key})
            detail.append({"id": case_id, "subject_id": subject, "profile": profile, "expected_score": expected,
                           "questions": expected_rows})

    (out_dir / "golden_synthetic.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    (out_dir / "expected_details.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"cases": len(cases), "subjects": len(SUBJECTS)}


def main(argv: Optional[List[str]] = None) -> int:
    out_dir = Path(__file__).resolve().parent / "synthetic"
    summary = build(out_dir)
    print(f"wrote {summary['cases']} cases for {summary['subjects']} subjects to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
