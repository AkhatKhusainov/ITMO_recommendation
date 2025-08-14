Чат‑бот для ответов по двум программам магистратуры ИТМО:
- Управление ИИ‑продуктами/AI Product — https://abit.itmo.ru/program/master/ai_product
- Искусственный интеллект — https://abit.itmo.ru/program/master/ai

Особенности:
1) Парсит страницы, вытаскивает ключевые поля (форма, длительность, язык, стоимость, контакты и т. п.), FAQ и ссылки.
2) Индексирует полный текст страниц (TF‑IDF) для робастного поиска ответов.
3) Комбинирует rule‑based ответы (на типовые вопросы) с полнотекстовым поиском.
4) FastAPI‑сервис с эндпоинтом /chat и Swagger UI.

Запуск:
    pip install fastapi "uvicorn[standard]" requests beautifulsoup4 lxml scikit-learn
    python itmo_bot.py
    # Открыть: http://localhost:8000/docs

Дополнительно:
- Код рассчитан на русскоязычные запросы; работает офлайн (без внешних LLM).
- При желании LLM легко подключается в функции `answer_with_llm` (заглушка нижe),
  где можно отправлять релевантные фрагменты в любую модель.

from __future__ import annotations

import re
import json
import time
import math
import html
from typing import Dict, List, Tuple, Optional

import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI
from pydantic import BaseModel

# === Конфигурация ===
PROGRAM_URLS = {
    "ai_product": "https://abit.itmo.ru/program/master/ai_product",
    "ai": "https://abit.itmo.ru/program/master/ai",
}

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
}

# === Утилиты парсинга ===

def fetch(url: str, timeout: int = 30) -> str:
    resp = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def text_tokens(s: str) -> List[str]:
    return [t.strip() for t in s.splitlines() if t and t.strip()]

LABELS = [
    "форма обучения",
    "длительность",
    "язык обучения",
    "стоимость контрактного обучения (год)",
    "общежитие",
    "военный учебный центр",
    "гос. аккредитация",
    "дополнительные возможности",
]

LABELS_LOWER = [l.lower() for l in LABELS]


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")
MONEY_RE = re.compile(r"\d[\d\s\u00A0]*₽")


class ProgramData(BaseModel):
    key: str
    url: str
    title: str
    institute: Optional[str] = None
    fields: Dict[str, str] = {}
    manager_name: Optional[str] = None
    manager_email: Optional[str] = None
    manager_phone: Optional[str] = None
    socials: Dict[str, str] = {}
    directions_raw: Optional[str] = None
    about: Optional[str] = None
    faq: List[Tuple[str, str]] = []  # (q,a)
    links: Dict[str, str] = {}  # важные ссылки по тексту
    full_text: str = ""


def parse_program(url_key: str, url: str) -> ProgramData:
    html_text = fetch(url)
    soup = BeautifulSoup(html_text, "lxml")

    # Заголовок
    h1 = soup.find(["h1", "h2"]) or soup.find("title")
    title = h1.get_text(strip=True) if h1 else url

    # Институт
    institute_link = soup.find("a", href=re.compile(r"itmo\.ru"))
    institute = institute_link.get_text(strip=True) if institute_link else None

    # Все текстовые токены
    tokens = text_tokens(soup.get_text("\n"))

    # Словарь полей по меткам
    fields: Dict[str, str] = {}
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in LABELS_LOWER:
            # брать следующий непустой токен как значение
            val = None
            for j in range(i + 1, min(i + 6, len(tokens))):
                cand = tokens[j]
                if cand.strip() and cand.lower() not in LABELS_LOWER:
                    val = cand.strip()
                    break
            if val:
                fields[tok] = val

    # Менеджер: ищем блок около слова "Менеджер программы"
    manager_block = None
    for i, tok in enumerate(tokens):
        if tok.strip().lower().startswith("менеджер программы"):
            manager_block = "\n".join(tokens[i : i + 15])
            break
    if manager_block is None:
        manager_block = "\n".join(tokens[:120])  # на всякий случай

    manager_email = None
    manager_phone = None
    manager_name = None

    emails = EMAIL_RE.findall(manager_block)
    if emails:
        manager_email = emails[0]

    phones = PHONE_RE.findall(manager_block)
    if phones:
        manager_phone = phones[0]

    # Имя менеджера — первая строка после заголовка блока, похожая на ФИО
    if manager_block:
        mb_lines = [l for l in manager_block.splitlines() if l.strip()]
        # ищем строку, которая содержит пробелы (минимум 2 слова) и не email/phone
        for l in mb_lines[1:6]:
            if EMAIL_RE.search(l) or PHONE_RE.search(l):
                continue
            if len(l.split()) >= 2:
                manager_name = l.strip()
                break

    # Соцсети (ищем ссылки с доменами vk.com, t.me, ai.itmo.ru и т.д.)
    socials: Dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        txt = a.get_text(strip=True)
        if any(dom in href for dom in ["vk.com", "t.me", "ai.itmo.ru", "alfabank.ru"]):
            socials[txt or href] = href

    # Направления подготовки — возьмем кусок текста от заголовка до следующего раздела
    directions_raw = None
    full_page_text = soup.get_text("\n")
    m = re.search(r"направления подготовки[\s\S]+?(?=##|#|\Z)", full_page_text, re.IGNORECASE)
    if m:
        directions_raw = m.group(0)

    # О программе — аналогично
    about = None
    m2 = re.search(r"о программе[\s\S]+?(?=##|#|\Z)", full_page_text, re.IGNORECASE)
    if m2:
        about = re.sub(r"\n{2,}", "\n\n", m2.group(0)).strip()

    # FAQ — берём пары последовательных строк «вопрос/ответ» из секции «Часто задаваемые вопросы»
    faq: List[Tuple[str, str]] = []
    faq_chunk = None
    m3 = re.search(r"Часто задаваемые вопросы[\s\S]+", full_page_text, re.IGNORECASE)
    if m3:
        faq_chunk = m3.group(0)
        lines = [l.strip() for l in faq_chunk.splitlines() if l.strip()]
        q = None
        for ln in lines:
            # эври��тика: вопросы заканчиваются на '?' или начинаются с '#####'
            if ln.endswith("?") or ln.lower().startswith("#####"):
                if q:
                    q = ln  # новый вопрос, старый пропустим если ответа не было
                else:
                    q = ln.lstrip('#').strip()
            else:
                if q:
                    faq.append((q, ln))
                    q = None

    # Полный текст — пригодится для поиска
    full_text = re.sub(r"\n{2,}", "\n\n", full_page_text)

    # Важные ссылки (вопросы к экзамену, учебный план)
    links: Dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        t = a.get_text(" ", strip=True).lower()
        if "вопросы для вступительного экзамена" in t or "вопросы для вступ" in t:
            links["exam_questions"] = a["href"]
        if "скачать учебный план" in t or "учебный план" == t:
            links["curriculum"] = a["href"]
        if "подать документы" in t:
            links["apply"] = a["href"]

    return ProgramData(
        key=url_key,
        url=url,
        title=title,
        institute=institute,
        fields=fields,
        manager_name=manager_name,
        manager_email=manager_email,
        manager_phone=manager_phone,
        socials=socials,
        directions_raw=directions_raw,
        about=about,
        faq=faq,
        links=links,
        full_text=full_text,
    )


# === Индексация (TF‑IDF) ===
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class KBIndex:
    def __init__(self, docs: List[Tuple[str, str]]):
        self.docs = docs  # (doc_id, text)
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            stop_words=None,
        )
        self.matrix = self.vectorizer.fit_transform([d[1] for d in docs])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        idxs = sims.argsort()[::-1][:top_k]
        return [(self.docs[i][0], float(sims[i])) for i in idxs]


# === Rule‑based ответы на типовые вопросы ===

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def make_rule_answers(pd: ProgramData) -> Dict[str, str]:
    f = {norm(k): v for k, v in pd.fields.items()}

    answers = {}

    # Стоимость
    cost = None
    for k, v in pd.fields.items():
        if "стоимость" in k.lower():
            cost = v
    if not cost:
        m = MONEY_RE.search(pd.full_text)
        if m:
            cost = m.group(0)
    if cost:
        answers["стоимость"] = f"Стоимость обучения: {cost}."

    # Форма, длительность, язык
    if any("форма обучения" in k.lower() for k in pd.fields):
        answers["форма обучения"] = f.get("форма обучения", "")
    if any("длительность" in k.lower() for k in pd.fields):
        answers["длительность"] = f.get("длительность", "")
    if any("язык обучения" in k.lower() for k in pd.fields):
        answers["язык обучения"] = f.get("язык обучения", "")

    # Общежитие/воен. центр/аккредитация
    if any("общежитие" in k.lower() for k in pd.fields):
        answers["общежитие"] = f.get("общежитие", "")
    if any("военный учебный центр" in k.lower() for k in pd.fields):
        answers["военный учебный центр"] = f.get("военный учебный центр", "")
    if any("гос. аккредитация" in k.lower() for k in pd.fields):
        answers["аккредитация"] = f.get("гос. аккредитация", "")

    # Контакты менеджера
    contact_bits = []
    if pd.manager_name:
        contact_bits.append(pd.manager_name)
    if pd.manager_email:
        contact_bits.append(pd.manager_email)
    if pd.manager_phone:
        contact_bits.append(pd.manager_phone)
    if contact_bits:
        answers["контакты"] = "Менеджер программы: " + ", ".join(contact_bits)

    # Учебный план, экзамен
    if pd.links.get("curriculum"):
        answers["учебный план"] = f"Учебный план: {pd.links['curriculum']}"
    if pd.links.get("exam_questions"):
        answers["вопросы экзамена"] = f"Вопросы для вступительного экзамена: {pd.links['exam_questions']}"
    if pd.links.get("apply"):
        answers["подача документов"] = f"Подать документы: {pd.links['apply']}"

    return answers


def rule_router(user_q: str, pd: ProgramData, rule_answers: Dict[str, str]) -> Optional[str]:
    q = norm(user_q)

    # Карта ключевых слов -> ключ ответа
    rules = [
        ("сколько стоит|стоимость|цена|платн", "стоимость"),
        ("форма|формат|очно|заочно|онлайн", "форма обучения"),
        ("сколько учить|длительно|срок", "длительность"),
        ("язык|english|русск", "язык обучения"),
        ("общежит|общаг", "общежитие"),
        ("военн|военком|учебн.*центр", "военный учебный центр"),
        ("аккредитац", "аккредитация"),
        ("контакт|менеджер|почта|телефон", "контакты"),
        ("учебн.*план|curriculum|силабус|syllabus", "учебный план"),
        ("экзамен|вступительн|вопрос.*экзам", "вопросы экзамена"),
        ("подать|подач.*документ|заявлен", "подача документов"),
    ]

    for pattern, key in rules:
        if re.search(pattern, q):
            ans = rule_answers.get(key)
            if ans:
                return ans
    return None


# === Комбинированный движок ответов ===

def build_kb(programs: List[ProgramData]) -> Tuple[KBIndex, Dict[str, ProgramData], Dict[str, Dict[str, str]]]:
    # Документы: разбиваем текст на абзацы/кусочки, чтобы поиск был точнее
    docs = []
    for pd in programs:
        chunks = [c.strip() for c in re.split(r"\n\n+", pd.full_text) if c.strip()]
        for i, ch in enumerate(chunks):
            doc_id = f"{pd.key}#para{i}"
            docs.append((doc_id, ch))
    index = KBIndex(docs)

    rule_map = {pd.key: make_rule_answers(pd) for pd in programs}
    pd_map = {pd.key: pd for pd in programs}
    return index, pd_map, rule_map


def pick_program(user_q: str, pd_map: Dict[str, ProgramData]) -> ProgramData:
    q = norm(user_q)
    if any(w in q for w in ["product", "продакт", "аiproduct", "управлен", "product manager"]):
        return pd_map["ai_product"]
    if any(w in q for w in ["искусствен", "ml", "ai", "машинн", "data", "aitalents"]):
        return pd_map["ai"]
    # по умолчанию — Искусственный интеллект (как более широкая)
    return pd_map["ai"]


def format_sources(doc_ids: List[str], pd_map: Dict[str, ProgramData]) -> str:
    urls = []
    for doc_id in doc_ids:
        key = doc_id.split("#", 1)[0]
        url = pd_map[key].url
        if url not in urls:
            urls.append(url)
    return "Источники: " + "; ".join(urls)


def answer_with_llm(query: str, context: str) -> str:
    """Заглушка: здесь можно подключить любую LLM (OpenAI, GigaChat, etc.).
    Сейчас просто вернём осмысленный экстракт без генерации.
    """
    # Обрезаем контекст и ответим коротко
    snippet = context.strip().split("\n")[:6]
    return "\n".join(snippet)


def engine_answer(user_q: str, index: KBIndex, pd_map: Dict[str, ProgramData], rule_map: Dict[str, Dict[str, str]]) -> str:
    # Определяем программу
    pd = pick_program(user_q, pd_map)

    # Пробуем rule‑based
    rule_ans = rule_router(user_q, pd, rule_map[pd.key])
    if rule_ans:
        return f"{rule_ans}\n\nИсточник: {pd.url}"

    # Иначе — полнотекстовый поиск по обеим программам
    hits = index.search(user_q, top_k=5)
    top_doc_ids = [doc_id for doc_id, _ in hits[:3]]

    # Составим краткий контекст
    context_parts = []
    for doc_id, sim in hits[:3]:
        key, para = doc_id.split("#", 1)
        text = next((t for i, (did, t) in enumerate(index.docs) if did == doc_id), "")
        context_parts.append(text)
    context = "\n\n".join(context_parts)

    draft = answer_with_llm(user_q, context)
    return f"{draft}\n\n{format_sources(top_doc_ids, pd_map)}"


# === FastAPI ===
class ChatIn(BaseModel):
    q: str

class ChatOut(BaseModel):
    answer: str

app = FastAPI(title="ITMO AI Programs Chatbot")

# Глобальные структуры
INDEX: Optional[KBIndex] = None
PD_MAP: Dict[str, ProgramData] = {}
RULE_MAP: Dict[str, Dict[str, str]] = {}


@app.on_event("startup")
def _startup():
    global INDEX, PD_MAP, RULE_MAP
    programs = [parse_program(k, u) for k, u in PROGRAM_URLS.items()]
    INDEX, PD_MAP, RULE_MAP = build_kb(programs)


@app.post("/chat", response_model=ChatOut)
def chat(inp: ChatIn):
    ans = engine_answer(inp.q, INDEX, PD_MAP, RULE_MAP)
    return ChatOut(answer=ans)


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
