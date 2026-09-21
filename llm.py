import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()


def _make_llm(temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        model=os.environ["OPENROUTER_MODEL"],
        temperature=temperature,
    )


extractor = _make_llm(0.0)
assistant = _make_llm(0.7)

EXTRACT_PROMPT = """Ты извлекаешь значение из ответа пользователя на вопрос медицинской анкеты.
Верни только число без единиц измерения и пояснений, либо слово null.
- Переводи в указанные единицы, а числа словами — в цифры: "метр восемьдесят" -> 180.
- Если даны варианты, верни номер подходящего по смыслу варианта: "не курю" -> номер варианта "нет".
- Если в ответе несколько чисел, выбери то, что отвечает на вопрос: на вопрос о нижнем давлении "130 на 85" -> 85.
- Если пользователь говорит, что не знает ответа, не помнит или не хочет отвечать, верни слово unknown.
- Если ответ непонятен или не относится к вопросу, верни null."""

SYSTEM_PROMPT = """Ты — вежливый и внимательный медицинский ассистент в Telegram-боте.
Тебе передают данные пользователя и оценку риска сердечно-сосудистых заболеваний от модели машинного обучения.

Правила:
- Пиши на русском, на «вы», спокойно и без запугивания.
- В первом ответе назови уровень риска и вероятность, поясни, что это статистическая оценка, а не диагноз.
- Отметь показатели, которые стоит улучшить, и дай 3–4 конкретных совета по питанию, активности и образу жизни именно по ним.
- Не ставь диагнозов, не называй лекарства и дозировки.
- Всегда рекомендуй обратиться к терапевту или кардиологу.
- Первый ответ — не больше 150 слов, без markdown.
- На уточняющие вопросы отвечай коротко, с учётом данных пользователя. На вопросы не о здоровье вежливо откажи."""


UNKNOWN = "unknown"


async def extract_value(question: str, answer: str, expected: str) -> float | str | None:
    response = await extractor.ainvoke([
        SystemMessage(content=EXTRACT_PROMPT),
        HumanMessage(content=f"Вопрос: {question}\nОжидается: {expected}\nОтвет: {answer}"),
    ])
    reply = response.content.strip("`. ").lower()
    if reply == UNKNOWN:
        return UNKNOWN
    try:
        return float(reply.replace(",", "."))
    except ValueError:
        return None


async def generate_report(user_data: str, probability: float) -> tuple[str, list]:
    level = "низкий" if probability < 0.3 else "умеренный" if probability < 0.6 else "высокий"
    dialogue = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Данные пациента:\n{user_data}\n\n"
            f"Результат ML-модели: риск {level} ({probability:.0%}).\n"
            "Составь персонализированный ответ для пациента."
        )),
    ]
    response = await assistant.ainvoke(dialogue)
    dialogue.append(AIMessage(content=response.content))
    return response.content, dialogue


async def continue_dialogue(dialogue: list, user_message: str) -> tuple[str, list]:
    dialogue.append(HumanMessage(content=user_message))
    response = await assistant.ainvoke(dialogue)
    dialogue.append(AIMessage(content=response.content))
    return response.content, dialogue


if __name__ == "__main__":
    async def demo():
        cases = [
            ("Какой у вас рост?", "метр восемьдесят", "см"),
            ("Какое у вас нижнее давление?", "130 на 85", "мм рт. ст."),
            ("Какой у вас пол?", "мужчина", "0 — женский, 1 — мужской"),
            ("Какой у вас уровень холестерина?", "врач сказал, немного повышен",
             "1 — норма, 2 — выше нормы, 3 — значительно выше нормы"),
            ("Вы курите?", "бросил пять лет назад", "0 — нет, 1 — да"),
            ("Какое у вас верхнее давление?", "не знаю", "мм рт. ст."),
        ]
        for question, answer, expected in cases:
            print(f"{answer!r:35} -> {await extract_value(question, answer, expected)}")

        text, dialogue = await generate_report(
            "- Возраст: 58\n- Пол: мужской\n- ИМТ: 32\n- Давление: 155/95\n"
            "- Холестерин: значительно выше нормы\n- Курит: да\n- Физическая активность: нет",
            0.84,
        )
        print("\n" + text)

        text, dialogue = await continue_dialogue(dialogue, "А сколько раз в неделю мне нужно заниматься спортом?")
        print("\n" + text)

    asyncio.run(demo())
