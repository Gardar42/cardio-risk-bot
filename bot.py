import asyncio
import os

import joblib
import pandas as pd
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message
from dotenv import load_dotenv

from llm import UNKNOWN, continue_dialogue, extract_value, generate_report

load_dotenv()

artifact = joblib.load("models/model.pkl")
model, FEATURES = artifact["pipeline"], artifact["features"]

LEVELS = {1: "норма", 2: "выше нормы", 3: "значительно выше нормы"}
YES_NO = {0: "нет", 1: "да"}

QUESTIONS = [
    {"key": "age", "label": "Возраст, лет", "text": "Сколько вам полных лет?", "unit": "лет", "range": (18, 100)},
    {"key": "is_male", "label": "Пол", "text": "Укажите ваш пол.", "options": {0: "женский", 1: "мужской"}},
    {"key": "height", "label": "Рост, см", "text": "Какой у вас рост?", "unit": "см", "range": (120, 220)},
    {"key": "weight", "label": "Вес, кг", "text": "Сколько вы весите?", "unit": "кг", "range": (30, 200)},
    {"key": "ap_hi", "label": "Верхнее давление", "text": "Какое у вас обычно верхнее (систолическое) давление?",
     "unit": "мм рт. ст.", "range": (80, 250)},
    {"key": "ap_lo", "label": "Нижнее давление", "text": "А нижнее (диастолическое)?",
     "unit": "мм рт. ст.", "range": (40, 160)},
    {"key": "cholesterol", "label": "Холестерин",
     "text": "Какой у вас уровень холестерина: в норме, выше нормы или значительно выше?", "options": LEVELS},
    {"key": "gluc", "label": "Глюкоза",
     "text": "Какой у вас уровень сахара (глюкозы) в крови: в норме, выше нормы или значительно выше?", "options": LEVELS},
    {"key": "smoke", "label": "Курит", "text": "Вы курите?", "options": YES_NO},
    {"key": "alco", "label": "Употребляет алкоголь", "text": "Вы регулярно употребляете алкоголь?", "options": YES_NO},
    {"key": "active", "label": "Физически активен", "text": "Вы регулярно занимаетесь физической активностью?",
     "options": YES_NO},
]

GREETING = (
    "Здравствуйте! Я помогу оценить риск сердечно-сосудистых заболеваний.\n"
    f"Задам {len(QUESTIONS)} коротких вопросов, отвечайте своими словами. "
    "Если не знаете ответа, так и напишите — вопрос пропустим.\n\n"
    "Важно: это статистическая оценка, а не медицинский диагноз."
)


class Form(StatesGroup):
    survey = State()
    chat = State()


def expected_of(q: dict) -> str:
    if "options" in q:
        return ", ".join(f"{code} — {text}" for code, text in q["options"].items())
    return q["unit"]


def is_valid(q: dict, value: float | None) -> bool:
    if value is None:
        return False
    if "options" in q:
        return value in q["options"]
    low, high = q["range"]
    return low <= value <= high


def predict(answers: dict) -> float:
    row = pd.DataFrame([answers])
    row["bmi"] = row["weight"] / (row["height"] / 100) ** 2
    return float(model.predict_proba(row[FEATURES])[0, 1])


def describe(answers: dict) -> str:
    lines = []
    for q in QUESTIONS:
        value = answers[q["key"]]
        if pd.isna(value):
            text = "не указано"
        elif "options" in q:
            text = q["options"][int(value)]
        else:
            text = f"{value:g}"
        lines.append(f"- {q['label']}: {text}")
    bmi = answers["weight"] / (answers["height"] / 100) ** 2
    lines.append(f"- Индекс массы тела: {'не указано' if pd.isna(bmi) else f'{bmi:.1f}'}")
    return "\n".join(lines)


dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: Message, state: FSMContext):
    await state.clear()
    await state.set_state(Form.survey)
    await state.update_data(step=0, answers={})
    await message.answer(GREETING)
    await message.answer(QUESTIONS[0]["text"])


@dp.message(Form.survey, F.text)
async def survey(message: Message, state: FSMContext):
    data = await state.get_data()
    step, answers = data["step"], data["answers"]
    q = QUESTIONS[step]

    await message.bot.send_chat_action(message.chat.id, "typing")
    try:
        value = await extract_value(q["text"], message.text, expected_of(q))
    except Exception:
        await message.answer("Не получилось связаться с сервисом, попробуйте отправить ответ ещё раз.")
        return

    if value == UNKNOWN:
        value = float("nan")
        await message.answer("Хорошо, пропустим этот вопрос.")
    elif not is_valid(q, value):
        await message.answer(f"Не совсем понял ответ. {q['text']}")
        return

    answers[q["key"]] = value
    step += 1
    await state.update_data(step=step, answers=answers)

    if step < len(QUESTIONS):
        await message.answer(QUESTIONS[step]["text"])
        return

    await message.answer("Спасибо! Анализирую ваши данные...")
    await message.bot.send_chat_action(message.chat.id, "typing")
    probability = predict(answers)
    try:
        text, dialogue = await generate_report(describe(answers), probability)
    except Exception:
        await message.answer("Не получилось связаться с сервисом, отправьте последний ответ ещё раз.")
        await state.update_data(step=step - 1)
        return

    await state.set_state(Form.chat)
    await state.update_data(dialogue=dialogue)
    await message.answer(text)
    await message.answer("Можете задать уточняющие вопросы. Чтобы пройти опрос заново, отправьте /start.")


@dp.message(Form.chat, F.text)
async def chat(message: Message, state: FSMContext):
    dialogue = (await state.get_data())["dialogue"]
    await message.bot.send_chat_action(message.chat.id, "typing")
    try:
        text, dialogue = await continue_dialogue(dialogue, message.text)
    except Exception:
        dialogue.pop()
        await message.answer("Не получилось связаться с сервисом, попробуйте ещё раз.")
        return
    await state.update_data(dialogue=dialogue)
    await message.answer(text)


@dp.message()
async def other(message: Message):
    await message.answer("Отправьте /start, чтобы начать опрос.")


async def main():
    bot = Bot(os.environ["TELEGRAM_TOKEN"])
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
