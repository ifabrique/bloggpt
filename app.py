import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import requests

# Инициализация FastAPI приложения
app = FastAPI(
    title="AI Blog Post Generator",
    description="Генерирует статьи по заданной теме с учётом свежих новостей",
    version="1.0.0"
)

# Получение API ключей из переменных окружения
openai.api_key = os.getenv("OPENAI_API_KEY")
currentsapi_key = os.getenv("CURRENTS_API_KEY")

# Проверка наличия необходимых переменных окружения
if not openai.api_key or not currentsapi_key:
    raise RuntimeError("Необходимо установить переменные окружения OPENAI_API_KEY и CURRENTS_API_KEY")

class TopicRequest(BaseModel):
    """Модель запроса для генерации поста"""
    topic: str

def get_recent_news(topic: str) -> str:
    """
    Получает свежие новости по заданной теме с помощью Currents API.
    Возвращает строку с заголовками новостей или сообщение об их отсутствии.
    """
    url = "https://api.currentsapi.services/v1/latest-news"
    params = {
        "language": "en",
        "keywords": topic,
        "apiKey": currentsapi_key
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        news_data = response.json().get("news", [])
        if not news_data:
            return "Свежих новостей не найдено."
        # Собираем заголовки первых 5 новостей
        return "\n".join([article.get("title", "") for article in news_data[:5]])
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ошибка при обращении к Currents API: {str(e)}")

def generate_content(topic: str) -> dict:
    """
    Генерирует заголовок, мета-описание и статью на основе темы и свежих новостей.
    """
    recent_news = get_recent_news(topic)

    try:
        # Генерация заголовка
        title_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Придумайте привлекательный и точный заголовок для статьи на тему '{topic}', с учётом актуальных новостей:\n{recent_news}. Заголовок должен быть интересным и ясно передавать суть темы. Не используй символы # и - в тексте, даже в заголовках."
            }],
            max_tokens=20,
            temperature=1.0,
            stop=["\n"]
        )
        title = title_response.choices[0].message.content.strip()

        # Генерация мета-описания
        meta_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Напишите мета-описание для статьи с заголовком: '{title}'. Оно должно быть полным, информативным и содержать основные ключевые слова. Не используй символы # и - в тексте, даже в заголовках."
            }],
            max_tokens=30,
            temperature=1.0,
            stop=["."]
        )
        meta_description = meta_response.choices[0].message.content.strip()

        # Генерация основной статьи
        content_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Напишите подробную статью на тему '{topic}', используя последние новости:\n{recent_news}.
                Статья должна быть:
                1. Информативной и логичной
                2. Содержать не менее 500 символов
                3. Иметь четкую структуру с подзаголовками
                4. Включать анализ текущих трендов
                5. Иметь вступление, основную часть и заключение
                6. Включать примеры из актуальных новостей
                7. Каждый абзац должен быть не менее 3-4 предложений
                8. Текст должен быть легким для восприятия и содержательным
                9. Не используй символ # в тексте, даже в заголовках."""
            }],
            max_tokens=1000,
            temperature=1.0,
            presence_penalty=0.6,
            frequency_penalty=0.6
        )
        post_content = content_response.choices[0].message.content.strip()

        return {
            "title": title,
            "meta_description": meta_description,
            "post_content": post_content
        }

    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обращении к OpenAI API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации контента: {str(e)}")

@app.post("/generate-post")
async def generate_post(topic_req: TopicRequest):
    """
    Эндпоинт для генерации поста по теме.
    Возвращает заголовок, мета-описание и текст статьи.
    """
    if not topic_req.topic or not topic_req.topic.strip():
        raise HTTPException(status_code=400, detail="Поле 'topic' не должно быть пустым")
    return generate_content(topic_req.topic.strip())

@app.get("/")
async def root():
    """
    Базовый эндпоинт для проверки состояния сервиса.
    """
    return {"message": "AI Blog Post Generator is running"}

@app.get("/heartbeat")
async def heartbeat():
    """
    Эндпоинт для проверки работоспособности сервиса.
    """
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    # Запуск приложения с портом из переменной окружения или по умолчанию 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
