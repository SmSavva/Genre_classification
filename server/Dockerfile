# Используем базовый образ Python
FROM python:3.9-slim

RUN apt-get update && apt-get install -y libpq-dev build-essential

# Устанавливаем рабочую директорию
WORKDIR /server

# Копируем все файлы проекта
COPY . .

# Устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Указываем порт, на котором будет слушать приложение
EXPOSE 5000

# Запускаем приложение
CMD ["python", "app.py"]