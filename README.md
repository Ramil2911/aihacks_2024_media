# Решение кейса «Подбор локаций для размещения рекламы» команды «ыыы»
Цифровой прорыв, 2024, СКФО

## Описание

- Сфера НТИ: None
- Бизнес-задача: Создание инструмента для прогнозирования охвата наружной рекламы в городе Москва
- Результат работы: модель предсказания охвата целевой аудитории рекламной кампании, сайт для определения охвата и поиска оптимального расположения баннеров

## Запуск решения
В папке frontend ```python app.py```. Зависимости указаны в requirements.txt.
Также можно использовать docker-compose

## Этапы работы

1. Анализ предоставленных данных.
2. Обучение нескольких моделей поверх предобработыннх данный бейзлайна.
3. Работа с OpenStreetMap, разработка деления города по административным районам.
4. Обучение модели Denoising Transformer Autoencoder + Catboost на данных OSM.
5. Добавление статистических данных и информации о точках интереса из OSM.
6. Обучение последней модели на новых данных.
7. Разработка веб-интерфейса и алгоритма оптимизации.

## Пайплайн обработки данных

1. Преобразуем и чистим входные данные.
2. Соединение статистическиз данных о населении каждого района Москвы со входными данными.
3. Загрузка данных о точках интереса вокруг точек конфигурации и добавление этих данных в датасет.

TODO: ссылка на ноутбук с пайплайном

## Команда «ыыы»
- [Рамиль Габдрахманов](https://github.com/Ramil2911) | Капитан команды | ML
- [Илья Емельянов](https://github.com/hornetio) | ML/DS Enginneer
- [Айнур Исмагилов](https://github.com/Hopper789) | ML/DS Engineer

## Используемый стек и технологии

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Markdown](https://img.shields.io/badge/markdown-%23000000.svg?style=for-the-badge&logo=markdown&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
