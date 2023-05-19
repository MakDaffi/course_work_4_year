# Исследование моделей рекомендательных систем на пример соревнования H&M

В данном репозитории находится моя курсовая работа за 4 курс университета.

Целью данной работы является обучение построянию рекомендательных систем. В качестве данных для построения рекомендательной системы был использован датасет kaggle соревнования [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/code "Ссылка на соревнование"), т.к. он содержит в себе много данных, которые интересно поиследовать.

## План курсовой работы

### Введение
Вопрос актуальности темы и ее применения в повседневной жизни
### Основные определения, обозначения и сокращения
### Теоретическая часть
1. Общие сведения о рекомендательных системах
2. Модели и их гиперпараметры:

Бейзлайны и модели основанные на правилах

Простые модели первого уровня:
* Content-base модели(TF-IDF, необычные модели и т.д.)
* Матричное разложение (ALS, LightFM, нейросетевые разложения)
* Гибридные модели

Модели второго уровня:

Зачем используются и почему не могут применятся отдельно.
* Бустинги
* Нейронные сети

Специфичные подходы к задаче:
* Графовые рекомендации
3. Метрики качества: 
* MAP@k
* nDSG
* Продвинутые метрики

### Практическая часть
1. Описание условий проведения экспериментов (железо, OS)
2. Описание инструментов
3. Описание набора данных
4. Описание результатов анализа данных
5. Описание разбиения на train/validation
6. Описание параметров модели, использованных для обучения
7. Описание результатов обучения
8. Размышления, куда нужно копать дальше


## План работы на семестр

- [X] Предварительный анализ данных, нахождение паттернов, создание новых признаков и т.д. (Сделано и находится на этапе проверки)
- [X] Создание baselinа (популярное), с которым будут сравниваться более сложные модели. (Сделано и находится на этапе проверки)
- [X] Улучшение базовой модели, до модели, которая работает по конкретным правилам. (Сделано и находится на этапе проверки)
- [X] Создание моделей первого уровня и оценка их качества:
* Матричное разложение
* Content-base модели
* Гибридные модели
- [ ] Выбор и обучение модели второго уровня (бустинг или нейронная сеть) для переранжирования результатов.
- [ ] Оценка качества всей системы на тестовых данных.
- [X] Написание теоретической части работы
- [X] Описание практической части работы и ее результатов
  
## Дополнительные пункты, если останется время
- [ ] Попробовать необычные подходы (графовые рекомендации и т.д.)
- [ ] Ускорение выдачи рекомендаций
- [ ] Докеризация проекта

## Информация о структуре репозитория

В директории text лежит текст курсовой работы.

В директории presentation лежит презентация для защиты курсовой работы.

В директории src лежит текст исходный код курсовой работы. Внутри нее лежат следующие директории: data_analysis - директория с ноутбуком для анализа и предобработки данных, директория get_embeddings директория с ноутбуком, в котором производились эксперименты с эмбеддинагми и директория models -директория с кодом для обучения моделей и их эффективного использования.