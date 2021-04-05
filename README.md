# Сравнение fine-tuned RoBERTa и zero-shot-classification

## Задание
1. Возьмите предобученую модель [RoBERTa](https://huggingface.co/transformers/model_summary.html#roberta) из библиотеки transformers от 🤗. Дообучите модель определять является ли твит расистским или сексистким с использованием соответствующего [датасета](https://huggingface.co/datasets/tweets_hate_speech_detection). Не забудьте поделить датасет на тренировочную и тестовую выборку.

2. Оцените качество тестовой модели с использованием метрики Accuracy до и после дообучения. Проанализируйте, как выбор порога классификации влияет на точность с помощью PR-curve.

3. (Опционально) Реализуйте простой CLI, который принимает на вход предложение и выводит в консоль результат оценки модели, а также время, которое понадобилось модели на обработку этого предложения.

## Уточнения
* _До дообучения_ - [zero shot classification](https://discuss.huggingface.co/t/new-pipeline-for-zero-shot-text-classification/681) из transformers
* В процессе EDA стало ясно, что имеется дизбаланс классов (0 - 93%; 1 - 7%), поэтому в качестве метрики кроме Accuracy я буду использовать также и F1-score.

## Файлы и что здесь вообще происходит
* `config.yml` - файл со всеми параметрами (hyperparameters, file paths, random state, etc.), которые используются в процессе обучения и сравнения результатов
* `requirements.txt` - использованные dependencies
* `utils.py` - вспомагательные функции, не содержащие главной логики алгоритмов