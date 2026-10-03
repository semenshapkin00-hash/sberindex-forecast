# Пульс муниципального потребления

Решение для онлайн-конкурса СберИндекса, номинация **«Прогнозирование»**: прогноз безналичных потребительских расходов на жителя в 2 190 муниципальных образованиях (6 категорий трат) на 1, 3, 6 и 12 месяцев, совмещённый с ранним обнаружением точек структурных изменений и интеграцией новостей.

- 📄 Методологический отчёт: [`reports/methodology_report.pdf`](reports/methodology_report.pdf) (исходник — [`reports/methodology_report.md`](reports/methodology_report.md))
- 🖥 Интерактивная презентация: <https://semenshapkin00-hash.github.io/sberindex-forecast/> (исходник — [`docs/index.html`](docs/index.html)), PDF-версия — [`reports/presentation.pdf`](reports/presentation.pdf)
- 📊 Все метрики: [`results/full/`](results/full) и [`results/cpd/`](results/cpd)

<!--RESULTS-->
## Итоги

Бэктест на 12 096 полных рядах, скользящая точка прогноза декабрь 2023 – ноябрь 2024 г. MAE — руб. на жителя в месяц; Ансамбль лучше Prophet на 45,6% / 28,4% / 16,3% / 46,3% (горизонты 1 / 3 / 6 / 12 мес.).

| Модель | MAE 1 мес. | MAE 3 мес. | MAE 6 мес. | MAE 12 мес. |
|---|--:|--:|--:|--:|
| **Ансамбль** | 326,0 | 405,3 | 496,1 | 480,7 |
| LightGBM | 374,5 | 450,7 | 540,0 | 538,9 |
| Chronos-2-SA + ковариаты | 409,1 | 564,2 | 698,2 | 1 117,3 |
| SNaive×Growth | 388,8 | 463,7 | 536,7 | 552,4 |
| Prophet | 599,5 | 566,2 | 592,5 | 894,3 |

Детекция шоков (синтетический тест, режим мониторинга): Chronos-2 интервал — F1 0,525 с обнаружением в месяц шока; схема «Chronos-2 → PELT» — точность 0,83 при 0,11 ложной тревоги на 100 ряд-месяцев.
<!--/RESULTS-->

## Что внутри

| Контур | Методы | Где в коде |
|---|---|---|
| Прогноз | Naive, сезонные эталоны, **Prophet (базовая модель)**, ETS/Theta с внешней сезонностью, глобальный **LightGBM** (прямой многошаговый, ~70 признаков), **Chronos-Bolt / Chronos-2** (zero-shot, ковариаты, очистка от сезонности, LoRA), стэкинг, онлайн-ансамбль | `src/sbx/forecast/` |
| Точки изменений | PELT, BinSeg, BottomUp, Window, KernelCPD, CUSUM, Page–Hinkley, BOCPD, BOCPD с новостным hazard, выход за интервал Chronos-2, двухуровневые схемы; синтетический бенчмарк, реальные кейсы, раннее предупреждение | `src/sbx/cpd.py`, `src/sbx/cpd_real.py` |
| Новости | сбор архивов Интерфакса и Lenta.ru, лемматизация, газетир МО из справочника СберИндекса, классификация событий, as-of индексы, event study | `src/sbx/sources/news.py`, `src/sbx/news.py` |
| Данные | датасет конкурса, справочник МО, SberIndex API, Росстат, ЦБ, календарь, NASA POWER | `src/sbx/sources/`, `src/sbx/data.py` |

## Быстрый старт

Требуется Python 3.12 (рекомендуется [uv](https://docs.astral.sh/uv/)), ~6 ГБ свободного места, интернет для загрузки данных. Проверено на macOS (Apple M, 16 ГБ RAM); на Linux работает без изменений.

```bash
git clone https://github.com/semenshapkin00-hash/sberindex-forecast.git && cd sberindex-forecast
make setup          # виртуальное окружение .venv и зависимости
make data           # загрузка всех данных (~1,5 ч; новости кэшируются по дням, загрузку можно прерывать)
make dataset news   # панель, признаки МО, внешние ряды, новостные индексы
make forecast       # бэктест всех моделей (~3 ч на 8 ядрах)
make cpd            # детекторы точек изменений, кейсы, раннее предупреждение (~30 мин)
make figures report # графики, отчёт PDF
python scripts/08_build_landing.py   # интерактивный лендинг docs/index.html
```

Быстрая проверка воспроизводимости на подвыборке 600 рядов (~15 мин): `make quick`.

Отдельный шаг или модель:

```bash
.venv/bin/python scripts/03_forecast_benchmark.py --models prophet lightgbm
.venv/bin/python scripts/03_forecast_benchmark.py --set backtest.sample_series=1000 models.lightgbm.num_boost_round=300
.venv/bin/python scripts/05_cpd_benchmark.py --parts real
```

**macOS и LightGBM.** LightGBM требует OpenMP (`libomp`). Если `brew install libomp` не выполнен, `make setup` подключает копию библиотеки из пакета torch.

**Сертификаты.** Сайты Сбера и Росстата используют сертификаты НУЦ Минцифры, а sberindex.ru не отдаёт промежуточный сертификат. Публичные корневой и промежуточные сертификаты лежат в `certs/`; скрипт загрузки собирает из них и certifi bundle — проверка TLS остаётся включённой.

## Конфигурация

Все пути, источники и гиперпараметры — в [`configs/default.yaml`](configs/default.yaml): горизонты и точки бэктеста, параметры Prophet, LightGBM (группы признаков, база-смещение, число деревьев), Chronos (модель, устройство, ковариаты, LoRA), состав ансамбля, сетки параметров детекторов, пороги раннего предупреждения. Свой конфиг наследует значения по умолчанию: `--config configs/quick.yaml`; отдельный ключ: `--set models.lightgbm.num_boost_round=500`.

## Структура

```
configs/            default.yaml, quick.yaml
scripts/            01_download_data … 08_build_landing — шаги конвейера
src/sbx/
  sources/          клиенты источников: sberindex_api, rosstat, misc (ЦБ, календарь, погода), news (архивы)
  data.py           сборка панели, характеристик МО, внешних рядов, национальной сезонности
  forecast/         base (контекст бэктеста, защита от утечки), baselines, gbm, foundation, ensemble
  backtest.py       скользящая точка прогноза, окна и шаги, метрики
  metrics.py        MAE, R², R² динамики, RMSE, WAPE, MASE, парные тесты
  cpd.py            сигнал (относительный индекс), офлайн/онлайн детекторы, синтетика, метрики
  cpd_real.py       реальные точки изменений, связь с новостями, event study, раннее предупреждение
  news.py           лемматизация, газетир, классификация событий, месячные индексы
  viz.py            стиль графиков
reports/            шаблон и итоговый отчёт, графики
landing/, docs/     интерактивный лендинг
results/            метрики, таблицы, кейсы (прогнозы целиком не хранятся)
certs/              публичные сертификаты для загрузки данных
```

## Данные и лицензии

Сырые файлы данных в репозиторий не включены — скрипт загружает их из первоисточников. Интерактивная презентация (`docs/index.html`) содержит ряды потребления из датасета СберИндекса для проводника по МО; эти данные распространяются по лицензии CC BY-SA 4.0 и приведены с указанием источника.

- Данные СберИндекса — CC BY-SA 4.0. Цитирование: *Потребительские безналичные расходы на уровне муниципальных образований по категориям трат. СберИндекс. https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim*; *Данные о границах и преобразованиях муниципальных образований. СберИндекс. https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities*.
- Росстат, Банк России — открытые данные; NASA POWER — общественное достояние; производственный календарь — isdayoff.ru.
- Новости: используются только метаданные заголовков (дата, заголовок, рубрика) из суточных архивов, разрешённых robots.txt; тексты статей не скачиваются и не публикуются.
- Модели: Chronos-Bolt и Chronos-2 (Amazon) — Apache-2.0; Prophet — MIT; LightGBM — MIT; ruptures — BSD-2. Платные или проприетарные технологии не используются.

Код — лицензия MIT.
