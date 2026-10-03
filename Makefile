# Полный воспроизводимый прогон: make all  (≈ 3–4 ч на ноутбуке Apple M-серии / 8 ядер)
# Быстрая проверка на подвыборке рядов: make quick (≈ 15 мин)

PY ?= .venv/bin/python
CFG ?= configs/default.yaml

UNAME := $(shell uname)

.PHONY: setup data dataset news forecast cpd figures report landing all quick clean

setup:
	uv venv --python 3.12 .venv || python3.12 -m venv .venv
	uv pip install --python $(PY) -r requirements.txt || $(PY) -m pip install -r requirements.txt
	# macOS: LightGBM требует libomp. Если он не установлен (brew install libomp), подключаем копию из torch
	@if [ "$(UNAME)" = "Darwin" ] && ! $(PY) -c "import lightgbm" 2>/dev/null; then \
		L=.venv/lib/python3.12/site-packages/lightgbm/lib/lib_lightgbm.dylib; \
		install_name_tool -add_rpath "@loader_path/../../torch/lib" $$L && codesign --force -s - $$L; fi

data:            ## шаг 1: загрузка открытых данных (новости — самый долгий этап, кэшируется по дням)
	$(PY) scripts/01_download_data.py --config $(CFG)

dataset:         ## шаг 2: панель, признаки МО, экзогенные ряды
	$(PY) scripts/02_build_dataset.py --config $(CFG)

news:            ## шаг 4: геопривязка и классификация новостей, месячные индексы
	$(PY) scripts/04_build_news_features.py --config $(CFG)

forecast:        ## шаг 3: бэктест моделей прогнозирования
	$(PY) scripts/03_forecast_benchmark.py --config $(CFG)

cpd:             ## шаг 5: сравнение детекторов точек изменений, раннее предупреждение
	$(PY) scripts/05_cpd_benchmark.py --config $(CFG)

figures:         ## шаг 6: графики и данные для отчёта и лендинга
	$(PY) scripts/06_make_figures.py --config $(CFG)

report:          ## шаг 7: методологический отчёт (Markdown → HTML → PDF)
	$(PY) scripts/07_build_report.py --config $(CFG)

landing:         ## шаг 8: интерактивный лендинг docs/index.html и его PDF-версия
	$(PY) scripts/08_build_landing.py --config $(CFG)

all: data dataset news forecast cpd figures report landing

quick: dataset news
	$(PY) scripts/03_forecast_benchmark.py --config configs/quick.yaml
	$(PY) scripts/05_cpd_benchmark.py --config configs/quick.yaml

test:            ## проверки ключевых свойств (утечка, метрики, детекторы)
	$(PY) -m pytest -q tests

clean:
	rm -rf results/sample* reports/figures/*.png
