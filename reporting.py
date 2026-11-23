from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def _esc(s) -> str:
    text = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for a, b in repl.items():
        text = text.replace(a, b)
    return text


def _matrix_table(routes: pd.DataFrame, warehouses: list[str], stores: list[str]) -> str:
    lines = [r"\begin{tabular}{l" + "c" * len(stores) + "}", r"\toprule", " & " + " & ".join(stores) + r" \\", r"\midrule"]
    for w in warehouses:
        vals = []
        for s in stores:
            row = routes[(routes["warehouse_id"] == w) & (routes["store_id"] == s)].iloc[0]
            if not bool(row["allowed"]):
                vals.append("X")
            else:
                vals.append(f"{row['unit_cost']:.2f}/{row['capacity_units']:.1f}")
        lines.append(w + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _rows_table(headers: list[str], rows: list[list[str]], align: str | None = None) -> str:
    if align is None:
        align = "l" + "r" * (len(headers) - 1)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def write_report_tex(root: Path) -> Path:
    data = root / "data"
    results = root / "results"
    plots = results / "plots"

    warehouses = pd.read_csv(data / "warehouses.csv")
    stores = pd.read_csv(data / "stores.csv")
    routes = pd.read_csv(data / "routes.csv")
    base = pd.read_csv(results / "base_solution.csv")
    failure = pd.read_csv(results / "failure_solution.csv")
    overload = pd.read_csv(results / "overload_solution.csv")
    scenarios = pd.read_csv(results / "scenarios.csv")
    bottlenecks = pd.read_csv(results / "bottlenecks.csv")
    potentials = pd.read_csv(results / "node_potentials.csv")
    changes = pd.read_csv(results / "failure_flow_changes.csv")
    overload_changes = pd.read_csv(results / "overload_flow_changes.csv")
    run_info = json.loads((results / "run_info.json").read_text(encoding="utf-8"))
    growth = json.loads((results / "demand_growth_diagnosis.json").read_text(encoding="utf-8"))
    greedy = json.loads((results / "greedy_summary.json").read_text(encoding="utf-8"))
    ver_base = json.loads((results / "verification_base.json").read_text(encoding="utf-8"))
    ver_failure = json.loads((results / "verification_failure.json").read_text(encoding="utf-8"))
    ver_overload = json.loads((results / "verification_overload.json").read_text(encoding="utf-8"))
    ver_greedy = json.loads((results / "verification_greedy.json").read_text(encoding="utf-8"))
    milp = pd.read_csv(results / "milp_solution.csv")
    milp_summary = json.loads((results / "milp_summary.json").read_text(encoding="utf-8"))
    milp_changes = pd.read_csv(results / "lp_milp_flow_changes.csv")
    ver_milp = json.loads((results / "verification_milp.json").read_text(encoding="utf-8"))

    wh_rows = [[r.warehouse_id, f"{r.supply_units:.0f}"] for r in warehouses.itertuples(index=False)]
    st_rows = [[r.store_id, f"{r.demand_units:.0f}"] for r in stores.itertuples(index=False)]
    forbidden = routes[~routes["allowed"]]
    forbidden_text = ", ".join(f"{r.warehouse_id}→{r.store_id}" for r in forbidden.itertuples(index=False))

    positive = base[base["flow_units"] > 1e-7]
    base_rows = [[
        f"{r.warehouse_id}→{r.store_id}",
        f"{r.flow_units:.1f}",
        f"{r.capacity_units:.1f}",
        f"{r.unit_cost:.2f}",
        f"{r.route_cost:.2f}",
    ] for r in positive.itertuples(index=False)]

    scen_rows = []
    for r in scenarios.itertuples(index=False):
        obj = "--" if pd.isna(r.objective_cost) else f"{r.objective_cost:.3f}"
        delta = "--" if pd.isna(r.delta_vs_base) else f"{r.delta_vs_base:.3f}"
        scen_rows.append([_esc(r.scenario), _esc(r.status), obj, delta, _esc(r.change)])

    change_rows = [[
        f"{r.warehouse_id}→{r.store_id}",
        f"{r.base_flow:.1f}",
        f"{r.failure_flow:.1f}",
        f"{r.delta_flow:+.1f}",
    ] for r in changes.itertuples(index=False)]

    overload_change_rows = [[
        f"{r.warehouse_id}→{r.store_id}",
        f"{r.base_flow:.2f}",
        f"{r.overload_flow:.2f}",
        f"{r.delta_flow:+.2f}",
    ] for r in overload_changes.itertuples(index=False)]

    bottleneck_rows = [[
        f"{r.warehouse_id}→{r.store_id}",
        f"{r.flow_units:.1f}",
        f"{r.capacity_units:.1f}",
        f"{r.capacity_shadow_benefit:.2f}",
    ] for r in bottlenecks.itertuples(index=False)]

    potential_rows = [[r.node_id, _esc(r.node_type), f"{r.dual_potential:.2f}"] for r in potentials.itertuples(index=False)]

    milp_positive = milp[milp["flow_units"] > 1e-7]
    milp_rows = [[
        f"{r.warehouse_id}→{r.store_id}",
        f"{r.flow_units:.0f}",
        f"{r.capacity_units:.1f}",
        f"{r.unit_cost:.2f}",
        f"{r.route_cost:.2f}",
    ] for r in milp_positive.itertuples(index=False)]

    milp_change_rows = [[
        f"{r.warehouse_id}→{r.store_id}",
        f"{r.lp_flow:.1f}",
        f"{r.milp_flow:.0f}",
        f"{r.delta_flow:+.1f}",
    ] for r in milp_changes.head(12).itertuples(index=False)]

    ver_rows = []
    for name, v in [("base", ver_base), ("arc failure", ver_failure), ("arc overload", ver_overload), ("greedy", ver_greedy), ("MILP", ver_milp)]:
        ver_rows.append([
            _esc(name),
            "да" if v["feasible"] else "нет",
            f"{v['max_warehouse_balance_error']:.2e}",
            f"{v['max_store_balance_error']:.2e}",
            f"{v['max_capacity_violation']:.2e}",
            f"{v['forbidden_route_violation']:.2e}",
        ])

    cost_cap_matrix = _matrix_table(routes, warehouses["warehouse_id"].tolist(), stores["store_id"].tolist())
    supply_table = _rows_table(["Склад", "Запас, ед."], wh_rows, "lr")
    demand_table = _rows_table(["Магазин", "Спрос, ед."], st_rows, "lr")
    base_table = _rows_table(["Маршрут", "Поток", "Пропускная", "Цена", "Стоимость"], base_rows, "lrrrr")
    scenario_table = _rows_table(["Сценарий", "Статус", "Стоимость", "Δ к базе", "Изменение"], scen_rows, "llrrl")
    changes_table = _rows_table(["Маршрут", "База", "После отказа", "Δ поток"], change_rows, "lrrr")
    overload_changes_table = _rows_table(["Маршрут", "База", "Перегрузка", "Δ поток"], overload_change_rows, "lrrr")
    bottleneck_table = _rows_table(["Маршрут", "Поток", "Предел", "Теневая цена"], bottleneck_rows, "lrrr")
    potentials_table = _rows_table(["Узел", "Тип", "Потенциал"], potential_rows, "llr")
    milp_table = _rows_table(["Маршрут", "Целый поток", "Пропускная", "Цена", "Стоимость"], milp_rows, "lrrrr")
    milp_changes_table = _rows_table(["Маршрут", "LP", "MILP", "Δ поток"], milp_change_rows, "lrrr")
    verification_table = _rows_table(["Решение", "Допустимо", "Ошибка склад", "Ошибка магазин", "Наруш. ёмкости", "Запрещ. дуги"], ver_rows, "llrrrr")

    tex = rf"""\documentclass[12pt,a4paper]{{article}}
\usepackage[margin=1.8cm]{{geometry}}
\usepackage{{fontspec}}
\setmainfont{{DejaVu Serif}}
\setsansfont{{DejaVu Sans}}
\usepackage{{polyglossia}}
\setdefaultlanguage{{russian}}
\newfontfamily\cyrillicfonttt{{DejaVu Sans Mono}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs,array,longtable}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{caption}}
\usepackage{{enumitem}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\hypersetup{{colorlinks=false,pdfborder={{0 0 0}}}}
\setlength{{\parindent}}{{1.25cm}}
\setlength{{\parskip}}{{0.25em}}
\captionsetup{{font=small}}

\begin{{document}}

\begin{{center}}
{{\large Национальный исследовательский ядерный университет «МИФИ»}}\\[1.4cm]
{{\Large \textbf{{Лабораторная работа №2}}}}\\[0.3cm]
{{\large «Транспортные и сетевые модели»}}\\[0.8cm]
Дисциплина «Методы оптимизации. Линейное программирование»\\[1.2cm]
\textbf{{Вариант 19}}\\[2.5cm]
\end{{center}}

\begin{{flushright}}
Выполнила: Федоровская Е. С.\\
Группа: Б24-517
\end{{flushright}}

\vfill
\begin{{center}}Москва, 2026\end{{center}}
\newpage

\section{{Постановка задачи}}
По формуле индивидуализации для $N=25$, $Q=1$, $Y=26$ получен вариант
\[
V=1+((17N+7Q+Y)\bmod 20)=19,
\]
seed генерации данных равен $10125$.

Вариант 19 --- транспортная задача с запретом части маршрутов. Рассматриваются четыре склада и шесть магазинов. Требуется распределить единый поток товара от складов к магазинам с минимальной суммарной стоимостью, выполнить спрос каждого магазина, не превысить запас каждого склада и пропускную способность маршрутов. Пять маршрутов запрещены.

\subsection{{Исходные данные}}
\begin{{minipage}}[t]{{0.46\textwidth}}
\centering
{supply_table}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.46\textwidth}}
\centering
{demand_table}
\end{{minipage}}

Общий запас равен общему базовому спросу: $227$ ед. Запрещённые маршруты: {forbidden_text}.

В таблице ниже в каждой разрешённой ячейке указано \emph{{стоимость/пропускная способность}}. Символ X означает запрещённый маршрут.

\begin{{center}}
{cost_cap_matrix}
\end{{center}}

Данные синтетические и воспроизводятся по seed $10125$. Стоимости формируются на основе расстояний между сгенерированными координатами узлов с небольшой добавкой, пропускные способности --- из запасов и спроса. Запрещённые дуги выбираются так, чтобы у каждого узла оставалось несколько допустимых связей.

\section{{Математическая модель}}
Пусть $I$ --- множество складов, $J$ --- множество магазинов, $A\subseteq I\times J$ --- множество разрешённых маршрутов.

Параметры:
\begin{{itemize}}[nosep]
\item $s_i$ --- запас склада $i$, ед. товара;
\item $d_j$ --- спрос магазина $j$, ед. товара;
\item $c_{{ij}}$ --- стоимость перевозки одной единицы по маршруту $(i,j)$, усл. ед./ед. товара;
\item $u_{{ij}}$ --- пропускная способность маршрута $(i,j)$, ед. товара.
\end{{itemize}}

Переменная $x_{{ij}}\ge0$ --- объём перевозки со склада $i$ в магазин $j$, ед. товара. Переменные непрерывные, что соответствует ЛП-постановке.

Целевая функция:
\[
\min z=\sum_{{(i,j)\in A}}c_{{ij}}x_{{ij}}.
\]

Баланс складов:
\[
\sum_{{j:(i,j)\in A}}x_{{ij}}=s_i,\qquad i\in I.
\]

Баланс магазинов:
\[
\sum_{{i:(i,j)\in A}}x_{{ij}}=d_j,\qquad j\in J.
\]

Пропускные способности:
\[
0\le x_{{ij}}\le u_{{ij}},\qquad (i,j)\in A.
\]

Для запрещённых маршрутов переменные в модель не включаются, что эквивалентно условию $x_{{ij}}=0$.

\section{{Реализация}}
Модель реализована на Python. Решение выполняется функцией \texttt{{scipy.optimize.linprog}} с HiGHS, методом dual simplex (\texttt{{highs-ds}}), presolve включён. Данные хранятся отдельно в CSV; результаты, проверки, логи и графики сохраняются в каталоге \texttt{{results}}. Полный диагностический вывод HiGHS для базовой задачи сохраняется в \texttt{{solver\_base\_full.log}}.

Независимая проверка реализована отдельным NumPy-кодом. Она заново читает CSV решения и проверяет неотрицательность, нулевой поток по запрещённым маршрутам, пропускные способности, балансы всех десяти узлов и пересчитывает целевую функцию без использования объекта солвера.

\section{{Базовый сценарий}}
HiGHS вернул статус \texttt{{Optimal}} за {run_info['base_iterations']} итераций. Минимальная стоимость:
\[
\boxed{{z^*={run_info['base_objective_cost']:.3f}}}.
\]

Ненулевые потоки оптимального плана:
\begin{{center}}
{base_table}
\end{{center}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{results/plots/network_base.png}}
\caption{{Базовый оптимальный поток. Пунктиром показаны запрещённые маршруты.}}
\end{{figure}}

\section{{Независимая проверка}}
Проверки выполнены с допуском $10^{{-7}}$.

\begin{{center}}
\resizebox{{\textwidth}}{{!}}{{%
{verification_table}
}}
\end{{center}}

Для базового решения максимальные нарушения ограничений равны нулю, а целевая функция при независимом пересчёте равна {ver_base['objective_recomputed']:.3f}. Таким образом, решение допустимо и согласуется с результатом HiGHS.

\section{{Сценарный анализ}}
\begin{{center}}
\resizebox{{\textwidth}}{{!}}{{%
{scenario_table}
}}
\end{{center}}

\subsection{{Отказ самой загруженной дуги}}
В базовом решении максимальный поток $29.4$ ед. проходит по маршруту {run_info['most_loaded_route'].replace('->','→')}. После его закрытия задача остаётся допустимой, но минимальная стоимость возрастает до
\[
{run_info['arc_failure_objective_cost']:.3f},
\]
то есть ущерб составляет
\[
\boxed{{{run_info['arc_failure_damage']:.3f}}}.
\]

Перестроение потока:
\begin{{center}}
{changes_table}
\end{{center}}

Основной поток с W1→M1 перераспределяется через W2 и W4; из-за этого меняются и другие маршруты, чтобы одновременно сохранить балансы складов, спрос магазинов и ограничения пропускной способности.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{results/plots/network_failure.png}}
\caption{{Оптимальный поток после отказа наиболее загруженного маршрута.}}
\end{{figure}}

\subsection{{Перегрузка дуги}}
Для отдельного сценария перегрузки выбран наиболее критичный насыщенный маршрут {run_info['overload_route'].replace('->','→')}, имеющий максимальную теневую цену среди насыщенных дуг. Его пропускная способность уменьшена на 20\%:
\[
{run_info['overload_capacity_base']:.2f} \rightarrow {run_info['overload_capacity_new']:.2f}.
\]
После повторной оптимизации сеть остаётся допустимой, а минимальная стоимость возрастает до
\[
{run_info['overload_objective_cost']:.3f},
\]
то есть ущерб относительно базы составляет
\[
\boxed{{{run_info['overload_damage']:.3f}}}.
\]

Перестроение потоков:
\begin{{center}}
{overload_changes_table}
\end{{center}}

Локальный прогноз по базовой теневой цене для такого уменьшения пропускной способности составляет примерно {run_info['overload_shadow_prediction']:.3f} усл. ед. Фактический рост стоимости равен {run_info['overload_damage']:.3f}. Различие показывает, что при уменьшении ёмкости сразу на 20\% структура оптимального решения меняется; теневая цена является локальной оценкой и не обязана оставаться точной на большом интервале.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{results/plots/network_overload.png}}
\caption{{Оптимальный поток после уменьшения пропускной способности критичной дуги на 20\%.}}
\end{{figure}}

\subsection{{Рост спроса на 20\%}}
После увеличения спроса всех магазинов на 20\% HiGHS возвращает статус \texttt{{{_esc(growth['status'])}}}. Общий спрос становится {growth['total_demand_units']:.1f} ед., тогда как суммарный запас остаётся {growth['total_supply_units']:.1f} ед. Уже на агрегированном уровне возникает дефицит
\[
{growth['aggregate_shortage_units']:.1f}\text{{ ед.}},
\]
поэтому выполнить все балансовые ограничения невозможно независимо от распределения по маршрутам. Это содержательное объяснение несовместимости сценария.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\textwidth]{{results/plots/scenario_costs.png}}
\caption{{Стоимость допустимых сценариев. Для роста спроса стоимость отсутствует, поскольку задача несовместима.}}
\end{{figure}}

\section{{Узкие места и двойственная интерпретация}}
Насыщенной считается дуга, для которой поток равен её пропускной способности. В базовом оптимуме таких маршрутов четыре:

\begin{{center}}
{bottleneck_table}
\end{{center}}

Наибольшая теневая цена у маршрута {bottlenecks.iloc[0]['warehouse_id']}→{bottlenecks.iloc[0]['store_id']}: {bottlenecks.iloc[0]['capacity_shadow_benefit']:.2f}. При небольшом увеличении его пропускной способности на одну единицу минимальная стоимость локально уменьшается примерно на {bottlenecks.iloc[0]['capacity_shadow_benefit']:.2f} усл. ед., пока структура оптимального базиса не изменится. Поэтому этот маршрут --- первый кандидат на расширение.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\textwidth]{{results/plots/bottleneck_shadow_prices.png}}
\caption{{Теневые цены насыщенных маршрутов.}}
\end{{figure}}

Маргинальные значения равенств баланса можно интерпретировать как один набор узловых потенциалов двойственной задачи:
\begin{{center}}
{potentials_table}
\end{{center}}

Поскольку в сбалансированной транспортной задаче одно балансовое равенство линейно зависимо от остальных, потенциалы определяются с точностью до общего сдвига; экономически существенны их комбинации и приведённые стоимости дуг.

\section{{Сравнение с жадной эвристикой}}
Для сравнения реализован простой алгоритм «сначала самые дешёвые маршруты». Разрешённые дуги сортируются по стоимости; на каждой дуге отправляется максимально возможный поток с учётом оставшегося запаса, спроса и пропускной способности.

Для текущих данных эвристика получила допустимый план со стоимостью
\[
{greedy['objective_cost']:.3f},
\]
что на {greedy['absolute_gap']:.3f} выше оптимума. Относительный зазор:
\[
\boxed{{{greedy['gap_percent']:.3f}\%}}.
\]
Жадный выбор локально дешёвых маршрутов не учитывает последствия для остальных узлов и потому уступает глобальному LP-оптимуму.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.68\textwidth]{{results/plots/optimal_vs_greedy.png}}
\caption{{HiGHS и жадная эвристика на одном наборе данных.}}
\end{{figure}}


\section{{Дополнительное задание: целочисленная версия}}
По методическим указаниям реализована целочисленная версия транспортной модели с неделимыми рейсами. Одна единица $x_{{ij}}$ интерпретируется как одна стандартная неделимая партия перевозки, поэтому
\[
x_{{ij}}\in\mathbb{{Z}}_{{\ge 0}}.
\]
Целевая функция, балансы, запрещённые маршруты и пропускные способности остаются теми же. Дробная ёмкость является верхней границей: например, при $x_{{ij}}\le18.6$ целочисленная модель допускает не более 18 рейсов. MILP решается функцией \texttt{{scipy.optimize.milp}} через HiGHS.

Непрерывная LP-модель даёт стоимость
\[
z_{{LP}}={milp_summary['lp_objective_cost']:.3f},
\]
а целочисленная модель ---
\[
z_{{MILP}}={milp_summary['milp_objective_cost']:.3f}.
\]
Абсолютное ухудшение составляет {milp_summary['absolute_gap']:.3f}, интегральный зазор
\[
\boxed{{\mathrm{{gap}}={milp_summary['integrality_gap_percent']:.3f}\%}}.
\]
Полученный MILP-статус --- \texttt{{Optimal}}; финальный MIP gap равен {milp_summary['mip_gap']:.3g}, число узлов branch-and-bound --- {milp_summary['mip_node_count']}. Независимая проверка подтверждает балансы, ограничения пропускной способности и целочисленность потоков: максимальное отклонение от ближайшего целого равно {ver_milp['max_integrality_violation']:.2e}.

Ненулевые потоки целочисленного плана:
\begin{{center}}
{milp_table}
\end{{center}}

Наиболее заметные изменения относительно LP-плана:
\begin{{center}}
{milp_changes_table}
\end{{center}}

Дробные значения LP на отдельных дугах в MILP заменяются целыми значениями. Из-за этого поток перераспределяется между маршрутами и стоимость немного возрастает. Разница небольшая, поэтому непрерывная модель даёт хорошую нижнюю оценку стоимости, но MILP формирует план, совместимый с требованием неделимости стандартных рейсов.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.52\textwidth]{{results/plots/lp_vs_milp.png}}
\caption{{Сравнение стоимости непрерывного LP- и целочисленного MILP-планов.}}
\end{{figure}}

\section{{Практическая интерпретация и ограничения модели}}
Найденный план можно использовать как агрегированный план перевозок одного однородного товара. Он показывает, какие маршруты загружать и где расширение пропускной способности даёт наибольший локальный эффект.

При практическом применении необходимо учитывать ограничения учебной модели:
\begin{{itemize}}[nosep]
\item рассматривается один вид товара и один период;
\item в основной LP-модели объёмы перевозок непрерывны; в дополнительном задании рассмотрена целочисленная версия;
\item стоимости и пропускные способности считаются детерминированными и линейными;
\item отсутствуют фиксированные затраты на запуск рейса, временные окна и парк транспортных средств;
\item сеть двухуровневая: склад → магазин, без промежуточных узлов;
\item запрещённые маршруты и ёмкости заданы заранее и не оптимизируются.
\end{{itemize}}

Поэтому значение {run_info['base_objective_cost']:.3f} является минимумом именно для сформулированной ЛП-модели, а не универсальным минимумом реальной логистической системы.

\section{{Вывод}}
Для варианта 19 построена транспортная ЛП-модель сети из 4 складов и 6 магазинов с пятью запрещёнными маршрутами и ограничениями пропускной способности. Базовая задача решена HiGHS со стоимостью {run_info['base_objective_cost']:.3f}; независимая проверка подтверждает допустимость. Отказ наиболее загруженной дуги повышает стоимость на {run_info['arc_failure_damage']:.3f}. При отдельном сценарии перегрузки критичной дуги {run_info['overload_route'].replace('->','→')} её пропускная способность уменьшена на 20\%, что повышает стоимость на {run_info['overload_damage']:.3f}. Рост спроса на 20\% приводит к несовместимости из-за дефицита {growth['aggregate_shortage_units']:.1f} ед. Выявлены четыре насыщенные дуги и их теневые цены. Жадная эвристика хуже оптимума на {greedy['gap_percent']:.3f}\%, что показывает пользу решения полной ЛП-модели. В дополнительном задании построена целочисленная версия с неделимыми рейсами; её стоимость равна {milp_summary['milp_objective_cost']:.3f}, интегральный зазор относительно LP составляет {milp_summary['integrality_gap_percent']:.3f}\%.

\end{{document}}
"""

    path = root / "report.tex"
    path.write_text(tex, encoding="utf-8")
    return path
