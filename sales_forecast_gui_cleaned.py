import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from datetime import datetime
import threading

# Глобальная переменная для окна логов
log_window = None
log_text = None
loaded_data = None  # Для хранения загруженных данных из CSV

# Функция для подключения к базе данных
def create_conn():
    return create_engine(
        'mssql+pyodbc://SRV/DB?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server'
    )
    return engine

# Функция для загрузки данных за один период
def fetch_data(year, month):
    date_key = f"{year}{month:02d}"
    query = f"""

    """
    engine = create_conn()
    try:
        return pd.read_sql_query(query, engine)
    except Exception as e:
        print(f"Ошибка при загрузке данных за {year}-{month:02d}: {e}")
        return pd.DataFrame()  # Возвращаем пустой DataFrame в случае ошибки
        
# Функция для многопоточной загрузки данных
def fetch_data_multithreaded(years, months):
    results = []
    total_tasks = len(years) * len(months)  # Общее количество задач
    completed_tasks = 0

    with ThreadPoolExecutor(max_workers=4) as executor:  # Параллельное выполнение
        futures = {
            executor.submit(fetch_data, year, month): (year, month) 
            for year in years for month in months
        }

        for future in futures:
            year, month = futures[future]
            try:
                result = future.result()
                completed_tasks += 1

                if not result.empty:
                    results.append(result)
                    add_log(f"Данные за {year}-{month:02d} успешно загружены. ({completed_tasks}/{total_tasks})")
                else:
                    add_log(f"Данные за {year}-{month:02d} отсутствуют. ({completed_tasks}/{total_tasks})")
            except Exception as e:
                add_log(f"Ошибка при загрузке данных за {year}-{month:02d}: {e} ({completed_tasks}/{total_tasks})")

    if results:
        combined_data = pd.concat(results, ignore_index=True)
        add_log(f"Успешно объединено {len(combined_data)} строк из {len(results)} загруженных периодов.")
        return combined_data
    else:
        add_log("Все периоды оказались пустыми, данные не загружены.")
        return pd.DataFrame()

# Функция добавления логов
def add_log(message):
    if log_text:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Формат без дробной части секунд
        log_text.config(state=tk.NORMAL)
        log_text.insert(tk.END, f"{current_time} - {message}\n")
        log_text.see(tk.END)
        log_text.config(state=tk.DISABLED)

# Функция для открытия окна логов
def show_log_window():
    global log_window, log_text
    if log_window is None or not log_window.winfo_exists():
        log_window = tk.Toplevel()
        log_window.title("Логи")
        log_window.geometry("600x400")
        log_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Отключаем кнопку закрытия
        log_text = tk.Text(log_window, bg="black", fg="white", state=tk.DISABLED)
        log_text.pack(fill=tk.BOTH, expand=True)
    else:
        log_window.deiconify()

# Функция сохранения полученных данных
def save_results(data, selected_years, selected_months, selected_categories):
    """
    Сохраняет результаты в CSV с меткой времени и параметрами.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings = (
        f"years_{'_'.join(map(str, selected_years))}_"
        f"months_{'_'.join(map(str, selected_months))}_"
        f"categories_{'_'.join(selected_categories)}"
    )
    filename = f"predictions_{settings}_{timestamp}.csv"
    data.to_csv(filename, index=False, encoding='utf-8-sig')
    add_log(f"Результаты сохранены в файл: {filename}")
        
def collect_sales_by_month_updated(years, months, use_csv):
    if use_csv:
        try:
            add_log("Чтение данных из файла 'exported_data.csv'...")
            return pd.read_csv("exported_data.csv")
        except FileNotFoundError:
            raise FileNotFoundError("Файл 'exported_data.csv' не найден. Переключитесь на запрос к базе данных или создайте выгрузку.")

    add_log(f"Начало многопоточной загрузки данных для {years}...")
    return fetch_data_multithreaded(years, months)


# Функция классификации категорий
def classify_final_category(name):
    if '_' in name:
        category = name.split('_')[0]
        if 'КОР' in name:
            return 'КОНФ_КОР'
        elif 'ВЕС' in name:
            return 'КОНФ_ВЕС'
        elif 'КАРАМЕЛЬ' in name.upper():
            return 'КАРАМЕЛЬ'
        elif 'ЗЕФИР' in name.upper():
            return 'ЗЕФИР'
        elif 'КОНД' in name.upper():
            if 'ИЗД' in name:
                return 'КОНД_ИЗД'
            elif 'ПЛИТКА' in name:
                return 'КОНД_ПЛИТКА'
        elif 'ВАФ' in name and 'ТОРТ' in name:
            return 'ВАФ_ТОРТ'
        return category
    return 'UNKNOWN'

# Функция подготовки данных
def prepare_data(sales_data):
    if sales_data.empty:
        raise ValueError("Данные для подготовки пусты.")
        
    sales_data = sales_data.drop_duplicates()
    
    # Проверка существования столбца SalesAmount
    if 'SalesAmount' not in sales_data.columns:
        sales_data['SalesAmount'] = sales_data.get('RegularSalesAmount', 0) + sales_data.get('DiscountSalesAmount', 0)
        add_log("Столбец 'SalesAmount' был создан из 'RegularSalesAmount' и 'DiscountSalesAmount'")
    
    add_log(f"Обработка данных: {len(sales_data)} строк")
    sales_data['DATE100KEY'] = pd.to_datetime(sales_data['DATE100KEY'].astype(str), format='%Y%m')
    sales_data['Year'] = sales_data['DATE100KEY'].dt.year
    sales_data['Month'] = sales_data['DATE100KEY'].dt.month
    sales_data['Category'] = sales_data['SkuNameUC'].apply(classify_final_category)
    
    # Исключение строк с неизвестными категориями
    sales_data = sales_data[sales_data['Category'] != 'UNKNOWN']
    add_log(f"Данные после фильтрации категорий: {len(sales_data)} строк")
    
    # Агрегация данных
    aggregated_data = sales_data.groupby(['Year', 'Month', 'WeekNumberOfYear', 'SkuCodeUC', 'Category']).agg(
        TotalSalesAmount=('SalesAmount', 'sum'),
        TotalSalesQty=('SalesQty', 'sum')
    ).reset_index()
    add_log(f"Данные после агрегации: {len(aggregated_data)} строк")
    
    # Удаление выбросов
    aggregated_data = aggregated_data[aggregated_data['TotalSalesQty'] < aggregated_data['TotalSalesQty'].quantile(0.99)]
    add_log(f"Данные после удаления выбросов: {len(aggregated_data)} строк")
    
    # Создание лагов
    aggregated_data['Lag_1'] = aggregated_data.groupby('SkuCodeUC')['TotalSalesQty'].shift(1)
    aggregated_data['Lag_2'] = aggregated_data.groupby('SkuCodeUC')['TotalSalesQty'].shift(2)
    aggregated_data = aggregated_data.dropna(subset=['Lag_1', 'Lag_2'])
    add_log(f"Данные после удаления пропусков: {len(aggregated_data)} строк")
    
    return aggregated_data


# Функция для обработки данных в отдельном потоке
def process_data(frame_graph, use_csv):
    show_log_window()  # Открываем окно логов
    try:
        add_log("Начало обработки данных за 2022–2024 годы...")
        years_train = [2022, 2023]
        year_test = [2024]

        # Загрузка данных
        train_data = prepare_data(collect_sales_by_month_updated(years_train, list(range(1, 13)), use_csv))
        test_data = prepare_data(collect_sales_by_month_updated(year_test, list(range(1, 13)), use_csv))
        add_log(f"Загружено данных для обучения: {len(train_data)} строк")
        add_log(f"Загружено данных для тестирования: {len(test_data)} строк")
        train_data = train_data[train_data['Year'].isin([2022, 2023])]
        if train_data.empty or test_data.empty:
            add_log("Ошибка: недостаточно данных для анализа.")
            return

        # Обучение моделей
        models = {
            'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6),
            'CatBoost': CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, silent=True)
        }

        X_train = train_data[['Month', 'WeekNumberOfYear', 'Lag_1', 'Lag_2']].astype(float)
        y_train = train_data['TotalSalesQty']
        X_test = test_data[['Month', 'WeekNumberOfYear', 'Lag_1', 'Lag_2']].astype(float)
        y_test = test_data['TotalSalesQty']

        best_model_name = None
        best_mae = float('inf')

        for name, model in models.items():
            model.fit(X_train, y_train)
            test_data[f"{name}_Predicted"] = model.predict(X_test)
            mae = mean_absolute_error(y_test, test_data[f"{name}_Predicted"])
            add_log(f"Модель {name} обучена. MAE: {mae:.4f}")

            if mae < best_mae:
                best_mae = mae
                best_model_name = name

        add_log(f"Выбрана лучшая модель: {best_model_name} с MAE: {best_mae:.4f}")

        # Использование лучшей модели для прогноза на 2025 год
        forecast_year = 2025
        forecast_data = prepare_data(collect_sales_by_month_updated([forecast_year], list(range(1, 13)), use_csv))

        # Создание прогноза
        forecast_data['Forecast'] = models[best_model_name].predict(
            forecast_data[['Month', 'WeekNumberOfYear', 'Lag_1', 'Lag_2']].astype(float)
        )

        # Нормализация прогноза, если требуется
        forecast_data['Forecast'] = forecast_data['Forecast'] / forecast_data['Forecast'].max() * forecast_data['TotalSalesQty'].max()
        add_log("Прогноз успешно создан для 2025 года.")

        # Построение графиков
        plot_forecast_with_actual(forecast_data, train_data, test_data, best_model_name, frame_graph)

        # Сохранение данных
        save_results(forecast_data, [forecast_year], list(range(1, 13)), [])

    except Exception as e:
        add_log(f"Ошибка при обработке данных: {e}")

def plot_forecast(data, best_model_name, frame_graph):
    """
    Построение графиков прогноза на 2025 год.
    """
    # Очистка предыдущих вкладок
    for widget in frame_graph.winfo_children():
        widget.destroy()

    tabs = ttk.Notebook(frame_graph)
    tabs.pack(fill="both", expand=True)

    for category in data['Category'].unique():
        subset = data[data['Category'] == category]
        if subset.empty:
            continue

        figure = plt.Figure(figsize=(10, 6), dpi=100)
        ax = figure.add_subplot(111)

        # Прогноз на 2025 год
        forecast_data = subset.groupby('Month')['Forecast'].sum()
        months = range(1, 13)
        ax.plot(months, [forecast_data.get(month, 0) for month in months], label=f'Прогноз ({best_model_name})', marker='x')

        ax.set_title(f"Прогноз на 2025 год. Категория: {category}")
        ax.set_xlabel("Месяцы")
        ax.set_ylabel("Продажи")
        ax.set_xticks(months)
        ax.set_xticklabels(
            [f"{i:02}-{'январь февраль март апрель май июнь июль август сентябрь октябрь ноябрь декабрь'.split()[i - 1]}"
             for i in months],
            rotation=45
        )
        ax.legend()

        tab = ttk.Frame(tabs)
        tabs.add(tab, text=category)
        canvas = FigureCanvasTkAgg(figure, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
def plot_forecast_with_actual(data, train_data, test_data, best_model_name, frame_graph):
    """
    Построение графиков прогноза на 2025 год с фактическими данными за 2024 год.
    """
    # Очистка предыдущих вкладок
    for widget in frame_graph.winfo_children():
        widget.destroy()

    tabs = ttk.Notebook(frame_graph)
    tabs.pack(fill="both", expand=True)

    for category in data['Category'].unique():
        subset = data[data['Category'] == category]
        test_subset = test_data[test_data['Category'] == category]

        if subset.empty and test_subset.empty:
            continue

        figure = plt.Figure(figsize=(10, 6), dpi=100)
        ax = figure.add_subplot(111)

        # Фактические данные за 2024 год
        if not test_subset.empty:
            test_actual = test_subset.groupby('Month')['TotalSalesQty'].sum()
            ax.plot(
                test_actual.index,
                test_actual,
                label='Фактические данные 2024',
                marker='o',
                color='blue'
            )

        # Прогноз на 2025 год
        forecast_data = subset.groupby('Month')['Forecast'].sum()
        ax.plot(
            forecast_data.index,
            forecast_data,
            label=f'Прогноз ({best_model_name}) 2025',
            color='red',
            marker='x',
            linestyle='--'
        )

        # Настройки графика
        ax.set_title(f"Прогноз на 2025 год. Категория: {category}")
        ax.set_xlabel("Месяцы")
        ax.set_ylabel("Продажи")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(
            [f"{i:02}-{'январь февраль март апрель май июнь июль август сентябрь октябрь ноябрь декабрь'.split()[i - 1]}"
             for i in range(1, 13)],
            rotation=45
        )
        ax.legend()

        # Добавление вкладки для графика
        tab = ttk.Frame(tabs)
        tabs.add(tab, text=category)
        canvas = FigureCanvasTkAgg(figure, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

# Функция построения графиков
def plot_predictions(data, models, frame_graph):
    """
    Построение графиков прогнозов по категориям.
    """
    # Очистка предыдущих вкладок
    for widget in frame_graph.winfo_children():
        widget.destroy()

    tabs = ttk.Notebook(frame_graph)
    tabs.pack(fill="both", expand=True)

    for category in data['Category'].unique():
        subset = data[data['Category'] == category]
        if subset.empty:
            continue

        # Создание графика
        figure = plt.Figure(figsize=(10, 6), dpi=100)
        ax = figure.add_subplot(111)

        # Фактические продажи
        actual_data = subset.groupby('Month')['TotalSalesQty'].sum()
        months = range(1, 13)  # Полный список месяцев
        ax.plot(months, [actual_data.get(month, 0) for month in months], label='Фактические продажи', marker='o')

        # Прогнозы моделей
        for name in models.keys():
            predicted_data = subset.groupby('Month')[f"{name}_Predicted"].sum()
            ax.plot(months, [predicted_data.get(month, 0) for month in months], label=f'Прогноз ({name})', marker='x')

        # Настройки осей и легенды
        ax.set_title(f"Категория: {category}")
        ax.set_xlabel("Месяцы")
        ax.set_ylabel("Продажи")
        ax.set_xticks(months)
        ax.set_xticklabels(
            [f"{i:02}-{'январь февраль март апрель май июнь июль август сентябрь октябрь ноябрь декабрь'.split()[i - 1]}"
             for i in months],
            rotation=45
        )
        ax.legend()

        # Добавление вкладки для графика
        tab = ttk.Frame(tabs)
        tabs.add(tab, text=category)
        canvas = FigureCanvasTkAgg(figure, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

# Функция загрузки данных из файла
def load_analyzed_data(frame_graph):
    global loaded_data
    try:
        filepath = filedialog.askopenfilename(
            title="Выберите CSV-файл",
            filetypes=(("CSV файлы", "*.csv"), ("Все файлы", "*.*"))
        )
        if not filepath:
            add_log("Файл не выбран.")
            return
        
        loaded_data = pd.read_csv(filepath)
        add_log(f"Данные успешно загружены из файла: {filepath}")
        
        # Построение графиков
        plot_loaded_data(loaded_data, frame_graph)
    except Exception as e:
        add_log(f"Ошибка при загрузке данных: {e}")

# Функция построения графиков для загруженных данных
def plot_loaded_data(data, frame_graph):
    if data is None or data.empty:
        add_log("Нет данных для построения графиков.")
        return

    # Очистка предыдущих вкладок
    for widget in frame_graph.winfo_children():
        widget.destroy()

    tabs = ttk.Notebook(frame_graph)
    tabs.pack(fill="both", expand=True)

    for category in data['Category'].unique():
        subset = data[data['Category'] == category]

        if subset.empty:
            continue

        figure = plt.Figure(figsize=(10, 6), dpi=100)
        ax = figure.add_subplot(111)

        # Фактические данные за 2024 год
        actual_2024 = subset[(subset['Year'] == 2024)].groupby('Month')['TotalSalesQty'].sum()
        if not actual_2024.empty:
            ax.plot(
                actual_2024.index,
                actual_2024,
                label='Фактические данные 2024',
                marker='o',
                color='blue'
            )

        # Прогноз на 2025 год
        forecast_2025 = subset[(subset['Year'] == 2025)].groupby('Month')['Forecast'].sum()
        if not forecast_2025.empty:
            ax.plot(
                forecast_2025.index,
                forecast_2025,
                label='Прогноз 2025',
                marker='x',
                color='red',
                linestyle='--'
            )

        # Настройки графика
        ax.set_title(f"Категория: {category}")
        ax.set_xlabel("Месяцы")
        ax.set_ylabel("Продажи")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(
            [f"{i:02}-{'январь февраль март апрель май июнь июль август сентябрь октябрь ноябрь декабрь'.split()[i - 1]}"
             for i in range(1, 13)],
            rotation=45
        )
        ax.legend()

        # Добавление вкладки для графика
        tab = ttk.Frame(tabs)
        tabs.add(tab, text=category)
        canvas = FigureCanvasTkAgg(figure, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        
#Функция сохранения всех данных        
def export_all_data():
    show_log_window()  # Открываем окно логов
    try:
        add_log("Начало выгрузки всех данных...")
        # Определение диапазона годов и месяцев для выгрузки
        current_year = datetime.now().year
        current_month = datetime.now().month
        years = range(2022, current_year + 1)

        all_data = []
        for year in years:
            months = range(1, 13)  # Все месяцы с января по декабрь
            if year == current_year:  # Для текущего года — до текущего месяца
                months = range(1, current_month + 1)
            
            # Выгрузка данных за текущий год и указанные месяцы
            data = collect_sales_by_month_updated([year], months, use_csv=False)
            if not data.empty:  # Добавляем только непустые DataFrame
                all_data.append(data)
            else:
                add_log(f"Нет данных за {year} год.")

        if all_data:  # Проверяем, что список не пустой
            combined_data = pd.concat(all_data, ignore_index=True)
            # Сохраняем в CSV
            filename = "exported_data.csv"
            combined_data.to_csv(filename, index=False, encoding="utf-8-sig")
            add_log(f"Данные успешно сохранены в файл: {filename}")
        else:
            add_log("Данные отсутствуют. Нечего сохранять.")
    except Exception as e:
        add_log(f"Ошибка при выгрузке данных: {e}")

# Основной интерфейс
def run_interface():
    root = tk.Tk()
    root.title("Прогнозирование продаж")

    # Определяем глобальную переменную для выбора источника данных
    use_csv_data = tk.BooleanVar(value=False)  # По умолчанию запросы из базы данных

    frame_controls = tk.Frame(root)
    frame_controls.pack(side="left", padx=10, pady=10)

    frame_graph = ttk.Frame(root)
    frame_graph.pack(side="right", fill="both", expand=True)

    # Элементы управления
    tk.Label(frame_controls, text="Источник данных:").pack()
    tk.Radiobutton(frame_controls, text="Запросы из базы данных", variable=use_csv_data, value=False).pack(anchor="w")
    tk.Radiobutton(frame_controls, text="Данные из файла exported_data.csv", variable=use_csv_data, value=True).pack(anchor="w")
    
    # Кнопка для выгрузки всех данных
    tk.Button(
        frame_controls,
        text="Сохранить все данные",
        command=lambda: threading.Thread(target=export_all_data).start()
    ).pack(pady=5)

    # Кнопка для запуска обработки данных
    tk.Button(
        frame_controls,
        text="Запустить прогноз",
        command=lambda: threading.Thread(target=process_data, args=(frame_graph, use_csv_data.get())).start()
    ).pack(pady=10)

    # Кнопка для загрузки сохраненных прогнозов
    tk.Button(
        frame_controls,
        text="Загрузить проанализированные данные",
        command=lambda: threading.Thread(target=load_analyzed_data, args=(frame_graph,)).start()
    ).pack(pady=5)

    root.mainloop()

# Запуск интерфейса
run_interface()