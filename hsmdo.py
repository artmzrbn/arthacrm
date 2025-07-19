import pandas as pd
import numpy as np
from cmdstanpy import install_cmdstan
install_cmdstan()
from cmdstanpy import CmdStanModel
import os


# =============================================================================
# ШАГ 1: ФУНКЦИЯ ДЛЯ ПОДГОТОВКИ ДАННЫХ
# =============================================================================

def prepare_data_for_hsmdo(df, holdout_months, analysis_date_str=None):
    """
    Подготавливает транзакционные данные для модели HSMDO.

    Args:
        df (pd.DataFrame): Исходный DataFrame со столбцами ['order_date', 'name', 'order_id'].
        holdout_months (int): Количество последних месяцев для тестовой выборки.
        analysis_date_str (str, optional): Дата, на которую проводится анализ (конец периода наблюдений),
                                         в формате 'YYYY-MM-DD'. Если None, используется последняя дата в df.

    Returns:
        dict: Словарь с подготовленными данными для STAN.
    """
    print("Шаг 1: Подготовка данных...")
    
    # Копируем исходный df, чтобы не изменять его
    df_copy = df.copy()
    df_copy['order_date'] = pd.to_datetime(df_copy['order_date'], dayfirst=True)

    # Определяем "настоящий момент" для анализа.
    if analysis_date_str:
        analysis_date = pd.to_datetime(analysis_date_str)
        df_processed = df_copy[df_copy['order_date'] <= analysis_date].copy()
    else:
        analysis_date = df_copy['order_date'].max()
        df_processed = df_copy
    
    print(f"Анализ проводится на дату: {analysis_date.date()}")

    # Используем 'name' как customer_id.
    df_processed = df_processed.rename(columns={'name': 'customer_id'})
    
    # Агрегация по месяцам.
    start_date = df_copy['order_date'].min()
    df_processed['month_number'] = (df_processed['order_date'].dt.year - start_date.year) * 12 + \
                                   (df_processed['order_date'].dt.month - start_date.month) + 1
    
    monthly_purchases = df_processed.groupby(['customer_id', 'month_number'])['order_id'].nunique().reset_index()
    monthly_purchases = monthly_purchases.rename(columns={'order_id': 'purchase_count'})

    # Общее количество месяцев от начала данных до даты анализа
    total_months = (analysis_date.year - start_date.year) * 12 + \
                   (analysis_date.month - start_date.month) + 1

    # Создаем полную матрицу N x T
    all_months = pd.RangeIndex(start=1, stop=total_months + 1, name='month_number')
    x_matrix = monthly_purchases.pivot_table(index='customer_id', 
                                             columns='month_number', 
                                             values='purchase_count').reindex(columns=all_months, fill_value=0)

    # Разделение на обучающую и тестовую выборки
    T_cal = total_months - holdout_months
    x_cal = x_matrix.iloc[:, :T_cal]
    
    # ИСПРАВЛЕНИЕ: Фильтруем покупателей, у которых не было покупок в обучающем периоде.
    # Модель не может работать с такими данными, и это приводит к ошибке.
    initial_customer_count = len(x_cal)
    customers_with_purchases = x_cal.sum(axis=1) > 0
    x_cal = x_cal[customers_with_purchases]
    if initial_customer_count > len(x_cal):
        print(f"Отфильтровано {initial_customer_count - len(x_cal)} покупателей без покупок в обучающем периоде.")

    # Расчет Recency (t) - номер последнего месяца с покупкой в обучающем периоде
    # Теперь эта функция никогда не вернет 0, так как мы отфильтровали нулевые строки.
    def get_recency(row):
        purchase_months = row[row > 0].index
        return purchase_months.max()
    
    t_cal = x_cal.apply(get_recency, axis=1)

    # Создаем карту для сопоставления id и индекса
    customer_map = pd.Series(x_cal.index, index=range(len(x_cal.index)))
    
    print("Подготовка данных завершена.")
    return {
        'x_cal': x_cal,
        't_cal': t_cal,
        'T_cal': T_cal,
        'customer_map': customer_map
    }

# =============================================================================
# ШАГ 2: ФУНКЦИЯ ДЛЯ ОБУЧЕНИЯ МОДЕЛИ
# =============================================================================

def run_hsmdo_model(prepared_data, holdout_months):
    """
    Запускает MCMC симуляцию для модели HSMDO и возвращает результаты.

    Args:
        prepared_data (dict): Словарь с данными, подготовленный функцией prepare_data_for_hsmdo.
        holdout_months (int): Количество месяцев в прогнозном периоде.

    Returns:
        CmdStanMCMC: Объект с результатами сэмплирования.
    """
    # Код модели HSMDO из диссертации
    hsmdo_stan_code = """
    data {
      int<lower=1> N; int<lower=1> K; int<lower=1> T; int<lower=1> F;
      array[T + F] int<lower=0> k;
      array[N, T] int<lower=0> x;
      array[N] int<lower=0> t;
    }
    transformed data {
      array[N] int<lower=0> X_total;
      for (i in 1:N) { X_total[i] = sum(x[i]); }
    }
    parameters {
      vector<lower=0>[N] lambda; vector[N] beta_adj; vector[K] s_adj;
      real<lower=0> r; real<lower=0> alpha;
      real<lower=0> tau_s; real<lower=0> tau_beta;
      real<lower=0.1> phi_c; real<lower=0, upper=1> phi_mu;
    }
    transformed parameters {
      vector[N] beta; vector[K] s;
      real<lower=0> sigma_s = 1 / sqrt(tau_s);
      real<lower=0> sigma_beta = 1 / sqrt(tau_beta);
      real<lower=0> a = phi_c * phi_mu;
      real<lower=0> b = phi_c * (1 - phi_mu);
      beta = (beta_adj - mean(beta_adj)) * sigma_beta + 1;
      s = s_adj - mean(s_adj);
    }
    model {
      vector[T] si_xi; vector[T] si_beta; vector[T + 1] l_sum;
      r ~ gamma(0.001, 0.001); alpha ~ gamma(0.001, 0.001);
      phi_c ~ pareto(0.1, 1.5); lambda ~ gamma(r, alpha);
      tau_s ~ gamma(0.001, 0.001); s_adj ~ normal(0, sigma_s);
      tau_beta ~ gamma(0.001, 0.001); beta_adj ~ normal(0, 1);
      for (i in 1:N) {
        for (j in 1:T) {
          si_xi[j] = x[i, j] * s[k[j]];
          si_beta[j] = exp(beta[i] * s[k[j]]);
        }
        for (tau in t[i]:T) {
          l_sum[tau] = lbeta(a + 1, b + tau - 1) - lambda[i] * sum(head(si_beta, tau));
        }
        l_sum[T + 1] = lbeta(a, b + T) - lambda[i] * sum(si_beta);
        target += (X_total[i] * log(lambda[i]) +
                   beta[i] * sum(si_xi) - lbeta(a, b) +
                   log_sum_exp(tail(l_sum, T - t[i] + 2)));
      }
    }
    generated quantities {
      matrix<lower=0>[N, F] f; vector[N] PZF; vector[N] PA;
      vector[T] xp_si_beta; vector[T + 1] xp_tau;
      vector[F + 2] xz_tau; vector[F] xpp_si_beta;
      for (i in 1:N) {
        for (j in 1:T) { xp_si_beta[j] = exp(beta[i] * s[k[j]]); }
        for (j in t[i]:T) {
          xp_tau[j] = lambda[i] * (sum(xp_si_beta) - sum(head(xp_si_beta, j))) +
                      lbeta(a + 1, b + j - 1) - lbeta(a, b + T);
        }
        xp_tau[T + 1] = 0;
        PA[i] = 1 / (exp(log_sum_exp(tail(xp_tau, T - t[i] + 2))));
        for (j in 1:F) {
          xpp_si_beta[j] = exp(beta[i] * s[k[T + j]]);
          f[i, j] = exp((log(lambda[i]) + beta[i] * s[k[j + T]] +
                       lbeta(a, b + j - 1) - lbeta(a, b))) * PA[i];
        }
        for (j in 1:F) {
          xz_tau[j] = -lambda[i] * sum(head(xpp_si_beta, j)) +
                      lbeta(a + 1, b + T + j - 1) - lbeta(a, b + T);
        }
        xz_tau[F + 1] = -lambda[i] * sum(xpp_si_beta) +
                        lbeta(a, b + T + F) - lbeta(a, b + T);
        xz_tau[F + 2] = log_sum_exp(segment(xp_tau, t[i], T - t[i] + 1));
        PZF[i] = exp(log_sum_exp(xz_tau) - log_sum_exp(tail(xp_tau, T - t[i] + 2)));
      }
    }
    """
    
    # Подготовка данных для STAN
    total_duration = prepared_data['T_cal'] + holdout_months
    k_vector = [(i % 12) + 1 for i in range(total_duration)]

    data_dict = {
        "N": len(prepared_data['x_cal']), "K": 12, "T": prepared_data['T_cal'],
        "F": holdout_months, "k": k_vector,
        "x": prepared_data['x_cal'].values.astype(int),
        "t": prepared_data['t_cal'].values.astype(int)
    }

    # Сохранение и компиляция модели
    stan_file = "hsmdo_model.stan"
    with open(stan_file, "w") as f:
        f.write(hsmdo_stan_code)
    
    print("Шаг 2: Запуск модели...")
    print("Компиляция модели STAN... Это может занять несколько минут.")
    model = CmdStanModel(stan_file=stan_file)
    
    # Запуск MCMC симуляции
    print("Запуск MCMC сэмплирования... Это может занять много времени.")
    fit = model.sample(
        data=data_dict, chains=4, iter_warmup=1000, iter_sampling=2000,
        seed=42, show_progress=True
    )
    
    print("Сэмплирование завершено.")
    os.remove(stan_file)
    
    return fit

# =============================================================================
# ШАГ 3: ФУНКЦИЯ ДЛЯ ПРОФИЛИРОВАНИЯ И СЕГМЕНТАЦИИ
# =============================================================================

def profile_and_segment_customers(fit_results, prepared_data):
    """
    Профилирует и сегментирует покупателей на основе результатов модели.

    Args:
        fit_results (CmdStanMCMC): Объект с результатами сэмплирования.
        prepared_data (dict): Словарь с данными, подготовленный на Шаге 1.

    Returns:
        pd.DataFrame: DataFrame с профилями и сегментами для каждого покупателя.
    """
    print("Шаг 3: Профилирование и сегментация...")
    
    # Пункт 3.1: Профилирование Покупателей
    posterior_samples = fit_results.draws_pd()
    customer_map = prepared_data['customer_map']
    
    # Извлекаем и усредняем ключевые метрики
    pzf_mean = posterior_samples.filter(regex=r'^PZF\[\d+\]$').mean()
    lambda_mean = posterior_samples.filter(regex=r'^lambda\[\d+\]$').mean()
    
    # Создаем итоговый DataFrame с профилями
    profiles_df = pd.DataFrame({
        'customer_id': customer_map.values,
        'lambda_mean': lambda_mean.values,
        'pzf_mean': pzf_mean.values
    }).set_index('customer_id')

    # Пункт 3.2: Создание Сегментов
    # Определяем пороги для "высоких" и "низких" значений (например, по квантилям)
    high_lambda_threshold = profiles_df['lambda_mean'].quantile(0.75)
    high_pzf_threshold = profiles_df['pzf_mean'].quantile(0.75)
    low_pzf_threshold = profiles_df['pzf_mean'].quantile(0.25)

    # Функция для присвоения сегмента
    def assign_segment(row):
        if row['lambda_mean'] >= high_lambda_threshold and row['pzf_mean'] <= low_pzf_threshold:
            return "Чемпионы"
        elif row['lambda_mean'] >= high_lambda_threshold and row['pzf_mean'] >= high_pzf_threshold:
            return "Лояльные в зоне риска"
        elif row['pzf_mean'] > 0.9: # Если вероятность не купить в будущем > 90%
            return "Спящие/Потерянные"
        elif row['pzf_mean'] <= low_pzf_threshold:
            return "Стабильные покупатели"
        else:
            return "Обычные покупатели"

    profiles_df['segment'] = profiles_df.apply(assign_segment, axis=1)
    
    print("Сегментация завершена.")
    return profiles_df

# =============================================================================
# ГЛАВНЫЙ БЛОК ДЛЯ ЗАПУСКА
# =============================================================================
if __name__ == '__main__':
    # Параметры
    HOLDOUT_MONTHS = 6
    # Укажите дату, на которую проводится анализ. Если None, будет взята последняя дата из данных.
    ANALYSIS_DATE = '' 

    # 1. Загрузка или создание сырых данных
    print("Загрузка сырых данных...")
    
    # --- НАЧАЛО БЛОКА ДЛЯ РЕДАКТИРОВАНИЯ ---
    # Закомментируйте блок с созданием фейковых данных
    # и раскомментируйте строку для загрузки вашего файла.
    
    # # Пример загрузки вашего файла CSV
    # # Убедитесь, что в файле есть колонки 'order_date', 'name', 'order_id'
    df_raw = pd.read_csv('cleaned_data.csv')

    # Для демонстрации создадим фейковые данные
    # customers = [f'Customer_{i}' for i in range(100)]
    # data = []
    # for month in range(1, 37): # 36 месяцев данных
    #     for cust in customers:
    #         if np.random.rand() > 0.6: 
    #             num_orders = np.random.poisson(1.2)
    #             if num_orders > 0:
    #                 for i in range(num_orders):
    #                   data.append({
    #                       'order_date': pd.to_datetime(f'2021-{((month-1)%12)+1}-15') + pd.DateOffset(years=(month-1)//12),
    #                       'name': cust,
    #                       'order_id': f'order_{month}_{cust}_{i}'
    #                   })
    # df_raw = pd.DataFrame(data)
    # print(f"Создано {len(df_raw)} транзакций для {len(customers)} покупателей.")
    # --- КОНЕЦ БЛОКА ДЛЯ РЕДАКТИРОВАНИЯ ---


    # 2. Вызов функции подготовки данных (Шаг 1)
    prepared_data = prepare_data_for_hsmdo(df_raw, 
                                           holdout_months=HOLDOUT_MONTHS, 
                                           analysis_date_str=ANALYSIS_DATE)

    # 3. Вызов функции обучения модели (Шаг 2)
    hsmdo_fit_results = run_hsmdo_model(prepared_data, holdout_months=HOLDOUT_MONTHS)

    # 4. Вызов функции профилирования и сегментации (Шаг 3)
    final_segments_df = profile_and_segment_customers(hsmdo_fit_results, prepared_data)

    # Вывод результатов
    print("\n--- Итоговая сегментация покупателей ---")
    print(final_segments_df.head(10))
    
    print("\n--- Распределение по сегментам ---")
    print(final_segments_df['segment'].value_counts())

    

    # сохранение файла с сегментами
    final_segments_df.to_csv('hsmdo_customer_segments.csv', encoding='utf-8-sig')
    print("\nИтоговая сегментация сохранена в файл: hsmdo_customer_segments.csv")
