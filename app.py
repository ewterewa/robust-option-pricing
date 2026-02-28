import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.stats import norm

# ---------- Настройка страницы (Красивые цвета) ----------
st.set_page_config(
    page_title="ROPD: Robust Option Pricing Dashboard",
    page_icon="📊",
    layout="wide"
)

# Цветовая схема (современная и научная)
COLOR_PRIMARY = "#4F6DF5"
COLOR_SECONDARY = "#F55D4F"
COLOR_SUCCESS = "#2ECC71"
COLOR_WARNING = "#F39C12"
COLOR_BG = "#F8FAFC"
COLOR_CARD = "#FFFFFF"

# Кастомный CSS для красоты
st.markdown(f"""
<style>
    .stApp {{
        background-color: {COLOR_BG};
    }}
    .main-header {{
        font-size: 2.5rem;
        color: #1E293B;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    .sub-header {{
        font-size: 1.2rem;
        color: #475569;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #E2E8F0;
    }}
    .card {{
        background-color: {COLOR_CARD};
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
    }}
    .metric-card {{
        background: linear-gradient(135deg, {COLOR_PRIMARY}10, {COLOR_SECONDARY}10);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid {COLOR_PRIMARY};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background-color: white;
        padding: 0.5rem;
        border-radius: 40px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 30px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLOR_PRIMARY} !important;
        color: white !important;
    }}
</style>
""", unsafe_allow_html=True)

# ---------- Заголовок ----------
st.markdown('<p class="main-header">📈 Интеграция робастного проектирования и теории реальных опционов</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Интерактивная модель, предложенная Евтеревой М.Д. (КМУ 26) — прототип для оценки эффективности НИОКР</p>', unsafe_allow_html=True)

# ---------- Боковая панель для ввода данных (Технические параметры) ----------
with st.sidebar:
    st.markdown("## ⚙️ Параметры модели")
    st.markdown("---")
    
    st.markdown("### 1. Инженерные параметры")
    sigma_lab = st.number_input(
        "σF0 (разброс в идеальных условиях)", 
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="Среднеквадратическое отклонение в лаборатории"
    )
    sigma_noise = st.number_input(
        "σF0X (разброс при внешних шумах)", 
        min_value=sigma_lab+0.1, max_value=10.0, value=3.0, step=0.1,
        help="Среднеквадратическое отклонение при воздействии шумов"
    )
    
    st.markdown("### 2. Финансовые параметры")
    k_sensitivity = st.slider(
        "k (чувствительность бизнеса к технической вариативности)", 
        min_value=0.1, max_value=2.0, value=0.8, step=0.05,
        help="Чем выше k, тем сильнее тех. разброс бьет по финансам"
    )
    sigma_min = st.slider(
        "σCF_min (рыночная волатильность, %)", 
        min_value=0.05, max_value=0.3, value=0.12, step=0.01,
        help="Нижний предел финансового риска (не зависит от тех. параметров)"
    )
    
    st.markdown("### 3. Опционные параметры")
    npv_project = st.number_input(
        "NPV базового сценария (млн руб.)", 
        min_value=-50.0, max_value=200.0, value=30.0, step=5.0
    )
    strike_price = st.number_input(
        "Инвестиции для запуска (X, млн руб.)", 
        min_value=0.0, max_value=200.0, value=25.0, step=5.0
    )
    time_to_maturity = st.slider(
        "Время до принятия решения (T, лет)", 
        min_value=0.5, max_value=5.0, value=2.0, step=0.5
    )
    risk_free_rate = st.slider(
        "Безрисковая ставка (r, %)", 
        min_value=0.01, max_value=0.10, value=0.05, step=0.01
    )

# ---------- Расчеты (Ядро модели) ----------
# Этап 1: Индекс робастности (R)
# Формула из статьи: R = 1 - (σF0X / σF0) - ИСПРАВЛЕНО на основе контекста автора.
# В статье опечатка. Правильный смысл: R тем выше, чем меньше влияние шума.
# Логичнее: R = σF0 / σF0X (отношение сигнал/шум Тагути), но для нормировки к [0,1] используем:
# R = 1 / (1 + (σF0X/σF0)) или предложенный автором вариант, но с исправлением знака.
# Следуя тексту: "R стремится к 1 при полной нечувствительности к шумам".
# Значит, R = σF0 / σF0X, но это может быть >1. Используем сигмоидную нормализацию или формулу:
# R = max(0, 1 - ( (σF0X - σF0) / σF0X ) ) - упрощенно.
# Для красоты демо используем: R = 1 / (1 + (σF0X/σF0))

if sigma_noise > sigma_lab:
    # Индекс робастности (от 0 до 1, где 1 - идеально)
    R = sigma_lab / sigma_noise
    # Но чтобы R не был линейным, оставим как есть для простоты.
else:
    R = 0.99

R = min(R, 0.99)  # Ограничим, чтобы избежать деления на ноль

# Этап 2: Волатильность денежного потока (σCF)
# Гиперболическая модель автора: σCF = k / R + σCF_min
if R > 0.01:
    sigma_cf = (k_sensitivity / R) + sigma_min
else:
    sigma_cf = 10.0  # Огромная волатильность при R=0

# Ограничим разумные пределы для графика
sigma_cf = min(sigma_cf, 2.5)

# Этап 3: Цена опциона (модель Блэка-Шоулза)
def black_scholes_call(S, K, T, r, sigma):
    """Цена европейского колл-опциона."""
    if sigma <= 0 or T <= 0:
        return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Базовый NPV - это стоимость базового актива (S)
S = npv_project
K = strike_price
T = time_to_maturity
r = risk_free_rate
sigma = sigma_cf

option_price = black_scholes_call(S, K, T, r, sigma)

# Цена опциона без учета технического риска (если бы R=1, то sigma_cf = sigma_min)
option_price_naive = black_scholes_call(S, K, T, r, sigma_min)

# ---------- Интерфейс: Вкладки (Рабочие вкладки) ----------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 1. Квантификация робастности", 
    "📉 2. Технический разброс → Финансовый риск", 
    "💼 3. Опционная модель",
    "📊 4. Интегральный калькулятор"
])

# --- Вкладка 1: Индекс робастности ---
with tab1:
    st.markdown("### 🔬 Этап 1: Расчет интегрального индекса робастности (R)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Исходные данные:**")
        st.markdown(f"- Разброс в идеале (σF0): **{sigma_lab:.2f}**")
        st.markdown(f"- Разброс с шумами (σF0X): **{sigma_noise:.2f}**")
        
        st.markdown("**Формула автора:**")
        st.latex(r'R = 1 - \frac{\sigma_{F0}X}{\sigma_{F0}}')
        st.caption("Примечание: В работе, вероятно, опечатка. Мы используем корректную нормировку для наглядности.")
        
        # Визуализация влияния шума
        fig = go.Figure()
        x_lab = np.random.normal(0, sigma_lab, 1000)
        x_noise = np.random.normal(0, sigma_noise, 1000)
        
        fig.add_trace(go.Histogram(x=x_lab, name="Идеальные условия (σF0)", 
                                   marker_color=COLOR_PRIMARY, opacity=0.7, nbinsx=40))
        fig.add_trace(go.Histogram(x=x_noise, name="Воздействие шумов (σF0X)", 
                                   marker_color=COLOR_SECONDARY, opacity=0.7, nbinsx=40))
        fig.update_layout(
            title="Распределение выходной характеристики",
            xaxis_title="Отклонение характеристики",
            yaxis_title="Частота",
            barmode='overlay',
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Результат")
        
        # Красивая метрика
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #1E293B; margin-bottom: 0;">Индекс робастности R</h3>
            <p style="font-size: 4rem; font-weight: 800; color: {COLOR_PRIMARY}; margin: 0;">{R:.3f}</p>
            <p style="color: #64748B;">{'🔵 Высокая устойчивость' if R > 0.7 else '🟡 Средняя устойчивость' if R > 0.4 else '🔴 Низкая устойчивость'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Горизонтальный бар
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = R * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Индекс робастности, %"},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': COLOR_PRIMARY},
                'steps' : [
                    {'range': [0, 40], 'color': "#FFEBEE"},
                    {'range': [40, 70], 'color': "#FFF9E6"},
                    {'range': [70, 100], 'color': "#E8F5E9"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 40}}))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"💡 **Интерпретация:** Продукт {'**устойчив**' if R > 0.7 else '**чувствителен**'} к внешним воздействиям. Индекс R={R:.2f} означает, что разброс характеристик в реальных условиях в {1/R:.1f} раз выше, чем в лаборатории.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- Вкладка 2: Трансляция риска ---
with tab2:
    st.markdown("### 📉 Этап 2: От технического разброса к волатильности денежных потоков")
    st.markdown("Неллинейная зависимость: снижение робастности резко увеличивает финансовый риск.")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Гиперболическая модель:**")
        st.latex(r'\sigma_{CF} = \frac{k}{R} + \sigma_{CF}^{min}')
        
        # График зависимости σCF от R
        r_range = np.linspace(0.1, 0.99, 100)
        sigma_range = (k_sensitivity / r_range) + sigma_min
        sigma_range = np.clip(sigma_range, 0, 3)  # Обрезаем для наглядности
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r_range, y=sigma_range, 
                                 mode='lines', name='σCF(R)',
                                 line=dict(color=COLOR_PRIMARY, width=4)))
        # Текущая точка
        fig.add_trace(go.Scatter(x=[R], y=[sigma_cf], 
                                 mode='markers', name='Текущий проект',
                                 marker=dict(color=COLOR_SECONDARY, size=15, line=dict(color='white', width=2))))
        
        fig.update_layout(
            title="Зависимость финансового риска от технической устойчивости",
            xaxis_title="Индекс робастности R (выше = лучше)",
            yaxis_title="Волатильность денежного потока σCF",
            template="plotly_white",
            height=450,
            hovermode="x"
        )
        fig.add_hline(y=sigma_min, line_dash="dash", line_color="gray", 
                     annotation_text=f"σ_min = {sigma_min:.2f}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Результат трансляции")
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin-bottom: 0;">Волатильность денежного потока (σCF)</h4>
            <p style="font-size: 3rem; font-weight: 700; color: {COLOR_SECONDARY};">{sigma_cf:.2f}</p>
            <p style="font-size: 0.9rem;">({sigma_cf*100:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Вклад технических факторов:**")
        tech_vol = sigma_cf - sigma_min
        st.progress(min(tech_vol/(sigma_cf+0.01), 1.0), text=f"Техническая составляющая: {tech_vol:.2f} ({tech_vol/sigma_cf*100:.1f}%)")
        st.progress(min(sigma_min/(sigma_cf+0.01), 1.0), text=f"Рыночная составляющая: {sigma_min:.2f} ({sigma_min/sigma_cf*100:.1f}%)")
        
        st.caption("Чем выше доля технической составляющей, тем сильнее инженерные решения влияют на финансовый результат.")
        
        # Риск-профиль
        if sigma_cf > 0.8:
            st.error("🔴 Высокий финансовый риск. Техническая неопределенность доминирует.")
        elif sigma_cf > 0.4:
            st.warning("🟡 Умеренный риск. Требуется баланс инженерных усилий.")
        else:
            st.success("🟢 Низкий риск. Проект предсказуем.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- Вкладка 3: Опционная модель ---
with tab3:
    st.markdown("### 💼 Этап 3: Оценка управленческой гибкости (Реальный опцион)")
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Параметры опциона:**")
        
        # Таблица параметров
        params_df = pd.DataFrame({
            "Параметр": ["Базовый актив (NPV)", "Цена исполнения (X)", "Время до решения (T)", "Безриск. ставка (r)", "Волатильность (σ)"],
            "Значение": [f"{S} млн руб.", f"{K} млн руб.", f"{T} лет", f"{r*100:.1f}%", f"{sigma*100:.1f}%"]
        })
        st.table(params_df)
        
        st.markdown("**Формула Блэка-Шоулза:**")
        st.latex(r'C = S \cdot N(d_1) - X \cdot e^{-rT} \cdot N(d_2)')
        st.latex(r'd_1 = \frac{\ln(S/X) + (r+\sigma^2/2)T}{\sigma\sqrt{T}}')
        
        # Сравнение с наивным подходом
        st.markdown("**Сравнение подходов:**")
        delta = option_price - option_price_naive
        st.metric("Цена опциона (с учетом тех. риска)", f"{option_price:.2f} млн руб.", 
                 delta=f"{delta:.2f} млн руб. vs безрисковый сценарий")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Визуализация распределения NPV и цены опциона
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Распределение NPV проекта с учетом технической волатильности", 
                           "Стоимость опциона на запуск проекта"),
            vertical_spacing=0.15
        )
        
        # Распределение NPV
        npv_sim = np.random.normal(S, S*sigma, 5000)
        fig.add_trace(go.Histogram(x=npv_sim, nbinsx=60, 
                                   marker_color=COLOR_PRIMARY, 
                                   opacity=0.7,
                                   name="NPV distribution"),
                     row=1, col=1)
        fig.add_vline(x=K, line_dash="dash", line_color=COLOR_SECONDARY,
                     annotation_text=f"Инвестиции (X={K})", row=1, col=1)
        
        # Цена опциона как функция от волатильности
        vol_range = np.linspace(0.1, 2.0, 50)
        price_range = [black_scholes_call(S, K, T, r, v) for v in vol_range]
        
        fig.add_trace(go.Scatter(x=vol_range, y=price_range,
                                 mode='lines', name='Цена опциона C(σ)',
                                 line=dict(color=COLOR_SUCCESS, width=3)),
                     row=2, col=1)
        fig.add_trace(go.Scatter(x=[sigma], y=[option_price],
                                 mode='markers', name='Текущий опцион',
                                 marker=dict(color=COLOR_SECONDARY, size=12)),
                     row=2, col=1)
        
        fig.update_layout(height=600, showlegend=False, template="plotly_white")
        fig.update_xaxes(title_text="NPV, млн руб.", row=1, col=1)
        fig.update_xaxes(title_text="Волатильность σ", row=2, col=1)
        fig.update_yaxes(title_text="Цена опциона, млн руб.", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Вкладка 4: Интегральный калькулятор и рекомендации ---
with tab4:
    st.markdown("### 📊 Интегральная оценка проекта")
    st.markdown("Сводка всех этапов и рекомендации по управлению.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔬 Индекс робастности (R)", f"{R:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📉 Волатильность (σCF)", f"{sigma_cf:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("💼 Цена опциона (C)", f"{option_price:.2f} млн руб.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Матрица решений
    st.markdown("### Матрица управленческих решений")
    
    if R > 0.7 and option_price > K * 0.3:
        decision = "🚀 **Запуск в серию**"
        comment = "Продукт устойчив, гибкость высокая. Опцион глубоко в деньгах."
        color = COLOR_SUCCESS
    elif R > 0.5 and option_price > 0:
        decision = "⏳ **Отложить / Доработать**"
        comment = "Средняя устойчивость. Есть смысл инвестировать в повышение R."
        color = COLOR_WARNING
    else:
        decision = "🛑 **Отказ / Перепроектирование**"
        comment = "Высокий технический риск съедает стоимость опциона."
        color = COLOR_SECONDARY
    
    st.markdown(f"""
    <div style="background-color: {color}20; padding: 2rem; border-radius: 15px; border-left: 10px solid {color};">
        <h2 style="margin:0;">{decision}</h2>
        <p style="font-size:1.2rem;">{comment}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Чек-лист для инженера
    with st.expander("🔧 Инженерный чек-лист для повышения стоимости опциона"):
        st.markdown("""
        - [ ] Снизить σF0X (чувствительность к шумам): использовать методы Тагути, экранирование.
        - [ ] Увеличить допуски на критически важные узлы (повышение R).
        - [ ] Провести параметрическую оптимизацию для снижения k (цены вариативности).
        
        **Эффект:** Увеличение R с {:.2f} до {:.2f} снизит σCF с {:.2f} до {:.2f} и повысит стоимость опциона до {:.2f} млн руб.
        """.format(R, min(R*1.3, 0.98), sigma_cf, (k_sensitivity/(R*1.3)+sigma_min), 
                   black_scholes_call(S, K, T, r, (k_sensitivity/(min(R*1.3, 0.98))+sigma_min)) )

    # Данные для отчета
    st.download_button(
        label="📥 Скачать отчет по проекту (CSV)",
        data=pd.DataFrame({
            "Параметр": ["R", "σCF", "C", "NPV", "X", "T"],
            "Значение": [R, sigma_cf, option_price, S, K, T]
        }).to_csv(index=False),
        file_name="project_report.csv",
        mime="text/csv"
    )

# Нижний колонтитул
st.markdown("---")
st.markdown("📐 **Модель разработана на основе работы Евтеревой М.Д. «Интеграция робастного проектирования и теории реальных опционов» (КМУ 26).**")
st.caption("Прототип демонстрирует связь микроуровня (технические допуски) и макроуровня (стоимость управленческой гибкости).")
