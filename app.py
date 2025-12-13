import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# =========================================================
# 🛠️ PHẦN 1: XỬ LÝ DỮ LIỆU & CACHING
# =========================================================

@st.cache_data(show_spinner=False)
def load_data(file_train, file_verify, col_res, col_day):
    try:
        df_train = pd.read_excel(file_train)
        df_verify = pd.read_excel(file_verify)
        df_train = df_train.dropna(subset=[col_res])
        df_verify = df_verify.dropna(subset=[col_res, col_day])
        return df_train, df_verify
    except Exception as e:
        return None, None

@st.cache_data(show_spinner=False)
def find_optimal_truncation(data_array, max_cut_percent=0.10, steps=10):
    calc_data = data_array
    if len(data_array) > 5000:
        np.random.seed(42)
        calc_data = np.random.choice(data_array, 5000, replace=False)
        
    best_p = -1
    best_range = (data_array.min(), data_array.max())
    
    cuts = np.linspace(0, max_cut_percent, steps)
    sorted_data = np.sort(calc_data)
    n = len(sorted_data)
    
    for left_cut in cuts:
        for right_cut in cuts:
            if left_cut + right_cut >= 0.5: continue
            s = int(n * left_cut)
            e = int(n * (1 - right_cut))
            subset = sorted_data[s:e]
            
            if len(subset) > 20:
                stat, p_val = stats.normaltest(subset)
                if p_val > best_p:
                    best_p = p_val
                    lower = np.percentile(data_array, left_cut * 100)
                    upper = np.percentile(data_array, (1 - right_cut) * 100)
                    best_range = (lower, upper)
    return best_range

# =========================================================
# 🧠 PHẦN 2: ENGINE MÔ PHỎNG THEO NGÀY (DAY-BASED)
# =========================================================

class PBRTQCEngine:
    def __init__(self, df_train, df_verify, col_res, col_day, trunc_range):
        self.trunc_min, self.trunc_max = trunc_range
        self.col_res = col_res
        self.col_day = col_day
        
        raw_train = df_train[col_res].values
        self.train_clean = raw_train[(raw_train >= self.trunc_min) & (raw_train <= self.trunc_max)]
        
        self.df_verify_clean = df_verify[
            (df_verify[col_res] >= self.trunc_min) & 
            (df_verify[col_res] <= self.trunc_max)
        ].copy()

    def calculate_ma(self, values, method, param):
        series = pd.Series(values)
        if method == 'SMA':
            return series.rolling(window=int(param)).mean().bfill().values
        elif method == 'EWMA':
            lam = 2 / (int(param) + 1)
            return series.ewm(alpha=lam, adjust=False).mean().values
        return values

    def determine_limits(self, method, param, target_fpr):
        ma_values = self.calculate_ma(self.train_clean, method, param)
        lower = np.percentile(ma_values, (target_fpr/2)*100)
        upper = np.percentile(ma_values, (1 - target_fpr/2)*100)
        return lower, upper

    def run_day_simulation(self, method, param, lcl, ucl, bias_pct, num_sims=None):
        grouped = self.df_verify_clean.groupby(self.col_day)
        
        total_days = 0
        detected_days = 0
        false_positive_days = 0
        nped_list = []
        plot_data = None 
        
        bias_factor = 1 + (bias_pct / 100.0)
        days_to_run = list(grouped.groups.keys())
        if num_sims and num_sims < len(days_to_run):
            days_to_run = days_to_run[:num_sims]

        for day_name in days_to_run:
            day_df = grouped.get_group(day_name)
            vals = day_df[self.col_res].values.astype(float)
            n = len(vals)
            if n < 5: continue 
            
            total_days += 1
            max_idx = min(40, n - 2) 
            if max_idx < 1: max_idx = 1
            injection_point = np.random.randint(1, max_idx + 1)
            
            # Check False Positive
            ma_clean_full = self.calculate_ma(vals, method, param)
            pre_bias_alarms = (ma_clean_full[:injection_point] < lcl) | (ma_clean_full[:injection_point] > ucl)
            
            if np.any(pre_bias_alarms):
                false_positive_days += 1
                if day_name == days_to_run[-1]:
                    plot_data = {
                        'day': day_name, 'vals_clean': vals, 'ma_clean': ma_clean_full,
                        'ma_sim': None, 'inject_idx': injection_point,
                        'alarm_idx': np.argmax(pre_bias_alarms), 'lcl': lcl, 'ucl': ucl, 'status': 'False Positive'
                    }
                continue 

            # Check Detection
            vals_biased = vals.copy()
            vals_biased[injection_point:] *= bias_factor
            ma_biased = self.calculate_ma(vals_biased, method, param)
            post_bias_region = ma_biased[injection_point:]
            post_alarms = (post_bias_region < lcl) | (post_bias_region > ucl)
            
            if np.any(post_alarms):
                detected_days += 1
                first_alarm_idx_rel = np.argmax(post_alarms)
                nped = first_alarm_idx_rel + 1 
                nped_list.append(nped)
                
                if day_name == days_to_run[-1]:
                     plot_data = {
                        'day': day_name, 'vals_clean': vals, 'ma_clean': ma_clean_full,
                        'ma_sim': ma_biased, 'inject_idx': injection_point,
                        'alarm_idx': injection_point + first_alarm_idx_rel, 'lcl': lcl, 'ucl': ucl, 'status': 'Detected'
                    }
            else:
                if day_name == days_to_run[-1]:
                     plot_data = {
                        'day': day_name, 'vals_clean': vals, 'ma_clean': ma_clean_full,
                        'ma_sim': ma_biased, 'inject_idx': injection_point, 'alarm_idx': None,
                        'lcl': lcl, 'ucl': ucl, 'status': 'Missed'
                    }

        metrics = {
            "Total Days": total_days,
            "Detected (%)": round(detected_days / total_days * 100, 1) if total_days > 0 else 0,
            "False Positive (%)": round(false_positive_days / total_days * 100, 1) if total_days > 0 else 0,
            "ANPed": round(np.mean(nped_list), 1) if nped_list else "N/A",
            "Median NPed": round(np.median(nped_list), 1) if nped_list else "N/A",
            "95th NPed": round(np.percentile(nped_list, 95), 1) if nped_list else "N/A"
        }
        return metrics, plot_data

# =========================================================
# 🖥️ PHẦN 3: GIAO DIỆN STREAMLIT (CẬP NHẬT INPUT)
# =========================================================

st.set_page_config(layout="wide", page_title="PBRTQC Day-Simulator")

st.title("📅 PBRTQC Day-by-Day Simulator")
st.markdown("Hệ thống mô phỏng PBRTQC theo từng ngày làm việc (Daily Run).")

with st.sidebar:
    st.header("1. Upload Data")
    f_train = st.file_uploader("Training Data (.xlsx)", type='xlsx')
    f_verify = st.file_uploader("Verify Data (.xlsx)", type='xlsx')
    
    st.divider()
    st.header("2. Settings")
    bias_pct = st.number_input("Bias (%)", value=5.0, step=0.5)
    target_fpr = st.slider("Target FPR (%)", 0.1, 10.0, 2.0, 0.1) / 100
    model = st.selectbox("Model", ["EWMA", "SMA"])
    max_days = st.slider("Giới hạn số ngày chạy", 10, 5000, 500)

if f_train and f_verify:
    df_temp = pd.read_excel(f_train, nrows=1)
    all_cols = df_temp.columns.tolist()
    
    c1, c2 = st.columns(2)
    col_res = c1.selectbox("Cột Kết quả (Results)", all_cols)
    col_day = c2.selectbox("Cột Ngày (Days)", all_cols)

    # --- [MỚI] PHẦN NHẬP BLOCK SIZE CHO 3 CASE ---
    st.divider()
    st.subheader(f"3. Cấu hình tham số cho mô hình {model}")
    
    col_case1, col_case2, col_case3 = st.columns(3)
    
    cases_config = []
    
    # Case 1
    with col_case1:
        st.markdown("**Case 1**")
        bs1 = st.number_input("Block Size (N)", value=20, key="bs1", min_value=2)
        freq1 = 1
        # Nếu là SMA thì hiện thêm ô nhập Frequency
        # Nếu là EWMA thì Frequency mặc định là 1 (vì EWMA tính liên tục từng điểm)
        # Tuy nhiên logic simulation của bạn có thể áp dụng frequency cho cả EWMA nếu muốn giảm tải
        # Ở đây mình để Frequency hiện ra cho SMA cho đúng chuẩn sách giáo khoa.
        if model == "SMA":
            freq1 = st.number_input("Frequency", value=1, key="freq1", min_value=1)
        cases_config.append({'bs': bs1, 'freq': freq1})

    # Case 2
    with col_case2:
        st.markdown("**Case 2**")
        bs2 = st.number_input("Block Size (N)", value=40, key="bs2", min_value=2)
        freq2 = 1
        if model == "SMA":
            freq2 = st.number_input("Frequency", value=1, key="freq2", min_value=1)
        cases_config.append({'bs': bs2, 'freq': freq2})

    # Case 3
    with col_case3:
        st.markdown("**Case 3**")
        bs3 = st.number_input("Block Size (N)", value=60, key="bs3", min_value=2)
        freq3 = 1
        if model == "SMA":
            freq3 = st.number_input("Frequency", value=1, key="freq3", min_value=1)
        cases_config.append({'bs': bs3, 'freq': freq3})
    # ---------------------------------------------

    if st.button("🚀 Run Simulation"):
        with st.spinner("Đang xử lý dữ liệu..."):
            df_train, df_verify = load_data(f_train, f_verify, col_res, col_day)
            
            if df_train is not None:
                trunc_range = find_optimal_truncation(df_train[col_res].values)
                st.info(f"Đã tối ưu Truncation Limit: [{trunc_range[0]:.2f} - {trunc_range[1]:.2f}]")
                
                engine = PBRTQCEngine(df_train, df_verify, col_res, col_day, trunc_range)
                
                results = []
                plots = []
                
                prog_bar = st.progress(0)
                
                # Dùng cases_config từ input người dùng
                for i, case in enumerate(cases_config):
                    lcl, ucl = engine.determine_limits(model, case['bs'], target_fpr)
                    
                    metrics, p_data = engine.run_day_simulation(
                        model, case['bs'], lcl, ucl, bias_pct, num_sims=max_days
                    )
                    
                    res_row = {
                        "Case": f"N={case['bs']}",
                        "LCL": round(lcl, 2), "UCL": round(ucl, 2),
                        **metrics
                    }
                    results.append(res_row)
                    plots.append({'name': f"Case N={case['bs']}", 'data': p_data})
                    
                    prog_bar.progress((i+1)/len(cases_config))
                
                st.subheader("📊 Kết quả Đánh giá")
                st.dataframe(pd.DataFrame(results).style.highlight_max(subset=['Detected (%)'], color='#d1ffbd'), use_container_width=True)
                
                st.divider()
                st.subheader("📈 Chi tiết 1 Ngày ngẫu nhiên")
                tabs = st.tabs([p['name'] for p in plots])
                for i, tab in enumerate(tabs):
                    with tab:
                        d = plots[i]['data']
                        if d:
                            fig, ax = plt.subplots(figsize=(12, 5))
                            ax.plot(d['ma_clean'], label='MA (Clean)', color='green', alpha=0.4)
                            if d['ma_sim'] is not None:
                                ax.plot(d['ma_sim'], label=f'MA (Bias {bias_pct}%)', color='orange')
                            ax.axhline(d['ucl'], color='red', ls='--')
                            ax.axhline(d['lcl'], color='red', ls='--')
                            ax.axvline(d['inject_idx'], color='black', ls=':', label='Thêm lỗi')
                            if d['alarm_idx'] is not None:
                                color = 'purple' if d['status'] == 'False Positive' else 'red'
                                shape = 'X' if d['status'] == 'False Positive' else '*'
                                y_val = d['ma_clean'][d['alarm_idx']] if d['ma_sim'] is None else d['ma_sim'][d['alarm_idx']]
                                ax.scatter(d['alarm_idx'], y_val, color=color, s=150, zorder=5, marker=shape, label=f'Alarm ({d["status"]})')
                            ax.set_title(f"Ngày: {d['day']} - {d['status']}")
                            ax.legend()
                            st.pyplot(fig)
                        else:
                            st.warning("Chưa có dữ liệu.")
            else:
                st.error("Lỗi file.")
