import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import io
import plotly.graph_objects as go

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

def find_optimal_truncation(data_array, max_cut_percent=0.10, steps=10):
    """Tìm khoảng cắt tối ưu (Auto Mode)"""
    calc_data = data_array
    if len(data_array) > 40000:
        np.random.seed(42)
        calc_data = np.random.choice(data_array, 40000, replace=False)
        
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
# 📈 PHẦN 2: HÀM VẼ BIỂU ĐỒ (PLOTLY)
# =========================================================

def draw_chart(df, method, lcl, ucl, title, direction='positive'):
    fig = go.Figure()

    # 1. Vẽ đường MA liên tục
    ma_col_name = f'{method}_Continuous'
    if ma_col_name in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[ma_col_name], 
            mode='lines', 
            name=f'{method} (Continuous)',
            line=dict(color='lightblue', width=1.5)
        ))

    # 2. Vẽ đường giới hạn
    fig.add_trace(go.Scatter(
        x=[df.index.min(), df.index.max()], 
        y=[ucl, ucl], 
        mode='lines', 
        name='UCL', 
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=[df.index.min(), df.index.max()], 
        y=[lcl, lcl], 
        mode='lines', 
        name='LCL', 
        line=dict(color='blue', width=2, dash='dash')
    ))

    # 3. Đánh dấu các điểm Alarm
    if direction == 'positive':
        alarm_points = df[(df['AON_Reported'] > ucl)]
        color = 'red'
        label = 'Alarm (> UCL)'
    else:
        alarm_points = df[(df['AON_Reported'] < lcl)]
        color = 'blue'
        label = 'Alarm (< LCL)'

    if not alarm_points.empty:
        fig.add_trace(go.Scatter(
            x=alarm_points.index, 
            y=alarm_points['AON_Reported'], 
            mode='markers', 
            name=label,
            marker=dict(color=color, size=8, symbol='circle')
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#cc0000')),
        xaxis_title="Data Point (Index)",
        yaxis_title="Value",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0.05)'
    )
    return fig

# =========================================================
# 🧠 PHẦN 3: ENGINE MÔ PHỎNG (DUAL DIRECTION)
# =========================================================

class PBRTQCEngine:
    def __init__(self, df_train, df_verify, col_res, col_day, trunc_range):
        self.trunc_min, self.trunc_max = trunc_range
        self.col_res = col_res
        self.col_day = col_day
        
        # Training Data
        raw_train = df_train[col_res].values
        self.train_clean = raw_train[(raw_train >= self.trunc_min) & (raw_train <= self.trunc_max)]
        
        # Verify Data (Apply Truncation Limit immediately for initial clean data)
        self.df_verify_clean = df_verify[
            (df_verify[col_res] >= self.trunc_min) & 
            (df_verify[col_res] <= self.trunc_max)
        ].copy()
        
        self.global_vals = self.df_verify_clean[col_res].values.astype(float)
        self.global_days = self.df_verify_clean[col_day].values

        # Map index theo ngày
        self.day_indices = {}
        unique_days = self.df_verify_clean[col_day].unique()
        current_idx = 0
        for day in unique_days:
            count = len(self.df_verify_clean[self.df_verify_clean[col_day] == day])
            self.day_indices[day] = (current_idx, current_idx + count)
            current_idx += count

    def get_data_stats(self):
        return {
            "Train Mean": np.mean(self.train_clean),
            "Train Median": np.median(self.train_clean),
            "Verify Mean": np.mean(self.global_vals),
            "Verify Median": np.median(self.global_vals),
            "Truncation Range": f"[{self.trunc_min:.2f} - {self.trunc_max:.2f}]"
        }

    def calculate_ma(self, values, method, param):
        """
        Tính MA có hỗ trợ xử lý NaN (dữ liệu bị loại bỏ).
        param: Là 'block_size' (nếu SMA) hoặc 'lambda' (nếu EWMA)
        """
        series = pd.Series(values)
        if method == 'SMA':
            # SMA dùng Block Size (N)
            # min_periods=1: Cho phép tính trung bình dù có NaN (bỏ qua NaN)
            return series.rolling(window=int(param), min_periods=1).mean().values
        elif method == 'EWMA':
            # EWMA dùng Lambda trực tiếp
            # ignore_na=True: Bỏ qua NaN trong quá trình tính trọng số
            return series.ewm(alpha=param, adjust=False, ignore_na=True).mean().values
        return values

    def get_report_mask(self, total_length, start_offset, frequency):
        """
        Tạo mask xác định các điểm Report.
        start_offset: Điểm bắt đầu report (SMA = N-1; EWMA = F-1)
        """
        mask = np.zeros(total_length, dtype=bool)
        start_idx = int(start_offset)
        if start_idx < total_length:
            report_indices = np.arange(start_idx, total_length, int(frequency))
            mask[report_indices] = True
        return mask

    def determine_limits(self, method, param, start_offset, frequency, target_fpr):
        """Tính Limit"""
        ma_values = self.calculate_ma(self.train_clean, method, param)
        mask = self.get_report_mask(len(ma_values), start_offset, frequency)
        valid_ma_values = ma_values[mask]
        
        if len(valid_ma_values) == 0:
            return 0, 0 

        lower = np.percentile(valid_ma_values, (target_fpr/2)*100)
        upper = np.percentile(valid_ma_values, (1 - target_fpr/2)*100)
        return lower, upper

    def run_simulation(self, method, param, start_offset, frequency, lcl, ucl, bias_pct, direction='positive', fixed_inject_idx=None, apply_trunc_on_bias=False, sim_mode='Standard'):
        """
        param: N (SMA) hoặc Lambda (EWMA)
        start_offset: Điểm bắt đầu report
        """
        total_days = 0
        detected_days = 0
        nped_list = []
        
        if direction == 'positive':
            bias_factor = 1 + (bias_pct / 100.0)
        else:
            bias_factor = 1 - (bias_pct / 100.0)
        
        # 1. BASELINE AUDIT
        global_ma_clean = self.calculate_ma(self.global_vals, method, param)
        global_report_mask = self.get_report_mask(len(self.global_vals), start_offset, frequency)
        
        baseline_aon_vals = global_ma_clean[global_report_mask]
        total_clean_checks = len(baseline_aon_vals)
        baseline_alarms = (baseline_aon_vals < lcl) | (baseline_aon_vals > ucl)
        total_false_alarms = np.sum(baseline_alarms)
        
        real_fpr_pct = 0.0
        if total_clean_checks > 0:
            real_fpr_pct = (total_false_alarms / total_clean_checks) * 100.0

        # 2. SIMULATION
        global_biased_export = self.global_vals.copy()
        injection_flags = np.zeros(len(self.global_vals), dtype=int)

        days_to_run = list(self.day_indices.keys())

        for day_name in days_to_run:
            start_idx, end_idx = self.day_indices[day_name]
            day_len = end_idx - start_idx
            
            # Kiểm tra độ dài ngày có đủ để report không
            # Với SMA cần ít nhất N mẫu. Với EWMA cần ít nhất Frequency mẫu (để có 1 điểm report)
            min_req = start_offset + 1
            if start_idx == 0 and day_len < min_req:
                continue

            if fixed_inject_idx is not None:
                local_inject = fixed_inject_idx
                if day_len <= local_inject: continue
            else:
                if day_len < 3: continue
                max_possible = day_len - 2
                max_limit_user = 40
                max_rnd = min(max_limit_user, max_possible)
                if max_rnd < 1: max_rnd = 1
                local_inject = np.random.randint(1, max_rnd + 1)
            
            total_days += 1
            global_inject_idx = start_idx + local_inject
            
            # --- TẠO DỮ LIỆU LỖI ---
            biased_chunk = self.global_vals[global_inject_idx : end_idx] * bias_factor
            
            if apply_trunc_on_bias:
                outlier_mask = (biased_chunk < self.trunc_min) | (biased_chunk > self.trunc_max)
                biased_chunk[outlier_mask] = np.nan
            
            global_biased_export[global_inject_idx : end_idx] = biased_chunk
            injection_flags[global_inject_idx : end_idx] = 1

            # --- TÍNH TOÁN DETECTION ---
            temp_global_vals = self.global_vals.copy()
            temp_global_vals[global_inject_idx : end_idx] = biased_chunk
            
            global_ma_biased_temp = self.calculate_ma(temp_global_vals, method, param)
            
            biased_check_mask = np.zeros(len(self.global_vals), dtype=bool)
            biased_check_mask[global_inject_idx : end_idx] = True
            
            final_biased_mask = biased_check_mask & global_report_mask
            check_vals_post = global_ma_biased_temp[final_biased_mask]
            
            if len(check_vals_post) > 0:
                if direction == 'positive':
                    alarms_post = (check_vals_post > ucl)
                else:
                    alarms_post = (check_vals_post < lcl)
                
                if np.any(alarms_post):
                    detected_days += 1
                    valid_indices = np.where(final_biased_mask)[0]
                    alarm_indices = valid_indices[alarms_post]
                    
                    if len(alarm_indices) > 0:
                        first_alarm_idx = alarm_indices[0] 
                        nped = first_alarm_idx - global_inject_idx + 1
                        nped_list.append(nped)
                        
                        # --- LOGIC REALITY MODE ---
                        if sim_mode == 'Reality (Fix on Alarm)':
                            revert_start = first_alarm_idx + 1
                            if revert_start < end_idx:
                                global_biased_export[revert_start : end_idx] = self.global_vals[revert_start : end_idx]
                                injection_flags[revert_start : end_idx] = 0

        metrics = {
            "Total Days": total_days,
            "Detected (%)": round(detected_days / total_days * 100, 1) if total_days > 0 else 0,
            "Real FPR (%)": round(real_fpr_pct, 2),
            "ANPed": round(np.mean(nped_list), 1) if nped_list else "N/A",
            "MNPed": round(np.median(nped_list), 1) if nped_list else "N/A",
            "95NPed": round(np.percentile(nped_list, 95), 1) if nped_list else "N/A"
        }
        
        # Export Data
        global_ma_biased_export_ma = self.calculate_ma(global_biased_export, method, param)
        aon_results = np.full(len(global_ma_biased_export_ma), np.nan)
        aon_results[global_report_mask] = global_ma_biased_export_ma[global_report_mask]

        export_data = pd.DataFrame({
            'Day': self.global_days,
            'Result_Original': self.global_vals,
            'Result_Biased': global_biased_export, 
            'Is_Injected': injection_flags,
            f'{method}_Continuous': global_ma_biased_export_ma,
            'AON_Reported': aon_results,
            'LCL': lcl,
            'UCL': ucl
        })
        
        return metrics, export_data, nped_list

# =========================================================
# 🖥️ PHẦN 4: GIAO DIỆN STREAMLIT
# =========================================================

st.set_page_config(layout="wide", page_title="PBRTQC Simulator Pro")

st.title("🏥 PBRTQC Simulator: Dual Bias Check & Visualization")
st.markdown("""
Hệ thống mô phỏng PBRTQC đa năng.
""")

with st.sidebar:
    st.header("1. Upload Data")
    f_train = st.file_uploader("Training Data (.xlsx)", type='xlsx')
    f_verify = st.file_uploader("Verify Data (.xlsx)", type='xlsx')
    
    st.divider()
    st.header("2. Settings")
    bias_pct = st.number_input("Bias (%)", value=5.0, step=0.5, help="Giá trị % dùng để cộng (Pos) và trừ (Neg).")
    
    apply_bias_trunc = st.checkbox("Áp dụng Truncation sau khi thêm Bias", value=False, 
                                   help="Nếu chọn: Giá trị bias vượt ngưỡng sẽ bị loại bỏ (NaN).")
    
    sim_mode = st.selectbox("Chế độ Mô phỏng (Simulation Mode)", 
                            ["Standard (Continuous Bias)", "Reality (Fix on Alarm)"],
                            help="Standard: Lỗi kéo dài hết ngày. Reality: Lỗi biến mất ngay sau khi có Alarm đầu tiên.")

    target_fpr = st.slider("Target FPR (%)", 0.0, 10.0, 2.0, 0.1) / 100
    
    # [NEW] MODEL SELECTION CHANGE
    model = st.selectbox("Model", ["EWMA", "SMA"])
    
    st.subheader("Injection Mode")
    inject_mode = st.radio("Chế độ thêm lỗi:", ["Ngẫu nhiên (Random 1-40)", "Cố định (Fixed Point)"])
    fixed_point = None
    if inject_mode == "Cố định (Fixed Point)":
        fixed_point = st.number_input("Vị trí mẫu bắt đầu lỗi:", min_value=1, value=20)

    st.divider()
    st.header("3. Truncation Limit")
    trunc_mode = st.radio("Phương pháp cắt:", ["Auto (Tự động)", "Manual (Thủ công)"])
    
    manual_min = 0.0
    manual_max = 1000.0
    
    if trunc_mode == "Manual (Thủ công)":
        c_min, c_max = st.columns(2)
        manual_min = c_min.number_input("Min Value", value=0.0)
        manual_max = c_max.number_input("Max Value", value=100.0)

if f_train and f_verify:
    df_temp = pd.read_excel(f_train, nrows=1)
    all_cols = df_temp.columns.tolist()
    
    c1, c2 = st.columns(2)
    col_res = c1.selectbox("Cột Kết quả (Results)", all_cols)
    col_day = c2.selectbox("Cột Ngày (Days)", all_cols)

    st.divider()
    st.subheader(f"4. Cấu hình tham số cho {model}")
    
    col_case1, col_case2, col_case3 = st.columns(3)
    cases_config = []
    
    # [NEW] UI LOGIC CHO SMA VÀ EWMA
    if model == 'SMA':
        # SMA: Dùng Block Size (N) và Frequency (F)
        default_configs = [(20, 2), (30, 3), (40, 4)]
        
        def create_sma_input(col, idx, default_n, default_f):
            with col:
                st.markdown(f"**Case {idx}**")
                bs = st.number_input(f"Block Size (N)", value=default_n, key=f"bs{idx}", min_value=2)
                freq = st.number_input("Frequency (F)", value=default_f, key=f"freq{idx}", min_value=1)
                # Với SMA, start offset là N - 1 (đợi đủ N mẫu)
                return {'param': bs, 'freq': freq, 'start_offset': bs - 1, 'label': f"N={bs}, F={freq}"}

        cases_config.append(create_sma_input(col_case1, 1, default_configs[0][0], default_configs[0][1]))
        cases_config.append(create_sma_input(col_case2, 2, default_configs[1][0], default_configs[1][1]))
        cases_config.append(create_sma_input(col_case3, 3, default_configs[2][0], default_configs[2][1]))

    else: # EWMA
        # EWMA: Dùng Lambda và Frequency (F)
        # Default Lambda = 0.4 như yêu cầu
        default_configs = [(0.4, 20), (0.4, 40), (0.4, 50)] 
        
        def create_ewma_input(col, idx, default_lam, default_f):
            with col:
                st.markdown(f"**Case {idx}**")
                lam = st.number_input(f"Lambda (λ)", value=default_lam, key=f"lam{idx}", min_value=0.01, max_value=1.0, step=0.01)
                freq = st.number_input("Frequency (F)", value=default_f, key=f"freq{idx}", min_value=1)
                # Với EWMA, start offset dựa vào F (F-1) để báo cáo tại cuối chu kỳ F đầu tiên
                return {'param': lam, 'freq': freq, 'start_offset': freq - 1, 'label': f"λ={lam}, F={freq}"}

        cases_config.append(create_ewma_input(col_case1, 1, default_configs[0][0], default_configs[0][1]))
        cases_config.append(create_ewma_input(col_case2, 2, default_configs[1][0], default_configs[1][1]))
        cases_config.append(create_ewma_input(col_case3, 3, default_configs[2][0], default_configs[2][1]))

    if st.button("🚀 Run Dual Simulation"):
        with st.spinner(f"Đang chạy mô phỏng ({sim_mode})..."):
            df_train, df_verify = load_data(f_train, f_verify, col_res, col_day)
            
            if df_train is not None:
                trunc_range = (0, 0)
                data_train_vals = df_train[col_res].dropna().values
                if trunc_mode == "Auto (Tự động)":
                    trunc_range = find_optimal_truncation(data_train_vals)
                    st.success(f"✅ Auto Truncation: [{trunc_range[0]:.2f} - {trunc_range[1]:.2f}]")
                else:
                    trunc_range = (manual_min, manual_max)
                    st.info(f"🔧 Manual Truncation: [{trunc_range[0]:.2f} - {trunc_range[1]:.2f}]")
                
                engine = PBRTQCEngine(df_train, df_verify, col_res, col_day, trunc_range)
                
                st.subheader("📋 Thống kê Dữ liệu (Sau Truncation)")
                stats_data = engine.get_data_stats()
                st.dataframe(pd.DataFrame([stats_data]), use_container_width=True)

                results_pos = []
                results_neg = []
                excel_sheets = {} 
                
                chart_container_pos = []
                chart_container_neg = []
                all_nped_data = {} 

                prog_bar = st.progress(0)
                
                for i, case in enumerate(cases_config):
                    # Gọi hàm với param và start_offset tương ứng
                    lcl, ucl = engine.determine_limits(
                        method=model, param=case['param'], start_offset=case['start_offset'], 
                        frequency=case['freq'], target_fpr=target_fpr
                    )
                    
                    # 1. Chạy Positive Bias
                    metrics_pos, df_pos, nped_list_pos = engine.run_simulation(
                        method=model, param=case['param'], start_offset=case['start_offset'], frequency=case['freq'],
                        lcl=lcl, ucl=ucl, bias_pct=bias_pct,
                        direction='positive',
                        fixed_inject_idx=fixed_point,
                        apply_trunc_on_bias=apply_bias_trunc,
                        sim_mode=sim_mode 
                    )
                    
                    # 2. Chạy Negative Bias
                    metrics_neg, df_neg, nped_list_neg = engine.run_simulation(
                        method=model, param=case['param'], start_offset=case['start_offset'], frequency=case['freq'],
                        lcl=lcl, ucl=ucl, bias_pct=bias_pct,
                        direction='negative',
                        fixed_inject_idx=fixed_point,
                        apply_trunc_on_bias=apply_bias_trunc,
                        sim_mode=sim_mode 
                    )
                    
                    # Lưu kết quả
                    row_base = {"Case": case['label'], "LCL": round(lcl, 2), "UCL": round(ucl, 2)}
                    results_pos.append({**row_base, **metrics_pos})
                    
                    metrics_neg_clean = metrics_neg.copy()
                    metrics_neg_clean.pop("Real FPR (%)", None) 
                    results_neg.append({**row_base, **metrics_neg_clean})
                    
                    # Tên Sheet an toàn (loại bỏ dấu = nếu có)
                    safe_label = case['label'].replace("=", "").replace(", ", "_")
                    case_key_pos = f"Pos_{safe_label}"
                    case_key_neg = f"Neg_{safe_label}"
                    
                    excel_sheets[case_key_pos] = df_pos
                    excel_sheets[case_key_neg] = df_neg
                    all_nped_data[case_key_pos] = nped_list_pos
                    all_nped_data[case_key_neg] = nped_list_neg
                    
                    fig_pos = draw_chart(df_pos, model, lcl, ucl, f"Case {i+1}: Positive Bias ({case['label']})", 'positive')
                    chart_container_pos.append(fig_pos)
                    
                    fig_neg = draw_chart(df_neg, model, lcl, ucl, f"Case {i+1}: Negative Bias ({case['label']})", 'negative')
                    chart_container_neg.append(fig_neg)

                    prog_bar.progress((i+1)/len(cases_config))
                
                # --- HIỂN THỊ KẾT QUẢ ---
                
                st.subheader("📈 Kết quả: Positive Bias Check (Check > UCL)")
                st.dataframe(pd.DataFrame(results_pos).style.highlight_max(subset=['Detected (%)'], color='#d1ffbd'), use_container_width=True)
                
                st.subheader("📉 Kết quả: Negative Bias Check (Check < LCL)")
                st.dataframe(pd.DataFrame(results_neg).style.highlight_max(subset=['Detected (%)'], color='#ffcccc'), use_container_width=True)

                st.divider()
                with st.expander("🔍 Xem Biểu đồ Positive Bias"):
                    for idx, fig in enumerate(chart_container_pos):
                        st.plotly_chart(fig, use_container_width=True)

                with st.expander("🔍 Xem Biểu đồ Negative Bias"):
                    for idx, fig in enumerate(chart_container_neg):
                        st.plotly_chart(fig, use_container_width=True)
                
                # --- DOWNLOAD ---
                st.divider()
                st.subheader("📥 Xuất dữ liệu")
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for sheet_name, df in excel_sheets.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                    df_audit_nped = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in all_nped_data.items() ]))
                    df_audit_nped.to_excel(writer, sheet_name="Audit_NPed_Raw", index=False)

                st.download_button(
                    label="Tải xuống báo cáo chi tiết (.xlsx)",
                    data=output.getvalue(),
                    file_name="PBRTQC_Dual_Simulation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error("Lỗi dữ liệu.")

