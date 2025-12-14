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
        
        # Verify Data (Apply Truncation Limit immediately)
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

    def calculate_ma(self, values, method, block_size):
        series = pd.Series(values)
        if method == 'SMA':
            return series.rolling(window=int(block_size)).mean().values
        elif method == 'EWMA':
            lam = 2 / (int(block_size) + 1)
            return series.ewm(alpha=lam, adjust=False).mean().values
        return values

    def get_report_mask(self, total_length, block_size, frequency):
        mask = np.zeros(total_length, dtype=bool)
        start_idx = int(block_size) - 1
        if start_idx < total_length:
            report_indices = np.arange(start_idx, total_length, int(frequency))
            mask[report_indices] = True
        return mask

    def determine_limits(self, method, block_size, frequency, target_fpr):
        ma_values = self.calculate_ma(self.train_clean, method, block_size)
        mask = self.get_report_mask(len(ma_values), block_size, frequency)
        valid_ma_values = ma_values[mask]
        
        if len(valid_ma_values) == 0:
            return 0, 0 

        lower = np.percentile(valid_ma_values, (target_fpr/2)*100)
        upper = np.percentile(valid_ma_values, (1 - target_fpr/2)*100)
        return lower, upper

    def run_simulation(self, method, block_size, frequency, lcl, ucl, bias_pct, direction='positive', fixed_inject_idx=None):
        total_days = 0
        detected_days = 0
        nped_list = []
        
        if direction == 'positive':
            bias_factor = 1 + (bias_pct / 100.0)
        else:
            bias_factor = 1 - (bias_pct / 100.0)
        
        # =======================================================
        # 1. AUDIT & FPR Calculation (Trên TOÀN BỘ dữ liệu sạch)
        # =======================================================
        # Tính MA cho toàn bộ file Verify (giả sử chưa có lỗi)
        global_ma_clean = self.calculate_ma(self.global_vals, method, block_size)
        global_report_mask = self.get_report_mask(len(self.global_vals), block_size, frequency)
        
        # Lấy tất cả các điểm AON report sạch
        baseline_aon_vals = global_ma_clean[global_report_mask]
        
        # Audit: Tổng số điểm kiểm tra
        total_clean_checks = len(baseline_aon_vals)
        
        # Audit: Tổng số Alarm giả (Check 2 đầu vì đây là dữ liệu sạch)
        baseline_alarms = (baseline_aon_vals < lcl) | (baseline_aon_vals > ucl)
        total_false_alarms = np.sum(baseline_alarms)
        
        # Tính FPR
        real_fpr_pct = 0.0
        if total_clean_checks > 0:
            real_fpr_pct = (total_false_alarms / total_clean_checks) * 100.0

        # =======================================================
        # 2. CHẠY MÔ PHỎNG TIÊM LỖI (Tính Detection)
        # =======================================================
        global_biased_export = self.global_vals.copy()
        injection_flags = np.zeros(len(self.global_vals), dtype=int)

        days_to_run = list(self.day_indices.keys())

        for day_name in days_to_run:
            start_idx, end_idx = self.day_indices[day_name]
            day_len = end_idx - start_idx
            
            # Logic: Ngày đầu phải đủ block, ngày sau chỉ cần đủ để chứa Injection Point
            if start_idx == 0 and day_len < block_size:
                continue

            # Xác định Injection Point
            if fixed_inject_idx is not None:
                local_inject = fixed_inject_idx
                if day_len <= local_inject: continue
            else:
                if day_len < 3: continue
                max_rnd = day_len - 2 
                if max_rnd < 1: max_rnd = 1
                local_inject = np.random.randint(1, max_rnd + 1)
            
            total_days += 1
            global_inject_idx = start_idx + local_inject
            
            # Tiêm lỗi vào Data Export
            global_biased_export[global_inject_idx : end_idx] *= bias_factor
            injection_flags[global_inject_idx : end_idx] = 1

            # CHECK DETECTION (Chỉ xét vùng sau khi tiêm lỗi)
            temp_global_vals = self.global_vals.copy()
            temp_global_vals[global_inject_idx : end_idx] *= bias_factor
            
            # Tính lại MA cục bộ (hoặc giả lập) cho detection
            global_ma_biased_temp = self.calculate_ma(temp_global_vals, method, block_size)
            
            # Mask vùng bị lỗi
            biased_check_mask = np.zeros(len(self.global_vals), dtype=bool)
            biased_check_mask[global_inject_idx : end_idx] = True
            
            # Kết hợp với Report Mask
            final_biased_mask = biased_check_mask & global_report_mask
            check_vals_post = global_ma_biased_temp[final_biased_mask]
            
            if len(check_vals_post) > 0:
                # Detection check theo hướng (1 chiều)
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

        metrics = {
            "Total Days": total_days,
            "Detected (%)": round(detected_days / total_days * 100, 1) if total_days > 0 else 0,
            "Real FPR (%)": round(real_fpr_pct, 2),
            "Detected_Count": detected_days,
            "False_Alarm_Count": total_false_alarms, # Global Count
            "Clean_Check_Count": total_clean_checks, # Global Count
            "ANPed": round(np.mean(nped_list), 1) if nped_list else "N/A",
            "MNPed": round(np.median(nped_list), 1) if nped_list else "N/A",
            "95NPed": round(np.percentile(nped_list, 95), 1) if nped_list else "N/A"
        }
        
        # Export Data
        global_ma_biased_export = self.calculate_ma(global_biased_export, method, block_size)
        aon_results = np.full(len(global_ma_biased_export), np.nan)
        aon_results[global_report_mask] = global_ma_biased_export[global_report_mask]

        export_data = pd.DataFrame({
            'Day': self.global_days,
            'Result_Original': self.global_vals,
            'Result_Biased': global_biased_export,
            'Is_Injected': injection_flags,
            f'{method}_Continuous': global_ma_biased_export,
            'AON_Reported': aon_results,
            'LCL': lcl,
            'UCL': ucl
        })
        
        return metrics, export_data

# =========================================================
# 🖥️ PHẦN 4: GIAO DIỆN STREAMLIT
# =========================================================

st.set_page_config(layout="wide", page_title="PBRTQC Simulator Pro")

st.title("🏥 PBRTQC Simulator: Dual Bias Check & Visualization")
st.markdown("""
Hệ thống mô phỏng 2 chiều + Biểu đồ trực quan:
1.  **Positive Bias (+):** Cộng thêm Bias -> Check > UCL.
2.  **Negative Bias (-):** Trừ đi Bias -> Check < LCL.
""")

with st.sidebar:
    st.header("1. Upload Data")
    f_train = st.file_uploader("Training Data (.xlsx)", type='xlsx')
    f_verify = st.file_uploader("Verify Data (.xlsx)", type='xlsx')
    
    st.divider()
    st.header("2. Settings")
    bias_pct = st.number_input("Bias (%)", value=5.0, step=0.5, help="Giá trị % dùng để cộng (Pos) và trừ (Neg).")
    target_fpr = st.slider("Target FPR (%)", 0.0, 10.0, 2.0, 0.1) / 100
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
    
    default_configs = []
    if model == 'SMA':
        default_configs = [(20, 2), (30, 3), (40, 4)]
    else: # EWMA
        default_configs = [(3, 3), (4, 4), (5, 5)]

    col_case1, col_case2, col_case3 = st.columns(3)
    cases_config = []
    
    def create_case_input(col, idx, default_n, default_f):
        with col:
            st.markdown(f"**Case {idx}**")
            bs = st.number_input(f"Block Size (N)", value=default_n, key=f"bs{idx}", min_value=2)
            freq = st.number_input("Frequency (F)", value=default_f, key=f"freq{idx}", min_value=1)
            return {'bs': bs, 'freq': freq}

    cases_config.append(create_case_input(col_case1, 1, default_configs[0][0], default_configs[0][1]))
    cases_config.append(create_case_input(col_case2, 2, default_configs[1][0], default_configs[1][1]))
    cases_config.append(create_case_input(col_case3, 3, default_configs[2][0], default_configs[2][1]))

    if st.button("🚀 Run Dual Simulation"):
        with st.spinner("Đang chạy mô phỏng và vẽ biểu đồ..."):
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

                prog_bar = st.progress(0)
                
                for i, case in enumerate(cases_config):
                    lcl, ucl = engine.determine_limits(model, case['bs'], case['freq'], target_fpr)
                    
                    # 1. Chạy Positive Bias
                    metrics_pos, df_pos = engine.run_simulation(
                        method=model, block_size=case['bs'], frequency=case['freq'],
                        lcl=lcl, ucl=ucl, bias_pct=bias_pct,
                        direction='positive',
                        fixed_inject_idx=fixed_point
                    )
                    
                    # 2. Chạy Negative Bias
                    metrics_neg, df_neg = engine.run_simulation(
                        method=model, block_size=case['bs'], frequency=case['freq'],
                        lcl=lcl, ucl=ucl, bias_pct=bias_pct,
                        direction='negative',
                        fixed_inject_idx=fixed_point
                    )
                    
                    # Lưu kết quả
                    row_base = {"Case": f"N={case['bs']}, F={case['freq']}", "LCL": round(lcl, 2), "UCL": round(ucl, 2)}
                    results_pos.append({**row_base, **metrics_pos})
                    
                    metrics_neg_clean = metrics_neg.copy()
                    metrics_neg_clean.pop("Real FPR (%)", None) 
                    results_neg.append({**row_base, **metrics_neg_clean})
                    
                    excel_sheets[f"Pos_N{case['bs']}_F{case['freq']}"] = df_pos
                    excel_sheets[f"Neg_N{case['bs']}_F{case['freq']}"] = df_neg
                    
                    fig_pos = draw_chart(df_pos, model, lcl, ucl, f"Case {i+1}: Positive Bias (N={case['bs']}, F={case['freq']})", 'positive')
                    chart_container_pos.append(fig_pos)
                    
                    fig_neg = draw_chart(df_neg, model, lcl, ucl, f"Case {i+1}: Negative Bias (N={case['bs']}, F={case['freq']})", 'negative')
                    chart_container_neg.append(fig_neg)

                    prog_bar.progress((i+1)/len(cases_config))
                
                # --- HIỂN THỊ KẾT QUẢ ---
                
                st.subheader("📈 Kết quả: Positive Bias Check (Check > UCL)")
                st.dataframe(pd.DataFrame(results_pos).style.highlight_max(subset=['Detected (%)'], color='#d1ffbd'), use_container_width=True)
                
                st.subheader("📉 Kết quả: Negative Bias Check (Check < LCL)")
                st.dataframe(pd.DataFrame(results_neg).style.highlight_max(subset=['Detected (%)'], color='#ffcccc'), use_container_width=True)

                # --- BẢNG AUDIT SỐ LIỆU THÔ ---
                st.divider()
                st.subheader("🕵️ Audit Data (Số liệu thô - Toàn bộ Dataset)")
                st.info("FPR được tính trên toàn bộ dữ liệu sạch (Baseline). Detection được tính theo từng ngày.")
                
                audit_cols = ['Case', 'Detected_Count', 'Total Days', 'False_Alarm_Count', 'Clean_Check_Count']
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Positive Bias Audit**")
                    df_audit_pos = pd.DataFrame(results_pos)[audit_cols]
                    st.dataframe(df_audit_pos, use_container_width=True)
                
                with c2:
                    st.markdown("**Negative Bias Audit**")
                    df_audit_neg = pd.DataFrame(results_neg)[audit_cols]
                    st.dataframe(df_audit_neg, use_container_width=True)
                
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
                
                st.download_button(
                    label="Tải xuống báo cáo chi tiết (.xlsx)",
                    data=output.getvalue(),
                    file_name="PBRTQC_Dual_Simulation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error("Lỗi dữ liệu.")
