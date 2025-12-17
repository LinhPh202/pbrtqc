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
    """Tìm khoảng cắt tối ưu (Auto Mode - Asymmetric)"""
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
            if e <= s: continue
            
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
    """Vẽ biểu đồ kiểm soát vận hành (Control Chart)"""
    fig = go.Figure()

    # 1. Vẽ đường MA liên tục (Nối các điểm đứt quãng do drop outlier)
    ma_col_name = f'{method}_Bias_Continuous'
    if ma_col_name in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[ma_col_name], 
            mode='lines', 
            name=f'{method} (Continuous)',
            line=dict(color='lightblue', width=1.5),
            connectgaps=True # Quan trọng: Nối liền các điểm bị NaN
        ))

    # 2. Vẽ đường giới hạn
    fig.add_trace(go.Scatter(
        x=[df.index.min(), df.index.max()], 
        y=[ucl, ucl], 
        mode='lines', name='UCL', 
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=[df.index.min(), df.index.max()], 
        y=[lcl, lcl], 
        mode='lines', name='LCL', 
        line=dict(color='blue', width=2, dash='dash')
    ))

    # 3. Đánh dấu các điểm Alarm
    alarm_points = pd.DataFrame()
    col_report = 'AON_Bias_Report'
    
    if col_report in df.columns:
        if direction == 'positive':
            alarm_points = df[(df[col_report] > ucl)]
            label = 'Alarm (> UCL)'
            color = 'red'
        elif direction == 'negative':
            alarm_points = df[(df[col_report] < lcl)]
            label = 'Alarm (< LCL)'
            color = 'blue'
    
        if not alarm_points.empty:
            fig.add_trace(go.Scatter(
                x=alarm_points.index, 
                y=alarm_points[col_report], 
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

def draw_combined_curve(all_data, y_col, title, y_label, x_label="Bias (%)", is_log=False):
    """Hàm vẽ biểu đồ gộp"""
    fig = go.Figure()
    
    for item in all_data:
        label = item['Label']
        df = item['Data']
        plot_data = df.copy()
        
        fig.add_trace(go.Scatter(
            x=plot_data['Bias_Value'], 
            y=plot_data[y_col],
            mode='lines+markers',
            name=label, 
            marker=dict(size=5)
        ))

    yaxis_config = dict(title=y_label)
    if is_log:
        yaxis_config['type'] = 'log'
        yaxis_config['dtick'] = 1 
    else:
        yaxis_config['range'] = [-5, 105]

    if y_col == 'Detection':
         fig.add_hline(y=90, line_dash="dash", line_color="gray", annotation_text="90%", annotation_position="bottom right")

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#003366')),
        xaxis_title=x_label,
        yaxis=yaxis_config,
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0.05)'
    )
    return fig

# =========================================================
# 🧠 PHẦN 3: ENGINE MÔ PHỎNG (STRICT MODE)
# =========================================================

class PBRTQCEngine:
    def __init__(self, df_train, df_verify, col_res, col_day, trunc_range):
        self.trunc_min, self.trunc_max = trunc_range
        self.col_res = col_res
        self.col_day = col_day
        
        raw_train = df_train[col_res].values
        # [STRICT] Training Clean: Loại bỏ hoàn toàn giá trị ngoài khoảng
        self.train_clean = raw_train[(raw_train >= self.trunc_min) & (raw_train <= self.trunc_max)]
        
        self.df_verify_raw = df_verify.copy()
        self.global_vals = self.df_verify_raw[col_res].values.astype(float)
        self.global_days = self.df_verify_raw[col_day].values
        
        self.day_indices = {}
        unique_days = self.df_verify_raw[col_day].unique()
        current_idx = 0
        for day in unique_days:
            count = len(self.df_verify_raw[self.df_verify_raw[col_day] == day])
            self.day_indices[day] = (current_idx, current_idx + count)
            current_idx += count

    def get_data_stats(self):
        # Stats tính trên dữ liệu hợp lệ
        verify_valid = self.global_vals[(self.global_vals >= self.trunc_min) & (self.global_vals <= self.trunc_max)]
        return {
            "Train Mean": np.mean(self.train_clean),
            "Train Median": np.median(self.train_clean),
            "Verify Mean": np.mean(verify_valid),
            "Truncation Range": f"[{self.trunc_min:.2f} - {self.trunc_max:.2f}]"
        }

    def calculate_ma_stream(self, values, method, param):
        """Tính MA trên chuỗi đã lọc (Reduced Stream - không chứa Outlier)"""
        series = pd.Series(values)
        if method == 'SMA':
            return series.rolling(window=int(param), min_periods=1).mean().values
        elif method == 'EWMA':
            return series.ewm(alpha=param, adjust=False).mean().values
        return values

    def _process_strict(self, raw_array, method, param, start_offset, frequency):
        """
        Core logic Strict Mode:
        1. Drop Outlier.
        2. Calculate MA.
        3. Frequency counting on valid samples.
        4. Map back to original indices.
        """
        # 1. Lọc Outlier
        mask_valid = (raw_array >= self.trunc_min) & (raw_array <= self.trunc_max)
        stream_vals = raw_array[mask_valid]
        original_indices = np.where(mask_valid)[0] # Lưu vị trí gốc
        
        # 2. Tính MA trên dòng sạch
        stream_ma = self.calculate_ma_stream(stream_vals, method, param)
        
        # 3. Tạo Report Mask trên dòng sạch (Chỉ đếm mẫu hợp lệ)
        n_stream = len(stream_ma)
        report_mask_stream = np.zeros(n_stream, dtype=bool)
        s_idx = int(start_offset)
        if s_idx < n_stream:
            report_mask_stream[s_idx::int(frequency)] = True
            
        # 4. Map ngược về mảng gốc (Full Size)
        # Continuous Array (Có giá trị tại mọi điểm Hợp lệ)
        ma_continuous_full = np.full(len(raw_array), np.nan)
        ma_continuous_full[original_indices] = stream_ma
        
        # Report Array (Chỉ có giá trị tại điểm Report Hợp lệ)
        ma_report_full = np.full(len(raw_array), np.nan)
        report_indices_original = original_indices[report_mask_stream]
        ma_report_full[report_indices_original] = stream_ma[report_mask_stream]
        
        return ma_continuous_full, ma_report_full, report_indices_original

    def determine_limits(self, method, param, start_offset, frequency, target_fpr):
        # Tính Limit dựa trên Training Data đã clean (Reduced Stream)
        ma_values = self.calculate_ma_stream(self.train_clean, method, param)
        
        # Frequency đếm trên mẫu hợp lệ
        total_len = len(ma_values)
        report_mask = np.zeros(total_len, dtype=bool)
        s_idx = int(start_offset)
        if s_idx < total_len:
            report_mask[s_idx::int(frequency)] = True
            
        valid_ma_values = ma_values[report_mask]
        
        if len(valid_ma_values) == 0:
            return 0, 0, 0, 0

        lower = np.percentile(valid_ma_values, (target_fpr/2)*100)
        upper = np.percentile(valid_ma_values, (1 - target_fpr/2)*100)
        
        train_mean = np.mean(valid_ma_values)
        train_median = np.median(valid_ma_values)
        
        return lower, upper, train_mean, train_median

    def run_simulation(self, method, param, start_offset, frequency, lcl, ucl, bias_pct, direction='positive', fixed_inject_idx=None, apply_trunc_on_bias=False, sim_mode='Standard'):
        total_days = 0
        detected_days = 0
        nped_list = []
        residual_alarms_list = [] 
        audit_logs = []
        
        if direction == 'positive':
            bias_factor = 1 + (bias_pct / 100.0)
        else:
            bias_factor = 1 - (bias_pct / 100.0)
        
        # 1. BASELINE AUDIT (Tính FPR trên dữ liệu Verify gốc - coi là sạch)
        # Sử dụng logic Strict: Lọc bỏ outlier trước khi tính
        ma_cont_clean, ma_rep_clean, report_idx_clean = self._process_strict(
            self.global_vals, method, param, start_offset, frequency
        )
        
        # Tính FPR trên các điểm report có giá trị (không nan)
        valid_checks = ma_rep_clean[~np.isnan(ma_rep_clean)]
        total_clean_checks = len(valid_checks)
        baseline_alarms = (valid_checks < lcl) | (valid_checks > ucl)
        total_false_alarms = np.sum(baseline_alarms)
        
        real_fpr_pct = 0.0
        if total_clean_checks > 0:
            real_fpr_pct = (total_false_alarms / total_clean_checks) * 100.0

        # 2. SIMULATION
        # Mảng để export (sẽ được cập nhật dần theo logic Reality)
        final_biased_vals = self.global_vals.copy()
        injection_flags = np.zeros(len(self.global_vals), dtype=int)
        
        days_to_run = list(self.day_indices.keys())

        for day_name in days_to_run:
            start_idx, end_idx = self.day_indices[day_name]
            day_len = end_idx - start_idx
            
            # Logic inject point
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
            
            # Start offset condition
            min_req = start_offset + 1
            if start_idx == 0 and day_len < min_req: continue

            total_days += 1
            global_inject_idx = start_idx + local_inject
            
            # --- TẠO DỮ LIỆU LỖI CHO NGÀY NÀY ---
            # Logic: Trước ngày này -> Clean. Trong ngày này -> Biased. Sau ngày này -> Clean.
            temp_global_vals = self.global_vals.copy()
            
            # Inject Bias
            biased_chunk = temp_global_vals[global_inject_idx : end_idx] * bias_factor
            temp_global_vals[global_inject_idx : end_idx] = biased_chunk
            
            # Update mảng export (Mặc định là Standard mode, nếu Reality sẽ revert sau)
            final_biased_vals[global_inject_idx : end_idx] = biased_chunk
            injection_flags[global_inject_idx : end_idx] = 1
            
            # --- TÍNH TOÁN STRICT ---
            # 1. Drop Outlier toàn bộ dataset -> 2. Calc MA -> 3. Map back
            ma_cont_day, ma_rep_day, report_idx_day = self._process_strict(
                temp_global_vals, method, param, start_offset, frequency
            )
            
            # Chỉ xét các điểm report nằm trong vùng bias của ngày này
            relevant_report_indices = report_idx_day[
                (report_idx_day >= global_inject_idx) & (report_idx_day < end_idx)
            ]
            
            detected = False
            first_alarm_idx = -1
            
            if len(relevant_report_indices) > 0:
                check_vals = ma_rep_day[relevant_report_indices]
                
                # [STRICT] One-sided Detection for Simulation
                if direction == 'positive':
                    alarms = (check_vals > ucl) 
                else:
                    alarms = (check_vals < lcl)
                
                if np.any(alarms):
                    detected = True
                    first_alarm_idx = relevant_report_indices[np.argmax(alarms)]
            
            if detected:
                detected_days += 1
                
                # Tính NPed: Số mẫu HỢP LỆ từ điểm inject đến điểm alarm
                segment = temp_global_vals[global_inject_idx : first_alarm_idx + 1]
                valid_count = np.sum((segment >= self.trunc_min) & (segment <= self.trunc_max))
                nped_list.append(valid_count)
                
                # --- REALITY MODE ---
                if sim_mode == 'Reality (Fix on Alarm)':
                    revert_start = first_alarm_idx + 1
                    if revert_start < end_idx:
                        # Fix: Revert data
                        temp_global_vals[revert_start : end_idx] = self.global_vals[revert_start : end_idx]
                        final_biased_vals[revert_start : end_idx] = self.global_vals[revert_start : end_idx]
                        injection_flags[revert_start : end_idx] = 0
                        
                        # Re-calc Strict trên dữ liệu đã fix để đếm Residual
                        _, _, report_idx_fixed = self._process_strict(
                            temp_global_vals, method, param, start_offset, frequency
                        )
                        # Re-calc MA values (vì _process_strict trả về full array nhưng ta cần value cụ thể)
                        # Để tối ưu, ta gọi lại hàm lấy MA continuous
                        ma_cont_fixed, ma_rep_fixed, _ = self._process_strict(
                            temp_global_vals, method, param, start_offset, frequency
                        )
                        
                        # Check Residuals (Sau khi fix)
                        check_indices = report_idx_fixed[
                            (report_idx_fixed >= revert_start) & (report_idx_fixed < end_idx)
                        ]
                        
                        res_count = 0
                        for idx in check_indices:
                            val = ma_rep_fixed[idx]
                            # Residual check thường xét 2 chiều (dao động) hoặc chiều lỗi cũ
                            # Để chặt chẽ, check 2 chiều
                            is_alarm = (val < lcl) or (val > ucl)
                            if is_alarm:
                                res_count += 1
                            else:
                                break
                        residual_alarms_list.append(res_count)
            
            # Log Audit
            audit_logs.append({
                "Day": day_name,
                "Injection_Index": global_inject_idx,
                "Detection_Index": first_alarm_idx if detected else "N/A",
                "NPed": nped_list[-1] if detected else "N/A",
                "Residual_Alarms": residual_alarms_list[-1] if (detected and sim_mode == 'Reality (Fix on Alarm)') else "N/A"
            })

        # Metrics Summary
        avg_residual = "N/A"
        med_residual = "N/A"
        if sim_mode == 'Reality (Fix on Alarm)' and residual_alarms_list:
            avg_residual = round(np.mean(residual_alarms_list), 1)
            med_residual = round(np.median(residual_alarms_list), 1)

        metrics = {
            "Total Days": total_days,
            "Detected (%)": round(detected_days / total_days * 100, 1) if total_days > 0 else 0,
            "Real FPR (%)": round(real_fpr_pct, 2),
            "ANPed": round(np.mean(nped_list), 1) if nped_list else "N/A",
            "MNPed": round(np.median(nped_list), 1) if nped_list else "N/A",
            "95NPed": round(np.percentile(nped_list, 95), 1) if nped_list else "N/A",
            "Avg_Residual": avg_residual,
            "Med_Residual": med_residual
        }
        
        # Final Recalculate for Export/Visualization (Continuous Bias View)
        # Nếu là Standard Mode, final_biased_vals đã chứa bias cả ngày.
        # Nếu là Reality Mode, final_biased_vals đã được revert sau alarm.
        ma_cont_final, ma_rep_final, _ = self._process_strict(
            final_biased_vals, method, param, start_offset, frequency
        )

        export_data = pd.DataFrame({
            'Day': self.global_days,
            'Result_Original': self.global_vals,
            'Result_Biased': final_biased_vals, 
            'Is_Injected': injection_flags,
            f'{method}_Bias_Continuous': ma_cont_final,
            'AON_Bias_Report': ma_rep_final,
            'LCL': lcl,
            'UCL': ucl
        })
        
        return metrics, export_data, nped_list, audit_logs

# =========================================================
# 🖥️ PHẦN 4: GIAO DIỆN STREAMLIT
# =========================================================

st.set_page_config(layout="wide", page_title="PBRTQC Simulator Pro")

st.title("🏥 PBRTQC Simulator: Dual Bias Check & Visualization")
st.markdown("Hệ thống mô phỏng PBRTQC (Logic Strict: Loại bỏ Outliers).")

with st.sidebar:
    st.header("1. Upload Data")
    f_train = st.file_uploader("Training Data (.xlsx)", type='xlsx')
    f_verify = st.file_uploader("Verify Data (.xlsx)", type='xlsx')
    
    st.divider()
    st.header("2. Settings")
    bias_pct = st.number_input("Bias (%)", value=5.0, step=0.5)
    
    # Checkbox này mặc định True và Disable vì Strict Mode luôn áp dụng
    apply_bias_trunc = st.checkbox("Áp dụng Truncation sau khi thêm Bias", value=True, disabled=True,
                                   help="Bắt buộc trong chế độ Strict Mode.")
    
    sim_mode = st.selectbox("Chế độ Mô phỏng", 
                            ["Standard (Continuous Bias)", "Reality (Fix on Alarm)"])

    model = st.selectbox("Model", ["EWMA", "SMA"])
    
    st.subheader("Injection Mode")
    inject_mode = st.radio("Chế độ thêm lỗi:", ["Ngẫu nhiên (Random 1-40)", "Cố định (Fixed Point)"])
    fixed_point = None
    if inject_mode == "Cố định (Fixed Point)":
        fixed_point = st.number_input("Vị trí mẫu bắt đầu lỗi:", min_value=1, value=20)

    st.divider()
    st.header("3. Truncation Limit")
    trunc_mode = st.radio("Phương pháp cắt:", ["Auto (Tự động)", "Manual (Thủ công)", "Percentile (Phân vị)"])
    
    manual_min = 0.0
    manual_max = 1000.0
    percent_cut = 0.0

    if trunc_mode == "Manual (Thủ công)":
        c_min, c_max = st.columns(2)
        manual_min = c_min.number_input("Min Value", value=0.0)
        manual_max = c_max.number_input("Max Value", value=100.0)
    elif trunc_mode == "Percentile (Phân vị)":
        percent_cut = st.slider("Cắt % đuôi dữ liệu (mỗi đầu)", 0.0, 10.0, 1.0, 0.1)
    
    st.divider()
    st.header("4. Control Limit Settings")
    cl_mode = st.radio("Chế độ giới hạn:", ["Auto (Dựa trên FPR)", "Manual (Thủ công)"])
    
    target_fpr = 0.02 
    manual_lcl = 0.0
    manual_ucl = 0.0
    
    if cl_mode == "Auto (Dựa trên FPR)":
        target_fpr = st.slider("Target FPR (%)", 0.0, 10.0, 2.0, 0.1) / 100
    else:
        c1, c2 = st.columns(2)
        manual_lcl = c1.number_input("LCL (Manual)", value=0.0)
        manual_ucl = c2.number_input("UCL (Manual)", value=0.0)

    with st.expander("⚙️ Cấu hình nâng cao: Power Curve"):
        run_power_curve = st.checkbox("Vẽ biểu đồ Power Curve", value=False)
        pc_steps = st.slider("Mật độ điểm vẽ (Steps)", 10, 100, 30)

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
    
    if model == 'SMA':
        default_configs = [(20, 2), (30, 3), (40, 4)]
        def create_sma_input(col, idx, default_n, default_f):
            with col:
                st.markdown(f"**Case {idx}**")
                bs = st.number_input(f"Block Size (N)", value=default_n, key=f"bs{idx}", min_value=2)
                freq = st.number_input("Frequency (F)", value=default_f, key=f"freq{idx}", min_value=1)
                return {'param': bs, 'freq': freq, 'start_offset': bs - 1, 'label': f"N={bs}, F={freq}"}

        cases_config.append(create_sma_input(col_case1, 1, default_configs[0][0], default_configs[0][1]))
        cases_config.append(create_sma_input(col_case2, 2, default_configs[1][0], default_configs[1][1]))
        cases_config.append(create_sma_input(col_case3, 3, default_configs[2][0], default_configs[2][1]))

    else: # EWMA
        default_configs = [(0.4, 20), (0.4, 40), (0.4, 50)] 
        def create_ewma_input(col, idx, default_lam, default_f):
            with col:
                st.markdown(f"**Case {idx}**")
                lam = st.number_input(f"Lambda (λ)", value=default_lam, key=f"lam{idx}", min_value=0.01, max_value=1.0, step=0.01)
                freq = st.number_input("Frequency (F)", value=default_f, key=f"freq{idx}", min_value=1)
                return {'param': lam, 'freq': freq, 'start_offset': freq - 1, 'label': f"λ={lam}, F={freq}"}

        cases_config.append(create_ewma_input(col_case1, 1, default_configs[0][0], default_configs[0][1]))
        cases_config.append(create_ewma_input(col_case2, 2, default_configs[1][0], default_configs[1][1]))
        cases_config.append(create_ewma_input(col_case3, 3, default_configs[2][0], default_configs[2][1]))

    if 'sim_results' not in st.session_state:
        st.session_state.sim_results = None

    if st.button("🚀 Run Dual Simulation"):
        with st.spinner(f"Đang chạy mô phỏng ({sim_mode})..."):
            df_train, df_verify = load_data(f_train, f_verify, col_res, col_day)
            
            if df_train is not None:
                # Truncation Logic
                data_train_vals = df_train[col_res].dropna().values
                trunc_range = (0, 0)
                if trunc_mode == "Auto (Tự động)":
                    trunc_range = find_optimal_truncation(data_train_vals)
                elif trunc_mode == "Manual (Thủ công)":
                    trunc_range = (manual_min, manual_max)
                else: 
                    lower = np.percentile(data_train_vals, percent_cut)
                    upper = np.percentile(data_train_vals, 100 - percent_cut)
                    trunc_range = (lower, upper)
                
                engine = PBRTQCEngine(df_train, df_verify, col_res, col_day, trunc_range)
                stats_data = engine.get_data_stats()
                
                results_pos, results_neg = [], []
                excel_sheets = {} 
                chart_container_pos, chart_container_neg = [], []
                all_nped_data, all_audit_data = {}, {}
                all_pc_datasets = []

                prog_bar = st.progress(0)
                
                for i, case in enumerate(cases_config):
                    # Limits
                    lcl_auto, ucl_auto, train_mean, train_median = engine.determine_limits(
                        method=model, param=case['param'], start_offset=case['start_offset'], 
                        frequency=case['freq'], target_fpr=target_fpr
                    )
                    lcl = manual_lcl if cl_mode == "Manual (Thủ công)" else lcl_auto
                    ucl = manual_ucl if cl_mode == "Manual (Thủ công)" else ucl_auto
                    
                    # Positive
                    m_pos, df_pos, nped_pos, audit_pos = engine.run_simulation(
                        method=model, param=case['param'], start_offset=case['start_offset'], frequency=case['freq'],
                        lcl=lcl, ucl=ucl, bias_pct=bias_pct, direction='positive',
                        fixed_inject_idx=fixed_point, apply_trunc_on_bias=True, sim_mode=sim_mode 
                    )
                    # Negative
                    m_neg, df_neg, nped_neg, audit_neg = engine.run_simulation(
                        method=model, param=case['param'], start_offset=case['start_offset'], frequency=case['freq'],
                        lcl=lcl, ucl=ucl, bias_pct=bias_pct, direction='negative',
                        fixed_inject_idx=fixed_point, apply_trunc_on_bias=True, sim_mode=sim_mode 
                    )
                    
                    row_base = {
                        "Case": case['label'], "LCL": round(lcl, 2), "UCL": round(ucl, 2),
                        "Train_Mean": round(train_mean, 2), "Train_Median": round(train_median, 2)
                    }
                    results_pos.append({**row_base, **m_pos})
                    results_neg.append({**row_base, **m_neg})
                    
                    safe_label = case['label'].replace("=", "").replace(", ", "_")
                    excel_sheets[f"Pos_{safe_label}"] = df_pos
                    excel_sheets[f"Neg_{safe_label}"] = df_neg
                    all_nped_data[f"Pos_{safe_label}"] = nped_pos
                    all_nped_data[f"Neg_{safe_label}"] = nped_neg
                    
                    for rec in audit_pos: rec['Case'] = f"Pos_{case['label']}"
                    for rec in audit_neg: rec['Case'] = f"Neg_{case['label']}"
                    all_audit_data[f"Pos_{safe_label}"] = audit_pos
                    all_audit_data[f"Neg_{safe_label}"] = audit_neg
                    
                    chart_container_pos.append(draw_chart(df_pos, model, lcl, ucl, f"Pos: {case['label']}", 'positive'))
                    chart_container_neg.append(draw_chart(df_neg, model, lcl, ucl, f"Neg: {case['label']}", 'negative'))

                    # Power Curve
                    if run_power_curve:
                        max_mag = bias_pct * 2
                        bias_range = np.linspace(-max_mag, max_mag, pc_steps)
                        pc_results = []
                        for b_val in bias_range:
                            curr_dir = 'positive' if b_val >= 0 else 'negative'
                            curr_mag = abs(b_val)
                            m_pc, _, _, _ = engine.run_simulation(
                                method=model, param=case['param'], start_offset=case['start_offset'], frequency=case['freq'],
                                lcl=lcl, ucl=ucl, bias_pct=curr_mag, direction=curr_dir, 
                                fixed_inject_idx=fixed_point, apply_trunc_on_bias=True, sim_mode=sim_mode
                            )
                            mnped_val = m_pc['MNPed']
                            anped_val = m_pc['ANPed']
                            if isinstance(mnped_val, str): mnped_val = 1500
                            if isinstance(anped_val, str): anped_val = 1500
                            pc_results.append({
                                'Bias_Value': b_val, 'Detection': m_pc['Detected (%)'],
                                'MNPed': mnped_val, 'ANPed': anped_val
                            })
                        df_pc = pd.DataFrame(pc_results)
                        all_pc_datasets.append({'Label': case['label'], 'Data': df_pc})
                        excel_sheets[f"PC_{safe_label}"] = df_pc

                    prog_bar.progress((i+1)/len(cases_config))
                
                st.session_state.sim_results = {
                    'stats': stats_data, 'trunc_range': trunc_range,
                    'res_pos': results_pos, 'res_neg': results_neg,
                    'charts_pos': chart_container_pos, 'charts_neg': chart_container_neg,
                    'pc_datasets': all_pc_datasets, 'excel_sheets': excel_sheets,
                    'nped_data': all_nped_data, 'audit_data': all_audit_data, 'bias_target': bias_pct 
                }
            else:
                st.error("Lỗi dữ liệu đầu vào.")

    if st.session_state.sim_results is not None:
        data = st.session_state.sim_results
        
        st.subheader("📋 Thống kê Dữ liệu")
        st.info(f"Truncation Range: [{data['trunc_range'][0]:.2f} - {data['trunc_range'][1]:.2f}]")
        st.dataframe(pd.DataFrame([data['stats']]), use_container_width=True)

        st.subheader("📈 Kết quả chi tiết: Positive Bias")
        st.dataframe(pd.DataFrame(data['res_pos']).style.highlight_max(subset=['Detected (%)'], color='#d1ffbd'), use_container_width=True)
        
        st.subheader("📉 Kết quả chi tiết: Negative Bias")
        st.dataframe(pd.DataFrame(data['res_neg']).style.highlight_max(subset=['Detected (%)'], color='#ffcccc'), use_container_width=True)
        
        with st.expander("🔍 Xem Biểu đồ Control Charts"):
            c1, c2 = st.columns(2)
            with c1: 
                for fig in data['charts_pos']: st.plotly_chart(fig, use_container_width=True)
            with c2:
                for fig in data['charts_neg']: st.plotly_chart(fig, use_container_width=True)

        if run_power_curve and data['pc_datasets']:
            st.divider()
            st.header("📊 Combined Power Function Graphs")
            c1, c2, c3 = st.columns(3)
            with c1:
                fig_det = draw_combined_curve(data['pc_datasets'], 'Detection', "Detection Rate (%)", "Detection (%)")
                st.plotly_chart(fig_det, use_container_width=True)
            with c2:
                fig_mnped = draw_combined_curve(data['pc_datasets'], 'MNPed', "Median NPed (Speed)", "MNPed", is_log=True)
                st.plotly_chart(fig_mnped, use_container_width=True)
            with c3:
                fig_anped = draw_combined_curve(data['pc_datasets'], 'ANPed', "Average NPed (Speed)", "ANPed", is_log=True)
                st.plotly_chart(fig_anped, use_container_width=True)
        
        st.divider()
        st.subheader("📥 Xuất dữ liệu")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df in data['excel_sheets'].items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            df_audit_nped = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in data['nped_data'].items()]))
            df_audit_nped.to_excel(writer, sheet_name="Audit_NPed_Raw", index=False)
            
            all_audit_logs = []
            for case_key, logs in data['audit_data'].items():
                all_audit_logs.extend(logs)
            if all_audit_logs:
                pd.DataFrame(all_audit_logs).to_excel(writer, sheet_name="Audit_Residual_Details", index=False)

        st.download_button(
            label="Tải xuống báo cáo chi tiết (.xlsx)",
            data=output.getvalue(),
            file_name="PBRTQC_Dual_Simulation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
