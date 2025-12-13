import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import io

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
# 🧠 PHẦN 2: ENGINE MÔ PHỎNG (STRIDE LOGIC & NPED CALC)
# =========================================================

class PBRTQCEngine:
    def __init__(self, df_train, df_verify, col_res, col_day, trunc_range):
        self.trunc_min, self.trunc_max = trunc_range
        self.col_res = col_res
        self.col_day = col_day
        
        # 1. Training Data
        raw_train = df_train[col_res].values
        self.train_clean = raw_train[(raw_train >= self.trunc_min) & (raw_train <= self.trunc_max)]
        
        # 2. Verify Data
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

    def calculate_ma(self, values, method, block_size):
        """Tính MA liên tục"""
        series = pd.Series(values)
        if method == 'SMA':
            return series.rolling(window=int(block_size)).mean().values
        elif method == 'EWMA':
            lam = 2 / (int(block_size) + 1)
            return series.ewm(alpha=lam, adjust=False).mean().values
        return values

    def get_report_mask(self, total_length, block_size, frequency):
        """Tạo mask xác định các điểm Report (N, N+F, N+2F...)"""
        mask = np.zeros(total_length, dtype=bool)
        start_idx = int(block_size) - 1
        if start_idx < total_length:
            report_indices = np.arange(start_idx, total_length, int(frequency))
            mask[report_indices] = True
        return mask

    def determine_limits(self, method, block_size, frequency, target_fpr):
        """Tính Limit dựa trên các điểm Report của Training Data"""
        ma_values = self.calculate_ma(self.train_clean, method, block_size)
        mask = self.get_report_mask(len(ma_values), block_size, frequency)
        valid_ma_values = ma_values[mask]
        
        if len(valid_ma_values) == 0:
            return 0, 0 

        lower = np.percentile(valid_ma_values, (target_fpr/2)*100)
        upper = np.percentile(valid_ma_values, (1 - target_fpr/2)*100)
        return lower, upper

    def run_simulation(self, method, block_size, frequency, lcl, ucl, bias_pct, num_sims=None, fixed_inject_idx=None):
        total_days = 0
        detected_days = 0
        nped_list = []
        
        total_clean_checks = 0    
        total_false_alarms = 0    

        bias_factor = 1 + (bias_pct / 100.0)
        
        # 1. Tính Clean MA & Report Mask
        global_ma_clean = self.calculate_ma(self.global_vals, method, block_size)
        global_report_mask = self.get_report_mask(len(self.global_vals), block_size, frequency)

        # 2. Chuẩn bị xuất Excel
        global_biased_export = self.global_vals.copy()
        injection_flags = np.zeros(len(self.global_vals), dtype=int)

        days_to_run = list(self.day_indices.keys())
        if num_sims and num_sims < len(days_to_run):
            days_to_run = days_to_run[:num_sims]

        for day_name in days_to_run:
            start_idx, end_idx = self.day_indices[day_name]
            day_len = end_idx - start_idx
            
            # --- LOGIC MỚI: LỌC NGÀY ---
            # 1. Nếu là ngày đầu tiên của toàn bộ dữ liệu (start_idx = 0):
            # Bắt buộc phải đủ Block Size để khởi tạo MA.
            if start_idx == 0 and day_len < block_size:
                continue

            # 2. Xác định điểm tiêm lỗi (Injection Point)
            if fixed_inject_idx is not None:
                local_inject = fixed_inject_idx
                # Nếu ngày quá ngắn, không tới được điểm Fixed Injection -> Bỏ qua
                if day_len <= local_inject:
                    continue
            else:
                # Random Logic: Tự động điều chỉnh theo độ dài ngày
                # Đảm bảo còn ít nhất 1 mẫu sau điểm tiêm lỗi
                if day_len < 3: continue # Quá ngắn để random
                max_rnd = day_len - 2 
                if max_rnd < 1: max_rnd = 1
                local_inject = np.random.randint(1, max_rnd + 1)
            
            # --- CHẠY MÔ PHỎNG NẾU ĐỦ ĐIỀU KIỆN ---
            total_days += 1
            global_inject_idx = start_idx + local_inject
            
            # Cập nhật Data Export
            global_biased_export[global_inject_idx : end_idx] *= bias_factor
            injection_flags[global_inject_idx : end_idx] = 1

            # ----------------------------------------------------
            # 1. CHECK FALSE POSITIVE (Vùng trước lỗi)
            # ----------------------------------------------------
            clean_check_mask = np.zeros(len(self.global_vals), dtype=bool)
            clean_check_mask[start_idx : global_inject_idx] = True
            
            final_clean_mask = clean_check_mask & global_report_mask
            check_vals = global_ma_clean[final_clean_mask]
            
            if len(check_vals) > 0:
                total_clean_checks += len(check_vals)
                alarms = (check_vals < lcl) | (check_vals > ucl)
                num_fp = np.sum(alarms)
                total_false_alarms += num_fp
                
                if num_fp > 0:
                    continue # False Alarm Day -> Skip

            # ----------------------------------------------------
            # 2. CHECK DETECTION (Vùng sau lỗi)
            # ----------------------------------------------------
            temp_global_vals = self.global_vals.copy()
            temp_global_vals[global_inject_idx : end_idx] *= bias_factor
            
            global_ma_biased = self.calculate_ma(temp_global_vals, method, block_size)
            
            biased_check_mask = np.zeros(len(self.global_vals), dtype=bool)
            biased_check_mask[global_inject_idx : end_idx] = True
            
            final_biased_mask = biased_check_mask & global_report_mask
            check_vals_post = global_ma_biased[final_biased_mask]
            
            if len(check_vals_post) > 0:
                alarms_post = (check_vals_post < lcl) | (check_vals_post > ucl)
                
                if np.any(alarms_post):
                    detected_days += 1
                    valid_indices = np.where(final_biased_mask)[0]
                    alarm_indices = valid_indices[alarms_post]
                    
                    if len(alarm_indices) > 0:
                        first_alarm_idx = alarm_indices[0]
                        nped = first_alarm_idx - global_inject_idx + 1
                        nped_list.append(nped)

        # --- TỔNG HỢP METRICS ---
        real_fpr_pct = 0.0
        if total_clean_checks > 0:
            real_fpr_pct = (total_false_alarms / total_clean_checks) * 100.0
            
        metrics = {
            "Total Days": total_days,
            "Detected (%)": round(detected_days / total_days * 100, 1) if total_days > 0 else 0,
            "Real FPR (%)": round(real_fpr_pct, 2),
            "ANPed": round(np.mean(nped_list), 1) if nped_list else "N/A",
            "MNPed": round(np.median(nped_list), 1) if nped_list else "N/A",
            "95NPed": round(np.percentile(nped_list, 95), 1) if nped_list else "N/A"
        }
        
        # --- EXCEL EXPORT ---
        global_ma_biased_export = self.calculate_ma(global_biased_export, method, block_size)
        aon_results = np.full(len(global_ma_biased_export), np.nan)
        aon_results[global_report_mask] = global_ma_biased_export[global_report_mask]

        export_data = pd.DataFrame({
            'Day': self.global_days,
            'Result_Original': self.global_vals,
            'Result_Biased': global_biased_export,
            'Is_Injected': injection_flags,
            f'{method}_Continuous': global_ma_clean,
            'AON_Reported': aon_results,
            'LCL': lcl,
            'UCL': ucl
        })
        
        return metrics, export_data

# =========================================================
# 🖥️ PHẦN 3: GIAO DIỆN STREAMLIT
# =========================================================

st.set_page_config(layout="wide", page_title="PBRTQC Simulator Pro")

st.title("🏥 PBRTQC Simulator: Stride Logic")
st.markdown("""
**Hệ thống tính toán:**
- **Continuous Logic:** Ngày sau nối tiếp ngày trước.
- **Filter Logic:** Chỉ loại bỏ ngày quá ngắn so với điểm tiêm lỗi. Ngày đầu tiên bắt buộc >= Block Size.
""")

with st.sidebar:
    st.header("1. Upload Data")
    f_train = st.file_uploader("Training Data (.xlsx)", type='xlsx')
    f_verify = st.file_uploader("Verify Data (.xlsx)", type='xlsx')
    
    st.divider()
    st.header("2. Settings")
    bias_pct = st.number_input("Bias (%)", value=5.0, step=0.5)
    target_fpr = st.slider("Target FPR (%)", 0.1, 10.0, 2.0, 0.1) / 100
    model = st.selectbox("Model", ["EWMA", "SMA"])
    max_days = st.slider("Max Simulation Days", 10, 5000, 100)
    
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
    
    def create_case_input(col, idx):
        with col:
            st.markdown(f"**Case {idx}**")
            bs = st.number_input(f"Block Size (N)", value=20*idx, key=f"bs{idx}", min_value=2,
                                 help="Kích thước cửa sổ tính toán (Window Size).")
            freq = st.number_input("Frequency (F)", value=1, key=f"freq{idx}", min_value=1,
                                 help="Bước nhảy báo cáo (Stride).")
            return {'bs': bs, 'freq': freq}

    cases_config.append(create_case_input(col_case1, 1))
    cases_config.append(create_case_input(col_case2, 2))
    cases_config.append(create_case_input(col_case3, 3))

    if st.button("🚀 Run Simulation"):
        with st.spinner("Đang chạy mô phỏng..."):
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
                
                results = []
                excel_sheets = {} 
                
                prog_bar = st.progress(0)
                
                for i, case in enumerate(cases_config):
                    lcl, ucl = engine.determine_limits(model, case['bs'], case['freq'], target_fpr)
                    
                    metrics, export_df = engine.run_simulation(
                        method=model, 
                        block_size=case['bs'], 
                        frequency=case['freq'],
                        lcl=lcl, ucl=ucl, 
                        bias_pct=bias_pct,
                        num_sims=max_days, 
                        fixed_inject_idx=fixed_point
                    )
                    
                    res_row = {
                        "Case": f"N={case['bs']}, F={case['freq']}",
                        "LCL": round(lcl, 2), "UCL": round(ucl, 2),
                        **metrics
                    }
                    results.append(res_row)
                    excel_sheets[f"Case_N{case['bs']}_F{case['freq']}"] = export_df
                    prog_bar.progress((i+1)/len(cases_config))
                
                st.subheader("📊 Bảng Kết quả Đánh giá")
                st.dataframe(pd.DataFrame(results).style.highlight_max(subset=['Detected (%)'], color='#d1ffbd'), use_container_width=True)
                
                st.divider()
                st.subheader("📥 Xuất dữ liệu")
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for sheet_name, df in excel_sheets.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                st.download_button(
                    label="Tải xuống chi tiết (.xlsx)",
                    data=output.getvalue(),
                    file_name="PBRTQC_Simulation_Results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error("Lỗi dữ liệu.")
