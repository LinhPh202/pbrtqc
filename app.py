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
        
        # Loại bỏ dòng trống
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
# 🧠 PHẦN 2: ENGINE MÔ PHỎNG (CONTINUOUS MODE)
# =========================================================

class PBRTQCEngine:
    def __init__(self, df_train, df_verify, col_res, col_day, trunc_range):
        self.trunc_min, self.trunc_max = trunc_range
        self.col_res = col_res
        self.col_day = col_day
        
        # 1. Training Data
        raw_train = df_train[col_res].values
        self.train_clean = raw_train[(raw_train >= self.trunc_min) & (raw_train <= self.trunc_max)]
        
        # 2. Verify Data (Continuous)
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

    def calculate_ma(self, values, method, param):
        """Tính MA trên toàn bộ chuỗi"""
        series = pd.Series(values)
        if method == 'SMA':
            return series.rolling(window=int(param)).mean().bfill().values
        elif method == 'EWMA':
            lam = 2 / (int(param) + 1)
            return series.ewm(alpha=lam, adjust=False).mean().values
        return values

    def determine_limits(self, method, param, target_fpr):
        """Tính Limit từ Training Data"""
        ma_values = self.calculate_ma(self.train_clean, method, param)
        lower = np.percentile(ma_values, (target_fpr/2)*100)
        upper = np.percentile(ma_values, (1 - target_fpr/2)*100)
        return lower, upper

    def run_continuous_simulation(self, method, param, lcl, ucl, bias_pct, frequency=1, num_sims=None, fixed_inject_idx=None):
        total_days = 0
        detected_days = 0
        false_positive_days = 0
        nped_list = []
        
        bias_factor = 1 + (bias_pct / 100.0)
        
        # Tính Global Clean MA (để xuất ra Excel và check FP)
        global_ma_clean = self.calculate_ma(self.global_vals, method, param)
        
        # Mảng Index check Frequency
        global_indices = np.arange(len(self.global_vals))
        valid_check_points = (global_indices % frequency == 0)

        days_to_run = list(self.day_indices.keys())
        if num_sims and num_sims < len(days_to_run):
            days_to_run = days_to_run[:num_sims]

        for day_name in days_to_run:
            start_idx, end_idx = self.day_indices[day_name]
            day_len = end_idx - start_idx
            if day_len < 5: continue
            total_days += 1
            
            # --- CHỌN ĐIỂM TIÊM LỖI ---
            if fixed_inject_idx is not None:
                local_inject = min(fixed_inject_idx, day_len - 1)
                local_inject = max(1, local_inject)
            else:
                max_rnd = min(40, day_len - 2)
                if max_rnd < 1: max_rnd = 1
                local_inject = np.random.randint(1, max_rnd + 1)
            
            global_inject_idx = start_idx + local_inject
            # ---------------------------

            # 1. CHECK FALSE POSITIVE (Trên Clean Run)
            region_mask = valid_check_points[start_idx : global_inject_idx]
            region_vals = global_ma_clean[start_idx : global_inject_idx]
            check_vals = region_vals[region_mask]
            
            if len(check_vals) > 0:
                alarms = (check_vals < lcl) | (check_vals > ucl)
                if np.any(alarms):
                    false_positive_days += 1
                    continue 

            # 2. CHECK DETECTION (Tính lại MA Biased)
            temp_global_vals = self.global_vals.copy()
            # Tiêm lỗi từ điểm bắt đầu đến hết ngày
            temp_global_vals[global_inject_idx : end_idx] *= bias_factor
            
            global_ma_biased = self.calculate_ma(temp_global_vals, method, param)
            
            region_mask_post = valid_check_points[global_inject_idx : end_idx]
            region_vals_post = global_ma_biased[global_inject_idx : end_idx]
            check_vals_post = region_vals_post[region_mask_post]
            
            if len(check_vals_post) > 0:
                alarms_post = (check_vals_post < lcl) | (check_vals_post > ucl)
                if np.any(alarms_post):
                    detected_days += 1
                    indices_in_region = np.arange(global_inject_idx, end_idx)
                    # Tìm điểm alarm thật sự
                    full_post_region = global_ma_biased[global_inject_idx:end_idx]
                    is_alarm = (full_post_region < lcl) | (full_post_region > ucl)
                    # Filter với frequency
                    valid_alarm_mask = is_alarm & valid_check_points[global_inject_idx:end_idx]
                    
                    if np.any(valid_alarm_mask):
                        # Lấy index đầu tiên thỏa mãn cả 2 điều kiện
                        first_valid_alarm_rel_idx = np.argmax(valid_alarm_mask)
                        nped = first_valid_alarm_rel_idx + 1 # +1 do tính từ điểm inject
                        nped_list.append(nped)

        metrics = {
            "Total Days": total_days,
            "Detected (%)": round(detected_days / total_days * 100, 1) if total_days > 0 else 0,
            "False Positive (%)": round(false_positive_days / total_days * 100, 1) if total_days > 0 else 0,
            "ANPed": round(np.mean(nped_list), 1) if nped_list else "N/A",
            "Median NPed": round(np.median(nped_list), 1) if nped_list else "N/A",
            "95th NPed": round(np.percentile(nped_list, 95), 1) if nped_list else "N/A"
        }
        
        # Trả về thêm data clean để xuất Excel
        export_data = pd.DataFrame({
            'Day': self.global_days,
            'Result': self.global_vals,
            f'{method}_Clean': global_ma_clean,
            'LCL': lcl,
            'UCL': ucl,
            'Is_False_Alarm': ((global_ma_clean < lcl) | (global_ma_clean > ucl))
        })
        
        return metrics, export_data

# =========================================================
# 🖥️ PHẦN 3: GIAO DIỆN STREAMLIT
# =========================================================

st.set_page_config(layout="wide", page_title="PBRTQC Simulator Pro")

st.title("🏥 PBRTQC Continuous Simulator")
st.markdown("""
Hệ thống mô phỏng PBRTQC (Continuous Logic).
- **EWMA:** Ẩn Frequency (mặc định check từng điểm).
- **Export:** Tải xuống file Excel chi tiết các giá trị MA và Limit sau khi chạy.
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

    # --- INPUT BLOCK SIZE (Ẩn Frequency nếu là EWMA) ---
    st.divider()
    st.subheader(f"4. Cấu hình tham số cho {model}")
    
    col_case1, col_case2, col_case3 = st.columns(3)
    cases_config = []
    
    def create_case_input(col, idx):
        with col:
            st.markdown(f"**Case {idx}**")
            bs = st.number_input(f"Block Size (N)", value=20*idx, key=f"bs{idx}", min_value=2)
            freq = 1
            # CHỈ HIỆN FREQUENCY NẾU LÀ SMA
            if model == "SMA":
                freq = st.number_input("Frequency", value=1, key=f"freq{idx}", min_value=1)
            return {'bs': bs, 'freq': freq}

    cases_config.append(create_case_input(col_case1, 1))
    cases_config.append(create_case_input(col_case2, 2))
    cases_config.append(create_case_input(col_case3, 3))

    if st.button("🚀 Run Simulation"):
        with st.spinner("Đang xử lý dữ liệu..."):
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
                # Dictionary để lưu data xuất Excel
                excel_sheets = {} 
                
                prog_bar = st.progress(0)
                
                for i, case in enumerate(cases_config):
                    lcl, ucl = engine.determine_limits(model, case['bs'], target_fpr)
                    
                    metrics, export_df = engine.run_continuous_simulation(
                        model, case['bs'], lcl, ucl, bias_pct,
                        frequency=case['freq'],
                        num_sims=max_days, 
                        fixed_inject_idx=fixed_point
                    )
                    
                    res_row = {
                        "Case": f"N={case['bs']}",
                        "LCL": round(lcl, 2), "UCL": round(ucl, 2),
                        **metrics
                    }
                    results.append(res_row)
                    
                    # Lưu dataframe vào dict để lát xuất excel
                    excel_sheets[f"Case_N{case['bs']}"] = export_df
                    
                    prog_bar.progress((i+1)/len(cases_config))
                
                st.subheader("📊 Bảng Kết quả Đánh giá")
                st.dataframe(pd.DataFrame(results).style.highlight_max(subset=['Detected (%)'], color='#d1ffbd'), use_container_width=True)
                
                # --- NÚT DOWNLOAD EXCEL ---
                st.divider()
                st.subheader("📥 Xuất dữ liệu")
                
                # Tạo file Excel trong bộ nhớ đệm
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for sheet_name, df in excel_sheets.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                st.download_button(
                    label="Tải xuống chi tiết kết quả (.xlsx)",
                    data=output.getvalue(),
                    file_name="PBRTQC_Simulation_Results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            else:
                st.error("Không đọc được dữ liệu.")
