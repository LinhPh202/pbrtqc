import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

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
        
        # Sắp xếp Verify theo thứ tự xuất hiện (quan trọng cho tính liên tục)
        # Giả định file excel đã sắp xếp theo thời gian, nếu không thì cần sort
        # df_verify = df_verify.sort_values(by=col_day) 
        
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
        
        # 1. Training Data (Để tính Limit)
        raw_train = df_train[col_res].values
        self.train_clean = raw_train[(raw_train >= self.trunc_min) & (raw_train <= self.trunc_max)]
        
        # 2. Verify Data (Xử lý chuỗi liên tục)
        # Lọc bỏ ngoại lai nhưng GIỮ NGUYÊN THỨ TỰ để đảm bảo tính liên tục của thời gian
        # Lưu ý: Nếu lọc bỏ dòng thì index sẽ bị nhảy, tuy nhiên SMA/EWMA sẽ tính trên các điểm dữ liệu còn lại liền kề nhau.
        self.df_verify_clean = df_verify[
            (df_verify[col_res] >= self.trunc_min) & 
            (df_verify[col_res] <= self.trunc_max)
        ].copy()
        
        # Tạo mảng dữ liệu toàn cục (Global Array)
        self.global_vals = self.df_verify_clean[col_res].values.astype(float)
        self.global_days = self.df_verify_clean[col_day].values

        # Tạo bản đồ index cho từng ngày: { "Day1": [start_idx, end_idx], ... }
        # Giúp truy xuất nhanh vị trí của từng ngày trong chuỗi toàn cục
        self.day_indices = {}
        unique_days = self.df_verify_clean[col_day].unique()
        
        # Vì dữ liệu đã sort hoặc liền mạch, ta tìm index start/end của từng ngày
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
        """
        Mô phỏng với logic: MA tính xuyên suốt, Frequency tính trên Index toàn cục.
        """
        total_days = 0
        detected_days = 0
        false_positive_days = 0
        nped_list = []
        
        bias_factor = 1 + (bias_pct / 100.0)
        
        # 1. Tính MA Sạch toàn cục (Global Clean MA)
        # Tính 1 lần dùng chung cho việc check False Positive
        global_ma_clean = self.calculate_ma(self.global_vals, method, param)
        
        # Mảng Index toàn cục để check Frequency
        # Chỉ những index nào chia hết cho Frequency mới được coi là điểm kiểm tra hợp lệ
        global_indices = np.arange(len(self.global_vals))
        valid_check_points = (global_indices % frequency == 0)

        # Lấy danh sách ngày cần chạy
        days_to_run = list(self.day_indices.keys())
        if num_sims and num_sims < len(days_to_run):
            days_to_run = days_to_run[:num_sims]

        for day_name in days_to_run:
            start_idx, end_idx = self.day_indices[day_name]
            day_len = end_idx - start_idx
            
            if day_len < 5: continue
            total_days += 1
            
            # --- XÁC ĐỊNH ĐIỂM TIÊM LỖI (LOCAL INDEX -> GLOBAL INDEX) ---
            if fixed_inject_idx is not None:
                local_inject = min(fixed_inject_idx, day_len - 1)
                local_inject = max(1, local_inject)
            else:
                max_rnd = min(40, day_len - 2)
                if max_rnd < 1: max_rnd = 1
                local_inject = np.random.randint(1, max_rnd + 1)
            
            # Chuyển đổi sang Index toàn cục
            global_inject_idx = start_idx + local_inject
            # ------------------------------------------------------------

            # 2. CHECK FALSE POSITIVE (Trên đường Global Clean)
            # Kiểm tra trong khoảng [start_idx, global_inject_idx)
            # VÀ phải thỏa mãn điều kiện Frequency
            
            # Cắt vùng cần check
            region_mask = valid_check_points[start_idx : global_inject_idx]
            region_vals = global_ma_clean[start_idx : global_inject_idx]
            
            # Lọc những điểm đúng Frequency
            check_vals = region_vals[region_mask]
            
            if len(check_vals) > 0:
                alarms = (check_vals < lcl) | (check_vals > ucl)
                if np.any(alarms):
                    false_positive_days += 1
                    continue # Ngày này coi như fail do báo giả, sang ngày tiếp

            # 3. CHECK DETECTION (Cần tính lại MA)
            # Tạo bản sao dữ liệu toàn cục và tiêm lỗi
            # Lưu ý: Ta chỉ cần tiêm lỗi từ global_inject_idx đến hết ngày đó (end_idx).
            # Vì logic là "Ngày hôm sau reset lỗi", nên ta không cần tiêm lỗi cho các ngày sau đó.
            # Tuy nhiên, MA cần được tính lại để phản ánh sự thay đổi.
            
            # Tối ưu: Để tính MA chính xác tại thời điểm global_inject_idx, ta cần lịch sử trước đó.
            # Cách an toàn nhất: Copy toàn bộ mảng, sửa đoạn bị lỗi, tính lại MA.
            
            temp_global_vals = self.global_vals.copy()
            # Tiêm lỗi từ điểm bắt đầu đến hết ngày hôm đó
            temp_global_vals[global_inject_idx : end_idx] *= bias_factor
            
            # Tính lại MA (Biased)
            global_ma_biased = self.calculate_ma(temp_global_vals, method, param)
            
            # Kiểm tra vùng [global_inject_idx, end_idx)
            # VÀ thỏa mãn Frequency
            region_mask_post = valid_check_points[global_inject_idx : end_idx]
            region_vals_post = global_ma_biased[global_inject_idx : end_idx]
            
            check_vals_post = region_vals_post[region_mask_post]
            
            if len(check_vals_post) > 0:
                alarms_post = (check_vals_post < lcl) | (check_vals_post > ucl)
                if np.any(alarms_post):
                    detected_days += 1
                    
                    # Tìm vị trí Alarm đầu tiên trong mảng đã filter (check_vals_post)
                    # Tuy nhiên để tính NPed chính xác, ta cần biết index thực
                    
                    # Lấy index thực trong vùng cắt
                    indices_in_region = np.arange(global_inject_idx, end_idx)
                    # Lọc index theo frequency và alarm
                    alarm_indices = indices_in_region[valid_check_points[global_inject_idx : end_idx] & ((global_ma_biased[global_inject_idx:end_idx] < lcl) | (global_ma_biased[global_inject_idx:end_idx] > ucl))]
                    
                    if len(alarm_indices) > 0:
                        first_alarm_idx = alarm_indices[0]
                        # NPed = Số mẫu bệnh nhân trôi qua kể từ lúc tiêm lỗi
                        nped = first_alarm_idx - global_inject_idx + 1
                        nped_list.append(nped)

        metrics = {
            "Total Days": total_days,
            "Detected (%)": round(detected_days / total_days * 100, 1) if total_days > 0 else 0,
            "False Positive (%)": round(false_positive_days / total_days * 100, 1) if total_days > 0 else 0,
            "ANPed": round(np.mean(nped_list), 1) if nped_list else "N/A",
            "Median NPed": round(np.median(nped_list), 1) if nped_list else "N/A",
            "95th NPed": round(np.percentile(nped_list, 95), 1) if nped_list else "N/A"
        }
        return metrics

# =========================================================
# 🖥️ PHẦN 3: GIAO DIỆN STREAMLIT
# =========================================================

st.set_page_config(layout="wide", page_title="PBRTQC Simulator Pro")

st.title("🏥 PBRTQC Continuous Simulator")
st.markdown("""
Hệ thống mô phỏng PBRTQC với logic **Continuous Monitoring**:
- **MA Calculation:** Tính xuyên suốt qua các ngày (Ngày 2 kế thừa dữ liệu Ngày 1).
- **Frequency:** Tính dựa trên Index toàn cục (Ví dụ Freq=5: check tại mẫu 5, 10, 15... bất kể ngày).
- **Simulation:** Reset trạng thái lỗi khi qua ngày mới.
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
    max_days = st.slider("Max Simulation Days", 10, 5000, 100, help="Số lượng ngày tối đa muốn chạy mô phỏng.")
    
    st.subheader("Injection Mode")
    inject_mode = st.radio("Chế độ thêm lỗi:", ["Ngẫu nhiên (Random 1-40)", "Cố định (Fixed Point)"])
    fixed_point = None
    if inject_mode == "Cố định (Fixed Point)":
        fixed_point = st.number_input("Vị trí mẫu bắt đầu lỗi (trong ngày):", min_value=1, value=20)

    # --- TRUNCATION SETTINGS ---
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

    # --- INPUT BLOCK SIZE ---
    st.divider()
    st.subheader(f"4. Cấu hình tham số (Block Size) cho {model}")
    
    col_case1, col_case2, col_case3 = st.columns(3)
    cases_config = []
    
    def create_case_input(col, idx):
        with col:
            st.markdown(f"**Case {idx}**")
            bs = st.number_input(f"Block Size (N)", value=20*idx, key=f"bs{idx}", min_value=2)
            freq = 1
            if model == "SMA":
                freq = st.number_input("Frequency", value=1, key=f"freq{idx}", min_value=1)
            # Nếu là EWMA, Frequency vẫn có thể áp dụng cho việc Check Alarm (Sampling)
            if model == "EWMA":
                 freq = st.number_input("Frequency (Check Interval)", value=1, key=f"freq_ewma{idx}", min_value=1)
            return {'bs': bs, 'freq': freq}

    cases_config.append(create_case_input(col_case1, 1))
    cases_config.append(create_case_input(col_case2, 2))
    cases_config.append(create_case_input(col_case3, 3))

    if st.button("🚀 Run Simulation"):
        with st.spinner("Đang xử lý dữ liệu và chạy mô phỏng (Có thể mất chút thời gian)..."):
            df_train, df_verify = load_data(f_train, f_verify, col_res, col_day)
            
            if df_train is not None:
                # --- XỬ LÝ TRUNCATION ---
                trunc_range = (0, 0)
                data_train_vals = df_train[col_res].dropna().values
                
                if trunc_mode == "Auto (Tự động)":
                    trunc_range = find_optimal_truncation(data_train_vals)
                    st.success(f"✅ Auto Truncation: [{trunc_range[0]:.2f} - {trunc_range[1]:.2f}]")
                else:
                    trunc_range = (manual_min, manual_max)
                    st.info(f"🔧 Manual Truncation: [{trunc_range[0]:.2f} - {trunc_range[1]:.2f}]")
                
                # Khởi tạo Engine
                engine = PBRTQCEngine(df_train, df_verify, col_res, col_day, trunc_range)
                
                results = []
                prog_bar = st.progress(0)
                
                for i, case in enumerate(cases_config):
                    lcl, ucl = engine.determine_limits(model, case['bs'], target_fpr)
                    
                    metrics = engine.run_continuous_simulation(
                        model, case['bs'], lcl, ucl, bias_pct,
                        frequency=case['freq'],
                        num_sims=max_days, 
                        fixed_inject_idx=fixed_point
                    )
                    
                    res_row = {
                        "Case": f"N={case['bs']}",
                        "Frequency": case['freq'],
                        "LCL": round(lcl, 2), "UCL": round(ucl, 2),
                        **metrics
                    }
                    results.append(res_row)
                    prog_bar.progress((i+1)/len(cases_config))
                
                st.subheader("📊 Bảng Kết quả Đánh giá")
                st.dataframe(pd.DataFrame(results).style.highlight_max(subset=['Detected (%)'], color='#d1ffbd'), use_container_width=True)
                
            else:
                st.error("Không đọc được dữ liệu Training/Verify.")
