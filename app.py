import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

# --- CLASS 1: XỬ LÝ DỮ LIỆU (GIỮ NGUYÊN) ---
class DataNormalizer:
    def __init__(self, data):
        self.original_data = np.array(data)
        self.original_data = self.original_data[~np.isnan(self.original_data)]

    def optimize_asymmetric(self, max_cut_percent=0.10, steps=10):
        best_p = -1
        best_range = (self.original_data.min(), self.original_data.max())
        
        cuts = np.linspace(0, max_cut_percent, steps)
        sorted_data = np.sort(self.original_data)
        n = len(sorted_data)
        
        for left_cut in cuts:
            for right_cut in cuts:
                if left_cut + right_cut >= 0.5: continue
                s = int(n * left_cut)
                e = int(n * (1 - right_cut))
                subset = sorted_data[s:e]
                
                if len(subset) > 20:
                    stat, p_val = stats.shapiro(subset) if len(subset) < 5000 else stats.normaltest(subset)
                    if p_val > best_p:
                        best_p = p_val
                        best_range = (subset.min(), subset.max())
        return best_range

# --- CLASS 2: PBRTQC ENGINE (CẬP NHẬT BIAS %) ---
class PBRTQCEngine:
    def __init__(self, train_data, verify_data, trunc_range):
        self.raw_train = np.array(train_data)
        self.raw_verify = np.array(verify_data)
        self.trunc_min, self.trunc_max = trunc_range
        
        self.train = self._apply_truncation(self.raw_train)
        self.verify = self._apply_truncation(self.raw_verify)

    def _apply_truncation(self, data):
        return data[(data >= self.trunc_min) & (data <= self.trunc_max)]

    def calculate_moving_metric(self, data, method, param):
        series = pd.Series(data)
        if method == 'SMA':
            return series.rolling(window=int(param)).mean().fillna(method='bfill').values
        elif method == 'EWMA':
            lam = 2 / (int(param) + 1)
            return series.ewm(alpha=lam, adjust=False).mean().values
        return series.values

    def determine_control_limits(self, method, param, target_fpr):
        ma_values = self.calculate_moving_metric(self.train, method, param)
        lower_percentile = (target_fpr / 2) * 100
        upper_percentile = 100 - (target_fpr / 2) * 100
        lcl = np.percentile(ma_values, lower_percentile)
        ucl = np.percentile(ma_values, upper_percentile)
        return lcl, ucl

    def run_simulation(self, method, param, lcl, ucl, bias_pct, frequency=1, num_sims=50):
        """
        bias_pct: Giá trị bias tính theo % (Ví dụ: 5.0 tương ứng 5%)
        """
        verify_data = self.verify
        n = len(verify_data)
        if n < 100: return {} 

        # 1. Tính FPR thực tế
        ma_clean = self.calculate_moving_metric(verify_data, method, param)
        alarms = 0
        checks = 0
        for i in range(n):
            if i % frequency == 0: 
                if ma_clean[i] < lcl or ma_clean[i] > ucl:
                    alarms += 1
                checks += 1
        real_fpr = alarms / checks if checks > 0 else 0

        # 2. Tính khả năng phát hiện lỗi (Simulation)
        detected_counts = []
        
        # Hệ số nhân để tạo Bias %
        # Ví dụ Bias = 10% -> bias_factor = 1.10
        bias_factor = 1 + (bias_pct / 100.0)

        for _ in range(num_sims):
            start_idx = np.random.randint(20, max(21, n - 50))
            
            # [THAY ĐỔI QUAN TRỌNG] Nhân theo tỷ lệ thay vì cộng
            sim_data = verify_data.copy()
            sim_data[start_idx:] = sim_data[start_idx:] * bias_factor
            
            ma_sim = self.calculate_moving_metric(sim_data, method, param)
            
            found = False
            for i in range(start_idx, n):
                if (i - start_idx) % frequency == 0: 
                    if ma_sim[i] < lcl or ma_sim[i] > ucl:
                        detected_counts.append(i - start_idx + 1)
                        found = True
                        break
            
            if not found:
                pass 

        if len(detected_counts) > 0:
            ped = len(detected_counts) / num_sims * 100
            anped = np.mean(detected_counts)
            mnped = np.median(detected_counts)
            nped95 = np.percentile(detected_counts, 95)
        else:
            ped = 0
            anped = mnped = nped95 = None

        return {
            "Real_FPR (%)": round(real_fpr * 100, 2),
            "Detection (%)": round(ped, 1),
            "ANPed": round(anped, 1) if anped else "N/A",
            "MNPed": round(mnped, 1) if mnped else "N/A",
            "95NPed": round(nped95, 1) if nped95 else "N/A"
        }

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(layout="wide", page_title="PBRTQC Simulator")

st.title("🏥 PBRTQC Analyzer: Real-Time Quality Control")
st.markdown("Hệ thống mô phỏng và tối ưu hóa PBRTQC (Bias tính theo %).")

# 1. Sidebar
with st.sidebar:
    st.header("1. Upload Dữ liệu")
    f_train = st.file_uploader("Dữ liệu Training (Xlsx)", type='xlsx')
    f_verify = st.file_uploader("Dữ liệu Verify (Xlsx)", type='xlsx')
    
    st.divider()
    st.header("2. Thông số Chung")
    
    # [CẬP NHẬT] Input Bias %
    bias_pct = st.number_input("Giá trị Bias (%) cần phát hiện", value=5.0, step=0.5, help="Nhập % sai số (Ví dụ 5.0 là 5%)")
    
    target_fpr = st.slider("Target False Positive Rate (%)", 0.1, 10.0, 2.0, 0.1) / 100
    model_type = st.selectbox("Chọn Mô hình", ["EWMA", "SMA"])

# 2. Xử lý chính
if f_train and f_verify:
    try:
        df_train = pd.read_excel(f_train)
        df_verify = pd.read_excel(f_verify)
        
        cols = df_train.columns
        col_res = st.selectbox("Chọn cột Kết quả (Results):", cols)
        
        data_train = df_train[col_res].values
        data_verify = df_verify[col_res].values

        with st.spinner("Đang tính toán Truncation Limit trên Training Data..."):
            normalizer = DataNormalizer(data_train)
            trunc_range = normalizer.optimize_asymmetric()
        
        st.success(f"✅ Đã xác định Truncation Range: {trunc_range[0]:.2f} đến {trunc_range[1]:.2f}")
        
        engine = PBRTQCEngine(data_train, data_verify, trunc_range)

        st.header(f"3. Cấu hình Tham số cho {model_type}")
        input_cols = st.columns(3)
        cases = []
        for i, col in enumerate(input_cols):
            with col:
                st.subheader(f"Case {i+1}")
                bs = st.number_input(f"Block Size (N) - Case {i+1}", min_value=2, value=20*(i+1))
                freq = 1
                if model_type == "SMA":
                    freq = st.number_input(f"Frequency - Case {i+1}", min_value=1, value=1)
                cases.append({'bs': bs, 'freq': freq})

        if st.button("🚀 Chạy Đánh giá (Evaluate)"):
            st.divider()
            results_table = []
            progress_bar = st.progress(0)
            
            for idx, case in enumerate(cases):
                lcl, ucl = engine.determine_control_limits(model_type, case['bs'], target_fpr)
                
                # Truyền bias_pct vào hàm simulation
                metrics = engine.run_simulation(
                    model_type, 
                    case['bs'], 
                    lcl, ucl, 
                    bias_pct,  # Đây là giá trị %
                    frequency=case['freq']
                )
                
                row = {
                    "Case": f"Case {idx+1}",
                    "Block Size": case['bs'],
                    "Frequency": case['freq'] if model_type == "SMA" else "N/A",
                    "Lower Limit": round(lcl, 3),
                    "Upper Limit": round(ucl, 3),
                    **metrics
                }
                results_table.append(row)
                progress_bar.progress((idx + 1) / 3)

            st.subheader(f"📊 Kết quả với mức Bias {bias_pct}%")
            res_df = pd.DataFrame(results_table)
            st.dataframe(res_df.style.highlight_max(axis=0, subset=['Detection (%)'], color='lightgreen'), use_container_width=True)
            
            st.markdown("""
            **Giải thích thuật ngữ:**
            * **Real_FPR:** Tỷ lệ báo động giả khi hệ thống bình thường.
            * **Detection (%):** Khả năng bắt được lỗi bias này.
            * **ANPed:** Số mẫu trung bình trôi qua trước khi bắt được lỗi.
            """)

    except Exception as e:
        st.error(f"Có lỗi xảy ra: {e}")
else:
    st.info("Vui lòng upload cả 2 file Training và Verify để bắt đầu.")