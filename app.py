import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# =========================================================
# 🚀 PHẦN 1: OPTIMIZED CACHING & DATA PROCESSING
# =========================================================

# Dùng decorator này để Streamlit nhớ kết quả, không phải tính lại mỗi lần
@st.cache_data(show_spinner=False)
def load_and_clean_data(file_train, file_verify, col_name):
    """Đọc file và tiền xử lý data (Cache lại)"""
    try:
        df_train = pd.read_excel(file_train)
        df_verify = pd.read_excel(file_verify)
        
        # Lấy dữ liệu dạng mảng numpy ngay lập tức để nhanh hơn
        data_train = df_train[col_name].dropna().values
        data_verify = df_verify[col_name].dropna().values
        
        return data_train, data_verify
    except Exception as e:
        return None, None

@st.cache_data(show_spinner=False)
def find_optimal_truncation(data, max_cut_percent=0.10, steps=10):
    """Tìm khoảng cắt tối ưu (Đã tối ưu hóa tốc độ)"""
    # Chỉ lấy mẫu tối đa 5000 điểm để tính Shapiro cho nhanh nếu data quá lớn
    # Data gốc vẫn giữ nguyên, chỉ dùng sample để tìm ngưỡng
    calc_data = data
    if len(data) > 5000:
        np.random.seed(42)
        calc_data = np.random.choice(data, 5000, replace=False)
        
    best_p = -1
    best_range = (data.min(), data.max())
    
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
                # Dùng normaltest nhanh hơn shapiro với data lớn
                stat, p_val = stats.normaltest(subset)
                if p_val > best_p:
                    best_p = p_val
                    # Map lại percentile vào data gốc
                    lower = np.percentile(data, left_cut * 100)
                    upper = np.percentile(data, (1 - right_cut) * 100)
                    best_range = (lower, upper)
    return best_range

# =========================================================
# 🚀 PHẦN 2: HIGH-PERFORMANCE ENGINE (VECTORIZED)
# =========================================================

class PBRTQCEngine:
    def __init__(self, train_data, verify_data, trunc_range):
        # Lưu trữ dưới dạng numpy array float64 để tính toán nhanh nhất
        self.raw_train = np.array(train_data, dtype=np.float64)
        self.raw_verify = np.array(verify_data, dtype=np.float64)
        self.trunc_min, self.trunc_max = trunc_range
        
        # Cắt gọt dữ liệu (Vectorized filtering)
        self.train = self.raw_train[(self.raw_train >= self.trunc_min) & (self.raw_train <= self.trunc_max)]
        self.verify = self.raw_verify[(self.raw_verify >= self.trunc_min) & (self.raw_verify <= self.trunc_max)]

    def calculate_moving_metric(self, data, method, param):
        """Tính toán MA bằng Pandas (Đã tối ưu C-backend)"""
        # Chuyển đổi nhanh sang Series để dùng hàm có sẵn
        series = pd.Series(data)
        if method == 'SMA':
            # fillna(method='bfill') để tránh lỗi NaN ở đầu
            return series.rolling(window=int(param)).mean().bfill().values
        elif method == 'EWMA':
            lam = 2 / (int(param) + 1)
            return series.ewm(alpha=lam, adjust=False).mean().values
        return data

    def determine_control_limits(self, method, param, target_fpr):
        ma_values = self.calculate_moving_metric(self.train, method, param)
        lower_percentile = (target_fpr / 2) * 100
        upper_percentile = 100 - (target_fpr / 2) * 100
        
        # np.percentile rất nhanh
        lcl = np.percentile(ma_values, lower_percentile)
        ucl = np.percentile(ma_values, upper_percentile)
        return lcl, ucl

    def run_simulation_vectorized(self, method, param, lcl, ucl, bias_pct, frequency=1, num_sims=50):
        """
        Phiên bản siêu tốc độ: Sử dụng NumPy Vectorization thay vì vòng lặp for
        """
        verify_data = self.verify
        n = len(verify_data)
        if n < 100: return {}, None

        # 1. Tính Real FPR (Vectorized)
        ma_clean = self.calculate_moving_metric(verify_data, method, param)
        
        # Tạo mảng chỉ số để check frequency
        indices = np.arange(n)
        freq_mask = (indices % frequency == 0) # Chỉ lấy các điểm đúng frequency
        
        # Tìm các điểm vi phạm
        violations = (ma_clean < lcl) | (ma_clean > ucl)
        
        # Kết hợp điều kiện: Vi phạm VÀ đúng frequency
        valid_alarms = violations & freq_mask
        
        alarms_count = np.sum(valid_alarms)
        checks_count = np.sum(freq_mask)
        real_fpr = alarms_count / checks_count if checks_count > 0 else 0

        # 2. Simulation (Vectorized Search)
        detected_counts = []
        bias_factor = 1 + (bias_pct / 100.0)
        
        last_run_data = {}

        # Pre-calculate random start indices (Vectorized random)
        # Giới hạn điểm bắt đầu để đảm bảo còn ít nhất 50 mẫu phía sau
        start_indices = np.random.randint(20, max(21, n - 50), size=num_sims)

        for i, start_idx in enumerate(start_indices):
            # Tạo data mô phỏng
            # Copy mảng tốn ít thời gian hơn là tính toán lại từ đầu
            sim_data = verify_data.copy()
            sim_data[start_idx:] *= bias_factor # Phép nhân tại chỗ (in-place) nhanh hơn
            
            # Tính lại MA cho toàn bộ chuỗi (Pandas C-optimized rất nhanh, 40k dòng chỉ mất ~2ms)
            ma_sim = self.calculate_moving_metric(sim_data, method, param)
            
            # --- ĐOẠN NÀY LÀ QUAN TRỌNG NHẤT (TỐI ƯU HÓA) ---
            # Thay vì for loop từng phần tử, ta dùng mask
            
            # Chỉ xét vùng dữ liệu từ start_idx trở đi
            region_of_interest = ma_sim[start_idx:]
            
            # 1. Tìm điểm vượt ngưỡng trong vùng này
            violation_mask = (region_of_interest < lcl) | (region_of_interest > ucl)
            
            # 2. Tìm điểm đúng Frequency trong vùng này
            # Cần tính lại index toàn cục cho vùng này
            global_indices_region = np.arange(start_idx, n)
            freq_mask_region = (global_indices_region % frequency == 0)
            
            # 3. Kết hợp điều kiện
            combined_mask = violation_mask & freq_mask_region
            
            # 4. Tìm vị trí True đầu tiên (Argmax trả về index đầu tiên của giá trị Max/True)
            if np.any(combined_mask):
                # np.argmax trả về index tương đối trong region
                relative_first_idx = np.argmax(combined_mask) 
                
                # Số bệnh nhân trôi qua = index tương đối + 1
                detected_counts.append(relative_first_idx + 1)
                
                # Lưu data lần cuối để vẽ
                if i == num_sims - 1:
                    global_alarm_idx = start_idx + relative_first_idx
                    last_run_data = {
                        'ma_clean': ma_clean,
                        'ma_sim': ma_sim,
                        'start_idx': start_idx,
                        'alarm_idx': global_alarm_idx,
                        'lcl': lcl, 'ucl': ucl
                    }
            else:
                 # Nếu không tìm thấy, vẫn lưu data để debug (không có alarm_idx)
                 if i == num_sims - 1:
                    last_run_data = {
                        'ma_clean': ma_clean,
                        'ma_sim': ma_sim,
                        'start_idx': start_idx,
                        'alarm_idx': None,
                        'lcl': lcl, 'ucl': ucl
                    }

        # Tổng hợp chỉ số
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
        }, last_run_data

# =========================================================
# 🚀 PHẦN 3: GIAO DIỆN STREAMLIT
# =========================================================

st.set_page_config(layout="wide", page_title="PBRTQC High-Performance")

st.title("⚡ PBRTQC Analyzer (High Performance Mode)")
st.markdown("Hệ thống tối ưu hóa cho dữ liệu lớn (100k+ dòng).")

with st.sidebar:
    st.header("1. Upload & Cấu hình")
    f_train = st.file_uploader("Dữ liệu Training", type='xlsx')
    f_verify = st.file_uploader("Dữ liệu Verify", type='xlsx')
    
    st.divider()
    bias_pct = st.number_input("Bias (%)", value=5.0, step=0.5)
    target_fpr = st.slider("Target FPR (%)", 0.1, 10.0, 2.0, 0.1) / 100
    model_type = st.selectbox("Mô hình", ["EWMA", "SMA"])
    
    # Thêm tùy chọn giảm số lần mô phỏng nếu máy yếu
    num_sims = st.slider("Số lần mô phỏng (Simulations)", 10, 100, 50, 10, help="Giảm xuống nếu thấy chạy chậm")

if f_train and f_verify:
    # Đọc tên cột trước (Để không cache sai cột)
    # Phần này đọc nhanh header thôi
    df_preview = pd.read_excel(f_train, nrows=5)
    col_res = st.selectbox("Chọn cột Kết quả:", df_preview.columns)
    
    # 1. LOAD DATA VỚI CACHE
    with st.spinner("Đang tải và xử lý dữ liệu lớn..."):
        data_train, data_verify = load_and_clean_data(f_train, f_verify, col_res)
        
    if data_train is not None:
        st.info(f"Đã tải: Training ({len(data_train):,} dòng) - Verify ({len(data_verify):,} dòng)")

        # 2. TÍNH TRUNCATION VỚI CACHE
        trunc_range = find_optimal_truncation(data_train)
        st.success(f"Truncation Range tối ưu: [{trunc_range[0]:.2f} - {trunc_range[1]:.2f}]")
        
        # 3. KHỞI TẠO ENGINE
        engine = PBRTQCEngine(data_train, data_verify, trunc_range)

        # 4. CẤU HÌNH CASES
        st.header("Cấu hình Cases")
        cols = st.columns(3)
        cases = []
        for i, col in enumerate(cols):
            with col:
                bs = st.number_input(f"Block Size Case {i+1}", value=20*(i+1))
                freq = 1
                if model_type == "SMA":
                    freq = st.number_input(f"Freq Case {i+1}", value=1, min_value=1)
                cases.append({'bs': bs, 'freq': freq})

        # 5. CHẠY SIMULATION
        if st.button("🚀 CHẠY ĐÁNH GIÁ NGAY"):
            st.divider()
            results_table = []
            plot_data_list = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, case in enumerate(cases):
                status_text.text(f"Đang chạy Case {idx+1}/{len(cases)} với {len(data_verify):,} dòng dữ liệu...")
                
                # a. Tính Limit
                lcl, ucl = engine.determine_control_limits(model_type, case['bs'], target_fpr)
                
                # b. Chạy Sim (Dùng hàm Vectorized mới)
                metrics, plot_data = engine.run_simulation_vectorized(
                    model_type, case['bs'], lcl, ucl, bias_pct, 
                    frequency=case['freq'], num_sims=num_sims
                )
                
                row = {
                    "Case": f"Case {idx+1}",
                    "N": case['bs'],
                    "LCL": round(lcl, 2), "UCL": round(ucl, 2),
                    **metrics
                }
                results_table.append(row)
                plot_data_list.append({'name': f"Case {idx+1}", 'data': plot_data})
                
                progress_bar.progress((idx + 1) / len(cases))
            
            status_text.text("Hoàn tất!")
            
            # HIỂN THỊ KẾT QUẢ
            st.subheader("📊 Kết quả")
            st.dataframe(pd.DataFrame(results_table).style.highlight_max(subset=['Detection (%)'], color='#d1ffbd'), use_container_width=True)
            
            # VẼ BIỂU ĐỒ
            st.divider()
            st.subheader("📈 Biểu đồ minh họa")
            tabs = st.tabs([p['name'] for p in plot_data_list])
            
            for i, tab in enumerate(tabs):
                with tab:
                    d = plot_data_list[i]['data']
                    if d:
                        fig, ax = plt.subplots(figsize=(12, 4))
                        # Vẽ sample khoảng 1000 điểm quanh điểm lỗi để đỡ lag khi vẽ
                        center = d['start_idx']
                        # Vẽ rộng ra 200 điểm trước và 500 điểm sau lỗi
                        s_plot = max(0, center - 200)
                        e_plot = min(len(d['ma_clean']), center + 500)
                        
                        x_axis = range(s_plot, e_plot)
                        
                        ax.plot(x_axis, d['ma_clean'][s_plot:e_plot], color='green', alpha=0.3, label='Sạch')
                        ax.plot(x_axis, d['ma_sim'][s_plot:e_plot], color='orange', label='Lỗi')
                        ax.axhline(d['ucl'], color='red', ls='--'); ax.axhline(d['lcl'], color='red', ls='--')
                        ax.axvline(d['start_idx'], color='black', ls=':', label='Bắt đầu lỗi')
                        
                        if d['alarm_idx'] and s_plot <= d['alarm_idx'] <= e_plot:
                            ax.scatter(d['alarm_idx'], d['ma_sim'][d['alarm_idx']], color='red', s=100, marker='*', zorder=5)
                        
                        ax.legend()
                        st.pyplot(fig)

    else:
        st.warning("Vui lòng tải file lên.")
