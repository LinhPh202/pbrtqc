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
    """Đọc file và lấy cột Days + Results"""
    try:
        df_train = pd.read_excel(file_train)
        df_verify = pd.read_excel(file_verify)
        
        # Lọc bỏ NaN
        df_train = df_train.dropna(subset=[col_res])
        df_verify = df_verify.dropna(subset=[col_res, col_day])
        
        return df_train, df_verify
    except Exception as e:
        return None, None

@st.cache_data(show_spinner=False)
def find_optimal_truncation(data_array, max_cut_percent=0.10, steps=10):
    """Tìm khoảng cắt tối ưu trên dữ liệu Training (1 chiều)"""
    # Lấy mẫu nếu data quá lớn để tăng tốc
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
        
        # 1. Xử lý Training Data (Để tính Limit)
        raw_train = df_train[col_res].values
        self.train_clean = raw_train[(raw_train >= self.trunc_min) & (raw_train <= self.trunc_max)]
        
        # 2. Xử lý Verify Data (Giữ nguyên cấu trúc DataFrame để group theo ngày)
        # Apply truncation cho verify: Các giá trị ngoại lai sẽ bị loại bỏ hoặc giữ nguyên tùy logic
        # Ở đây ta lọc bỏ dòng ngoại lai để không làm nhiễu biểu đồ verify
        self.df_verify_clean = df_verify[
            (df_verify[col_res] >= self.trunc_min) & 
            (df_verify[col_res] <= self.trunc_max)
        ].copy()

    def calculate_ma(self, values, method, param):
        """Tính MA cho 1 mảng dữ liệu"""
        series = pd.Series(values)
        if method == 'SMA':
            return series.rolling(window=int(param)).mean().bfill().values
        elif method == 'EWMA':
            lam = 2 / (int(param) + 1)
            return series.ewm(alpha=lam, adjust=False).mean().values
        return values

    def determine_limits(self, method, param, target_fpr):
        """Tính Limit dựa trên Training Data"""
        ma_values = self.calculate_ma(self.train_clean, method, param)
        lower = np.percentile(ma_values, (target_fpr/2)*100)
        upper = np.percentile(ma_values, (1 - target_fpr/2)*100)
        return lower, upper

    def run_day_simulation(self, method, param, lcl, ucl, bias_pct, num_sims=None):
        """
        Logic: Duyệt qua từng ngày.
        Với mỗi ngày:
        1. Chọn điểm random k (1-40).
        2. Check Alarm trước k (Clean) -> Nếu có -> False Positive.
        3. Thêm Bias từ k -> Check Alarm sau k -> Nếu có -> Detection.
        """
        
        # Group dữ liệu theo ngày
        grouped = self.df_verify_clean.groupby(self.col_day)
        
        total_days = 0
        detected_days = 0
        false_positive_days = 0
        nped_list = []
        
        plot_data = None # Lưu data ngày cuối để vẽ
        
        bias_factor = 1 + (bias_pct / 100.0)

        # Lặp qua các ngày
        # Note: num_sims ở đây có thể hiểu là giới hạn số ngày chạy thử nếu data quá lớn
        # Nếu None thì chạy hết các ngày có trong file verify
        
        days_to_run = list(grouped.groups.keys())
        if num_sims and num_sims < len(days_to_run):
            days_to_run = days_to_run[:num_sims]

        for day_name in days_to_run:
            # Lấy dữ liệu của ngày đó
            day_df = grouped.get_group(day_name)
            vals = day_df[self.col_res].values.astype(float)
            n = len(vals)
            
            if n < 5: continue # Bỏ qua ngày quá ít mẫu
            
            total_days += 1
            
            # 1. Chọn điểm tiêm lỗi (Random 1 - 40)
            # Nếu ngày đó ít hơn 40 mẫu, chọn random trong khoảng độ dài của nó
            max_idx = min(40, n - 2) 
            if max_idx < 1: max_idx = 1
            
            injection_point = np.random.randint(1, max_idx + 1)
            
            # 2. Check False Positive (Kiểm tra Run sạch TRƯỚC điểm tiêm lỗi)
            # Tính MA cho đoạn clean đầu tiên
            # Lưu ý: PBRTQC thường chạy liên tục, nhưng ở đây ta giả định reset theo ngày hoặc chạy nối tiếp.
            # Để đơn giản và cô lập, ta tính MA cho ngày hiện tại.
            
            ma_clean_full = self.calculate_ma(vals, method, param)
            
            # Kiểm tra xem có alarm nào xuất hiện TRƯỚC injection_point không?
            # Vùng an toàn: index 0 đến injection_point - 1
            pre_bias_alarms = (ma_clean_full[:injection_point] < lcl) | (ma_clean_full[:injection_point] > ucl)
            
            if np.any(pre_bias_alarms):
                # Đã báo động TRƯỚC KHI có lỗi -> Báo động giả
                false_positive_days += 1
                
                # Lưu data để debug/vẽ nếu là ngày cuối
                if day_name == days_to_run[-1]:
                    plot_data = {
                        'day': day_name,
                        'vals_clean': vals,
                        'ma_clean': ma_clean_full,
                        'ma_sim': None,
                        'inject_idx': injection_point,
                        'alarm_idx': np.argmax(pre_bias_alarms), # Vị trí báo giả đầu tiên
                        'lcl': lcl, 'ucl': ucl,
                        'status': 'False Positive'
                    }
                continue # Dừng xử lý ngày này (theo yêu cầu user)

            # 3. Tiêm Bias và Check Detection (Sau điểm tiêm lỗi)
            vals_biased = vals.copy()
            vals_biased[injection_point:] *= bias_factor
            
            ma_biased = self.calculate_ma(vals_biased, method, param)
            
            # Chỉ xét vùng SAU injection_point
            post_bias_region = ma_biased[injection_point:]
            post_alarms = (post_bias_region < lcl) | (post_bias_region > ucl)
            
            if np.any(post_alarms):
                detected_days += 1
                first_alarm_idx_rel = np.argmax(post_alarms) # Index tương đối
                nped = first_alarm_idx_rel + 1 # Số mẫu trôi qua
                nped_list.append(nped)
                
                # Lưu data vẽ
                if day_name == days_to_run[-1]:
                     plot_data = {
                        'day': day_name,
                        'vals_clean': vals,
                        'ma_clean': ma_clean_full,
                        'ma_sim': ma_biased,
                        'inject_idx': injection_point,
                        'alarm_idx': injection_point + first_alarm_idx_rel,
                        'lcl': lcl, 'ucl': ucl,
                        'status': 'Detected'
                    }
            else:
                # Missed
                if day_name == days_to_run[-1]:
                     plot_data = {
                        'day': day_name,
                        'vals_clean': vals,
                        'ma_clean': ma_clean_full,
                        'ma_sim': ma_biased,
                        'inject_idx': injection_point,
                        'alarm_idx': None,
                        'lcl': lcl, 'ucl': ucl,
                        'status': 'Missed'
                    }

        # 4. Tổng hợp chỉ số
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
# 🖥️ PHẦN 3: GIAO DIỆN STREAMLIT
# =========================================================

st.set_page_config(layout="wide", page_title="PBRTQC Day-Simulator")

st.title("📅 PBRTQC Day-by-Day Simulator")
st.markdown("""
Hệ thống mô phỏng theo logic **Daily Run**:
1. Duyệt qua từng ngày trong dữ liệu Verify.
2. Tại mỗi ngày, chọn ngẫu nhiên thời điểm (1-40) để thêm Bias.
3. Nếu báo động xuất hiện **trước** khi thêm Bias -> **False Positive**.
4. Nếu báo động xuất hiện **sau** khi thêm Bias -> **Detection**.
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
    
    st.divider()
    max_days = st.slider("Giới hạn số ngày chạy mô phỏng", 10, 5000, 500, help="Giảm số này nếu thấy chạy chậm")

if f_train and f_verify:
    # Preview columns
    df_temp = pd.read_excel(f_train, nrows=1)
    all_cols = df_temp.columns.tolist()
    
    c1, c2 = st.columns(2)
    col_res = c1.selectbox("Cột Kết quả (Results)", all_cols)
    col_day = c2.selectbox("Cột Ngày (Days)", all_cols)

    if st.button("🚀 Run Simulation"):
        with st.spinner("Đang xử lý dữ liệu..."):
            # 1. Load Data
            df_train, df_verify = load_data(f_train, f_verify, col_res, col_day)
            
            if df_train is not None:
                # 2. Truncation
                trunc_range = find_optimal_truncation(df_train[col_res].values)
                st.info(f"Đã tối ưu Truncation Limit trên bộ Training: [{trunc_range[0]:.2f} - {trunc_range[1]:.2f}]")
                
                # 3. Init Engine
                engine = PBRTQCEngine(df_train, df_verify, col_res, col_day, trunc_range)
                
                # 4. Define Cases
                cases = [
                    {'bs': 20, 'freq': 1},
                    {'bs': 40, 'freq': 1},
                    {'bs': 60, 'freq': 1}
                ]
                
                results = []
                plots = []
                
                # 5. Run Loop
                prog_bar = st.progress(0)
                for i, case in enumerate(cases):
                    # a. Limit
                    lcl, ucl = engine.determine_limits(model, case['bs'], target_fpr)
                    
                    # b. Sim
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
                    
                    prog_bar.progress((i+1)/len(cases))
                
                # 6. Display
                st.subheader("📊 Kết quả Đánh giá")
                st.dataframe(pd.DataFrame(results).style.highlight_max(subset=['Detected (%)'], color='#d1ffbd'), use_container_width=True)
                
                st.divider()
                st.subheader("📈 Chi tiết 1 Ngày ngẫu nhiên (Ngày cuối cùng trong mô phỏng)")
                
                tabs = st.tabs([p['name'] for p in plots])
                for i, tab in enumerate(tabs):
                    with tab:
                        d = plots[i]['data']
                        if d:
                            fig, ax = plt.subplots(figsize=(12, 5))
                            
                            # Vẽ Clean MA
                            ax.plot(d['ma_clean'], label='MA (Clean Run)', color='green', alpha=0.4)
                            
                            # Vẽ Biased MA (Chỉ vẽ nếu có)
                            if d['ma_sim'] is not None:
                                ax.plot(d['ma_sim'], label=f'MA (Bias {bias_pct}%)', color='orange')
                            
                            # Limits
                            ax.axhline(d['ucl'], color='red', ls='--')
                            ax.axhline(d['lcl'], color='red', ls='--')
                            
                            # Injection Line
                            ax.axvline(d['inject_idx'], color='black', ls=':', label='Thời điểm thêm lỗi')
                            
                            # Alarm Point
                            if d['alarm_idx'] is not None:
                                marker_color = 'purple' if d['status'] == 'False Positive' else 'red'
                                marker_shape = 'X' if d['status'] == 'False Positive' else '*'
                                ax.scatter(d['alarm_idx'], d['ma_clean'][d['alarm_idx']] if d['ma_sim'] is None else d['ma_sim'][d['alarm_idx']], 
                                           color=marker_color, s=150, zorder=5, marker=marker_shape, label=f'Alarm ({d["status"]})')

                            ax.set_title(f"Mô phỏng ngày: {d['day']} - Trạng thái: {d['status']}")
                            ax.legend()
                            st.pyplot(fig)
                        else:
                            st.warning("Chưa có dữ liệu vẽ.")

            else:
                st.error("Lỗi định dạng file.")
