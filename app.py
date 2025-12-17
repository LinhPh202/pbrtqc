import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import io
import plotly.graph_objects as go

# =========================================================
# 🛠️ PHẦN 1: CORE FUNCTIONS (GIỐNG HỆT OPTIMIZER)
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

def calculate_ma_vectorized(values, method, param):
    series = pd.Series(values)
    if method == 'SMA':
        return series.rolling(window=int(param), min_periods=1).mean().values
    elif method == 'EWMA':
        return series.ewm(alpha=param, adjust=False, ignore_na=False).mean().values
    return values

def find_optimal_truncation(data_array, max_cut_percent=0.10, steps=10):
    calc_data = data_array
    if len(data_array) > 40000:
        np.random.seed(42)
        calc_data = np.random.choice(data_array, 40000, replace=False)
    best_p = -1
    best_range = (data_array.min(), data_array.max())
    cuts = np.linspace(0, max_cut_percent, steps)
    n = len(np.sort(calc_data)) # Fix sort length
    sorted_data = np.sort(calc_data)
    
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
# 📈 PHẦN 2: ENGINE MÔ PHỎNG (STRICT + ONE-SIDED)
# =========================================================

class PBRTQCEngine:
    def __init__(self, df_train, df_verify, col_res, col_day, trunc_range):
        self.trunc_min, self.trunc_max = trunc_range
        self.col_res = col_res
        self.col_day = col_day
        
        raw_train = df_train[col_res].values
        # Training Clean: Drop hoàn toàn
        self.train_clean = raw_train[(raw_train >= self.trunc_min) & (raw_train <= self.trunc_max)]
        
        # Verify: Giữ raw để xử lý từng ngày
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
        return {
            "Train Mean": np.mean(self.train_clean),
            "Train Median": np.median(self.train_clean),
            "Verify Mean": np.mean(self.global_vals),
            "Truncation Range": f"[{self.trunc_min:.2f} - {self.trunc_max:.2f}]"
        }

    def determine_limits(self, method, param, start_offset, frequency, target_fpr):
        # Tính Limit từ Training Data đã Clean (Reduced Stream)
        ma_values = calculate_ma_vectorized(self.train_clean, method, param)
        
        # Frequency đếm trên mẫu hợp lệ
        total_len = len(ma_values)
        report_mask = np.zeros(total_len, dtype=bool)
        s_idx = int(start_offset)
        if s_idx < total_len:
            report_mask[s_idx::int(frequency)] = True
            
        valid_ma_values = ma_values[report_mask]
        
        if len(valid_ma_values) == 0: return 0, 0, 0, 0

        lower = np.percentile(valid_ma_values, (target_fpr/2)*100)
        upper = np.percentile(valid_ma_values, (1 - target_fpr/2)*100)
        train_mean = np.mean(valid_ma_values)
        train_median = np.median(valid_ma_values)
        
        return lower, upper, train_mean, train_median

    def run_simulation(self, method, param, start_offset, frequency, lcl, ucl, bias_pct, direction='positive', fixed_inject_idx=None, apply_trunc_on_bias=False, sim_mode='Standard'):
        # --- BƯỚC 1: TÍNH DÒNG SẠCH (Clean Stream) ---
        # Lọc bỏ Outlier từ toàn bộ Verify Data
        mask_valid_global = (self.global_vals >= self.trunc_min) & (self.global_vals <= self.trunc_max)
        stream_clean_global = self.global_vals[mask_valid_global]
        indices_clean_global = np.where(mask_valid_global)[0] # Map index reduced -> original
        
        # Tính MA Clean Full
        ma_clean_full = calculate_ma_vectorized(stream_clean_global, method, param)
        
        # Tạo Map: Mỗi ngày bắt đầu từ index nào trong dòng Clean này
        daily_start_idx_clean = {}
        curr_clean = 0
        for day_name in self.day_indices:
            s, e = self.day_indices[day_name]
            # Đếm số mẫu hợp lệ trong ngày này
            day_valid_count = np.sum(mask_valid_global[s:e])
            daily_start_idx_clean[day_name] = curr_clean
            curr_clean += day_valid_count

        # --- BƯỚC 2: TÍNH FPR (Trên dòng sạch) ---
        # Mask Report Clean
        n_clean = len(ma_clean_full)
        report_mask_clean = np.zeros(n_clean, dtype=bool)
        s_off = int(start_offset)
        if s_off < n_clean:
            report_mask_clean[s_off::int(frequency)] = True
            
        valid_checks = ma_clean_full[report_mask_clean]
        fpr = 0.0
        if len(valid_checks) > 0:
            # FPR vẫn xét 2 chiều
            fpr = (np.sum((valid_checks < lcl) | (valid_checks > ucl)) / len(valid_checks)) * 100.0

        # --- BƯỚC 3: SIMULATION TỪNG NGÀY ---
        bias_factor = 1 + (bias_pct / 100.0) if direction == 'positive' else 1 - (bias_pct / 100.0)
        
        detected_days = 0
        total_valid_days = 0 # Ngày đủ điều kiện để test
        nped_list = []
        
        # Mảng để lưu kết quả vẽ biểu đồ (Map ngược về size gốc)
        # Khởi tạo bằng NaN
        plot_ma_values = np.full(len(self.global_vals), np.nan)
        plot_is_alarm = np.zeros(len(self.global_vals), dtype=bool)
        
        # Fill MA Clean vào trước (để những ngày không lỗi vẫn có hình)
        plot_ma_values[indices_clean_global] = ma_clean_full

        days_to_run = list(self.day_indices.keys())
        
        for day_name in days_to_run:
            start_idx, end_idx = self.day_indices[day_name]
            day_raw_vals = self.global_vals[start_idx:end_idx]
            n_raw = len(day_raw_vals)
            
            # Logic Inject
            local_inject = fixed_inject_idx if fixed_inject_idx is not None else np.random.randint(1, min(40, max(1, n_raw-2)) + 1)
            
            # Điều kiện độ dài (dựa trên raw để nhất quán input)
            if n_raw <= local_inject: continue
            min_req = s_off + 1
            # Start_idx = 0 nghĩa là ngày đầu tiên của file
            if start_idx == 0 and n_raw < min_req: continue
            
            total_valid_days += 1
            
            # A. Tạo dữ liệu ngày Biased
            day_vals_biased = day_raw_vals.copy()
            day_vals_biased[local_inject:] *= bias_factor
            
            # B. Lọc Strict (Drop Outlier)
            mask_valid_day = (day_vals_biased >= self.trunc_min) & (day_vals_biased <= self.trunc_max)
            day_vals_reduced = day_vals_biased[mask_valid_day]
            
            # C. Xác định điểm Inject trong dòng Reduced
            # Là số lượng mẫu hợp lệ nằm TRƯỚC điểm inject
            # (Phần trước inject chưa bị bias nên dùng mask clean hay biased đều giống nhau, trừ phi outlier gốc)
            inject_pos_reduced = np.sum(mask_valid_day[:local_inject])
            
            # D. Tính MA Ngày (Nối tiếp Clean History)
            start_clean_global = daily_start_idx_clean[day_name]
            
            ma_day = np.array([])
            if len(day_vals_reduced) > 0:
                if method == 'EWMA':
                    prev_ma = ma_clean_full[start_clean_global - 1] if start_clean_global > 0 else day_vals_reduced[0]
                    # Manual Calc
                    ma_day = np.empty(len(day_vals_reduced))
                    curr = prev_ma
                    for k, v in enumerate(day_vals_reduced):
                        curr = (1-param)*curr + param*v
                        ma_day[k] = curr
                elif method == 'SMA':
                    buffer_size = int(param) - 1
                    if start_clean_global >= buffer_size:
                        prev_buffer = stream_clean_global[start_clean_global - buffer_size : start_clean_global]
                        combined = np.concatenate([prev_buffer, day_vals_reduced])
                        full_roll = pd.Series(combined).rolling(window=int(param)).mean().values
                        ma_day = full_roll[buffer_size:]
                    else:
                        ma_day = pd.Series(day_vals_reduced).rolling(window=int(param), min_periods=1).mean().values
            
            # E. Check Alarm
            detected = False
            first_alarm_idx_reduced = -1
            
            # Lưu lại vào mảng Plot (Cần map index reduced -> raw local -> raw global)
            # Index của các mẫu hợp lệ trong ngày (tương đối với start_idx)
            valid_indices_local = np.where(mask_valid_day)[0]
            valid_indices_global = start_idx + valid_indices_local
            
            # Ghi đè MA ngày biased vào mảng plot
            if len(ma_day) > 0:
                plot_ma_values[valid_indices_global] = ma_day
            
            for k in range(len(ma_day)):
                # Chỉ xét sau khi inject
                if k >= inject_pos_reduced:
                    # Global Counter ảo: Tính từ đầu file clean
                    global_counter = start_clean_global + k
                    
                    if global_counter >= s_off and (global_counter - s_off) % int(frequency) == 0:
                        val = ma_day[k]
                        
                        # One-sided check cho Detection (Positive Bias -> > UCL)
                        is_alarm = False
                        if direction == 'positive': is_alarm = (val > ucl)
                        else: is_alarm = (val < lcl)
                        
                        if is_alarm:
                            if not detected: # Found first alarm
                                detected = True
                                first_alarm_idx_reduced = k
                            # Đánh dấu điểm báo động để vẽ
                            plot_is_alarm[valid_indices_global[k]] = True

            if detected:
                detected_days += 1
                nped = first_alarm_idx_reduced - inject_pos_reduced + 1
                nped_list.append(nped)

        # Kết quả
        metrics = {
            "Total Days": total_valid_days,
            "Detected (%)": round(detected_days / total_valid_days * 100, 1) if total_valid_days > 0 else 0,
            "Real FPR (%)": round(fpr, 2),
            "ANPed": round(np.mean(nped_list), 1) if nped_list else "N/A",
            "MNPed": round(np.median(nped_list), 1) if nped_list else "N/A",
            "95NPed": round(np.percentile(nped_list, 95), 1) if nped_list else "N/A",
            "Avg_Residual": "N/A" # Giản lược cho mode này
        }
        
        # Export Dataframe
        # Tạo mảng AON Report để vẽ scatter
        plot_aon_report = np.full(len(self.global_vals), np.nan)
        # Chỉ những điểm là Alarm mới hiện? Hoặc hiện tất cả điểm report?
        # Để giống biểu đồ cũ: Hiện giá trị tại các điểm Report
        # Cần tái tạo lại logic report mask trên toàn bộ mảng plot
        # Nhưng vì Drop logic làm index nhảy, ta chỉ có thể hiện những điểm ĐÃ ĐƯỢC TÍNH
        # Đơn giản nhất: Lấy plot_ma_values, lọc theo điều kiện >UCL/<LCL để tô màu
        
        # Để vẽ scatter đúng các điểm report, ta cần biết điểm nào là report
        # (Đã xử lý trong loop nhưng phức tạp để lưu lại hết)
        # Workaround: Chỉ export giá trị Continuous, User tự soi Alarm trên chart
        
        # Mảng AON_Reported chỉ chứa giá trị tại điểm Alarm để vẽ chấm đỏ
        plot_aon_report[plot_is_alarm] = plot_ma_values[plot_is_alarm]

        export_data = pd.DataFrame({
            'Day': self.global_days,
            'Result_Original': self.global_vals,
            f'{method}_Bias_Continuous': plot_ma_values,
            'AON_Bias_Report': plot_aon_report, # Chỉ chứa các điểm Alarm thực sự
            'LCL': lcl,
            'UCL': ucl
        })
        
        return metrics, export_data, nped_list, []

# =========================================================
# 🖥️ PHẦN 4: GIAO DIỆN (GIỮ NGUYÊN DRAW CHART CŨ)
# =========================================================
# (Phần code giao diện draw_chart và Streamlit UI giữ nguyên như code cũ của bạn)
# Chỉ thay class PBRTQCEngine bằng class mới ở trên.

def draw_chart(df, method, lcl, ucl, title, direction='positive'):
    fig = go.Figure()
    ma_col = f'{method}_Bias_Continuous'
    if ma_col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[ma_col], mode='lines', name='MA (Valid)', connectgaps=True, line=dict(color='lightblue')))
    
    fig.add_trace(go.Scatter(x=[df.index.min(), df.index.max()], y=[ucl, ucl], mode='lines', name='UCL', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=[df.index.min(), df.index.max()], y=[lcl, lcl], mode='lines', name='LCL', line=dict(color='blue', dash='dash')))
    
    # Vẽ điểm Alarm
    alarm_col = 'AON_Bias_Report'
    if alarm_col in df.columns:
        alarms = df.dropna(subset=[alarm_col])
        if not alarms.empty:
            fig.add_trace(go.Scatter(x=alarms.index, y=alarms[alarm_col], mode='markers', name='Alarm', marker=dict(color='red', size=8)))
            
    fig.update_layout(title=title, height=500)
    return fig

# ... (Phần UI Streamlit phía sau giữ nguyên)
# Lưu ý copy phần UI từ code cũ và paste xuống dưới class PBRTQCEngine này.
