def run_simulation(self, method, block_size, frequency, lcl, ucl, bias_pct, direction='positive', fixed_inject_idx=None):
        total_days = 0
        detected_days = 0
        nped_list = []
        
        if direction == 'positive':
            bias_factor = 1 + (bias_pct / 100.0)
        else:
            bias_factor = 1 - (bias_pct / 100.0)
        
        # 1. TÍNH TOÁN BASELINE (AUDIT TRÊN TOÀN BỘ DỮ LIỆU SẠCH)
        # -------------------------------------------------------
        # Tính MA cho toàn bộ dữ liệu sạch
        global_ma_clean = self.calculate_ma(self.global_vals, method, block_size)
        # Lấy mask các điểm report trên toàn bộ dữ liệu
        global_report_mask = self.get_report_mask(len(self.global_vals), block_size, frequency)
        
        # Lọc ra các giá trị AON sạch
        baseline_aon_vals = global_ma_clean[global_report_mask]
        
        # Audit: Tổng số điểm kiểm tra (Mẫu số FPR)
        total_clean_checks = len(baseline_aon_vals)
        
        # Audit: Tổng số báo động giả (Tử số FPR) - Check cả 2 đầu vì đây là dữ liệu sạch
        baseline_alarms = (baseline_aon_vals < lcl) | (baseline_aon_vals > ucl)
        total_false_alarms = np.sum(baseline_alarms)
        
        # Tính FPR ngay tại đây (Dựa trên toàn bộ dữ liệu)
        real_fpr_pct = 0.0
        if total_clean_checks > 0:
            real_fpr_pct = (total_false_alarms / total_clean_checks) * 100.0
        # -------------------------------------------------------

        # 2. CHẠY MÔ PHỎNG (CHỈ ĐỂ TÍNH DETECTION)
        global_biased_export = self.global_vals.copy()
        injection_flags = np.zeros(len(self.global_vals), dtype=int)

        days_to_run = list(self.day_indices.keys())

        for day_name in days_to_run:
            start_idx, end_idx = self.day_indices[day_name]
            day_len = end_idx - start_idx
            
            if start_idx == 0 and day_len < block_size:
                continue

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
            
            # Tiêm lỗi
            global_biased_export[global_inject_idx : end_idx] *= bias_factor
            injection_flags[global_inject_idx : end_idx] = 1

            # CHECK DETECTION (Vùng sau lỗi)
            temp_global_vals = self.global_vals.copy()
            temp_global_vals[global_inject_idx : end_idx] *= bias_factor
            
            global_ma_biased_temp = self.calculate_ma(temp_global_vals, method, block_size)
            
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

        metrics = {
            "Total Days": total_days,
            "Detected (%)": round(detected_days / total_days * 100, 1) if total_days > 0 else 0,
            "Real FPR (%)": round(real_fpr_pct, 2),
            "Detected_Count": detected_days,
            "False_Alarm_Count": total_false_alarms, # <--- Giờ là số liệu toàn bộ Dataset
            "Clean_Check_Count": total_clean_checks, # <--- Giờ là số liệu toàn bộ Dataset
            "ANPed": round(np.mean(nped_list), 1) if nped_list else "N/A",
            "MNPed": round(np.median(nped_list), 1) if nped_list else "N/A",
            "95NPed": round(np.percentile(nped_list, 95), 1) if nped_list else "N/A"
        }
        
        # Export Data Logic (Giữ nguyên)
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
