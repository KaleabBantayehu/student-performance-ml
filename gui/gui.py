import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import numpy as np

# Use TkAgg backend for Matplotlib
matplotlib.use("TkAgg")

# Add project root to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import PerformancePredictor

class AIStudentPerformancePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Student Performance Predictor Pro")
        self.root.geometry("1000x800")
        self.root.minsize(900, 750)
        
        # Proper Window Close Protocol
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)
        
        # Initialize Predictor
        try:
            self.predictor = PerformancePredictor()
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load ML models: {e}")
            self.root.destroy()
            return

        # Configure Styles
        self._setup_styles()
        
        # Main Container
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self._setup_header()
        
        # Tabbed Interface
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Prediction Dashboard
        self.predict_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_tab, text=" Prediction Dashboard ")
        
        # Tab 2: Model Analytics
        self.analytics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_tab, text=" Model Insights ")
        
        # Setup Tabs
        self._setup_predict_tab()
        self._setup_analytics_tab()

    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Colors
        self.colors = {
            "primary": "#1A252F",    # Deep Navy
            "secondary": "#2C3E50",  # Dark Blue-Grey
            "accent": "#3498DB",     # Bright blue
            "background": "#F4F7F6", # Off White/Grey
            "surface": "#FFFFFF",    # White
            "text": "#2C3E50",
            "success": "#27AE60",    # Green
            "warning": "#F39C12",    # Orange
            "danger": "#E74C3C"      # Red
        }
        
        # Font Configuration
        self.fonts = {
            "title": ("Segoe UI", 22, "bold"),
            "header": ("Segoe UI", 12, "bold"),
            "body": ("Segoe UI", 10),
            "label": ("Segoe UI", 10, "bold"),
            "result_val": ("Segoe UI", 14, "bold"),
            "category": ("Segoe UI", 18, "bold")
        }

        # Apply Styles
        self.root.configure(background=self.colors["background"])
        self.style.configure("TFrame", background=self.colors["background"])
        self.style.configure("TNotebook", background=self.colors["background"])
        self.style.configure("TNotebook.Tab", font=self.fonts["label"], padding=[15, 5])
        
        self.style.configure("Header.TLabel", 
                           font=self.fonts["title"], 
                           foreground=self.colors["primary"], 
                           background=self.colors["background"])
        
        self.style.configure("Section.TLabelframe", 
                           background=self.colors["surface"])
        self.style.configure("Section.TLabelframe.Label", 
                           font=self.fonts["header"], 
                           foreground=self.colors["secondary"],
                           background=self.colors["background"])
        
        self.style.configure("Input.TLabel", 
                           font=self.fonts["label"], 
                           background=self.colors["surface"])
        
        self.style.configure("Surface.TFrame", background=self.colors["surface"])
        self.style.configure("Surface.TLabel", background=self.colors["surface"])
        
        self.style.configure("Action.TButton", 
                           font=self.fonts["label"], 
                           padding=8)
        
        self.style.configure("Predict.TButton", 
                           font=self.fonts["label"], 
                           foreground="white", 
                           background=self.colors["accent"],
                           padding=10)
        self.style.map("Predict.TButton", 
                      background=[('active', '#2980B9')])

    def _setup_header(self):
        header_frame = ttk.Frame(self.main_container)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(header_frame, 
                               text="AI Student Performance Predictor", 
                               style="Header.TLabel")
        title_label.pack(side=tk.LEFT)
        
        status_dot = tk.Label(header_frame, text="●", fg=self.colors["success"], bg=self.colors["background"], font=("Arial", 14))
        status_dot.pack(side=tk.LEFT, padx=(20, 5), pady=(5, 0))
        
        status_lbl = ttk.Label(header_frame, text="Model Active", font=("Segoe UI", 9, "bold"), foreground=self.colors["success"])
        status_lbl.pack(side=tk.LEFT, pady=(8, 0))

    def _setup_predict_tab(self):
        # Two-column layout
        left_col = ttk.Frame(self.predict_tab, padding=10)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        
        right_col = ttk.Frame(self.predict_tab, padding=10)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Left Column: Inputs ---
        self.input_lf = ttk.LabelFrame(left_col, text=" Student Metrics ", padding="15", style="Section.TLabelframe")
        self.input_lf.pack(fill=tk.X, pady=(0, 10))
        
        self.input_lf.columnconfigure(1, weight=1)
        
        inputs = [
            ("Study Hours (Weekly):", "study_hours"),
            ("Attendance (%):", "attendance"),
            ("Sleep Hours (Daily):", "sleep_hours"),
            ("Previous GPA (0.0 - 4.0):", "prev_gpa")
        ]
        
        self.vars = {}
        for i, (label_text, var_name) in enumerate(inputs):
            lbl = ttk.Label(self.input_lf, text=label_text, style="Input.TLabel")
            lbl.grid(row=i, column=0, sticky=tk.W, pady=8, padx=(0, 10))
            
            var = tk.StringVar()
            entry = ttk.Entry(self.input_lf, textvariable=var, font=self.fonts["body"])
            entry.grid(row=i, column=1, sticky=tk.EW, pady=8)
            self.vars[var_name] = var

        lbl_asgn = ttk.Label(self.input_lf, text="Assignments Completed:", style="Input.TLabel")
        lbl_asgn.grid(row=len(inputs), column=0, sticky=tk.W, pady=8, padx=(0, 10))
        
        self.vars["assignments"] = tk.StringVar(value="Select...")
        asgn_dropdown = ttk.Combobox(self.input_lf, 
                                    textvariable=self.vars["assignments"], 
                                    values=["Yes", "No"], 
                                    state="readonly",
                                    font=self.fonts["body"])
        asgn_dropdown.grid(row=len(inputs), column=1, sticky=tk.EW, pady=8)

        # Buttons
        btn_frame = ttk.Frame(left_col, padding=10)
        btn_frame.pack(fill=tk.X)
        
        self.predict_btn = ttk.Button(btn_frame, text="RUN AI PREDICTION", style="Predict.TButton", command=self._on_predict)
        self.predict_btn.pack(fill=tk.X, pady=5)
        
        sub_btn_frame = ttk.Frame(btn_frame)
        sub_btn_frame.pack(fill=tk.X)
        
        self.clear_btn = ttk.Button(sub_btn_frame, text="Reset", style="Action.TButton", command=self._on_clear)
        self.clear_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.exit_btn = ttk.Button(sub_btn_frame, text="Exit System", style="Action.TButton", command=self._on_exit)
        self.exit_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Right Column: Results & Visualization ---
        self.results_lf = ttk.LabelFrame(right_col, text=" Prediction Output ", padding="15", style="Section.TLabelframe")
        self.results_lf.pack(fill=tk.X, pady=(0, 10))
        
        # Primary Result Display
        res_header = ttk.Frame(self.results_lf, style="Surface.TFrame")
        res_header.pack(fill=tk.X, pady=5)
        
        self.res_labels = {}
        
        # Grade Label (Large)
        self.grade_val = ttk.Label(res_header, text="---", font=self.fonts["category"], 
                                  foreground=self.colors["secondary"], style="Surface.TLabel")
        self.grade_val.pack(pady=5)
        
        metrics_frame = ttk.Frame(self.results_lf, style="Surface.TFrame")
        metrics_frame.pack(fill=tk.X, pady=10)
        
        for i, (text, key, color) in enumerate([("Confidence:", "confidence", self.colors["accent"]), 
                                               ("Risk Level:", "risk", self.colors["warning"])]):
            lbl = ttk.Label(metrics_frame, text=text, font=self.fonts["label"], style="Surface.TLabel")
            lbl.grid(row=0, column=i*2, padx=(20 if i>0 else 0, 5), sticky=tk.W)
            
            val = ttk.Label(metrics_frame, text="---", font=self.fonts["result_val"], 
                            foreground=color, style="Surface.TLabel")
            val.grid(row=0, column=i*2+1, sticky=tk.W)
            self.res_labels[key] = val

        # Probability Progress Bars
        self.prob_frame = ttk.LabelFrame(right_col, text=" Class Probabilities ", padding="15", style="Section.TLabelframe")
        self.prob_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.prob_bars = {}
        classes = ["Excellent", "Good", "Average", "At Risk"]
        for i, cls in enumerate(classes):
            lbl = ttk.Label(self.prob_frame, text=f"{cls}:", font=self.fonts["label"], style="Surface.TLabel", width=12)
            lbl.grid(row=i, column=0, sticky=tk.W, pady=5)
            
            bar = ttk.Progressbar(self.prob_frame, length=300, mode='determinate')
            bar.grid(row=i, column=1, sticky=tk.EW, padx=10, pady=5)
            
            val_lbl = ttk.Label(self.prob_frame, text="0%", font=self.fonts["body"], style="Surface.TLabel", width=6)
            val_lbl.grid(row=i, column=2, sticky=tk.W, pady=5)
            
            self.prob_bars[cls] = (bar, val_lbl)
        self.prob_frame.columnconfigure(1, weight=1)

        # Explanation
        exp_lf = ttk.LabelFrame(right_col, text=" Automated Explanation ", padding="15", style="Section.TLabelframe")
        exp_lf.pack(fill=tk.BOTH, expand=True)
        
        self.explanation_box = scrolledtext.ScrolledText(exp_lf, 
                                                       height=6, 
                                                       font=self.fonts["body"],
                                                       wrap=tk.WORD,
                                                       bg="white",
                                                       relief=tk.FLAT)
        self.explanation_box.pack(fill=tk.BOTH, expand=True)
        self.explanation_box.insert(tk.INSERT, "Perform a prediction to see the analysis...")
        self.explanation_box.config(state=tk.DISABLED)

    def _setup_analytics_tab(self):
        analytics_container = ttk.Frame(self.analytics_tab, padding=20)
        analytics_container.pack(fill=tk.BOTH, expand=True)
        
        # Top Section (Stats + Importance)
        top_frame = ttk.Frame(analytics_container)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Top Left: Session Statistics
        stats_lf = ttk.LabelFrame(top_frame, text=" Session Statistics & Live Analytics ", padding="15", style="Section.TLabelframe")
        stats_lf.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        self.stats_labels = {}
        stats_keys = [
            ("Total Predictions", "total_preds"), 
            ("Average Confidence", "avg_conf"),
            ("Highest Confidence", "high_conf"),
            ("Lowest Confidence", "low_conf"),
            ("Confidence Interpretation", "conf_interp"),
            ("Top Contributors (Current)", "top_features")
        ]
        
        for i, (label_text, key) in enumerate(stats_keys):
            ttk.Label(stats_lf, text=f"{label_text}:", font=self.fonts["label"], style="Surface.TLabel").grid(row=i, column=0, sticky=tk.W, pady=8)
            val_lbl = ttk.Label(stats_lf, text="---", font=self.fonts["body"], style="Surface.TLabel", wraplength=200)
            val_lbl.grid(row=i, column=1, sticky=tk.W, padx=10, pady=8)
            self.stats_labels[key] = val_lbl
            
        self.session_data = {"preds": [], "confidences": []}
        
        # Top Right: Feature Importance
        importance_lf = ttk.LabelFrame(top_frame, text=" Global Feature Importance ", padding="15", style="Section.TLabelframe")
        importance_lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.importance_canvas_frame = ttk.Frame(importance_lf, style="Surface.TFrame")
        self.importance_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Bottom Section (History Table)
        history_lf = ttk.LabelFrame(analytics_container, text=" Session Prediction History ", padding="15", style="Section.TLabelframe")
        history_lf.pack(fill=tk.BOTH, expand=True)
        
        columns = ("Time", "Study", "Att(%)", "Sleep", "Asgn", "GPA", "Prediction", "Confidence")
        self.history_tree = ttk.Treeview(history_lf, columns=columns, show="headings", height=5)
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=70, anchor=tk.CENTER)
            
        scrollbar = ttk.Scrollbar(history_lf, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscroll=scrollbar.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initial Plot
        self._plot_feature_importances()

    def _plot_feature_importances(self):
        # Clear existing
        for widget in self.importance_canvas_frame.winfo_children():
            widget.destroy()
            
        importances = self.predictor.get_feature_importances()
        if not importances:
            ttk.Label(self.importance_canvas_frame, text="Feature importance data not available.", style="Surface.TLabel").pack()
            return

        # Prepare Data
        features = list(importances.keys())
        scores = list(importances.values())
        
        # Create Figure
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        fig.patch.set_facecolor('white')
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(features)))
        bars = ax.barh(features, scores, color=colors)
        
        ax.set_xlabel('Importance Score')
        ax.set_title('Impact of Features on Performance Prediction', fontweight='bold', pad=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add labels to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center')

        plt.tight_layout()
        
        # Embed
        canvas = FigureCanvasTkAgg(fig, master=self.importance_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _validate_inputs(self):
        try:
            for key, var in self.vars.items():
                val = var.get().strip()
                if not val or val == "Select...":
                    messagebox.showwarning("Incomplete Data", f"Field '{key.replace('_', ' ').title()}' is required.")
                    return None

            sh = float(self.vars["study_hours"].get())
            att = float(self.vars["attendance"].get())
            slh = float(self.vars["sleep_hours"].get())
            gpa = float(self.vars["prev_gpa"].get())
            
            if any(x < 0 for x in [sh, att, slh, gpa]):
                messagebox.showerror("Validation Error", "Metrics cannot be negative values.")
                return None
            if att > 100:
                messagebox.showerror("Validation Error", "Attendance percentage cannot exceed 100%.")
                return None
            if gpa > 4.0:
                messagebox.showerror("Validation Error", "GPA must be between 0.0 and 4.0.")
                return None

            asgn = 1 if self.vars["assignments"].get() == "Yes" else 0
            return [sh, att, slh, asgn, gpa]
        except ValueError:
            messagebox.showerror("Type Error", "Please ensure all metrics are numeric values.")
            return None

    def _on_predict(self):
        features = self._validate_inputs()
        if not features: return
        
        try:
            # Main Prediction
            grade, confidence = self.predictor.predict_with_proba(features)
            all_probs = self.predictor.get_all_probabilities(features)
            importances = self.predictor.get_feature_importances() or {}
            
            # Risk/Color Logic
            status_config = {
                "Excellent": (self.colors["success"], "LOW"),
                "Good": (self.colors["success"], "LOW"),
                "Average": (self.colors["warning"], "MODERATE"),
                "At Risk": (self.colors["danger"], "HIGH")
            }
            color, risk = status_config.get(grade, (self.colors["secondary"], "UNKNOWN"))

            # Update Main Labels
            self.grade_val.config(text=grade.upper(), foreground=color)
            self.res_labels["confidence"].config(text=f"{confidence*100:.1f}%")
            self.res_labels["risk"].config(text=risk, foreground=color)
            
            # Update Probability Bars
            for cls, (bar, lbl) in self.prob_bars.items():
                prob = all_probs.get(cls, 0)
                bar['value'] = prob * 100
                lbl.config(text=f"{prob*100:.1f}%")

            # Update Explanation
            self._update_explanation(grade, risk, confidence, all_probs, importances, features)
            
            # Update Live Analytics Dashboard
            self._update_analytics_dashboard(grade, confidence, importances, features)

        except Exception as e:
            messagebox.showerror("System Error", f"Prediction engine failed: {e}")

    def _update_analytics_dashboard(self, grade, confidence, importances, features):
        import datetime
        import numpy as np
        sh, att, slh, asgn, gpa = features
        
        # 1. Update Session Data
        self.session_data["preds"].append(grade)
        self.session_data["confidences"].append(confidence)
        
        # 2. Update Stats Labels
        confs = self.session_data["confidences"]
        self.stats_labels["total_preds"].config(text=str(len(confs)))
        self.stats_labels["avg_conf"].config(text=f"{np.mean(confs)*100:.1f}%")
        self.stats_labels["high_conf"].config(text=f"{np.max(confs)*100:.1f}%")
        self.stats_labels["low_conf"].config(text=f"{np.min(confs)*100:.1f}%")
        
        # Confidence Interpretation
        if confidence >= 0.85:
            interp_text = "High Confidence"
            interp_color = self.colors["success"]
        elif confidence < 0.60:
            interp_text = "Low Confidence"
            interp_color = self.colors["danger"]
        else:
            interp_text = "Moderate Confidence"
            interp_color = self.colors["warning"]
            
        self.stats_labels["conf_interp"].config(text=interp_text, foreground=interp_color)
        
        # Top 2 Contributors for Current Prediction
        feature_status = {
            'Attendance': {'value': att, 'score': importances.get('Attendance', 0)},
            'Study Hours': {'value': sh, 'score': importances.get('Study Hours', 0)},
            'Prev GPA': {'value': gpa, 'score': importances.get('Previous Gpa', 0)}, # Match exact model keys if possible
            'Assignments': {'value': asgn, 'score': importances.get('Assignments Completed', 0)},
            'Sleep Hours': {'value': slh, 'score': importances.get('Sleep Hours', 0)}
        }
        
        # We sort by importance score (we can't easily do local shap, so we use global weighting)
        sorted_features = sorted(feature_status.items(), key=lambda x: x[1]['score'], reverse=True)
        top_2 = [name for name, data in sorted_features[:2]]
        if not top_2: top_2 = ["Data Unavailable"]
        self.stats_labels["top_features"].config(text=", ".join(top_2))
        
        # 3. Update History Table
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        asgn_str = "Yes" if asgn == 1 else "No"
        self.history_tree.insert("", 0, values=(timestamp, sh, f"{att}%", slh, asgn_str, gpa, grade, f"{confidence*100:.1f}%"))

    def _update_explanation(self, grade, risk, confidence, all_probs, importances, features):
        sh, att, slh, asgn, gpa = features
        
        explanation = f"AI ANALYSIS SUMMARY\n" + "="*20 + "\n\n"
        explanation += f"Predicted Outcome: {grade}\n"
        explanation += f"Strategic Risk Level: {risk}\n\n"
        
        explanation += f"Confidence Assessment ({confidence*100:.1f}%):\n"
        if confidence > 0.85:
            explanation += "The AI is highly confident in this prediction, indicating your metrics strongly align with historical data for this outcome.\n\n"
        elif confidence < 0.60:
            explanation += "The AI detects significant uncertainty. Your profile exhibits overlapping characteristics with other performance tiers. Small behavioral changes could quickly shift your outcome.\n\n"
        else:
            explanation += "The AI is moderately confident. Your profile generally matches this outcome, though some variables introduce slight variation.\n\n"
        
        explanation += "Data-Driven Insights:\n"
        
        feature_status = {
            'Attendance': {'value': att, 'status': 'Strong' if att >= 90 else 'Weak' if att < 80 else 'Average', 'score': importances.get('Attendance', 0)},
            'Study Hours': {'value': sh, 'status': 'Strong' if sh >= 15 else 'Weak' if sh < 10 else 'Average', 'score': importances.get('Study Hours', 0)},
            'Prev GPA': {'value': gpa, 'status': 'Strong' if gpa >= 3.5 else 'Weak' if gpa < 2.5 else 'Average', 'score': importances.get('Prev GPA', 0)},
            'Assignments': {'value': asgn, 'status': 'Strong' if asgn == 1 else 'Weak', 'score': importances.get('Assignments', 0)},
            'Sleep Hours': {'value': slh, 'status': 'Strong' if slh >= 7 else 'Weak' if slh < 6 else 'Average', 'score': importances.get('Sleep Hours', 0)}
        }
        
        sorted_features = sorted(feature_status.items(), key=lambda x: x[1]['score'], reverse=True)
        weak_features = [name for name, data in sorted_features if data['status'] == 'Weak']
        strong_features = [name for name, data in sorted_features if data['status'] == 'Strong']
        
        if sorted_features and sorted_features[0][1]['score'] > 0:
            top_feature = sorted_features[0][0]
            explanation += f"• Primary Driver: '{top_feature}' is the most heavily weighted factor in this model.\n"
        
        if strong_features:
            explanation += f"• Greatest Strengths: Your performance is bolstered by your {', '.join(strong_features)}.\n"
        if weak_features:
            explanation += f"• Critical Vulnerabilities: The model flagged {', '.join(weak_features)} as negatively impacting your score.\n"
            
        explanation += "\nPERSONALIZED RECOMMENDATION:\n"
        
        if grade == "Excellent":
            explanation += "Outstanding trajectory. "
            if weak_features:
                explanation += f"To ensure you maintain this, address minor vulnerabilities in your {weak_features[0].lower()}. "
            explanation += "Consider engaging in advanced coursework or peer tutoring to solidify your mastery."
        elif grade == "Good":
            explanation += "Solid performance. You are on track, but there is room to elevate to the top tier. "
            if weak_features:
                explanation += f"Focus your immediate efforts on improving your {weak_features[0].lower()}. "
            else:
                explanation += "Focus on incremental improvements across study habits and attendance."
        elif grade == "Average":
            explanation += "You are currently maintaining a baseline academic standing. To shift out of the 'Average' bracket, "
            if weak_features:
                explanation += f"it is imperative to directly address your {weak_features[0].lower()} and {weak_features[1].lower() if len(weak_features)>1 else 'overall engagement'}. "
            explanation += "Set specific weekly goals for study time and assignment completion."
        else:
            explanation += "URGENT INTERVENTION REQUIRED. Your current metrics align with students who face academic probation. "
            if weak_features:
                explanation += f"Immediate action must be taken regarding your {', '.join(weak_features)}. "
            explanation += "Schedule an emergency meeting with an academic advisor today to construct a recovery plan."

        self.explanation_box.config(state=tk.NORMAL)
        self.explanation_box.delete(1.0, tk.END)
        self.explanation_box.insert(tk.END, explanation)
        self.explanation_box.config(state=tk.DISABLED)

    def _on_clear(self):
        for var in self.vars.values():
            if var.get() in ["Yes", "No", "Select..."]: var.set("Select...")
            else: var.set("")
        
        self.grade_val.config(text="---", foreground=self.colors["secondary"])
        self.res_labels["confidence"].config(text="---")
        self.res_labels["risk"].config(text="---", foreground=self.colors["warning"])
        
        for bar, lbl in self.prob_bars.values():
            bar['value'] = 0
            lbl.config(text="0%")
            
        self.explanation_box.config(state=tk.NORMAL)
        self.explanation_box.delete(1.0, tk.END)
        self.explanation_box.insert(tk.END, "Perform a prediction to see the analysis...")
        self.explanation_box.config(state=tk.DISABLED)

    def _on_exit(self):
        if messagebox.askokcancel("Exit", "Shutdown AI Student Performance Predictor?"):
            self.root.quit()
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    # High DPI support for Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    
    app = AIStudentPerformancePredictor(root)
    root.mainloop()
