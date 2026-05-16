import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

class AIStudentPerformancePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Student Performance Predictor")
        self.root.geometry("800x700")
        self.root.minsize(700, 600)
        
        # Configure Styles
        self._setup_styles()
        
        # Main Container
        self.main_container = ttk.Frame(self.root, padding="20")
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self._setup_header()
        
        # Content Layout: Two columns (Inputs and Results)
        self.content_frame = ttk.Frame(self.main_container)
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self._setup_input_frame()
        self._setup_results_frame()
        
        # Button Frame
        self._setup_button_frame()

    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Using 'clam' as a base for better customization
        
        # Colors
        self.colors = {
            "primary": "#2C3E50",    # Dark Blue
            "secondary": "#34495E",  # Slightly lighter blue
            "accent": "#3498DB",     # Bright blue
            "background": "#ECF0F1", # Light Gray
            "text": "#2C3E50",
            "success": "#27AE60",    # Green
            "warning": "#F39C12",    # Orange
            "danger": "#E74C3C"      # Red
        }
        
        # Font Configuration
        self.fonts = {
            "title": ("Segoe UI", 18, "bold"),
            "header": ("Segoe UI", 12, "bold"),
            "body": ("Segoe UI", 10),
            "label": ("Segoe UI", 10, "bold"),
            "result_val": ("Segoe UI", 11, "bold")
        }

        # Apply Styles
        self.style.configure("TFrame", background=self.colors["background"])
        self.root.configure(background=self.colors["background"])
        
        self.style.configure("Header.TLabel", 
                           font=self.fonts["title"], 
                           foreground=self.colors["primary"], 
                           background=self.colors["background"])
        
        self.style.configure("Section.TLabelframe", 
                           background=self.colors["background"])
        self.style.configure("Section.TLabelframe.Label", 
                           font=self.fonts["header"], 
                           foreground=self.colors["secondary"],
                           background=self.colors["background"])
        
        self.style.configure("Input.TLabel", 
                           font=self.fonts["label"], 
                           background=self.colors["background"])
        
        self.style.configure("Predict.TButton", 
                           font=self.fonts["label"], 
                           foreground="white", 
                           background=self.colors["accent"])
        self.style.map("Predict.TButton", 
                      background=[('active', '#2980B9')])

    def _setup_header(self):
        header_frame = ttk.Frame(self.main_container)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(header_frame, 
                               text="AI Student Performance Predictor", 
                               style="Header.TLabel")
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = ttk.Label(header_frame, 
                                  text="Smart Academic Analytics", 
                                  font=("Segoe UI", 9, "italic"),
                                  foreground="#7F8C8D")
        subtitle_label.pack(side=tk.LEFT, padx=15, pady=(8, 0))

    def _setup_input_frame(self):
        # Input LabelFrame
        self.input_lf = ttk.LabelFrame(self.content_frame, text=" Student Data Input ", padding="15", style="Section.TLabelframe")
        self.input_lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Grid Configuration
        self.input_lf.columnconfigure(1, weight=1)
        
        # Input Fields
        inputs = [
            ("Study Hours (Weekly):", "study_hours"),
            ("Attendance (%):", "attendance"),
            ("Sleep Hours (Daily):", "sleep_hours"),
            ("Previous GPA (0.0 - 4.0):", "prev_gpa")
        ]
        
        self.vars = {}
        for i, (label_text, var_name) in enumerate(inputs):
            lbl = ttk.Label(self.input_lf, text=label_text, style="Input.TLabel")
            lbl.grid(row=i, column=0, sticky=tk.W, pady=10, padx=(0, 10))
            
            var = tk.StringVar()
            entry = ttk.Entry(self.input_lf, textvariable=var)
            entry.grid(row=i, column=1, sticky=tk.EW, pady=10)
            self.vars[var_name] = var

        # Assignments Completed Dropdown
        lbl_asgn = ttk.Label(self.input_lf, text="Assignments Completed:", style="Input.TLabel")
        lbl_asgn.grid(row=len(inputs), column=0, sticky=tk.W, pady=10, padx=(0, 10))
        
        self.vars["assignments"] = tk.StringVar(value="Select...")
        asgn_dropdown = ttk.Combobox(self.input_lf, 
                                    textvariable=self.vars["assignments"], 
                                    values=["Yes", "No"], 
                                    state="readonly")
        asgn_dropdown.grid(row=len(inputs), column=1, sticky=tk.EW, pady=10)

    def _setup_results_frame(self):
        # Results LabelFrame
        self.results_lf = ttk.LabelFrame(self.content_frame, text=" Prediction Results ", padding="15", style="Section.TLabelframe")
        self.results_lf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Grid Configuration
        self.results_lf.columnconfigure(1, weight=1)
        
        # Result Fields placeholders
        self.res_labels = {}
        
        result_items = [
            ("Predicted Grade:", "grade", self.colors["primary"]),
            ("Confidence Score:", "confidence", self.colors["accent"]),
            ("Risk Level:", "risk", self.colors["warning"])
        ]
        
        for i, (label_text, key, color) in enumerate(result_items):
            lbl = ttk.Label(self.results_lf, text=label_text, style="Input.TLabel")
            lbl.grid(row=i, column=0, sticky=tk.W, pady=8)
            
            val_lbl = ttk.Label(self.results_lf, text="---", font=self.fonts["result_val"], foreground=color)
            val_lbl.grid(row=i, column=1, sticky=tk.W, padx=10)
            self.res_labels[key] = val_lbl
            
        # Explanation Section
        exp_lbl = ttk.Label(self.results_lf, text="Analysis Explanation:", style="Input.TLabel")
        exp_lbl.grid(row=len(result_items), column=0, columnspan=2, sticky=tk.W, pady=(15, 5))
        
        self.explanation_box = scrolledtext.ScrolledText(self.results_lf, 
                                                       height=8, 
                                                       font=self.fonts["body"],
                                                       wrap=tk.WORD,
                                                       bg="white",
                                                       relief=tk.FLAT)
        self.explanation_box.grid(row=len(result_items)+1, column=0, columnspan=2, sticky=tk.NSEW, pady=5)
        self.explanation_box.insert(tk.INSERT, "Perform a prediction to see the analysis...")
        self.explanation_box.config(state=tk.DISABLED)
        
        self.results_lf.rowconfigure(len(result_items)+1, weight=1)

    def _setup_button_frame(self):
        btn_frame = ttk.Frame(self.main_container)
        btn_frame.pack(fill=tk.X, pady=20)
        
        # Spacer
        ttk.Frame(btn_frame).pack(side=tk.LEFT, expand=True)
        
        self.predict_btn = ttk.Button(btn_frame, text="Predict Performance", style="Predict.TButton", command=self._on_predict)
        self.predict_btn.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=5)
        
        self.clear_btn = ttk.Button(btn_frame, text="Clear All", command=self._on_clear)
        self.clear_btn.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=5)
        
        self.exit_btn = ttk.Button(btn_frame, text="Exit", command=self._on_exit)
        self.exit_btn.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=5)
        
        # Spacer
        ttk.Frame(btn_frame).pack(side=tk.LEFT, expand=True)

    def _on_predict(self):
        """
        Placeholder logic for prediction. 
        Will be connected to the ML pipeline in the next milestone.
        """
        # Simple validation check (placeholders)
        for key, var in self.vars.items():
            val = var.get().strip()
            if not val or val == "Select...":
                messagebox.showwarning("Missing Data", f"Please provide a valid value for {key.replace('_', ' ').title()}.")
                return
        
        # Update placeholders with dummy data
        self.res_labels["grade"].config(text="A- (Excellent)", foreground=self.colors["success"])
        self.res_labels["confidence"].config(text="92.4%")
        self.res_labels["risk"].config(text="LOW", foreground=self.colors["success"])
        
        self.explanation_box.config(state=tk.NORMAL)
        self.explanation_box.delete(1.0, tk.END)
        self.explanation_box.insert(tk.END, 
            "The model predicts a high performance level based on your consistent study hours and high attendance. "
            "Maintaining a GPA above 3.5 while completing all assignments significantly reduces academic risk. "
            "\n\nRecommendation: Continue your current study routine and ensure you get at least 7 hours of sleep."
        )
        self.explanation_box.config(state=tk.DISABLED)

    def _on_clear(self):
        """Resets all input fields and result displays."""
        for var in self.vars.values():
            if isinstance(var, tk.StringVar) and var.get() in ["Yes", "No", "Select..."]:
                var.set("Select...")
            else:
                var.set("")
        
        # Reset labels
        self.res_labels["grade"].config(text="---", foreground=self.colors["primary"])
        self.res_labels["confidence"].config(text="---")
        self.res_labels["risk"].config(text="---", foreground=self.colors["warning"])
        
        self.explanation_box.config(state=tk.NORMAL)
        self.explanation_box.delete(1.0, tk.END)
        self.explanation_box.insert(tk.END, "Perform a prediction to see the analysis...")
        self.explanation_box.config(state=tk.DISABLED)

    def _on_exit(self):
        """Safely closes the application."""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit the application?"):
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    # Attempt to set an icon if available (optional)
    # root.iconbitmap("path_to_icon.ico") 
    app = AIStudentPerformancePredictor(root)
    root.mainloop()
