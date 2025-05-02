from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(200, 10, "Churn Analysis Report", ln=True, align="C")
        self.ln(10)
    
    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True, align="L")
        self.ln(5)
    
    def chapter_body(self, body):
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 5, body)
        self.ln(5)
    
    def add_image(self, image_path, w=150):
        if os.path.exists(image_path):
            self.image(image_path, x=30, w=w)
            self.ln(5)
        else:
            self.chapter_body(f"Image not found: {image_path}")

# Create PDF document
pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Report Content
pdf.chapter_title("1. Overview")
pdf.chapter_body("This report provides insights from the churn analysis, including key metrics and visualizations.")

pdf.chapter_title("2. Key Metrics")
metrics_text = f"""
- Accuracy: 85%
- Precision: 78%
- Recall: 82%
- F1 Score: 80%
"""
pdf.chapter_body(metrics_text)

# Add Plots
pdf.chapter_title("3. Data Visualizations")
plot_files = [
    "plots/churn_distribution.png", 
    "plots/tenure_vs_churn.png", 
    "plots/monthly_charges_vs_churn.png", 
    "plots/confusion_matrix.png", 
    "plots/roc_curve.png", 
    "plots/precision_recall_curve.png", 
    "plots/shap_summary.png",
    "plots/shap_dependence_monthlycharges.png"
]

for plot in plot_files:
    pdf.add_image(plot)

# Save PDF
pdf.output("Churn_Analysis_Report.pdf")
print("Report saved as Churn_Analysis_Report.pdf")
