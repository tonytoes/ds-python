import tkinter as tk
from tkinter import filedialog
from customtkinter import CTk, CTkEntry, CTkButton, CTkLabel, CTkOptionMenu, CTkCheckBox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import random
import threading
import time


class ModernDataVisualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Visualization")
        self.root.geometry("1200x800")

        # Initialize data storage
        self.data = []
        self.df = None  # For imported data
        self.columns = []  # Columns in the dataset
        self.numerical_columns = []  # Numerical columns for calculations
        self.categorical_columns = []  # Categorical columns for grouping

        # Sidebar for controls
        self.sidebar = tk.Frame(root, width=200, bg="#2E3440")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        # Main content area
        self.main_content = tk.Frame(root, bg="#3B4252")
        self.main_content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Input field for data
        self.input_label = CTkLabel(self.sidebar, text="Enter Data (comma-separated):", font=("Arial", 12))
        self.input_label.grid(row=0, column=0, padx=10, pady=10)

        self.input_entry = CTkEntry(self.sidebar, width=180, font=("Arial", 12))
        self.input_entry.grid(row=1, column=0, padx=10, pady=10)

        # Dropdown for chart type
        self.chart_type_label = CTkLabel(self.sidebar, text="Select Chart Type:", font=("Arial", 12))
        self.chart_type_label.grid(row=2, column=0, padx=10, pady=10)

        self.chart_type = CTkOptionMenu(self.sidebar, values=["Line Graph", "Bar Chart", "Pie Chart", "Scatter Plot", "Histogram", "Area Chart"])
        self.chart_type.grid(row=3, column=0, padx=10, pady=10)

        # Dropdown for column selection
        self.column_label = CTkLabel(self.sidebar, text="Select Column:", font=("Arial", 12))
        self.column_label.grid(row=4, column=0, padx=10, pady=10)

        self.column_option = CTkOptionMenu(self.sidebar, values=[])
        self.column_option.grid(row=5, column=0, padx=10, pady=10)

        # Button to update visualization
        self.update_button = CTkButton(self.sidebar, text="Update Visualization", command=self.update_visualization)
        self.update_button.grid(row=6, column=0, padx=10, pady=10)

        # Import/Export buttons
        self.import_button = CTkButton(self.sidebar, text="Import Data", command=self.import_data)
        self.import_button.grid(row=7, column=0, padx=10, pady=10)

        self.export_button = CTkButton(self.sidebar, text="Export Graph", command=self.export_graph)
        self.export_button.grid(row=8, column=0, padx=10, pady=10)

        # Real-time data streaming
        self.streaming_var = tk.BooleanVar()
        self.streaming_check = CTkCheckBox(self.sidebar, text="Enable Real-Time Streaming", variable=self.streaming_var)
        self.streaming_check.grid(row=9, column=0, padx=10, pady=10)

        # Machine Learning options
        self.ml_label = CTkLabel(self.sidebar, text="Machine Learning:", font=("Arial", 12))
        self.ml_label.grid(row=10, column=0, padx=10, pady=10)

        self.ml_option = CTkOptionMenu(self.sidebar, values=["None", "Trend Line", "Clustering"])
        self.ml_option.grid(row=11, column=0, padx=10, pady=10)

        # Frame for overall calculations
        self.calculation_frame = tk.Frame(self.main_content, bg="#4C566A")
        self.calculation_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Matplotlib figure for graph
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_content)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add a toolbar for zoom/pan
        self.toolbar_frame = tk.Frame(self.main_content)
        self.toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # Start real-time data streaming thread
        self.streaming_thread = threading.Thread(target=self.real_time_streaming, daemon=True)
        self.streaming_thread.start()

    def update_visualization(self):
        if self.df is None:
            return

        # Get selected column
        selected_column = self.column_option.get()
        if selected_column not in self.df.columns:
            return

        # Get data for the selected column
        self.data = self.df[selected_column]

        # Clear previous plot
        self.ax.clear()

        # Get selected chart type
        chart_type = self.chart_type.get()

        # Plot based on selected chart type
        if chart_type == "Line Graph":
            self.ax.plot(self.data, marker='o', color='b')
            self.ax.set_title(f"Line Graph ({selected_column})")
            self.ax.set_xlabel("Index")
            self.ax.set_ylabel(selected_column)
        elif chart_type == "Bar Chart":
            self.ax.bar(range(len(self.data)), self.data, color='g')
            self.ax.set_title(f"Bar Chart ({selected_column})")
            self.ax.set_xlabel("Index")
            self.ax.set_ylabel(selected_column)
        elif chart_type == "Pie Chart":
            if self.data.dtype == "object":  # For categorical data
                counts = self.data.value_counts()
                self.ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
            else:  # For numerical data
                self.ax.pie(self.data, labels=[f"Data {i+1}" for i in range(len(self.data))], autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
            self.ax.set_title(f"Pie Chart ({selected_column})")
        elif chart_type == "Scatter Plot":
            if len(self.df.columns) > 1:
                x_column = self.df.columns[0]  # Use the first column for X-axis
                self.ax.scatter(self.df[x_column], self.data, color='r')
                self.ax.set_title(f"Scatter Plot ({x_column} vs {selected_column})")
                self.ax.set_xlabel(x_column)
                self.ax.set_ylabel(selected_column)
            else:
                self.ax.scatter(range(len(self.data)), self.data, color='r')
                self.ax.set_title(f"Scatter Plot ({selected_column})")
                self.ax.set_xlabel("Index")
                self.ax.set_ylabel(selected_column)
        elif chart_type == "Histogram":
            self.ax.hist(self.data, bins=10, color='purple', edgecolor='black')
            self.ax.set_title(f"Histogram ({selected_column})")
            self.ax.set_xlabel(selected_column)
            self.ax.set_ylabel("Frequency")
        elif chart_type == "Area Chart":
            self.ax.fill_between(range(len(self.data)), self.data, color='orange', alpha=0.4)
            self.ax.set_title(f"Area Chart ({selected_column})")
            self.ax.set_xlabel("Index")
            self.ax.set_ylabel(selected_column)

        # Apply Machine Learning
        self.apply_machine_learning()

        # Redraw canvas
        self.canvas.draw()

    def apply_machine_learning(self):
        ml_option = self.ml_option.get()
        if ml_option == "Trend Line" and len(self.data) > 1:
            X = np.array(range(len(self.data))).reshape(-1, 1)
            y = np.array(self.data)
            model = LinearRegression()
            model.fit(X, y)
            trend = model.predict(X)
            self.ax.plot(X, trend, color='red', linestyle='--', label="Trend Line")
            self.ax.legend()
        elif ml_option == "Clustering":
          if self.df is not None and len(self.df.columns) >= 2:

              # Take first two numerical columns
              cols = self.numerical_columns[:2]

              X = self.df[cols].values

              kmeans = KMeans(n_clusters=4, random_state=42)
              labels = kmeans.fit_predict(X)
              centroids = kmeans.cluster_centers_

              self.ax.clear()

              self.ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=60)
              self.ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=120, marker='X')

              self.ax.set_title("K-Means Clustering")
              self.ax.set_xlabel(cols[0])
              self.ax.set_ylabel(cols[1])

    def import_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            self.columns = self.df.columns.tolist()
            self.column_option.configure(values=self.columns)
            self.column_option.set(self.columns[0])  # Set default column

            # Detect numerical and categorical columns
            self.numerical_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_columns = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

            # Display overall calculations
            self.show_overall_calculations()

    def show_overall_calculations(self):
        # Clear previous calculations
        for widget in self.calculation_frame.winfo_children():
            widget.destroy()

        # Display calculations for numerical columns
        for col in self.numerical_columns:
            stats = self.df[col].describe()
            stats_text = f"{col}:\n" + "\n".join([f"{key}: {value:.2f}" for key, value in stats.items()])
            label = CTkLabel(self.calculation_frame, text=stats_text, font=("Arial", 12))
            label.pack(side=tk.LEFT, padx=10, pady=10)

        # Display counts for categorical columns
        for col in self.categorical_columns:
            counts = self.df[col].value_counts()
            counts_text = f"{col}:\n" + "\n".join([f"{key}: {value}" for key, value in counts.items()])
            label = CTkLabel(self.calculation_frame, text=counts_text, font=("Arial", 12))
            label.pack(side=tk.LEFT, padx=10, pady=10)

    def export_graph(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("PDF Files", "*.pdf")])
        if file_path:
            self.fig.savefig(file_path)

    def real_time_streaming(self):
        while True:
            if self.streaming_var.get():
                self.data.append(random.randint(1, 100))
                if len(self.data) > 20:  # Limit data points
                    self.data.pop(0)
                self.update_visualization()
            time.sleep(1)

if __name__ == "__main__":
    root = CTk()
    app = ModernDataVisualizationApp(root)
    root.mainloop()

Data = { 'x': [25, 34, 22, 27, 33, 33, 31, 22, 35, 34, 67, 54, 57, 43, 50, 57, 59, 52, 65, 47, 49, 48, 35, 33, 44, 45, 38, 43, 51, 46], 'y': [79, 51, 53, 78, 59, 74, 73, 57, 69, 75, 51, 32, 40, 47, 53, 36, 35, 58, 59, 50, 25, 20, 14, 12, 20, 5, 29, 27, 8, 7]
 }

# 2. Create the DataFrame [cite: 106, 163] 
df = pd.DataFrame(Data, columns=['x', 'y'])
# 3. Apply K-Means Clustering (Example: 3 Clusters) [cite: 157, 158, 164] 
# Change n_clusters to 4 to see the second part of the activity [cite: 183, 184] 

kmeans = KMeans(n_clusters=4).fit(df) 
centroids = kmeans.cluster_centers_

# 4. Output the Centroids [cite: 166, 195] 
print("Centroid Coordinates:") 
print(centroids) 
# 5. Graphical Visualization [cite: 167, 196, 197] 
# Scatter plot of data points colored by cluster [cite: 167]    
plt.scatter(df['x'], df['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5) 

plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, label='Centroids') 

plt.title('K-Means Clustering Visualization') 
plt.xlabel('X Coordinate') 
plt.ylabel('Y Coordinate') 
plt.show()


