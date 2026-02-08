# 🎓 UOL-SE Timetable Generator

This tool automates the process of generating personalized class schedules for UOL Software Engineering students. It parses raw, unstructured Excel timetables released by the university and converts them into clean, visual, and conflict-free PNG schedules.


![Timetable Demo](initial%20timetables%20generated/timetable_section_S.png)

## 🚀 Overview


The University of Lahore (UOL) releases class schedules in a consolidated Excel format where multiple sections and semesters are mixed within single sheets. Students manually enroll in courses, often resulting in cross-section or cross-semester schedules that are tedious to map manually.

This project solves typically time-consuming timetable conflicts by:
1.  **Parsing** the raw Excel file using `pandas` and `openpyxl`.
2.  **Extracting** subject and section information via Regex pattern matching.
3.  **Visualizing** the schedule as a high-quality PNG using `matplotlib`.
4.  **Handling Exceptions**: Allowing students to select specific subjects from different sections (e.g., repeating a course or resolving a clash).

## ✨ Key Features

-   **Excel Parsing Engine**: Reads complex, merged-cell Excel sheets to extract time slots, room numbers, and course details.
-   **Regex-Based Identification**: robustly identifies subjects and sections (e.g., "Data Structures (Section A)" vs "Data Structures Lab") despite inconsistent naming conventions.
-   **Smart Time Slot Handling**: converting non-standard time ranges (e.g., 8:30-10:00) into a standardized grid system.
-   **Interactive Web UI**: Built with **Streamlit** for easy file uploads and interactive filtering.
-   **Customizable Output**: Generates download-ready PNG timetables with color-coded theory and lab slots.

## 🛠 Tech Stack

-   **Python 3.10+**
-   **Streamlit**: Frontend UI.
-   **Pandas**: Data manipulation and cleaning.
-   **Matplotlib**: Grid generation and plotting.
-   **OpenPyXL**: Excel file reading.

## 📦 Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/timetbale-generator.git
    cd timetbale-generator
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

4.  **Generate Your Timetable**
    -   Upload the official UOL Timetable Excel file.
    -   Select your primary **Section** (e.g., Section A).
    -   (Optional) Select your **Semester** to auto-populate core courses.
    -   Add/Remove specific subjects if you have irregular enrollment.
    -   Click **Download Timetable Image**.

## 🤝 Contribution

Contributions are welcome! If you find edge cases in the Excel parsing logic or want to improve the visualization:
1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

