import streamlit as st
import pandas as pd
import tempfile
import os
import re
from io import BytesIO
from datetime import time as dtime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

# -------------------------------------------------------------
# Streamlit page config
# -------------------------------------------------------------
st.set_page_config(
    page_title="🎓 UOL-SE Timetable Generator",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# Default Curriculum Data
# -------------------------------------------------------------
DEFAULT_CURRICULUM = {
    "Semester 1": [
        "Applications of Information and Communication Technology",
        "Discrete Structures", 
        "Islamic Studies",
        "Applied Physics",
        "Functional English",
        "Ideology and Constitution of Pakistan",
        "Pre-Calculus I"
    ],
    "Semester 2": [
        "Programming Fundamentals",
        "Digital Logic Design",
        "Expository Writing",
        "Professional Practices",
        "Civics and Community Engagement",
        "Probability & Statistics",
        "Translation of the Holy Quran",
        "Pre-Calculus II"
    ],
    "Semester 3": [
        "Computer Organization and Assembly Language",
        "Object Oriented Programming",
        "Computer Networks",
        "Software Engineering",
        "Calculus and Analytic Geometry",
        "Understanding of the Holy Quran"
    ],
    "Semester 4": [
        "Data Structures",
        "Linear Algebra",
        "Database Systems",
        "Entrepreneurship",
        "Information Security",
        "Introduction to Psychology"
    ],
    "Semester 5": [
        "Operating Systems",
        "Artificial Intelligence",
        "Software Requirement Engineering",
        "Analysis of Algorithms",
        "Software Design and Architecture",
        "Domain Elective 1"
    ],
    "Semester 6": [
        "Multivariable Calculus",
        "Software Construction and Development",
        "Software Quality Engineering",
        "Software Project Management",
        "Domain Elective 2",
        "Domain Elective 3"
    ],
    "Semester 7": [
        "Technical & Business Writing",
        "Parallel and Distributed Computing",
        "Elective Supporting Course",
        "Domain Elective 4",
        "Domain Elective 5",
        "Final Year Project I",
        "Career Development"
    ],
    "Semester 8": [
        "Domain Elective 6",
        "Domain Elective 7",
        "Pakistan Studies",
        "Capstone Project II"
    ]
}

# -------------------------------------------------------------
# Styles
# -------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-header { font-size: 2.2rem; color: #1f77b4; text-align: center; margin-bottom: 1.25rem; font-weight: 800; }
        .section-header { color: #ff7f0e; font-size: 1.25rem; font-weight: 700; margin-top: 1.25rem; margin-bottom: .5rem; }
        .muted { color:#6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def _fmt_time_cell(val):
    """Format an Excel time cell to the exact style used in your sheet (no AM/PM, 12-hour style)."""
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        h, m = val.hour, val.minute
    elif isinstance(val, dtime):
        h, m = val.hour, val.minute
    else:
        return str(val).strip()
    h12 = 12 if h == 0 else (h - 12 if h > 12 else h)
    return f"{h12}:{m:02d}"

def match_subject_to_curriculum(timetable_subject: str, semester_subjects: list) -> str:
    """
    Match a subject from timetable to semester curriculum using strict matching.
    Only matches core subjects, ignores electives. Returns curriculum subject if matched, else None.
    """
    timetable_lower = timetable_subject.lower().strip()
    
    # Skip elective subjects entirely
    if any(word in timetable_lower for word in ['elective', 'domain elective']):
        return None
    
    # Direct exact match first
    for curr_subject in semester_subjects:
        if timetable_lower == curr_subject.lower():
            return curr_subject
    
    # Special case mappings for common variations (only core subjects)
    core_mappings = {
        "discrete structures": "Discrete Structures",
        "qr1": "Discrete Structures", 
        "applied physics": "Applied Physics",
        "natural sciences": "Applied Physics",
        "programming fundamentals": "Programming Fundamentals",
        "object oriented programming": "Object Oriented Programming",
        "oop": "Object Oriented Programming",
        "database systems": "Database Systems",
        "data structures": "Data Structures",
        "linear algebra": "Linear Algebra",
        "computer networks": "Computer Networks",
        "artificial intelligence": "Artificial Intelligence",
        "operating systems": "Operating Systems",
        "information security": "Information Security",
        "software requirement engineering": "Software Requirement Engineering",
        "software design and architecture": "Software Design and Architecture",
        "software construction and development": "Software Construction and Development",
        "software quality engineering": "Software Quality Engineering",
        "software project management": "Software Project Management",
        "technical & business writing": "Technical & Business Writing",
        "technical and business writing": "Technical & Business Writing",
        "ideology and constitution of pakistan": "Ideology and Constitution of Pakistan",
        "functional english": "Functional English",
        "islamic studies": "Islamic Studies",
        "probability & statistics": "Probability & Statistics",
        "translation of the holy quran": "Translation of the Holy Quran",
        "pakistan studies": "Pakistan Studies",
        "pre-calculus": "Pre-Calculus I",
        "pre-calculus i": "Pre-Calculus I",
        "pre-calculus ii": "Pre-Calculus II",
        "multivariable calculus": "Multivariable Calculus",
        "calculus and analytic geometry": "Calculus and Analytic Geometry",
        "qr2": "Calculus and Analytic Geometry",
        "understanding of the holy quran": "Understanding of the Holy Quran",
        "fehm-e-quran": "Understanding of the Holy Quran",
        "expository writing": "Expository Writing",
        "professional practices": "Professional Practices",
        "civics and community engagement": "Civics and Community Engagement",
        "digital logic design": "Digital Logic Design",
        "computer organization and assembly language": "Computer Organization and Assembly Language",
        "entrepreneurship": "Entrepreneurship",
        "introduction to psychology": "Introduction to Psychology",
        "analysis of algorithms": "Analysis of Algorithms",
        "parallel and distributed computing": "Parallel and Distributed Computing",
        "final year project i": "Final Year Project I",
        "career development": "Career Development",
        "capstone project ii": "Capstone Project II"
    }
    
    # Check mappings only if the mapped subject is in the semester
    for key, mapped_subject in core_mappings.items():
        if key in timetable_lower and mapped_subject in semester_subjects:
            return mapped_subject
    
    # Strict matching for "Software Engineering" - only match exact or very close
    if "software engineering" in timetable_lower:
        # Only match if it's exactly "Software Engineering" or very close
        if timetable_lower in ["software engineering", "se"] and "Software Engineering" in semester_subjects:
            return "Software Engineering"
        # Don't match "Advance Software Engineering", "Software Engineering Lab", etc.
        return None
    
    # For other subjects, only do partial matching if curriculum subject is exactly contained
    for curr_subject in semester_subjects:
        curr_lower = curr_subject.lower()
        # Only match if curriculum subject name is exactly contained (not the other way around)
        if curr_lower == timetable_lower:
            return curr_subject
    
    # If no match found, return None (don't include in auto-selection)
    return None

def get_all_curriculum_subjects():
    """Get all subjects from all semesters."""
    all_subjects = []
    for semester_subjects in DEFAULT_CURRICULUM.values():
        all_subjects.extend(semester_subjects)
    return list(set(all_subjects))  # Remove duplicates

# -------------------------------------------------------------
# Extractor — follows your sheet layout
# -------------------------------------------------------------

class SimpleSubjectExtractor:
    def __init__(self, excel_file: str, sheet_name: str | int | None = None):
        self.excel_file = excel_file
        self.sheet_name = sheet_name
        self.df: pd.DataFrame | None = None
        self.extracted_subjects: list[dict] = []
        self.timeslots: list[dict] = []

    def load_excel_file(self) -> bool:
        try:
            self.df = pd.read_excel(self.excel_file, sheet_name=self.sheet_name, header=None)
            # Gentle fill to cope with merged cells; avoids many NAs in day/room/headers
            self.df.iloc[:, 0] = self.df.iloc[:, 0].ffill()  # only forward-fill the day column
            return True
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return False

    def _find_subject_columns_and_times(self):
        assert self.df is not None
        df = self.df

        # Subject columns are every 3rd column starting at index 2, if there is at least one value below row 6
        subject_cols = [
            j for j in range(2, df.shape[1])
            if (j - 2) % 3 == 0 and df.iloc[6:, j].notna().any()
        ]

        timeslots = []
        for idx, j in enumerate(subject_cols):
            header_row = 4 if idx % 2 == 0 else 5  # rows 5/6 in Excel (0-indexed 4/5)
            start = _fmt_time_cell(df.iat[header_row, j - 1])
            end = _fmt_time_cell(df.iat[header_row, j + 1])
            label = f"{start}-{end}" if start and end else (start or end or "")
            timeslots.append({
                "slot_index": idx,
                "col": j,
                "header_row": header_row,
                "start": start,
                "end": end,
                "label": label,
            })
        self.timeslots = timeslots
        return subject_cols, timeslots

    def get_timeslot_labels(self) -> list[str]:
        """Return a list of time slot labels from the currently loaded timetable."""
        if not self.timeslots and self.df is not None:
            self._find_subject_columns_and_times()
        return [slot["label"] for slot in self.timeslots]

    def extract_subjects_from_timetable(self) -> list[dict]:
        if not self.load_excel_file():
            return []
        df = self.df
        assert df is not None

        _, timeslots = self._find_subject_columns_and_times()

        # Forward-fill day names in first column (already done in load, safe to re-ensure)
        day_series = df.iloc[:, 0].ffill().apply(lambda x: str(x).strip() if pd.notna(x) else x)

        subjects: list[dict] = []
        for r in range(6, df.shape[0]):  # start from Excel row 7 (0-indexed 6)
            day = day_series.iat[r]
            room = df.iat[r, 1] if df.shape[1] > 1 else None
            if pd.isna(day) or pd.isna(room) or str(room).strip() == "":
                continue
            room = str(room).strip()

            for slot in timeslots:
                c = slot["col"]
                if c >= df.shape[1]:
                    continue
                cell = df.iat[r, c]
                if pd.isna(cell):
                    continue

                text = str(cell).strip()
                if not text:
                    continue

                # Prefer "Subject (X)" form
                m = re.search(r"^(.+?)\s*\(([A-Za-z]+)\)\s*$", text)
                if m:
                    subject_name, section = m.group(1).strip(), m.group(2).strip()
                else:
                    # Allow entries like "Data Structures Lab U" (no parentheses)
                    tokens = re.split(r"\s+", text)
                    possible_section = tokens[-1].upper() if tokens else ""
                    if possible_section in ["U", "V", "W", "X", "Y", "S", "T", "R", "Q", "P", "O", "N", "M", "L", "K", "J", "I", "H", "G", "F", "E", "D", "C", "B", "A"]:
                        subject_name = " ".join(tokens[:-1]).strip()
                        section = possible_section
                        if not subject_name:
                            # If subject_name is empty after split, fall back to the whole cell (skip)
                            continue
                    else:
                        # Skip dummy department labels like "CS&IT" that don't carry a section
                        continue

                is_lab = "lab" in subject_name.lower()

                # Record this slot
                subjects.append({
                    "day": day,
                    "room": room,
                    "subject": subject_name,
                    "section": section,
                    "time": slot["label"],
                    "slot_index": slot["slot_index"],
                    "is_lab": is_lab,
                })

                # Labs span 2 consecutive slots — add the next one too if it exists
                if is_lab:
                    next_slot_idx = slot["slot_index"] + 1
                    if next_slot_idx < len(timeslots):
                        next_slot_label = timeslots[next_slot_idx]["label"]
                        subjects.append({
                            "day": day,
                            "room": room,
                            "subject": subject_name,
                            "section": section,
                            "time": next_slot_label,
                            "slot_index": next_slot_idx,
                            "is_lab": is_lab,
                        })

        self.extracted_subjects = subjects
        return subjects

    def get_all_subjects(self) -> list[str]:
        """Get all unique subjects from the timetable."""
        return sorted(list(set(s.get('subject', '') for s in self.extracted_subjects if s.get('subject'))))

    def get_subjects_for_section(self, target_section: str) -> list[str]:
        """Get unique subjects for a specific section."""
        section_subjects = [s for s in self.extracted_subjects if s.get("section") == target_section]
        return sorted(list(set(s.get('subject', '') for s in section_subjects if s.get('subject'))))

    def create_timetable_for_section_subjects(self, section: str, selected_subjects: list[str]) -> list[dict]:
        return [
            s for s in self.extracted_subjects
            if s.get("section") == section and s.get("subject") in selected_subjects
        ]

    def get_sections_summary(self) -> dict:
        """
        Return dict:
        {
          "A": {"count": int, "subjects": [...], "days": [...], "labs": int, "regular_classes": int},
          ...
        }
        """
        summary: dict[str, dict] = {}
        for s in self.extracted_subjects:
            sec = s.get("section", "") or "?"
            if sec not in summary:
                summary[sec] = {
                    "count": 0,
                    "subjects": set(),
                    "days": set(),
                    "labs": 0,
                    "regular_classes": 0,
                }
            summary[sec]["count"] += 1
            summary[sec]["subjects"].add(s.get("subject", ""))
            summary[sec]["days"].add((s.get("day", "") or "").strip())
            if s.get("is_lab"):
                summary[sec]["labs"] += 1
            else:
                summary[sec]["regular_classes"] += 1

        # Convert sets to sorted lists for display
        for sec, info in summary.items():
            info["subjects"] = sorted(x for x in info["subjects"] if x)
            info["days"] = sorted(x for x in info["days"] if x)

        return summary

# -------------------------------------------------------------
# Visualization
# -------------------------------------------------------------
def create_visual_timetable(schedule_data: list[dict], section_name: str, times: list[str]):
    if not schedule_data:
        return None

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    times = [t for t in times if t]
    fig, ax = plt.subplots(figsize=(16, 10))

    cell_w, cell_h = 2.5, 1.0
    plt.subplots_adjust(top=0.88, bottom=0.06, left=0.08, right=0.98)
    fig.suptitle(f"Timetable for Section {section_name}", fontsize=18, fontweight='bold', y=0.96)

    ax.set_xlim(0, len(days) * cell_w)
    ax.set_ylim(0, len(times) * cell_h)

    for i, day in enumerate(days):
        x = i * cell_w + cell_w / 2
        y = len(times) * cell_h + 0.15
        ax.text(x, y, day, ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax.axvline(x=i * cell_w, color='black', linewidth=1)

    for i, label in enumerate(times):
        y = (len(times) - i - 1) * cell_h + cell_h / 2
        ax.text(-0.5, y, label, ha='right', va='center', fontweight='bold', fontsize=10)
        ax.axhline(y=(len(times) - i) * cell_h, color='black', linewidth=1)

    ax.axvline(x=len(days) * cell_w, color='black', linewidth=2)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=len(times) * cell_h, color='black', linewidth=2)
    ax.axvline(x=0, color='black', linewidth=2)

    day_idx = {d: i for i, d in enumerate(days)}
    time_idx = {t: i for i, t in enumerate(times)}
    colors = {
        'regular': '#E3F2FD',
        'lab': '#FFECEC',
        'border_regular': '#1976D2',
        'border_lab': '#D32F2F',
    }

    for s in schedule_data:
        d, t = s.get('day', '').strip(), s.get('time')
        if d not in day_idx or t not in time_idx:
            continue
        dx, tx = day_idx[d], time_idx[t]
        x, y = dx * cell_w, (len(times) - tx - 1) * cell_h
        is_lab = bool(s.get('is_lab'))
        fill, border = (colors['lab'], colors['border_lab']) if is_lab else (colors['regular'], colors['border_regular'])
        rect = Rectangle((x, y), cell_w, cell_h, facecolor=fill, edgecolor=border, linewidth=2)
        ax.add_patch(rect)

        subj = s.get('subject', '')
        subj_txt = (subj[:22] + '…') if len(subj) > 22 else subj
        sec, room = s.get('section', ''), s.get('room', '')
        ax.text(x + cell_w/2, y + cell_h*0.68, subj_txt, ha='center', va='center', fontweight='bold', fontsize=9, wrap=True)
        ax.text(x + cell_w/2, y + cell_h*0.40, f"({sec})", ha='center', va='center', fontsize=8)
        ax.text(x + cell_w/2, y + cell_h*0.14, room, ha='center', va='center', fontsize=7, style='italic')

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ['top','right','bottom','left']:
        ax.spines[sp].set_visible(False)

    # Legend moved to top-left (so it doesn't block subjects)
    # regular_patch = patches.Patch(color=colors['regular'], label='📖 Regular Classes')
    # lab_patch = patches.Patch(color=colors['lab'], label='🧪 Lab Classes')
    # ax.legend(handles=[regular_patch, lab_patch], loc='upper left', bbox_to_anchor=(0.005, 0.98))

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

# -------------------------------------------------------------
# Session state
# -------------------------------------------------------------
def init_session_state():
    if 'extractor' not in st.session_state:
        st.session_state.extractor = None
    if 'subjects_data' not in st.session_state:
        st.session_state.subjects_data = []
    if 'sections_summary' not in st.session_state:
        st.session_state.sections_summary = {}
    if 'timeslot_labels' not in st.session_state:
        st.session_state.timeslot_labels = []
    if 'uploaded_file_processed' not in st.session_state:
        st.session_state.uploaded_file_processed = False

# -------------------------------------------------------------
# File processing
# -------------------------------------------------------------
def process_uploaded_file(uploaded_file, sheet_name=None):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        with st.spinner("🔍 Processing timetable…"):
            extractor = SimpleSubjectExtractor(tmp_path, sheet_name=sheet_name or 0)
            subjects_data = extractor.extract_subjects_from_timetable()
            sections_summary = extractor.get_sections_summary()
            timeslot_labels = extractor.get_timeslot_labels()
        os.unlink(tmp_path)
        st.session_state.extractor = extractor
        st.session_state.subjects_data = subjects_data
        st.session_state.sections_summary = sections_summary
        st.session_state.timeslot_labels = timeslot_labels
        st.session_state.uploaded_file_processed = True
        return True, len(subjects_data)
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        return False, 0

# -------------------------------------------------------------
# UI blocks
# -------------------------------------------------------------
def display_file_upload():
    st.markdown('<div class="main-header">🎓 UOL-SE Timetable Generator</div>', unsafe_allow_html=True)
    st.markdown(
        """
        ### 📂 Upload SE Timetable
        Upload your Excel timetable to generate a visual schedule:
        - ⏱️ Reads time slots exactly from **rows 5 & 6**
        - 🏫 Uses **rooms from column B**
        - 🧪 Detects labs (based on subject name)
        - 📚 Smart semester-based filtering with manual override
        - 🖼️ Generates a downloadable image
        """
    )
    up = st.file_uploader("Choose your Excel timetable", type=["xlsx", "xls"], help="Upload timetable Excel file")
    if up is not None:
        st.info(f"📄 File uploaded: {up.name}")
        if st.button("🚀 Process Timetable", type="primary"):
            ok, n = process_uploaded_file(up)
            if ok:
                st.success(f"✅ Processed {n} subject entries.")
                st.rerun()

def display_section_selector():
    if not st.session_state.sections_summary:
        return None
    
    st.markdown('<div class="section-header">🎯 Choose Your Section</div>', unsafe_allow_html=True)

    # Exclude '?' pseudo-section if it exists
    sections = [""] + sorted(k for k in st.session_state.sections_summary.keys() if k != "?")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_section = st.selectbox("Select your section:", sections, format_func=lambda x: "All Sections" if x == "" else f"Section {x}")
        
    with col2:
        if selected_section:
            info = st.session_state.sections_summary[selected_section]
            st.info(
                f"""
**Section {selected_section} Overview**  
• 📚 {info['count']} total classes  
• 📖 {info['regular_classes']} regular  
• 🧪 {info['labs']} labs  
• 📅 Days: {', '.join(info['days'])}
                """
            )
        else:
            total_subjects = len(st.session_state.extractor.get_all_subjects()) if st.session_state.extractor else 0
            st.info(
                f"""
**All Sections Overview**  
• 📚 {total_subjects} unique subjects  
• 📊 {len(st.session_state.sections_summary)} sections available  
• 🎯 Select a section to filter subjects
                """
            )
    
    return selected_section if selected_section != "" else None

def display_semester_selector():
    st.markdown('<div class="section-header">📚 Choose Your Semester (Optional)</div>', unsafe_allow_html=True)
    
    semesters = [""] + list(DEFAULT_CURRICULUM.keys())
    
    selected_semester = st.selectbox(
        "Select semester for auto-selection:", 
        semesters,
        format_func=lambda x: "No Semester Filter" if x == "" else x,
        help="This will automatically check subjects for the selected semester, but you can still manually adjust."
    )
    
    if selected_semester:
        subjects = DEFAULT_CURRICULUM[selected_semester]
        st.info(
            f"""
**{selected_semester} Auto-Selection**  
• 📚 Will auto-check {len(subjects)} subjects  
• ✅ You can still manually add/remove subjects  
• 🎯 Helps you quickly select most relevant subjects
            """
        )
    
    return selected_semester if selected_semester != "" else None

def get_subjects_for_semester_matching(semester: str, available_subjects: list[str]) -> list[str]:
    """Get subjects from available list that match the semester curriculum (core subjects only)."""
    if not semester:
        return []
    
    semester_subjects = DEFAULT_CURRICULUM.get(semester, [])
    matched_subjects = []
    
    for subject in available_subjects:
        # Try to match this subject to semester curriculum (core subjects only)
        matched_curriculum = match_subject_to_curriculum(subject, semester_subjects)
        if matched_curriculum:  # Only add if there's a valid match (not None)
            matched_subjects.append(subject)
    
    return matched_subjects

def display_subject_selector(selected_section: str, selected_semester: str):
    if not st.session_state.extractor:
        return []
    
    st.markdown('<div class="section-header">📚 Select Your Subjects</div>', unsafe_allow_html=True)
    
    # Get available subjects based on section selection
    if selected_section:
        available_subjects = st.session_state.extractor.get_subjects_for_section(selected_section)
        context = f"Section {selected_section}"
    else:
        available_subjects = st.session_state.extractor.get_all_subjects()
        context = "All Sections"
    
    if not available_subjects:
        st.warning(f"⚠️ No subjects found for {context}")
        return []
    
    # Auto-select subjects based on semester
    auto_selected = []
    if selected_semester:
        auto_selected = get_subjects_for_semester_matching(selected_semester, available_subjects)
        st.success(f"🎯 Auto-selected {len(auto_selected)} subjects for {selected_semester}")
    
    # Create session state key for this selection
    key_suffix = f"{selected_section or 'all'}_{selected_semester or 'none'}"
    session_key = f"selected_subjects_{key_suffix}"
    
    # Initialize or update session state based on auto-selection
    if session_key not in st.session_state or selected_semester:
        st.session_state[session_key] = auto_selected.copy()
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Select All", key=f"select_all_{key_suffix}"):
            st.session_state[session_key] = available_subjects.copy()
            st.rerun()
    with col2:
        if st.button("❌ Clear All", key=f"clear_all_{key_suffix}"):
            st.session_state[session_key] = []
            st.rerun()
    with col3:
        if selected_semester and st.button("🎯 Auto-Select Semester", key=f"auto_select_{key_suffix}"):
            st.session_state[session_key] = auto_selected.copy()
            st.rerun()
    
    # Display subject checkboxes
    chosen = []
    cols = st.columns(2)
    
    for i, subject in enumerate(available_subjects):
        # Check if subject has labs
        section_subjects = st.session_state.extractor.get_subjects_for_section(selected_section) if selected_section else available_subjects
        is_lab = any(s.get('is_lab', False) for s in st.session_state.subjects_data 
                    if s.get('subject') == subject and (not selected_section or s.get('section') == selected_section))
        
        icon = "🧪" if is_lab else "📖"
        
        # Check if this subject was auto-selected
        is_auto_selected = subject in auto_selected
        auto_indicator = " 🎯" if is_auto_selected and selected_semester else ""
        
        default_checked = subject in st.session_state[session_key]
        
        with cols[i % 2]:
            if st.checkbox(
                f"{icon} {subject}{auto_indicator}", 
                value=default_checked, 
                key=f"subj_{key_suffix}_{i}",
                help=f"Auto-selected for {selected_semester}" if is_auto_selected and selected_semester else None
            ):
                chosen.append(subject)
    
    # Update session state
    st.session_state[session_key] = chosen
    return chosen

def display_visual_timetable(selected_section: str, selected_subjects: list[str]):
    if not selected_subjects or not selected_section:
        if not selected_section:
            st.warning("⚠️ Please select a section to generate timetable")
        else:
            st.warning("⚠️ Please select at least one subject to generate timetable")
        return
    
    st.markdown('<div class="section-header">🗓️ Your Visual Timetable</div>', unsafe_allow_html=True)
    
    filtered = st.session_state.extractor.create_timetable_for_section_subjects(selected_section, selected_subjects)
    if not filtered:
        st.warning("⚠️ No classes found for selected subjects in this section")
        return
    
    day_order = {d: i for i, d in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])}
    filtered = sorted(filtered, key=lambda x: (day_order.get(x['day'].strip(), 99), x.get('slot_index', 0)))
    
    img = create_visual_timetable(filtered, selected_section, st.session_state.timeslot_labels)
    if img:
        st.image(img, caption=f"Timetable for Section {selected_section}", use_column_width=True)
        st.download_button(
            label="📷 Download Timetable Image",
            data=img.getvalue(),
            file_name=f"timetable_section_{selected_section}.png",
            mime="image/png",
        )

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    init_session_state()
    
    with st.sidebar:
        st.markdown("### 🎓 Navigation")
        if st.session_state.uploaded_file_processed:
            st.success("✅ File processed!")
            if st.button("🔄 Upload New File"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
            if st.session_state.sections_summary:
                st.markdown("### 📊 Quick Stats")
                total_subjects = len(st.session_state.extractor.get_all_subjects()) if st.session_state.extractor else 0
                st.metric("Total Unique Subjects", total_subjects)
                st.metric("Available Sections", len(st.session_state.sections_summary))
                st.markdown("### 🎯 Available Sections")
                for sec, info in sorted(st.session_state.sections_summary.items()):
                    if sec != "?":
                        st.write(f"**Section {sec}:** {len(info['subjects'])} subjects")
                    
                st.markdown("### 📚 Available Semesters")
                for semester in DEFAULT_CURRICULUM.keys():
                    subject_count = len(DEFAULT_CURRICULUM[semester])
                    st.write(f"**{semester}:** {subject_count} subjects")
        else:
            st.info("📂 Upload timetable to start")

    if not st.session_state.uploaded_file_processed:
        display_file_upload()
    else:
        # Step 1: Section Selection (optional)
        selected_section = display_section_selector()
        
        # Step 2: Semester Selection (optional)
        selected_semester = display_semester_selector()
        
        # Step 3: Subject Selection (always show, filtered by section/semester)
        selected_subjects = display_subject_selector(selected_section, selected_semester)
        
        # Step 4: Generate Timetable (only if section and subjects are selected)
        if selected_subjects:
            display_visual_timetable(selected_section, selected_subjects)
        elif selected_section is None:
            st.info("💡 **Next Steps:** Select a section to filter subjects and generate your timetable")
        elif not selected_subjects:
            st.info("💡 **Next Steps:** Select at least one subject to generate your timetable")

if __name__ == "__main__":
    main()