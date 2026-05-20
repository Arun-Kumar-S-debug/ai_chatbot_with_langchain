import streamlit as st
import cv2
import numpy as np
import pandas as pd
import base64
import json
import os
import time
from datetime import datetime, date
from PIL import Image
import io
from openai import OpenAI
from dotenv import load_dotenv
import pickle

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Paths ─────────────────────────────────────────────────────────────────────
FACES_DIR        = "registered_faces"
ATTENDANCE_FILE  = "attendance.csv"
FACE_DB_FILE     = "face_database.pkl"

os.makedirs(FACES_DIR, exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

  html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: #0a0e1a;
    color: #00f5ff;
  }
  .main { background-color: #0a0e1a; }
  .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a0e1a 100%); }

  h1, h2, h3 {
    font-family: 'Orbitron', monospace;
    color: #00f5ff;
    text-shadow: 0 0 20px #00f5ff55;
  }

  .metric-card {
    background: linear-gradient(135deg, #0d1b2a, #1a2a4a);
    border: 1px solid #00f5ff33;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 20px #00f5ff15;
  }
  .metric-number {
    font-family: 'Orbitron', monospace;
    font-size: 2.5rem;
    font-weight: 900;
    color: #00f5ff;
  }
  .metric-label { font-size: 0.8rem; color: #7ba7bc; }

  .success-box {
    background: linear-gradient(135deg, #0d2a1a, #1a4a2a);
    border: 1px solid #00ff8844;
    border-radius: 8px;
    padding: 16px;
    color: #00ff88;
    text-align: center;
    font-family: 'Orbitron', monospace;
  }
  .warning-box {
    background: linear-gradient(135deg, #2a1a0d, #4a2a1a);
    border: 1px solid #ff880044;
    border-radius: 8px;
    padding: 16px;
    color: #ff8800;
    text-align: center;
  }
  .error-box {
    background: linear-gradient(135deg, #2a0d0d, #4a1a1a);
    border: 1px solid #ff004444;
    border-radius: 8px;
    padding: 16px;
    color: #ff0044;
    text-align: center;
  }

  .stButton > button {
    background: linear-gradient(135deg, #003344, #005566);
    color: #00f5ff;
    border: 1px solid #00f5ff55;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
    transition: all 0.3s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #00f5ff, #0088aa);
    color: #0a0e1a;
    border-color: #00f5ff;
    box-shadow: 0 0 20px #00f5ff55;
  }

  .stTextInput > div > div > input,
  .stSelectbox > div > div > div {
    background: #0d1b2a;
    color: #00f5ff;
    border: 1px solid #00f5ff33;
    border-radius: 4px;
  }

  .scan-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    color: #00f5ff99;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_face_database() -> dict:
    if os.path.exists(FACE_DB_FILE):
        with open(FACE_DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_face_database(db: dict):
    with open(FACE_DB_FILE, "wb") as f:
        pickle.dump(db, f)


def load_attendance() -> pd.DataFrame:
    if os.path.exists(ATTENDANCE_FILE):
        return pd.read_csv(ATTENDANCE_FILE)
    return pd.DataFrame(columns=["Name", "Student_ID", "Date", "Time", "Status"])


def save_attendance(df: pd.DataFrame):
    df.to_csv(ATTENDANCE_FILE, index=False)


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# OpenAI Vision helpers
# ══════════════════════════════════════════════════════════════════════════════

def describe_face(b64_img: str) -> str:
    """Get a detailed facial description from GPT-4o Vision."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe the person's face in this image in great detail for identity matching. "
                            "Include: face shape, skin tone, eye color/shape, eyebrow shape, nose shape, "
                            "lip shape, hair color/style, any distinctive features (scars, moles, glasses, beard, etc.), "
                            "approximate age range, and gender presentation. "
                            "Be very specific and precise. Output ONLY the description, no preamble."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "high"}
                    }
                ]
            }],
            max_tokens=400
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def image_path_to_base64(path: str) -> str:
    """Load a saved image file and return base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def compare_faces(b64_live: str, face_db: dict) -> dict:
    """
    Compare live photo directly against each registered photo using GPT-4o Vision.
    Sends BOTH images side-by-side so GPT-4o does real visual comparison.
    """
    if not face_db:
        return {"matched": False, "name": None, "student_id": None,
                "confidence": 0, "reason": "No registered faces in database."}

    best = {"matched": False, "name": None, "student_id": None,
            "confidence": 0, "reason": "No match found."}

    for key, info in face_db.items():
        img_path = info.get("image_path", "")
        if not img_path or not os.path.exists(img_path):
            continue
        try:
            b64_registered = image_path_to_base64(img_path)
        except Exception:
            continue

        prompt = (
            "You are a face verification system.\n"
            "Image 1 = REGISTERED photo of a known person.\n"
            "Image 2 = LIVE photo of the person trying to mark attendance.\n\n"
            "Compare the two faces carefully:\n"
            "- Look at face shape, eyes, nose, mouth, skin tone, hair, and distinctive features.\n"
            "- Ignore lighting differences, angles, and image quality.\n"
            "- Focus only on whether these are the SAME person.\n\n"
            "Respond ONLY with valid JSON (no markdown, no extra text):\n"
            "{\"same_person\": true/false, \"confidence\": 0-100, \"reason\": \"one sentence\"}"
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64_registered}", "detail": "high"}},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64_live}", "detail": "high"}},
                    ]
                }],
                max_tokens=120
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)

            confidence = result.get("confidence", 0)
            if result.get("same_person") and confidence > best["confidence"]:
                best = {
                    "matched": True,
                    "name": info["name"],
                    "student_id": info["student_id"],
                    "confidence": confidence,
                    "reason": result.get("reason", "")
                }
        except Exception:
            continue

    return best


def detect_face_in_image(b64_img: str) -> bool:
    """Quick check: does this image contain a face?"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Does this image contain a human face? Answer only YES or NO."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "low"}}
                ]
            }],
            max_tokens=5
        )
        return "YES" in resp.choices[0].message.content.strip().upper()
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Attendance logic
# ══════════════════════════════════════════════════════════════════════════════

def mark_attendance(name: str, student_id: str):
    df = load_attendance()
    today = str(date.today())
    already = df[(df["Name"] == name) & (df["Student_ID"] == student_id) & (df["Date"] == today)]
    if not already.empty:
        return False, f"{name} already marked present today."

    now = datetime.now()
    new_row = {
        "Name": name,
        "Student_ID": student_id,
        "Date": today,
        "Time": now.strftime("%H:%M:%S"),
        "Status": "Present"
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_attendance(df)
    return True, f"Attendance marked at {now.strftime('%H:%M:%S')}"


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("""
<div style='text-align:center; padding: 10px 0 20px;'>
  <div style='font-family:Orbitron,monospace; font-size:1.1rem; color:#00f5ff; letter-spacing:3px;'>
    ◈ SMART ATTEND ◈
  </div>
  <div style='font-size:0.65rem; color:#7ba7bc; letter-spacing:5px; margin-top:4px;'>
    FACIAL RECOGNITION SYSTEM
  </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "NAVIGATE",
    ["🏠 Dashboard", "📷 Mark Attendance", "👤 Register Face", "📊 View Records", "⚙️ Manage Database"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
face_db = load_face_database()
att_df  = load_attendance()

today_count = len(att_df[att_df["Date"] == str(date.today())]) if not att_df.empty else 0

st.sidebar.markdown(f"""
<div class='metric-card' style='margin-bottom:10px;'>
  <div class='metric-number'>{len(face_db)}</div>
  <div class='metric-label'>REGISTERED FACES</div>
</div>
<div class='metric-card'>
  <div class='metric-number'>{today_count}</div>
  <div class='metric-label'>PRESENT TODAY</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.markdown("<h1>◈ SMART ATTENDANCE SYSTEM</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:#7ba7bc; letter-spacing:2px; margin-bottom:30px;'>"
        f"DATE: {datetime.now().strftime('%A, %B %d %Y')} &nbsp;|&nbsp; "
        f"TIME: {datetime.now().strftime('%H:%M')}</div>",
        unsafe_allow_html=True
    )

    total_reg  = len(face_db)
    total_sess = len(att_df["Date"].unique()) if not att_df.empty else 0
    pct        = round((today_count / total_reg * 100) if total_reg else 0, 1)

    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl in [
        (c1, today_count, "PRESENT TODAY"),
        (c2, total_reg,   "REGISTERED"),
        (c3, total_sess,  "SESSIONS"),
        (c4, f"{pct}%",   "ATTENDANCE RATE"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-number'>{num}</div>
          <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if not att_df.empty:
        st.markdown("### 📋 RECENT ACTIVITY")
        recent = att_df.sort_values(["Date", "Time"], ascending=False).head(10)
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div class='warning-box'>
          ⚡ No attendance records yet. Register faces and start marking attendance.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 QUICK START")
    q1, q2, q3 = st.columns(3)
    q1.markdown("**Step 1 · Register**\nGo to *Register Face* and add students with their photo.")
    q2.markdown("**Step 2 · Scan**\nGo to *Mark Attendance* and use your webcam or upload a photo.")
    q3.markdown("**Step 3 · Review**\nGo to *View Records* to export and analyse attendance data.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Mark Attendance
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📷 Mark Attendance":
    st.markdown("<h1>📷 MARK ATTENDANCE</h1>", unsafe_allow_html=True)

    if len(face_db) == 0:
        st.markdown("""<div class='error-box'>⚠ No faces registered yet. Please register faces first.</div>""",
                    unsafe_allow_html=True)
        st.stop()

    st.markdown(
        f"<div style='color:#7ba7bc;'>System has <b style='color:#00f5ff;'>{len(face_db)}</b> "
        f"registered face(s).</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    mode = st.radio("INPUT MODE", ["📸 Webcam", "🖼️ Upload Image"], horizontal=True)

    # Clear stored image when mode changes
    if st.session_state.get("last_att_mode") != mode:
        st.session_state.att_image_b64 = None
        st.session_state.last_att_mode = mode

    if mode == "📸 Webcam":
        st.markdown("<div class='scan-label'>▶ LIVE CAPTURE — click the camera button, then press Identify</div>", unsafe_allow_html=True)
        cam_img = st.camera_input("", label_visibility="collapsed")
        if cam_img is not None:
            pil_img = Image.open(cam_img).convert("RGB")
            st.session_state.att_image_b64 = pil_to_base64(pil_img)
    else:
        uploaded = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            st.session_state.att_image_b64 = pil_to_base64(pil_img)
            st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    # Show preview + status
    if st.session_state.get("att_image_b64"):
        st.success("✅ Image captured — ready to identify!")
    else:
        st.info("📷 Capture or upload a photo first, then click Identify.")

    live_image = st.session_state.get("att_image_b64")

    if st.button("🔍 IDENTIFY & MARK ATTENDANCE", use_container_width=True, disabled=(not live_image)):
        with st.spinner("🔬 Scanning face with AI Vision... comparing against all registered photos..."):
            current_image = st.session_state.get("att_image_b64")

            if not detect_face_in_image(current_image):
                st.markdown(
                    "<div class='error-box'>❌ No human face detected. Please try again with a clearer photo.</div>",
                    unsafe_allow_html=True
                )
            else:
                result = compare_faces(current_image, face_db)

                if result.get("matched") and result.get("confidence", 0) >= 60:
                    name       = result["name"]
                    student_id = result["student_id"]
                    confidence = result["confidence"]
                    marked, msg = mark_attendance(name, student_id)

                    st.markdown(f"""
                    <div class='success-box'>
                      ✅ IDENTITY CONFIRMED<br>
                      <span style='font-size:1.4rem;'>{name}</span><br>
                      ID: {student_id} &nbsp;|&nbsp; Confidence: {confidence}%<br><br>
                      {msg}
                    </div>""", unsafe_allow_html=True)

                    if not marked:
                        st.info(f"ℹ️ {name} was already marked present today.")
                else:
                    reason = result.get("reason", "Low confidence or no match found.")
                    st.markdown(f"""
                    <div class='error-box'>
                      ❌ FACE NOT RECOGNIZED<br>
                      <small>{reason}</small><br>
                      Confidence: {result.get('confidence', 0)}%
                    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Register Face
# ══════════════════════════════════════════════════════════════════════════════

elif page == "👤 Register Face":
    st.markdown("<h1>👤 REGISTER NEW FACE</h1>", unsafe_allow_html=True)

    # NOTE: camera_input cannot live inside st.form — use plain widgets instead
    c1, c2 = st.columns(2)
    name       = c1.text_input("Full Name",             placeholder="e.g. Priya Sharma")
    student_id = c2.text_input("Student / Employee ID",  placeholder="e.g. CS2024001")
    dept       = c1.text_input("Department / Class",     placeholder="e.g. CSE - B")
    email      = c2.text_input("Email (optional)",       placeholder="e.g. priya@college.edu")

    st.markdown("---")
    st.markdown("<div class='scan-label'>▶ FACE CAPTURE</div>", unsafe_allow_html=True)

    reg_mode = st.radio("Capture method", ["📸 Webcam", "🖼️ Upload"], horizontal=True)
    if st.session_state.get("last_reg_mode") != reg_mode:
        st.session_state.reg_image_b64 = None
        st.session_state.last_reg_mode = reg_mode

    if reg_mode == "📸 Webcam":
        cam_pic = st.camera_input("Take photo", label_visibility="collapsed")
        if cam_pic is not None:
            pil_preview = Image.open(cam_pic).convert("RGB")
            st.session_state.reg_image_b64 = pil_to_base64(pil_preview)
    else:
        reg_file = st.file_uploader("Upload clear face photo", type=["jpg", "jpeg", "png"])
        if reg_file is not None:
            pil_preview = Image.open(reg_file).convert("RGB")
            st.session_state.reg_image_b64 = pil_to_base64(pil_preview)
            st.image(pil_preview, caption="Preview", width=250)

    if st.session_state.get("reg_image_b64"):
        st.success("✅ Photo ready — fill in details above then click Register.")
    else:
        st.info("📷 Capture or upload a photo to continue.")

    reg_image_b64 = st.session_state.get("reg_image_b64")

    if st.button("✅ REGISTER PERSON", use_container_width=True, disabled=(not reg_image_b64)):
        if not name or not student_id:
            st.error("Name and Student ID are required.")
        else:
            key = f"{name}_{student_id}"
            if key in face_db:
                st.warning(f"⚠️ {name} (ID: {student_id}) is already registered.")
            else:
                with st.spinner("🧠 Analysing face with GPT-4o Vision..."):
                    if not detect_face_in_image(reg_image_b64):
                        st.error("❌ No human face detected. Use a clear, well-lit frontal photo.")
                    else:
                        description = describe_face(reg_image_b64)
                        if description.startswith("ERROR"):
                            st.error(f"API error: {description}")
                        else:
                            img_bytes = base64.b64decode(reg_image_b64)
                            pil_save  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            img_path  = os.path.join(FACES_DIR, f"{key}.jpg")
                            pil_save.save(img_path)

                            face_db[key] = {
                                "name":        name,
                                "student_id":  student_id,
                                "department":  dept,
                                "email":       email,
                                "description": description,
                                "registered":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "image_path":  img_path
                            }
                            save_face_database(face_db)
                            st.session_state.reg_image_b64 = None

                            st.markdown(f"""
                            <div class='success-box'>
                              ✅ REGISTRATION COMPLETE<br>
                              <span style='font-size:1.2rem;'>{name}</span><br>
                              ID: {student_id} &nbsp;|&nbsp; Dept: {dept or 'N/A'}
                            </div>""", unsafe_allow_html=True)

                            with st.expander("🔍 Facial Description (stored for AI matching)"):
                                st.text(description)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: View Records
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 View Records":
    st.markdown("<h1>📊 ATTENDANCE RECORDS</h1>", unsafe_allow_html=True)

    df = load_attendance()
    if df.empty:
        st.markdown("<div class='warning-box'>⚡ No attendance records found.</div>", unsafe_allow_html=True)
        st.stop()

    f1, f2, f3 = st.columns(3)
    all_dates  = sorted(df["Date"].unique(), reverse=True)
    all_names  = ["All"] + sorted(df["Name"].unique())
    sel_date   = f1.selectbox("Filter by Date",   ["All"] + all_dates)
    sel_name   = f2.selectbox("Filter by Name",   all_names)
    sel_status = f3.selectbox("Filter by Status", ["All", "Present", "Absent"])

    filtered = df.copy()
    if sel_date   != "All": filtered = filtered[filtered["Date"]   == sel_date]
    if sel_name   != "All": filtered = filtered[filtered["Name"]   == sel_name]
    if sel_status != "All": filtered = filtered[filtered["Status"] == sel_status]

    st.markdown(
        f"<div style='color:#7ba7bc; margin:10px 0;'>Showing "
        f"<b style='color:#00f5ff;'>{len(filtered)}</b> records</div>",
        unsafe_allow_html=True
    )
    st.dataframe(filtered.sort_values(["Date", "Time"], ascending=False), use_container_width=True, hide_index=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ EXPORT CSV",
        data=csv,
        file_name=f"attendance_{date.today()}.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("### 📈 ATTENDANCE BY DATE")
    by_date = df.groupby("Date").size().reset_index(name="Count")
    st.bar_chart(by_date.set_index("Date"))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Manage Database
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Manage Database":
    st.markdown("<h1>⚙️ MANAGE DATABASE</h1>", unsafe_allow_html=True)

    face_db = load_face_database()

    if not face_db:
        st.markdown("<div class='warning-box'>⚡ No registered faces in database.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"### Registered People ({len(face_db)})")
        for key, info in face_db.items():
            with st.expander(f"👤 {info['name']}  ·  ID: {info['student_id']}"):
                c1, c2 = st.columns([1, 3])
                img_path = info.get("image_path", "")
                if img_path and os.path.exists(img_path):
                    c1.image(img_path, width=120)
                else:
                    c1.markdown("📷 No photo")
                c2.markdown(f"""
**Name:** {info['name']}  
**Student ID:** {info['student_id']}  
**Department:** {info.get('department', 'N/A')}  
**Email:** {info.get('email', 'N/A')}  
**Registered:** {info.get('registered', 'N/A')}
""")
                if st.button(f"🗑️ Remove {info['name']}", key=f"del_{key}"):
                    del face_db[key]
                    save_face_database(face_db)
                    if img_path and os.path.exists(img_path):
                        os.remove(img_path)
                    st.success(f"Removed {info['name']}.")
                    st.rerun()

    st.markdown("---")
    st.markdown("### ⚠️ DANGER ZONE")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("🗑️ CLEAR ALL ATTENDANCE", use_container_width=True):
            if st.session_state.get("confirm_att"):
                save_attendance(pd.DataFrame(columns=["Name", "Student_ID", "Date", "Time", "Status"]))
                st.success("All attendance records cleared.")
                st.session_state.confirm_att = False
                st.rerun()
            else:
                st.session_state.confirm_att = True
                st.warning("Click again to confirm deletion of ALL attendance records.")

    with c2:
        if st.button("💣 RESET ENTIRE DATABASE", use_container_width=True):
            if st.session_state.get("confirm_db"):
                save_face_database({})
                save_attendance(pd.DataFrame(columns=["Name", "Student_ID", "Date", "Time", "Status"]))
                import shutil
                shutil.rmtree(FACES_DIR, ignore_errors=True)
                os.makedirs(FACES_DIR, exist_ok=True)
                st.success("Database fully reset.")
                st.session_state.confirm_db = False
                st.rerun()
            else:
                st.session_state.confirm_db = True
                st.warning("Click again to confirm FULL RESET (faces + attendance).")