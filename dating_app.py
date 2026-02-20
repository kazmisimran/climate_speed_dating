import streamlit as st
import pandas as pd
import networkx as nx
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="Climate Speed Dating & Friendship", layout="wide")

# Custom CSS for UI
st.markdown("""
<style>
    .match-card { 
        background-color: #333333; color: white; padding: 15px; 
        border-radius: 8px; margin-bottom: 10px; border-left: 5px solid #ff4b4b;
    }
    .friend-card { 
        background-color: #2e3d33; color: white; padding: 15px; 
        border-radius: 8px; margin-bottom: 10px; border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'blocked_pairs' not in st.session_state:
    st.session_state['blocked_pairs'] = set()

# --- HELPER FUNCTIONS ---
def clean_age_value(val):
    if pd.isna(val): return None
    val_str = str(val).lower()
    if 'no limit' in val_str: return 100
    nums = re.findall(r'\d+', val_str)
    return int(nums[0]) if nums else None

def get_display_name(full_name):
    if pd.isna(full_name): return "Unknown"
    parts = str(full_name).strip().split()
    return f"{parts[0]} {parts[-1][0]}." if len(parts) >= 2 else parts[0]

def parse_multi_select(entry):
    if pd.isna(entry): return []
    return [x.strip().lower() for x in str(entry).split(',')]

def calculate_dating_score(person_A, person_B, enforce_goals=False):
    # Orientation Check
    poly_keywords = ['polyamorous', 'open', 'non-monogamy', 'polyamory']
    mono_keywords = ['monogamous', 'monogamy']
    a_or, b_or = str(person_A['Orientation']).lower(), str(person_B['Orientation']).lower()
    a_is_poly, a_is_mono = any(k in a_or for k in poly_keywords), any(k in a_or for k in mono_keywords)
    b_is_poly, b_is_mono = any(k in b_or for k in poly_keywords), any(k in b_or for k in mono_keywords)
    if (a_is_mono and not a_is_poly and b_is_poly and not b_is_mono) or \
       (b_is_mono and not b_is_poly and a_is_poly and not a_is_mono): return 0

    # Gender Match
    a_genders, b_genders = set(parse_multi_select(person_A['Gender'])), set(parse_multi_select(person_B['Gender']))
    a_wants, b_wants = set(parse_multi_select(person_A['Interested_In'])), set(parse_multi_select(person_B['Interested_In']))
    match_a_to_b = ("all genders" in a_wants) or ("all" in a_wants) or not a_wants.isdisjoint(b_genders)
    match_b_to_a = ("all genders" in b_wants) or ("all" in b_wants) or not b_wants.isdisjoint(a_genders)
    if not (match_a_to_b and match_b_to_a): return 0

    # Age Match
    if not (person_B['Min_Age'] <= person_A['Age'] <= person_B['Max_Age']): return 0
    if not (person_A['Min_Age'] <= person_B['Age'] <= person_A['Max_Age']): return 0

    # Goals
    if enforce_goals:
        a_goals, b_goals = set(parse_multi_select(person_A['Goals'])), set(parse_multi_select(person_B['Goals']))
        if not (not a_goals.isdisjoint({"open to possibilities", "open to anything"}) or not b_goals.isdisjoint({"open to possibilities", "open to anything"}) or not a_goals.isdisjoint(b_goals)):
            return 0
    
    # Scoring
    a_int, b_int = set(parse_multi_select(person_A['Interests'])), set(parse_multi_select(person_B['Interests']))
    union = len(a_int.union(b_int))
    jaccard = (len(a_int.intersection(b_int)) / union) if union > 0 else 0
    return 50 + (jaccard * 50)

# --- LOADING DATA ---
def load_data(uploaded_file):
    df_raw = pd.read_csv(uploaded_file)
    df_raw['Public_ID'] = df_raw.index + 1
    
    # Filter for Approved only
    if 'approval_status' in df_raw.columns:
        df_raw = df_raw[df_raw['approval_status'].str.lower() == 'approved'].copy()
    
    col_map = {
        'name': 'Name', 'How old are you?': 'Age', 'What gender do you identify as?': 'Gender',
        'Which genders are you interested in dating?': 'Interested_In',
        'What relationship styles are you interested in?': 'Orientation',
        "What is the lowest age you're open to dating?": 'Min_Age',
        "What's the oldest you're open to dating?": 'Max_Age',
        'Tell us about your interests! ': 'Interests', 'What are your dating goals?': 'Goals'
    }
    df = df_raw.rename(columns=col_map)
    df['Age'] = df['Age'].apply(clean_age_value).fillna(30)
    df['Min_Age'] = df['Min_Age'].apply(clean_age_value).fillna(18)
    df['Max_Age'] = df['Max_Age'].apply(clean_age_value).fillna(100)
    df['Display_Name'] = df['Name'].apply(get_display_name)
    df = df.sort_values(by='Display_Name').reset_index(drop=True)
    df['Internal_ID'] = df.index
    df['Present'] = False
    return df

# --- MAIN UI ---
st.title("Climate Speed Dating: Dating + Friendship")
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file:
    full_df = load_data(uploaded_file)
    if full_df.empty:
        st.warning("No 'Approved' participants found.")
    else:
        full_df['Check_In_Label'] = full_df['Display_Name'] + " (#" + full_df['Public_ID'].astype(str) + ")"
        edited_view = st.data_editor(full_df[['Present', 'Check_In_Label']], hide_index=True, use_container_width=True)
        
        df = full_df.loc[edited_view[edited_view['Present']].index].copy()
        
        # Sidebar Config
        st.sidebar.header("Configuration")
        num_rounds = st.sidebar.slider("Rounds", 1, 12, 5)
        enforce_goals = st.sidebar.checkbox("Strict Goal Match", value=False)
        
        # Blocking Pair Logic
        st.sidebar.subheader("Manual Pair Blocking")
        options = df[['Internal_ID', 'Display_Name', 'Public_ID']].to_dict('records')
        label_map = {p['Internal_ID']: f"{p['Display_Name']} (#{p['Public_ID']})" for p in options}
        
        c1, c2 = st.sidebar.columns(2)
        p1_block = c1.selectbox("Person A", options=label_map.keys(), format_func=lambda x: label_map[x], key="ba")
        p2_block = c2.selectbox("Person B", options=label_map.keys(), format_func=lambda x: label_map[x], key="bb")
        
        if st.sidebar.button("Block Pair"):
            if p1_block != p2_block:
                st.session_state['blocked_pairs'].add(tuple(sorted((p1_block, p2_block))))
                st.sidebar.success("Pair Blocked!")

        if st.session_state['blocked_pairs']:
            for pair in list(st.session_state['blocked_pairs']):
                try:
                    p1_n = df.loc[df['Internal_ID'] == pair[0], 'Display_Name'].values[0]
                    p1_id = df.loc[df['Internal_ID'] == pair[0], 'Public_ID'].values[0]
                    p2_n = df.loc[df['Internal_ID'] == pair[1], 'Display_Name'].values[0]
                    p2_id = df.loc[df['Internal_ID'] == pair[1], 'Public_ID'].values[0]
                    st.sidebar.text(f"❌ {p1_n}(#{p1_id}) & {p2_n}(#{p2_id})")
                except: pass
            if st.sidebar.button("Clear All Blocks"):
                st.session_state['blocked_pairs'] = set(); st.rerun()

        # GENERATE SEATING
        if st.button("Generate Seating Plan", type="primary"):
            platonic_only = []
            flexible_dating = []
            for idx, row in df.iterrows():
                goals = parse_multi_select(row['Goals'])
                if len(goals) == 1 and 'friendship' in goals[0]:
                    platonic_only.append(row['Internal_ID'])
                else:
                    flexible_dating.append(row['Internal_ID'])
            
            history = {uid: [] for uid in df['Internal_ID']}
            blocked = st.session_state['blocked_pairs']
            master_log = []
            tabs = st.tabs([f"Round {i+1}" for i in range(num_rounds)])

            for r in range(num_rounds):
                round_num = r + 1
                with tabs[r]:
                    st.header(f"Round {round_num}")
                    matched_this_round = set()

                    # Dating Pool
                    G_date = nx.Graph()
                    for i in flexible_dating:
                        for j in flexible_dating:
                            if i >= j or j in history[i] or tuple(sorted((i, j))) in blocked: continue
                            score = calculate_dating_score(df.loc[i], df.loc[j], enforce_goals)
                            if score > 0: G_date.add_edge(i, j, weight=score)
                    
                    date_matches = list(nx.max_weight_matching(G_date, maxcardinality=True))
                    
                    st.subheader("🔥 Dating Tables")
                    d_cols = st.columns(4)
                    for k, (u, v) in enumerate(date_matches):
                        matched_this_round.update([u, v])
                        history[u].append(v); history[v].append(u)
                        p1, p2 = df.loc[u], df.loc[v]
                        master_log.append({"Round": round_num, "Table": f"D-{k+1}", "A": f"{p1['Display_Name']} (#{p1['Public_ID']})", "B": f"{p2['Display_Name']} (#{p2['Public_ID']})", "Type": "Date"})
                        with d_cols[k % 4]:
                            st.markdown(f'<div class="match-card"><strong>Table D-{k+1}</strong><br>{p1["Display_Name"]} (#{p1["Public_ID"]})<br>{p2["Display_Name"]} (#{p2["Public_ID"]})</div>', unsafe_allow_html=True)

                    # Friendship Pool
                    rem_dating = [uid for uid in flexible_dating if uid not in matched_this_round]
                    friend_pool = platonic_only + rem_dating
                    
                    G_friend = nx.Graph()
                    for i in friend_pool:
                        for j in friend_pool:
                            if i >= j or j in history[i] or tuple(sorted((i, j))) in blocked: continue
                            a_int, b_int = set(parse_multi_select(df.loc[i]['Interests'])), set(parse_multi_select(df.loc[j]['Interests']))
                            common = len(a_int.intersection(b_int))
                            G_friend.add_edge(i, j, weight=10 + common)
                    
                    friend_matches = list(nx.max_weight_matching(G_friend, maxcardinality=True))

                    st.subheader("🤝 Friendship Tables")
                    f_cols = st.columns(4)
                    for k, (u, v) in enumerate(friend_matches):
                        matched_this_round.update([u, v])
                        history[u].append(v); history[v].append(u)
                        p1, p2 = df.loc[u], df.loc[v]
                        master_log.append({"Round": round_num, "Table": f"F-{k+1}", "A": f"{p1['Display_Name']} (#{p1['Public_ID']})", "B": f"{p2['Display_Name']} (#{p2['Public_ID']})", "Type": "Friendship"})
                        with f_cols[k % 4]:
                            st.markdown(f'<div class="friend-card"><strong>Table F-{k+1}</strong><br>{p1["Display_Name"]} (#{p1["Public_ID"]})<br>{p2["Display_Name"]} (#{p2["Public_ID"]})</div>', unsafe_allow_html=True)

                    # Lounge
                    unmatched = [uid for uid in df['Internal_ID'] if uid not in matched_this_round]
                    if unmatched:
                        st.markdown("#### 🛋️ Lounge")
                        st.info(", ".join([f"{df.loc[uid]['Display_Name']} (#{df.loc[uid]['Public_ID']})" for uid in unmatched]))
            
            if master_log:
                st.markdown("---")
                csv_data = pd.DataFrame(master_log).to_csv(index=False).encode('utf-8')
                st.download_button("Download Schedule", csv_data, "event_schedule.csv", "text/csv")