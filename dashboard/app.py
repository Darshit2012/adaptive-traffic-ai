"""
Professional Streamlit Dashboard for Adaptive AI-Based Traffic Signal Optimization System
Designed for final-year B.Tech project evaluation with multi-tab layout, animations, and explanations.
"""

import pathlib
import time
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import datetime

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="AI Traffic Signal Optimizer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS STYLING ==========
st.markdown("""
    <style>
        /* Main background gradient */
        .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px;}
        
        /* Card styling */
        .info-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        
        /* Traffic light animation */
        .traffic-light {
            width: 60px;
            height: 180px;
            background: #333;
            border-radius: 10px;
            padding: 10px;
            display: inline-block;
            margin: 5px;
        }
        
        .light {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin: 10px auto;
            opacity: 0.3;
        }
        
        .light.active {opacity: 1; box-shadow: 0 0 20px currentColor;}
        .red {background-color: #ef4444;}
        .yellow {background-color: #fbbf24;}
        .green {background-color: #10b981;}
        
        /* Metric boxes */
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin: 5px;
        }
        
        /* Explanation box */
        .explain-box {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        /* Success highlight */
        .success-box {
            background: #d1fae5;
            border-left: 4px solid #10b981;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        /* Section headers */
        .section-header {
            color: #1e293b;
            font-size: 24px;
            font-weight: 600;
            margin: 20px 0 10px 0;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
    </style>
""", unsafe_allow_html=True)

# ========== SIDEBAR CONFIGURATION ==========
with st.sidebar:
    st.markdown("<div style='text-align: center;'><span style='font-size: 80px;'>🚦</span></div>", unsafe_allow_html=True)
    st.title("AI Traffic Optimizer")
    st.markdown("---")
    st.markdown("""
    **Project:** Adaptive AI-Based Traffic Signal Optimization System
    
    **Objective:** Use AI to reduce traffic congestion by dynamically adjusting signal timings based on real-time conditions.
    
    **Technologies:**
    - 🧠 Neural Networks
    - 🎯 Reinforcement Learning
    - 📊 Real-time Analytics
    """)
    st.markdown("---")
    
    # Navigation
    st.subheader("📍 Navigation")
    tab_selection = st.radio(
        "Select Dashboard Section:",
        ["🏠 Project Overview", "🚦 Live Simulation", "🧠 AI Decision Engine", 
         "📊 Performance Analytics", "🚑 Special Scenarios", "📈 Learning Progress"],
        label_visibility="collapsed"
    )

# ========== DATA LOADING ==========
LOG_PATH = pathlib.Path("data/run_log.csv")
COMP_PATH = pathlib.Path("data/comparison.csv")

@st.cache_data
def load_data():
    if not LOG_PATH.exists():
        return None, None
    log_df = pd.read_csv(LOG_PATH)
    comp_df = pd.read_csv(COMP_PATH, index_col=0) if COMP_PATH.exists() else None
    return log_df, comp_df

log_df, comp_df = load_data()

# ========== TAB 1: PROJECT OVERVIEW ==========
def show_overview():
    st.markdown("<h1 style='text-align: center; color: white;'>🚦 Adaptive AI-Based Traffic Signal Optimization</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>Final Year B.Tech AI Project</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Problem statement
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🚨 The Problem")
        st.markdown("""
        <div class='info-card'>
        <h4>Why Traditional Traffic Signals Fail</h4>
        <ul>
            <li><b>Fixed Timing:</b> Signals operate on pre-programmed cycles regardless of actual traffic</li>
            <li><b>No Adaptation:</b> Cannot respond to emergencies, accidents, or sudden congestion</li>
            <li><b>Waste:</b> Vehicles wait at red lights even when no cross-traffic exists</li>
            <li><b>Frustration:</b> Peak hours cause massive delays and pollution</li>
        </ul>
        <p><b>Result:</b> Average urban driver wastes <b>54 hours per year</b> in traffic!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 The Solution")
        st.markdown("""
        <div class='success-box'>
        <h4>AI-Powered Adaptation</h4>
        <ul>
            <li>📊 Real-time queue monitoring</li>
            <li>🧠 Neural network predictions</li>
            <li>🎯 Reinforcement learning</li>
            <li>🚑 Emergency prioritization</li>
            <li>🌅 Time-of-day intelligence</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("---")
    st.markdown("### ⚙️ How Our System Works")
    
    step1, step2, step3 = st.columns(3)
    with step1:
        st.markdown("""
        <div class='info-card' style='background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); color: white;'>
        <h4>1️⃣ Sense</h4>
        <p>📹 Monitor traffic density, queue lengths, waiting times at each intersection in real-time.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with step2:
        st.markdown("""
        <div class='info-card' style='background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white;'>
        <h4>2️⃣ Decide</h4>
        <p>🧠 AI controller analyzes conditions and chooses optimal signal timing and phase.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with step3:
        st.markdown("""
        <div class='info-card' style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white;'>
        <h4>3️⃣ Act</h4>
        <p>🚦 Adjust signal timing dynamically to minimize waiting and maximize throughput.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Animated demo
    st.markdown("---")
    st.markdown("### 🎬 Live Demo: Fixed vs AI-Controlled")
    
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        st.markdown("#### ⏱️ Fixed Timer (Traditional)")
        fixed_progress = st.progress(0)
        fixed_status = st.empty()
        
        for i in range(101):
            fixed_progress.progress(i)
            if i < 50:
                fixed_status.markdown("🔴 **Red Light** - Vehicles waiting even with no cross-traffic")
            else:
                fixed_status.markdown("🟢 **Green Light** - Fixed duration regardless of queue")
            time.sleep(0.02)
    
    with demo_col2:
        st.markdown("#### 🧠 AI Controller (Adaptive)")
        ai_progress = st.progress(0)
        ai_status = st.empty()
        
        for i in range(101):
            ai_progress.progress(i)
            if i < 30:
                ai_status.markdown("🟢 **Green Extended** - Heavy queue detected, extending green")
            elif i < 60:
                ai_status.markdown("🟡 **Quick Amber** - Light traffic, short transition")
            else:
                ai_status.markdown("🟢 **Smart Green** - Adjusted based on real-time demand")
            time.sleep(0.02)
    
    # Key features
    st.markdown("---")
    st.markdown("### ✨ What Makes This Project Unique")
    
    feat1, feat2, feat3, feat4 = st.columns(4)
    with feat1:
        st.metric("🧠 AI Controllers", "3", help="Fixed, Neural Network, Q-Learning")
    with feat2:
        st.metric("🚑 Emergency Priority", "Yes", help="Instant override for emergency vehicles")
    with feat3:
        st.metric("🌅 Time Profiles", "3", help="Morning, Afternoon, Night patterns")
    with feat4:
        st.metric("📊 Explainability", "Built-in", help="Every decision is logged and explained")

# ========== TAB 2: LIVE SIMULATION ==========
def show_live_simulation():
    st.markdown("<h2 style='color: white;'>🚦 Live Traffic Simulation</h2>", unsafe_allow_html=True)
    
    if log_df is None:
        st.warning("⚠️ No simulation data found. Run `python main.py` first!")
        return
    
    # Control panel
    st.markdown("### 🎮 Simulation Controls")
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    
    with ctrl1:
        density = st.slider("🚗 Traffic Density", 1, 10, 5, help="Adjust incoming vehicle rate")
    with ctrl2:
        emergency = st.checkbox("🚑 Emergency Vehicle Active", help="Trigger emergency override")
    with ctrl3:
        time_period = st.selectbox("🌅 Time of Day", ["morning", "afternoon", "night"])
    
    st.markdown("---")
    
    # Get latest intersection states
    latest_step = log_df['step'].max()
    latest_data = log_df[log_df['step'] == latest_step]
    
    st.markdown("### 🛣️ Current Intersection States")
    
    for _, row in latest_data.iterrows():
        inter_id = int(row['intersection'])
        phase = row['phase_used']
        queue_ns = int(row['queue_ns'])
        queue_ew = int(row['queue_ew'])
        duration = round(row['duration'], 1)
        
        with st.expander(f"🚦 Intersection {inter_id} - Phase: {phase}", expanded=True):
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                # Traffic light visualization
                ns_class = "active" if phase == "NS" else ""
                ew_class = "active" if phase == "EW" else ""
                
                st.markdown(f"""
                <div style='text-align: center;'>
                    <p><b>North-South</b></p>
                    <div class='traffic-light'>
                        <div class='light red {"active" if phase != "NS" else ""}'></div>
                        <div class='light yellow'></div>
                        <div class='light green {ns_class}'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("🚗 NS Queue", queue_ns)
                m2.metric("🚗 EW Queue", queue_ew)
                m3.metric("⏱️ Duration", f"{duration}s")
                
                # Progress bar for countdown
                st.markdown("**Green Light Countdown:**")
                countdown_progress = st.progress(0)
                for i in range(int(duration * 10)):
                    countdown_progress.progress(i / (duration * 10))
                    time.sleep(0.05)
            
            with col3:
                # Traffic light visualization
                st.markdown(f"""
                <div style='text-align: center;'>
                    <p><b>East-West</b></p>
                    <div class='traffic-light'>
                        <div class='light red {"active" if phase != "EW" else ""}'></div>
                        <div class='light yellow'></div>
                        <div class='light green {ew_class}'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Queue visualization
            st.markdown("**Vehicle Queues:**")
            q_col1, q_col2 = st.columns(2)
            q_col1.progress(min(queue_ns / 20, 1.0))
            q_col1.caption(f"NS: {queue_ns} vehicles")
            q_col2.progress(min(queue_ew / 20, 1.0))
            q_col2.caption(f"EW: {queue_ew} vehicles")

# ========== TAB 3: AI DECISION ENGINE ==========
def show_ai_engine():
    st.markdown("<h2 style='color: white;'>🧠 AI Decision Engine</h2>", unsafe_allow_html=True)
    
    if log_df is None:
        st.warning("⚠️ No simulation data found. Run `python main.py` first!")
        return
    
    # Detect controller type
    sample_explanation = log_df['explanation'].iloc[0]
    if "RL:" in sample_explanation:
        controller_type = "Reinforcement Learning (Q-Learning)"
        controller_icon = "🎯"
        controller_color = "#8b5cf6"
    elif "Neural" in sample_explanation:
        controller_type = "Neural Network"
        controller_icon = "🧠"
        controller_color = "#3b82f6"
    else:
        controller_type = "Fixed Timer"
        controller_icon = "⏱️"
        controller_color = "#f59e0b"
    
    st.markdown(f"""
    <div class='info-card' style='background: {controller_color}; color: white; text-align: center;'>
        <h2>{controller_icon} Active Controller: {controller_type}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Latest decision
    latest = log_df.iloc[-1]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📥 Input Features (State)")
        st.markdown("""
        <div class='info-card'>
        <p><b>The AI observes these features to make decisions:</b></p>
        """, unsafe_allow_html=True)
        
        feat1, feat2, feat3 = st.columns(3)
        feat1.metric("🚗 NS Queue", int(latest['queue_ns']))
        feat2.metric("🚗 EW Queue", int(latest['queue_ew']))
        feat3.metric("🚦 Current Phase", latest['phase_used'])
        
        feat4, feat5, feat6 = st.columns(3)
        feat4.metric("🌅 Time Period", latest['time_of_day'])
        feat5.metric("🚑 Emergency", "Yes" if latest['emergency'] else "No")
        feat6.metric("📊 Step", int(latest['step']))
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("### 📤 Output Decision (Action)")
        st.markdown(f"""
        <div class='success-box'>
        <h4>✅ Signal Duration: {round(latest['duration'], 1)} seconds</h4>
        <p>Phase: <b>{latest['phase_used']}</b> gets green light</p>
        <p>Expected to serve <b>{int(latest['throughput'])}</b> vehicles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔍 How It Works")
        
        if "RL" in controller_type:
            st.markdown("""
            <div class='explain-box'>
            <h4>Q-Learning Process:</h4>
            <ol>
                <li><b>State:</b> Current queues, phase, time</li>
                <li><b>Action Space:</b> Duration choices (10s, 12s, 14s)</li>
                <li><b>Reward:</b> Throughput - Wait Time - Stops</li>
                <li><b>Learning:</b> Updates Q-values after each step</li>
                <li><b>Policy:</b> Choose action with highest Q-value</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        elif "Neural" in controller_type:
            st.markdown("""
            <div class='explain-box'>
            <h4>Neural Network Process:</h4>
            <ol>
                <li><b>Input Layer:</b> 5 features (queues, phase, time, emergency)</li>
                <li><b>Hidden Layer:</b> 6 neurons with sigmoid activation</li>
                <li><b>Output:</b> Signal duration (4-16s)</li>
                <li><b>Learning:</b> Backpropagation with reward feedback</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='explain-box'>
            <h4>Fixed Timer Process:</h4>
            <ol>
                <li><b>Morning:</b> 12s green (peak traffic)</li>
                <li><b>Afternoon:</b> 10s green (moderate)</li>
                <li><b>Night:</b> 6s green (low traffic)</li>
                <li><b>No Learning:</b> Static plan</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
    
    # Explainability section
    st.markdown("---")
    st.markdown("### 💡 Decision Explainability")
    
    recent_decisions = log_df.tail(10)[['step', 'intersection', 'queue_ns', 'queue_ew', 'duration', 'explanation']]
    st.dataframe(recent_decisions, use_container_width=True, height=300)
    
    st.markdown("""
    <div class='explain-box'>
    <h4>🎓 Viva Tip: Why Explainability Matters</h4>
    <p>Unlike black-box AI, our system provides clear reasoning for every decision. This is crucial for:</p>
    <ul>
        <li><b>Trust:</b> Traffic authorities can understand and validate AI decisions</li>
        <li><b>Debugging:</b> Identify when the system needs improvement</li>
        <li><b>Safety:</b> Ensure emergency overrides work correctly</li>
        <li><b>Compliance:</b> Meet regulatory requirements for autonomous systems</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ========== TAB 4: PERFORMANCE ANALYTICS ==========
def show_analytics():
    st.markdown("<h2 style='color: white;'>📊 Performance Analytics</h2>", unsafe_allow_html=True)
    
    if log_df is None:
        st.warning("⚠️ No simulation data found. Run `python main.py` first!")
        return
    
    # Aggregate metrics by step
    step_metrics = log_df.groupby('step').agg({
        'throughput': 'sum',
        'avg_wait_proxy': 'mean',
        'stops': 'sum',
        'emergency': 'sum'
    }).reset_index()
    
    # Overall metrics
    st.markdown("### 📈 Overall Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Throughput", f"{int(step_metrics['throughput'].sum())} vehicles", 
              help="Total vehicles served across all intersections")
    m2.metric("Avg Wait Time", f"{round(step_metrics['avg_wait_proxy'].mean(), 2)}s",
              help="Average waiting time proxy (lower is better)")
    m3.metric("Total Stops", int(step_metrics['stops'].sum()),
              help="Total number of vehicle stops (lower is better)")
    m4.metric("Emergencies Handled", int(step_metrics['emergency'].sum()),
              help="Emergency vehicle priority events")
    
    st.markdown("---")
    
    # Time series charts
    st.markdown("### 📉 Performance Over Time")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        throughput_chart = alt.Chart(step_metrics).mark_area(
            color='#10b981',
            opacity=0.7
        ).encode(
            x=alt.X('step:Q', title='Simulation Step'),
            y=alt.Y('throughput:Q', title='Vehicles Served'),
            tooltip=['step', 'throughput']
        ).properties(
            title='Vehicle Throughput',
            height=300
        )
        st.altair_chart(throughput_chart, use_container_width=True)
    
    with chart_col2:
        wait_chart = alt.Chart(step_metrics).mark_line(
            color='#ef4444',
            strokeWidth=3
        ).encode(
            x=alt.X('step:Q', title='Simulation Step'),
            y=alt.Y('avg_wait_proxy:Q', title='Average Wait (s)'),
            tooltip=['step', 'avg_wait_proxy']
        ).properties(
            title='Average Waiting Time',
            height=300
        )
        st.altair_chart(wait_chart, use_container_width=True)
    
    # Controller comparison
    if comp_df is not None:
        st.markdown("---")
        st.markdown("### 🏆 Controller Comparison")
        
        st.markdown("""
        <div class='info-card'>
        <p>Comparing different control strategies on the same traffic scenario:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Highlight best performer
        best_throughput = comp_df['throughput'].idxmax()
        best_wait = comp_df['avg_wait_proxy'].idxmin()
        best_stops = comp_df['stops'].idxmin()
        
        comp_styled = comp_df.style.highlight_max(subset=['throughput'], color='lightgreen')\
                                    .highlight_min(subset=['avg_wait_proxy', 'stops'], color='lightgreen')\
                                    .format({'avg_wait_proxy': '{:.3f}', 'throughput': '{:.0f}', 'stops': '{:.0f}'})
        
        st.dataframe(comp_styled, use_container_width=True)
        
        st.markdown(f"""
        <div class='success-box'>
        <h4>🎯 Performance Insights:</h4>
        <ul>
            <li><b>Best Throughput:</b> {best_throughput} ({int(comp_df.loc[best_throughput, 'throughput'])} vehicles)</li>
            <li><b>Lowest Wait Time:</b> {best_wait} ({comp_df.loc[best_wait, 'avg_wait_proxy']:.3f}s average)</li>
            <li><b>Fewest Stops:</b> {best_stops} ({int(comp_df.loc[best_stops, 'stops'])} stops)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Bar chart comparison
        st.markdown("### 📊 Visual Comparison")
        
        comp_melted = comp_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
        comp_melted = comp_melted[comp_melted['Metric'].isin(['throughput', 'avg_wait_proxy', 'stops'])]
        
        comparison_chart = alt.Chart(comp_melted).mark_bar().encode(
            x=alt.X('index:N', title='Controller'),
            y=alt.Y('Value:Q', title='Value'),
            color=alt.Color('index:N', legend=None),
            column=alt.Column('Metric:N', title=None),
            tooltip=['index', 'Metric', 'Value']
        ).properties(
            width=200,
            height=300
        )
        
        st.altair_chart(comparison_chart)

# ========== TAB 5: SPECIAL SCENARIOS ==========
def show_special_scenarios():
    st.markdown("<h2 style='color: white;'>🚑 Special Scenarios</h2>", unsafe_allow_html=True)
    
    if log_df is None:
        st.warning("⚠️ No simulation data found. Run `python main.py` first!")
        return
    
    st.markdown("### 🎯 Testing Edge Cases")
    st.markdown("""
    <div class='info-card'>
    <p>Our system handles challenging real-world scenarios that fixed timers cannot adapt to:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Emergency vehicle analysis
    st.markdown("---")
    st.markdown("### 🚑 Emergency Vehicle Priority")
    
    emergency_events = log_df[log_df['emergency'] == True]
    
    if len(emergency_events) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class='success-box'>
            <h4>✅ Emergency Override Active</h4>
            <p><b>{len(emergency_events)}</b> emergency events handled in this simulation</p>
            <p>Average response: Signal changed immediately to prioritize emergency vehicle direction</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show emergency event timeline
            emergency_timeline = emergency_events.groupby('step').size().reset_index(name='count')
            emergency_chart = alt.Chart(emergency_timeline).mark_circle(
                size=200,
                color='#ef4444'
            ).encode(
                x=alt.X('step:Q', title='Simulation Step'),
                y=alt.Y('count:Q', title='Emergency Events'),
                tooltip=['step', 'count']
            ).properties(
                title='Emergency Event Timeline',
                height=250
            )
            st.altair_chart(emergency_chart, use_container_width=True)
        
        with col2:
            st.markdown("#### How It Works:")
            st.markdown("""
            <div class='explain-box'>
            <ol>
                <li><b>Detection:</b> Emergency flag triggered</li>
                <li><b>Override:</b> Normal timing ignored</li>
                <li><b>Priority:</b> Green given to heaviest queue (proxy for emergency direction)</li>
                <li><b>Logging:</b> Event recorded for analysis</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No emergency events in current simulation. Increase emergency rate in main.py.")
    
    # Peak hour congestion
    st.markdown("---")
    st.markdown("### 🌅 Time-of-Day Adaptation")
    
    time_comparison = log_df.groupby('time_of_day').agg({
        'throughput': 'sum',
        'avg_wait_proxy': 'mean',
        'queue_ns': 'mean',
        'queue_ew': 'mean'
    }).reset_index()
    
    st.dataframe(time_comparison, use_container_width=True)
    
    st.markdown("""
    <div class='explain-box'>
    <h4>💡 Adaptive Behavior:</h4>
    <ul>
        <li><b>Morning Peak:</b> Higher traffic → Longer greens → More throughput</li>
        <li><b>Afternoon Normal:</b> Balanced traffic → Moderate timing</li>
        <li><b>Night Low:</b> Minimal traffic → Shorter greens → Less waiting</li>
    </ul>
    <p><b>Why Fixed Timers Fail:</b> They use the same timing regardless of actual demand, wasting time at night and causing congestion during peaks.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Congestion waves
    st.markdown("---")
    st.markdown("### 🌊 Multi-Intersection Coordination")
    
    if log_df['intersection'].nunique() > 1:
        st.markdown("""
        <div class='info-card'>
        <p>Our system coordinates multiple intersections to prevent congestion waves:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show coordination effect
        inter_comparison = log_df.groupby('intersection').agg({
            'throughput': 'sum',
            'avg_wait_proxy': 'mean'
        }).reset_index()
        
        coord_chart = alt.Chart(inter_comparison).mark_bar().encode(
            x=alt.X('intersection:N', title='Intersection ID'),
            y=alt.Y('throughput:Q', title='Total Throughput'),
            color=alt.Color('intersection:N', legend=None),
            tooltip=['intersection', 'throughput']
        ).properties(
            title='Throughput by Intersection',
            height=300
        )
        st.altair_chart(coord_chart, use_container_width=True)
        
        st.markdown("""
        <div class='success-box'>
        <h4>🔗 Coordination Logic:</h4>
        <p>Upstream intersection shares anticipated outflow with downstream neighbors, allowing them to pre-adjust timings and prevent queue spillback.</p>
        </div>
        """, unsafe_allow_html=True)

# ========== TAB 6: LEARNING PROGRESS ==========
def show_learning_progress():
    st.markdown("<h2 style='color: white;'>📈 Learning Progress</h2>", unsafe_allow_html=True)
    
    if log_df is None:
        st.warning("⚠️ No simulation data found. Run `python main.py` first!")
        return
    
    st.markdown("""
    <div class='info-card'>
    <h3>🧠 How AI Controllers Learn and Improve</h3>
    <p>Unlike fixed timers, AI controllers continuously adapt based on experience and feedback.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate rolling metrics to show improvement
    window = 50
    log_df_copy = log_df.copy()
    log_df_copy['throughput_ma'] = log_df_copy.groupby('intersection')['throughput'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    log_df_copy['wait_ma'] = log_df_copy.groupby('intersection')['avg_wait_proxy'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    
    st.markdown("---")
    st.markdown("### 📊 Performance Improvement Over Time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        throughput_trend = alt.Chart(log_df_copy).mark_line(
            color='#10b981',
            strokeWidth=2
        ).encode(
            x=alt.X('step:Q', title='Simulation Step'),
            y=alt.Y('throughput_ma:Q', title='Throughput (Moving Avg)'),
            tooltip=['step', 'throughput_ma']
        ).properties(
            title=f'Throughput Trend (Rolling {window}-step average)',
            height=300
        )
        st.altair_chart(throughput_trend, use_container_width=True)
    
    with col2:
        wait_trend = alt.Chart(log_df_copy).mark_line(
            color='#ef4444',
            strokeWidth=2
        ).encode(
            x=alt.X('step:Q', title='Simulation Step'),
            y=alt.Y('wait_ma:Q', title='Wait Time (Moving Avg)'),
            tooltip=['step', 'wait_ma']
        ).properties(
            title=f'Wait Time Trend (Rolling {window}-step average)',
            height=300
        )
        st.altair_chart(wait_trend, use_container_width=True)
    
    # Learning insights
    st.markdown("---")
    st.markdown("### 🎓 Learning Behavior Analysis")
    
    initial_perf = log_df.head(100)
    final_perf = log_df.tail(100)
    
    initial_throughput = initial_perf['throughput'].mean()
    final_throughput = final_perf['throughput'].mean()
    throughput_improvement = ((final_throughput - initial_throughput) / initial_throughput) * 100 if initial_throughput > 0 else 0
    
    initial_wait = initial_perf['avg_wait_proxy'].mean()
    final_wait = final_perf['avg_wait_proxy'].mean()
    wait_improvement = ((initial_wait - final_wait) / initial_wait) * 100 if initial_wait > 0 else 0
    
    col1, col2 = st.columns(2)
    col1.metric("Throughput Change", f"{throughput_improvement:+.1f}%", 
                delta=f"{final_throughput - initial_throughput:.1f} vehicles",
                help="Change from first 100 steps to last 100 steps")
    col2.metric("Wait Time Change", f"{wait_improvement:+.1f}%",
                delta=f"{final_wait - initial_wait:.2f}s",
                delta_color="inverse",
                help="Change from first 100 steps to last 100 steps")
    
    if "RL" in log_df['explanation'].iloc[0]:
        st.markdown("""
        <div class='explain-box'>
        <h4>🎯 Q-Learning Behavior:</h4>
        <ul>
            <li><b>Exploration Phase (Early):</b> Tries different actions to discover rewards</li>
            <li><b>Exploitation Phase (Later):</b> Uses learned Q-values for optimal decisions</li>
            <li><b>Epsilon Decay:</b> Gradually reduces random exploration over time</li>
            <li><b>Q-Value Updates:</b> Continuously refines action values based on rewards</li>
        </ul>
        <p><b>Result:</b> Controller converges to near-optimal policy after sufficient experience.</p>
        </div>
        """, unsafe_allow_html=True)
    elif "Neural" in log_df['explanation'].iloc[0]:
        st.markdown("""
        <div class='explain-box'>
        <h4>🧠 Neural Network Learning:</h4>
        <ul>
            <li><b>Forward Pass:</b> Predicts signal duration from current state</li>
            <li><b>Reward Feedback:</b> Receives reward based on throughput and waiting</li>
            <li><b>Backpropagation:</b> Updates weights to improve future predictions</li>
            <li><b>Online Learning:</b> Adapts continuously during simulation</li>
        </ul>
        <p><b>Result:</b> Network learns to balance queue lengths and minimize waiting.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Exploration vs Exploitation balance
    if "RL" in log_df['explanation'].iloc[0]:
        st.markdown("---")
        st.markdown("### 🔍 Exploration vs Exploitation")
        
        exploration_count = log_df[log_df['explanation'].str.contains('exploring', case=False)].shape[0]
        exploitation_count = log_df[log_df['explanation'].str.contains('exploiting', case=False)].shape[0]
        total = exploration_count + exploitation_count
        
        if total > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("🔍 Exploration Steps", exploration_count, 
                       help="Random actions to discover new strategies")
            col2.metric("🎯 Exploitation Steps", exploitation_count,
                       help="Using learned Q-values for optimal actions")
            col3.metric("📊 Exploitation Rate", f"{(exploitation_count/total)*100:.1f}%",
                       help="Higher rate means more confidence in learned policy")
            
            # Show progression
            log_df_copy2 = log_df.copy()
            log_df_copy2['is_exploit'] = log_df_copy2['explanation'].str.contains('exploiting', case=False).astype(int)
            log_df_copy2['exploit_ma'] = log_df_copy2['is_exploit'].rolling(window=50, min_periods=1).mean()
            
            exploit_chart = alt.Chart(log_df_copy2).mark_line(
                color='#8b5cf6',
                strokeWidth=2
            ).encode(
                x=alt.X('step:Q', title='Simulation Step'),
                y=alt.Y('exploit_ma:Q', title='Exploitation Rate', scale=alt.Scale(domain=[0, 1])),
                tooltip=['step', 'exploit_ma']
            ).properties(
                title='Exploitation Rate Over Time (Rolling 50-step average)',
                height=250
            )
            st.altair_chart(exploit_chart, use_container_width=True)
            
            st.markdown("""
            <div class='success-box'>
            <p><b>📈 Insight:</b> As the controller learns, exploitation rate increases, showing confidence in the learned policy.</p>
            </div>
            """, unsafe_allow_html=True)

# ========== MAIN APP ROUTER ==========
def main():
    if "Project Overview" in tab_selection:
        show_overview()
    elif "Live Simulation" in tab_selection:
        show_live_simulation()
    elif "AI Decision Engine" in tab_selection:
        show_ai_engine()
    elif "Performance Analytics" in tab_selection:
        show_analytics()
    elif "Special Scenarios" in tab_selection:
        show_special_scenarios()
    elif "Learning Progress" in tab_selection:
        show_learning_progress()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: white; padding: 20px;'>
        <p><b>Adaptive AI-Based Traffic Signal Optimization System</b></p>
        <p>Final Year B.Tech AI Project | Built with Python, Streamlit, and Reinforcement Learning</p>
        <p>🚦 Making cities smarter, one signal at a time</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
