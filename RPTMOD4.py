# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Rock Physics Analyzer",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RockPhysicsAnalyzer:
    def __init__(self, data):
        self.data = data.copy()
        self.depth = None
        self.results = {}
    
    def load_data(self, depth_col='Depth'):
        self.depth = self.data[depth_col]
        return f"Data loaded successfully. Columns: {list(self.data.columns)}"
    
    def quality_control(self):
        logs_to_check = ['Depth', 'Vp', 'Vs', 'NPHI', 'GR', 'RT', 'SW', 'VSH']
        qc_results = {}
        
        for log in logs_to_check:
            if log in self.data.columns:
                valid_data = self.data[log].notna().sum()
                total_data = len(self.data[log])
                percentage = (valid_data / total_data) * 100
                qc_results[log] = {
                    'valid': valid_data,
                    'total': total_data,
                    'percentage': percentage
                }
        return qc_results
    
    def calculate_elastic_properties(self, rhob_matrix=2.65, rhob_fluid=1.0):
        if 'NPHI' in self.data.columns:
            phi = self.data['NPHI'].clip(0, 0.4)
            self.data['RHOB'] = rhob_matrix * (1 - phi) + rhob_fluid * phi
        
        if 'Vp' in self.data.columns and 'RHOB' in self.data.columns:
            self.data['AI'] = self.data['Vp'] * self.data['RHOB']
        
        if 'Vp' in self.data.columns and 'Vs' in self.data.columns:
            self.data['Vp_Vs'] = self.data['Vp'] / self.data['Vs']
            self.data['PR'] = (self.data['Vp']**2 - 2 * self.data['Vs']**2) / (2 * (self.data['Vp']**2 - self.data['Vs']**2))
        
        if all(log in self.data.columns for log in ['Vp', 'Vs', 'RHOB']):
            self.data['LambdaRho'] = self.data['RHOB'] * (self.data['Vp']**2 - 2 * self.data['Vs']**2)
            self.data['MuRho'] = self.data['RHOB'] * self.data['Vs']**2
        
        return "Elastic properties calculated successfully"

class RockPhysicsModeler:
    def __init__(self, data):
        self.data = data.copy()
        self.model_results = {}
        self.fluid_substitution_results = {}
        self.depth_track_results = {}
    
    def kuster_toksoz_model(self, phi, aspect_ratios, mineral_mixture, fluid_properties):
        mineral_props = {
            'quartz': {'K': 37, 'G': 44, 'rho': 2.65},
            'clay': {'K': 21, 'G': 7, 'rho': 2.58},
            'calcite': {'K': 76.8, 'G': 32, 'rho': 2.71},
            'dolomite': {'K': 94.9, 'G': 45, 'rho': 2.87}
        }
        
        K_mineral, G_mineral, rho_mineral = 0, 0, 0
        
        for mineral, fraction in mineral_mixture.items():
            if mineral in mineral_props:
                K_mineral += fraction * mineral_props[mineral]['K']
                G_mineral += fraction * mineral_props[mineral]['G']
                rho_mineral += fraction * mineral_props[mineral]['rho']
        
        K_dry = K_mineral * (1 - phi)**2.5
        G_dry = G_mineral * (1 - phi)**2.5
        
        K_fluid = fluid_properties['K_fluid']
        K_sat = K_dry + (1 - K_dry/K_mineral)**2 / (phi/K_fluid + (1 - phi)/K_mineral - K_dry/K_mineral**2)
        G_sat = G_dry
        
        rho_sat = rho_mineral * (1 - phi) + fluid_properties['rho_fluid'] * phi
        
        Vp_sat = np.sqrt((K_sat + 4/3 * G_sat) / rho_sat) * 1000
        Vs_sat = np.sqrt(G_sat / rho_sat) * 1000
        
        return Vp_sat, Vs_sat, rho_sat, K_sat, G_sat

    def model_depth_range(self, depth_min, depth_max, mineral_params, fluid_params, aspect_params):
        """
        Model Vp, Vs, RHOB for a depth range using porosity log
        """
        depth_mask = (self.data['Depth'] >= depth_min) & (self.data['Depth'] <= depth_max)
        depth_data = self.data[depth_mask].copy()
        
        if depth_data.empty:
            return None
        
        depth_values = depth_data['Depth'].values
        phi_values = depth_data['NPHI'].values if 'NPHI' in depth_data.columns else np.full_like(depth_values, 0.15)
        
        Vp_modeled = np.zeros_like(depth_values)
        Vs_modeled = np.zeros_like(depth_values)
        RHOB_modeled = np.zeros_like(depth_values)
        
        mineral_mixture = {
            'quartz': mineral_params['quartz'],
            'clay': mineral_params['clay'], 
            'calcite': mineral_params['calcite']
        }
        
        aspect_ratios = {
            'cracks': {'alpha': aspect_params['crack_ar'], 'fraction': 0.3},
            'pores': {'alpha': aspect_params['pore_ar'], 'fraction': 0.7}
        }
        
        for i, (depth, phi) in enumerate(zip(depth_values, phi_values)):
            Vp_m, Vs_m, rho_m, _, _ = self.kuster_toksoz_model(
                phi, aspect_ratios, mineral_mixture, fluid_params)
            Vp_modeled[i] = Vp_m
            Vs_modeled[i] = Vs_m
            RHOB_modeled[i] = rho_m
        
        results_df = pd.DataFrame({
            'Depth': depth_values,
            'Vp_modeled': Vp_modeled,
            'Vs_modeled': Vs_modeled, 
            'RHOB_modeled': RHOB_modeled
        })
        
        # Add measured data if available
        if 'Vp' in depth_data.columns:
            results_df['Vp_measured'] = depth_data['Vp'].values
        if 'Vs' in depth_data.columns:
            results_df['Vs_measured'] = depth_data['Vs'].values
        if 'RHOB' in depth_data.columns:
            results_df['RHOB_measured'] = depth_data['RHOB'].values
        if 'NPHI' in depth_data.columns:
            results_df['NPHI'] = depth_data['NPHI'].values
        if 'SW' in depth_data.columns:
            results_df['SW'] = depth_data['SW'].values
        
        self.depth_track_results = results_df
        return results_df

def create_sample_data():
    np.random.seed(42)
    n_points = 200
    
    depth = np.linspace(1500, 2500, n_points)
    
    clean_sand_brine = (depth >= 1500) & (depth < 1700)
    shaly_sand_oil = (depth >= 1700) & (depth < 2000)
    carbonate_gas = (depth >= 2000) & (depth <= 2500)
    
    sample_data = pd.DataFrame({
        'Depth': depth,
        'Vp': np.where(clean_sand_brine, np.random.normal(3200, 100, n_points),
              np.where(shaly_sand_oil, np.random.normal(2800, 150, n_points),
                      np.random.normal(4200, 120, n_points))),
        'Vs': np.where(clean_sand_brine, np.random.normal(1800, 80, n_points),
              np.where(shaly_sand_oil, np.random.normal(1600, 100, n_points),
                      np.random.normal(2400, 100, n_points))),
        'RHOB': np.where(clean_sand_brine, np.random.normal(2.25, 0.08, n_points),
                np.where(shaly_sand_oil, np.random.normal(2.35, 0.12, n_points),
                        np.random.normal(2.60, 0.10, n_points))),
        'NPHI': np.where(clean_sand_brine, np.random.normal(0.22, 0.03, n_points),
                np.where(shaly_sand_oil, np.random.normal(0.18, 0.04, n_points),
                        np.random.normal(0.08, 0.02, n_points))),
        'GR': np.where(clean_sand_brine, np.random.normal(25, 4, n_points),
              np.where(shaly_sand_oil, np.random.normal(65, 12, n_points),
                      np.random.normal(40, 6, n_points))),
        'RT': np.random.lognormal(2.5, 0.5, n_points),
        'SW': np.where(clean_sand_brine, np.random.normal(0.95, 0.02, n_points),
              np.where(shaly_sand_oil, np.random.normal(0.35, 0.08, n_points),
                      np.random.normal(0.60, 0.10, n_points))),
        'VSH': np.where(clean_sand_brine, np.random.normal(0.05, 0.01, n_points),
               np.where(shaly_sand_oil, np.random.normal(0.35, 0.06, n_points),
                       np.random.normal(0.15, 0.04, n_points)))
    })
    
    return sample_data

def calculate_r2(measured, modeled):
    """Calculate R-squared value"""
    if len(measured) < 2:
        return 0.0
    correlation_matrix = np.corrcoef(measured, modeled)
    return correlation_matrix[0,1] ** 2

def gassmann_fluid_substitution(vp_orig, vs_orig, rhob_orig, phi, sw_orig, sw_new, 
                              k_mineral=37, rho_mineral=2.65,
                              k_brine=2.8, rho_brine=1.05,
                              k_oil=1.2, rho_oil=0.85,
                              k_gas=0.1, rho_gas=0.25,
                              original_fluid='brine', new_fluid='gas'):
    """
    Gassmann fluid substitution
    """
    # Fluid properties
    fluid_props = {
        'brine': {'K': k_brine, 'rho': rho_brine},
        'oil': {'K': k_oil, 'rho': rho_oil},
        'gas': {'K': k_gas, 'rho': rho_gas}
    }
    
    # Calculate mixed fluid properties using Wood's equation
    def mixed_fluid_properties(sw, fluid_type):
        if fluid_type == 'brine':
            return fluid_props['brine']['K'], fluid_props['brine']['rho']
        elif fluid_type == 'oil':
            return fluid_props['oil']['K'], fluid_props['oil']['rho']
        elif fluid_type == 'gas':
            return fluid_props['gas']['K'], fluid_props['gas']['rho']
        else:
            # Mixed fluid (simplified)
            k_fluid = 1 / (sw/fluid_props['brine']['K'] + (1-sw)/fluid_props['oil']['K'])
            rho_fluid = sw * fluid_props['brine']['rho'] + (1-sw) * fluid_props['oil']['rho']
            return k_fluid, rho_fluid
    
    # Get fluid properties
    k_fluid_orig, rho_fluid_orig = mixed_fluid_properties(sw_orig, original_fluid)
    k_fluid_new, rho_fluid_new = mixed_fluid_properties(sw_new, new_fluid)
    
    # Calculate dry rock modulus using inverse Gassmann
    k_sat_orig = rhob_orig * (vp_orig**2 - 4/3 * vs_orig**2) * 1e-9  # Convert to GPa
    k_dry = (k_sat_orig * (phi * k_mineral / k_fluid_orig + 1 - phi) - k_mineral) / \
            (phi * k_mineral / k_fluid_orig + k_sat_orig / k_mineral - 1 - phi)
    
    # Gassmann substitution to new fluid
    k_sat_new = k_dry + (1 - k_dry/k_mineral)**2 / (phi/k_fluid_new + (1-phi)/k_mineral - k_dry/k_mineral**2)
    
    # Shear modulus unchanged
    g_sat = rhob_orig * vs_orig**2 * 1e-9  # GPa
    
    # New density
    rhob_new = rho_mineral * (1 - phi) + rho_fluid_new * phi
    
    # New velocities
    vp_new = np.sqrt((k_sat_new + 4/3 * g_sat) / rhob_new * 1e9)  # Convert back to m/s
    vs_new = np.sqrt(g_sat / rhob_new * 1e9)
    
    return vp_new, vs_new, rhob_new

def main():
    st.title("🪨 Rock Physics Analysis Tool")
    st.markdown("""
    This interactive tool performs comprehensive rock physics analysis including:
    - **Data Quality Control**
    - **Elastic Properties Calculation**
    - **Rock Physics Modeling (Single Point & Depth Range)**
    - **Fluid Substitution**
    - **Interactive Visualization**
    """)
    
    # Sidebar for data upload and parameters
    st.sidebar.header("Data Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        use_sample_data = False
    else:
        st.sidebar.info("Using sample data. Upload your own CSV file to use your data.")
        data = create_sample_data()
        use_sample_data = True
    
    # Initialize analyzer and modeler
    analyzer = RockPhysicsAnalyzer(data)
    modeler = RockPhysicsModeler(data)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🔍 Quality Control", 
        "📈 Basic Analysis",
        "🔄 Rock Physics Modeling",
        "💧 Fluid Substitution"
    ])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
        with col2:
            st.subheader("Data Information")
            st.write(f"**Shape:** {data.shape}")
            st.write(f"**Columns:** {list(data.columns)}")
            st.write(f"**Depth Range:** {data['Depth'].min():.1f} - {data['Depth'].max():.1f}")
            
        # Depth plot
        st.subheader("Depth Distribution")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Depth'], y=data.index, mode='lines', name='Depth'))
        fig.update_layout(
            title="Depth Profile",
            xaxis_title="Depth",
            yaxis_title="Index",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Data Quality Control")
        
        qc_results = analyzer.quality_control()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Completeness")
            qc_df = pd.DataFrame(qc_results).T
            qc_df.columns = ['Valid Points', 'Total Points', 'Percentage (%)']
            st.dataframe(qc_df.style.format({'Percentage (%)': '{:.1f}%'}))
        
        with col2:
            st.subheader("Quality Metrics")
            for log, metrics in qc_results.items():
                st.metric(
                    label=f"{log} Data Quality",
                    value=f"{metrics['percentage']:.1f}%",
                    delta=f"{metrics['valid']}/{metrics['total']} points"
                )
        
        # Log statistics
        st.subheader("Log Statistics")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        st.dataframe(data[numeric_cols].describe(), use_container_width=True)
    
    with tab3:
        st.header("Basic Rock Physics Analysis")
        
        # Calculate elastic properties
        if st.button("Calculate Elastic Properties"):
            result = analyzer.calculate_elastic_properties()
            st.success(result)
        
        if 'AI' in analyzer.data.columns:
            st.subheader("Elastic Properties Calculated")
            elastic_props = ['AI', 'Vp_Vs', 'PR', 'LambdaRho', 'MuRho']
            available_props = [prop for prop in elastic_props if prop in analyzer.data.columns]
            
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            
            for i, prop in enumerate(available_props):
                with cols[i % 3]:
                    st.metric(
                        label=prop,
                        value=f"{analyzer.data[prop].mean():.2f}",
                        delta=f"±{analyzer.data[prop].std():.2f} std"
                    )
        
        # Cross-plot
        st.subheader("Cross-Plot Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_log = st.selectbox("X-axis", options=data.columns, index=list(data.columns).index('Vp') if 'Vp' in data.columns else 0)
        with col2:
            y_log = st.selectbox("Y-axis", options=data.columns, index=list(data.columns).index('Vs') if 'Vs' in data.columns else 1)
        with col3:
            color_log = st.selectbox("Color by", options=['None'] + list(data.columns), index=0)
        
        fig = go.Figure()
        
        if color_log != 'None' and color_log in data.columns:
            fig.add_trace(go.Scatter(
                x=data[x_log], y=data[y_log],
                mode='markers',
                marker=dict(
                    size=8,
                    color=data[color_log],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=color_log)
                ),
                hovertemplate=f"<b>{x_log}</b>: %{{x}}<br><b>{y_log}</b>: %{{y}}<br><b>{color_log}</b>: %{{marker.color}}<extra></extra>"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=data[x_log], y=data[y_log],
                mode='markers',
                marker=dict(size=8, color='blue', opacity=0.6),
                hovertemplate=f"<b>{x_log}</b>: %{{x}}<br><b>{y_log}</b>: %{{y}}<extra></extra>"
            ))
        
        fig.update_layout(
            title=f"{y_log} vs {x_log}",
            xaxis_title=x_log,
            yaxis_title=y_log,
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Depth plots
        st.subheader("Depth Plots")
        
        logs_to_plot = st.multiselect(
            "Select logs to plot vs depth",
            options=data.columns,
            default=['GR', 'Vp', 'Vs', 'RHOB'] if all(x in data.columns for x in ['GR', 'Vp', 'Vs', 'RHOB']) else data.columns[:4]
        )
        
        if logs_to_plot:
            fig = make_subplots(
                rows=1, 
                cols=len(logs_to_plot),
                subplot_titles=logs_to_plot,
                shared_yaxes=True
            )
            
            for i, log in enumerate(logs_to_plot, 1):
                fig.add_trace(
                    go.Scatter(x=data[log], y=data['Depth'], mode='lines', name=log),
                    row=1, col=i
                )
                fig.update_xaxes(title_text=log, row=1, col=i)
            
            fig.update_yaxes(title_text="Depth", row=1, col=1)
            fig.update_layout(
                title="Logs vs Depth",
                template="plotly_white",
                height=600,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Rock Physics Modeling")
        
        st.markdown("""
        Model rock elastic properties using the Kuster-Toksoz model. Choose between single point modeling or depth range modeling.
        """)
        
        modeling_type = st.radio("Modeling Type", ["Single Point", "Depth Range"], horizontal=True)
        
        # Quick Presets Section
        st.subheader("🎯 Quick Presets for Common Rock Types")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🟡 Clean Sandstone Preset", use_container_width=True):
                st.session_state.min_quartz = 0.75
                st.session_state.min_clay = 0.20
                st.session_state.min_calcite = 0.05
                st.session_state.rock_porosity = 0.20
                st.session_state.rock_crack_ar = 0.02
                st.session_state.rock_pore_ar = 0.8
                st.session_state.fluid_type = "brine"
                st.session_state.fluid_sw = 0.8
                st.success("Clean Sandstone preset applied! Typical: High porosity, quartz-rich")

        with col2:
            if st.button("🟤 Shaly Sandstone Preset", use_container_width=True):
                st.session_state.min_quartz = 0.60
                st.session_state.min_clay = 0.35
                st.session_state.min_calcite = 0.05
                st.session_state.rock_porosity = 0.15
                st.session_state.rock_crack_ar = 0.01
                st.session_state.rock_pore_ar = 0.7
                st.session_state.fluid_type = "mixed"
                st.session_state.fluid_sw = 0.5
                st.success("Shaly Sandstone preset applied! Typical: Medium porosity, clay-rich")

        with col3:
            if st.button("⚪ Carbonate Preset", use_container_width=True):
                st.session_state.min_quartz = 0.10
                st.session_state.min_clay = 0.10
                st.session_state.min_calcite = 0.80
                st.session_state.rock_porosity = 0.10
                st.session_state.rock_crack_ar = 0.005
                st.session_state.rock_pore_ar = 0.9
                st.session_state.fluid_type = "brine"
                st.session_state.fluid_sw = 1.0
                st.success("Carbonate preset applied! Typical: Low porosity, calcite-rich")
        
        # Parameter Tuning Guide
        with st.expander("🎛️ Parameter Tuning Guide (Click to expand)"):
            st.markdown("""
            **Quick Fixes for Better R²:**
            
            | Problem | Solution |
            |---------|----------|
            | **Vp too high** | ↑ Porosity, ↑ Clay, ↓ Quartz, ↑ Crack AR |
            | **Vp too low** | ↓ Porosity, ↑ Quartz, ↓ Clay, ↑ Pore AR |
            | **Vp/Vs too high** | ↓ Crack AR, ↑ Clay, ↓ Porosity |
            | **Vp/Vs too low** | ↑ Pore AR, ↓ Clay, ↑ Quartz |
            
            **Expected R² Values:**
            - 🟢 Excellent: R² > 0.9
            - 🟡 Good: R² = 0.7-0.9  
            - 🟠 Reasonable: R² = 0.5-0.7
            - 🔴 Poor: R² < 0.5
            """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Mineral Composition")
            quartz = st.slider("Quartz Fraction", 0.0, 1.0, 0.6, 0.1, key="min_quartz",
                              help="Increase for higher Vp, decrease for lower Vp")
            clay = st.slider("Clay Fraction", 0.0, 1.0, 0.3, 0.1, key="min_clay",
                            help="Increase for lower Vp/Vs, decrease for higher Vp/Vs")
            calcite = st.slider("Calcite Fraction", 0.0, 1.0, 0.1, 0.1, key="min_calcite",
                               help="Increase for carbonates, typically high Vp")
        
        with col2:
            st.subheader("Rock Properties")
            porosity = st.slider("Porosity", 0.01, 0.4, 0.15, 0.01, key="rock_porosity",
                                help="Most sensitive parameter! High porosity = lower Vp")
            crack_ar = st.slider("Crack Aspect Ratio", 0.001, 0.1, 0.01, 0.001, key="rock_crack_ar",
                                help="Lower values = more cracks = lower Vp/Vs")
            pore_ar = st.slider("Pore Aspect Ratio", 0.1, 1.0, 0.8, 0.1, key="rock_pore_ar",
                               help="Higher values = spherical pores = higher Vp/Vs")
        
        with col3:
            st.subheader("Fluid Properties")
            fluid_type = st.selectbox("Fluid Type", ['brine', 'oil', 'gas', 'mixed'], key="fluid_type",
                                     help="Brine: highest Vp, Gas: lowest Vp")
            water_sat = st.slider("Water Saturation", 0.0, 1.0, 0.8, 0.1, key="fluid_sw",
                                 help="High Sw = brine-like, Low Sw = hydrocarbon-like")
            
            if modeling_type == "Depth Range":
                st.subheader("Depth Range")
                depth_min = st.number_input("Start Depth", 
                                          value=float(data['Depth'].min()),
                                          min_value=float(data['Depth'].min()),
                                          max_value=float(data['Depth'].max()),
                                          key="depth_min")
                depth_max = st.number_input("End Depth", 
                                          value=float(data['Depth'].max()),
                                          min_value=float(data['Depth'].min()),
                                          max_value=float(data['Depth'].max()),
                                          key="depth_max")
        
        # Normalize mineral fractions
        total = quartz + clay + calcite
        quartz /= total
        clay /= total
        calcite /= total
        
        mineral_mixture = {
            'quartz': quartz,
            'clay': clay,
            'calcite': calcite
        }
        
        aspect_ratios = {
            'cracks': {'alpha': crack_ar, 'fraction': 0.3},
            'pores': {'alpha': pore_ar, 'fraction': 0.7}
        }
        
        fluid_scenarios = {
            'brine': {'K_fluid': 2.8, 'rho_fluid': 1.05},
            'oil': {'K_fluid': 1.2, 'rho_fluid': 0.85},
            'gas': {'K_fluid': 0.1, 'rho_fluid': 0.25},
            'mixed': {'K_fluid': 1.5, 'rho_fluid': 0.95}
        }
        
        fluid_properties = fluid_scenarios[fluid_type]
        
        if st.button("Run Modeling", type="primary"):
            if modeling_type == "Single Point":
                # Single point modeling
                Vp_model, Vs_model, rho_model, K_sat, G_sat = modeler.kuster_toksoz_model(
                    porosity, aspect_ratios, mineral_mixture, fluid_properties)
                
                modeler.model_results = {
                    'Vp_model': Vp_model,
                    'Vs_model': Vs_model,
                    'rho_model': rho_model,
                    'mineral_composition': mineral_mixture,
                    'porosity': porosity,
                    'fluid_properties': fluid_properties
                }
                
                # Display single point results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Modeled Vp", f"{Vp_model:.0f} m/s")
                with col2:
                    st.metric("Modeled Vs", f"{Vs_model:.0f} m/s")
                with col3:
                    st.metric("Modeled RHOB", f"{rho_model:.2f} g/cc")
                
                # Single point comparison plots
                if all(log in data.columns for log in ['Vp', 'Vs', 'RHOB']):
                    st.subheader("Single Point Comparison")
                    
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=['Vp Distribution', 'Vs Distribution', 'RHOB Distribution', 'Mineral Composition'],
                        specs=[[{"type": "histogram"}, {"type": "histogram"}],
                               [{"type": "histogram"}, {"type": "pie"}]]
                    )
                    
                    # Vp histogram
                    fig.add_trace(
                        go.Histogram(x=data['Vp'], name='Measured Vp', opacity=0.7, nbinsx=30),
                        row=1, col=1
                    )
                    fig.add_vline(x=Vp_model, line_dash="dash", line_color="red", row=1, col=1,
                                 annotation_text="Modeled", annotation_position="top")
                    
                    # Vs histogram
                    fig.add_trace(
                        go.Histogram(x=data['Vs'], name='Measured Vs', opacity=0.7, nbinsx=30),
                        row=1, col=2
                    )
                    fig.add_vline(x=Vs_model, line_dash="dash", line_color="red", row=1, col=2,
                                 annotation_text="Modeled", annotation_position="top")
                    
                    # RHOB histogram
                    fig.add_trace(
                        go.Histogram(x=data['RHOB'], name='Measured RHOB', opacity=0.7, nbinsx=30),
                        row=2, col=1
                    )
                    fig.add_vline(x=rho_model, line_dash="dash", line_color="red", row=2, col=1,
                                 annotation_text="Modeled", annotation_position="top")
                    
                    # Mineral composition pie chart
                    labels = [f'{mineral.capitalize()}' for mineral in mineral_mixture.keys()]
                    values = list(mineral_mixture.values())
                    fig.add_trace(
                        go.Pie(labels=labels, values=values, name="Mineral Composition"),
                        row=2, col=2
                    )
                    
                    fig.update_layout(height=600, showlegend=False, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Depth range modeling
                mineral_params = {
                    'quartz': quartz,
                    'clay': clay,
                    'calcite': calcite
                }
                aspect_params = {
                    'crack_ar': crack_ar,
                    'pore_ar': pore_ar
                }
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_df = modeler.model_depth_range(depth_min, depth_max, mineral_params, fluid_properties, aspect_params)
                
                if results_df is not None:
                    st.success(f"Depth range modeling completed for {len(results_df)} points!")
                    
                    # Display depth range results
                    st.subheader("Depth Range Modeling Results")
                    
                    # Calculate R² values
                    if all(col in results_df.columns for col in ['Vp_measured', 'Vp_modeled', 'Vs_measured', 'Vs_modeled', 'RHOB_measured', 'RHOB_modeled']):
                        r2_vp = calculate_r2(results_df['Vp_measured'], results_df['Vp_modeled'])
                        r2_vs = calculate_r2(results_df['Vs_measured'], results_df['Vs_modeled'])
                        r2_rhob = calculate_r2(results_df['RHOB_measured'], results_df['RHOB_modeled'])
                        
                        # R² metrics with color coding
                        st.subheader("Model Quality Metrics (R²)")
                        col1, col2, col3 = st.columns(3)
                        
                        # Color code based on R² value
                        vp_color = "green" if r2_vp > 0.7 else "orange" if r2_vp > 0.5 else "red"
                        vs_color = "green" if r2_vs > 0.7 else "orange" if r2_vs > 0.5 else "red"
                        rhob_color = "green" if r2_rhob > 0.7 else "orange" if r2_rhob > 0.5 else "red"
                        
                        with col1:
                            st.metric("Vp R²", f"{r2_vp:.4f}", delta_color="off")
                            st.markdown(f"<p style='color: {vp_color}; font-weight: bold;'>"
                                      f"{'Excellent' if r2_vp > 0.9 else 'Good' if r2_vp > 0.7 else 'Reasonable' if r2_vp > 0.5 else 'Poor'}"
                                      f"</p>", unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("Vs R²", f"{r2_vs:.4f}", delta_color="off")
                            st.markdown(f"<p style='color: {vs_color}; font-weight: bold;'>"
                                      f"{'Excellent' if r2_vs > 0.9 else 'Good' if r2_vs > 0.7 else 'Reasonable' if r2_vs > 0.5 else 'Poor'}"
                                      f"</p>", unsafe_allow_html=True)
                        
                        with col3:
                            st.metric("RHOB R²", f"{r2_rhob:.4f}", delta_color="off")
                            st.markdown(f"<p style='color: {rhob_color}; font-weight: bold;'>"
                                      f"{'Excellent' if r2_rhob > 0.9 else 'Good' if r2_rhob > 0.7 else 'Reasonable' if r2_rhob > 0.5 else 'Poor'}"
                                      f"</p>", unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        avg_vp = results_df['Vp_modeled'].mean()
                        st.metric("Average Modeled Vp", f"{avg_vp:.0f} m/s")
                    with col2:
                        avg_vs = results_df['Vs_modeled'].mean()
                        st.metric("Average Modeled Vs", f"{avg_vs:.0f} m/s")
                    with col3:
                        avg_rhob = results_df['RHOB_modeled'].mean()
                        st.metric("Average Modeled RHOB", f"{avg_rhob:.2f} g/cc")
                    with col4:
                        st.metric("Depth Points", len(results_df))
                    
                    # Export results
                    st.subheader("Export Modeling Results")
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Modeling Results CSV",
                        data=csv,
                        file_name="rock_physics_modeling_results.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
                    # Store results in session state
                    st.session_state.depth_range_results = results_df
                    
                else:
                    st.error("No data found in the selected depth range. Please adjust depth range.")
    
    with tab5:
        st.header("Fluid Substitution Analysis")
        
        st.markdown("""
        Gassmann fluid substitution analysis. Select a depth range and fluid scenario to model Vp, Vs, and RHOB changes.
        Compare modeled results with measured data and export the results.
        """)
        
        # Fluid Substitution Quick Presets
        st.subheader("🎯 Fluid Substitution Presets")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Brine → Gas", use_container_width=True):
                st.session_state.fs_orig_sw = 1.0
                st.session_state.fs_orig_fluid = "brine"
                st.session_state.fs_new_sw = 0.0
                st.session_state.fs_new_fluid = "gas"
                st.session_state.fs_scenario = "Brine to Gas"
                st.success("Brine to Gas: Largest Vp decrease expected")

        with col2:
            if st.button("Oil → Brine", use_container_width=True):
                st.session_state.fs_orig_sw = 0.2
                st.session_state.fs_orig_fluid = "oil"
                st.session_state.fs_new_sw = 1.0
                st.session_state.fs_new_fluid = "brine"
                st.session_state.fs_scenario = "Oil to Brine"
                st.success("Oil to Brine: Moderate Vp increase expected")

        with col3:
            if st.button("Gas → Brine", use_container_width=True):
                st.session_state.fs_orig_sw = 0.0
                st.session_state.fs_orig_fluid = "gas"
                st.session_state.fs_new_sw = 1.0
                st.session_state.fs_new_fluid = "brine"
                st.session_state.fs_scenario = "Gas to Brine"
                st.success("Gas to Brine: Largest Vp increase expected")
        
        # Parameter Tuning Guide for Fluid Substitution
        with st.expander("🎛️ Fluid Substitution Tuning Guide"):
            st.markdown("""
            **Parameter Sensitivity:**
            
            | Parameter | Effect on Vp | Typical Range |
            |-----------|-------------|---------------|
            | **Brine K** | High K = High Vp | 2.2-3.0 GPa |
            | **Oil K** | Medium K = Medium Vp | 0.8-1.5 GPa |
            | **Gas K** | Low K = Low Vp | 0.05-0.2 GPa |
            | **Water Sat** | High Sw = Brine-like | 0.0-1.0 |
            
            **Quick Tips:**
            - For **larger Vp changes**: Increase contrast between original and new fluid K
            - For **better R²**: Adjust fluid K values to match your reservoir conditions
            - **Gas has biggest effect** (lowest K), **Brine has smallest effect** (highest K)
            """)
        
        # Depth range selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_depth = st.number_input("Start Depth", 
                                      value=float(data['Depth'].min()),
                                      min_value=float(data['Depth'].min()),
                                      max_value=float(data['Depth'].max()),
                                      key="fs_depth_min")
        
        with col2:
            max_depth = st.number_input("End Depth", 
                                      value=float(data['Depth'].max()),
                                      min_value=float(data['Depth'].min()),
                                      max_value=float(data['Depth'].max()),
                                      key="fs_depth_max")
        
        with col3:
            fluid_scenario = st.selectbox(
                "Fluid Scenario",
                options=['Full Brine', 'Full Oil', 'Full Gas', 'Oil to Gas', 'Brine to Oil', 'Gas to Brine'],
                help="Select the fluid substitution scenario",
                key="fs_scenario"
            )
        
        # Model parameters
        st.subheader("Model Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Fluid Properties**")
            original_sw = st.slider("Original Water Saturation", 0.0, 1.0, 0.8, 0.01, key="fs_orig_sw",
                                   help="High = brine-dominated, Low = hydrocarbon-dominated")
            original_fluid = st.selectbox("Original Fluid Type", ['brine', 'oil', 'gas'], key="fs_orig_fluid",
                                         help="Original fluid in the reservoir")
        
        with col2:
            st.markdown("**New Fluid Properties**")
            new_sw = st.slider("New Water Saturation", 0.0, 1.0, 0.2, 0.01, key="fs_new_sw",
                              help="New fluid saturation after substitution")
            new_fluid = st.selectbox("New Fluid Type", ['brine', 'oil', 'gas'], key="fs_new_fluid",
                                    help="New fluid replacing the original")
        
        # Advanced parameters
        with st.expander("Advanced Fluid Parameters"):
            st.info("Adjust these for fine-tuning fluid substitution results")
            col1, col2, col3 = st.columns(3)
            with col1:
                k_brine = st.number_input("Brine Bulk Modulus (GPa)", 2.0, 3.0, 2.8, 0.1, key="adv_k_brine",
                                         help="Higher values = higher Vp, typical: 2.2-3.0 GPa")
                rho_brine = st.number_input("Brine Density (g/cc)", 1.0, 1.1, 1.05, 0.01, key="adv_rho_brine",
                                           help="Typical: 1.0-1.1 g/cc")
            with col2:
                k_oil = st.number_input("Oil Bulk Modulus (GPa)", 0.5, 1.5, 1.2, 0.1, key="adv_k_oil",
                                       help="Medium values = medium Vp, typical: 0.8-1.5 GPa")
                rho_oil = st.number_input("Oil Density (g/cc)", 0.7, 0.9, 0.85, 0.01, key="adv_rho_oil",
                                         help="Typical: 0.7-0.9 g/cc")
            with col3:
                k_gas = st.number_input("Gas Bulk Modulus (GPa)", 0.01, 0.2, 0.1, 0.01, key="adv_k_gas",
                                       help="Lower values = lower Vp, typical: 0.05-0.2 GPa")
                rho_gas = st.number_input("Gas Density (g/cc)", 0.1, 0.3, 0.25, 0.01, key="adv_rho_gas",
                                         help="Typical: 0.2-0.3 g/cc")
        
        if st.button("Run Fluid Substitution Modeling", type="primary"):
            # Filter data for selected depth range
            depth_mask = (data['Depth'] >= min_depth) & (data['Depth'] <= max_depth)
            depth_data = data[depth_mask].copy()
            
            if depth_data.empty:
                st.error("No data found in the selected depth range. Please adjust depth range.")
            else:
                # Initialize progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Perform fluid substitution for each depth point
                results = []
                total_points = len(depth_data)
                
                for i, (idx, row) in enumerate(depth_data.iterrows()):
                    status_text.text(f"Processing depth {row['Depth']:.1f} ({i+1}/{total_points})")
                    
                    # Check if required logs are available
                    if all(log in row for log in ['Vp', 'Vs', 'RHOB', 'NPHI', 'SW']):
                        try:
                            vp_new, vs_new, rhob_new = gassmann_fluid_substitution(
                                vp_orig=row['Vp'],
                                vs_orig=row['Vs'], 
                                rhob_orig=row['RHOB'],
                                phi=row['NPHI'],
                                sw_orig=original_sw,
                                sw_new=new_sw,
                                original_fluid=original_fluid,
                                new_fluid=new_fluid,
                                k_brine=k_brine, rho_brine=rho_brine,
                                k_oil=k_oil, rho_oil=rho_oil,
                                k_gas=k_gas, rho_gas=rho_gas
                            )
                            
                            result_point = {
                                'Depth': row['Depth'],
                                'Vp_measured': row['Vp'],
                                'Vs_measured': row['Vs'],
                                'RHOB_measured': row['RHOB'],
                                'Vp_modeled': vp_new,
                                'Vs_modeled': vs_new,
                                'RHOB_modeled': rhob_new,
                                'NPHI': row['NPHI'],
                                'SW_original': original_sw,
                                'SW_new': new_sw,
                                'Fluid_Scenario': fluid_scenario
                            }
                            
                            # Add VSH if available
                            if 'VSH' in row:
                                result_point['VSH'] = row['VSH']
                            # Add current SW if available
                            if 'SW' in row:
                                result_point['SW_current'] = row['SW']
                                
                            results.append(result_point)
                        except Exception as e:
                            st.warning(f"Error at depth {row['Depth']:.1f}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / total_points)
                
                if results:
                    results_df = pd.DataFrame(results)
                    modeler.fluid_substitution_results = results_df
                    
                    st.success(f"Fluid substitution completed for {len(results_df)} depth points!")
                    
                    # Display results
                    st.subheader("Modeling Results")
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        vp_change = ((results_df['Vp_modeled'] - results_df['Vp_measured']) / results_df['Vp_measured'] * 100).mean()
                        st.metric("Average Vp Change", f"{vp_change:+.1f}%")
                    
                    with col2:
                        vs_change = ((results_df['Vs_modeled'] - results_df['Vs_measured']) / results_df['Vs_measured'] * 100).mean()
                        st.metric("Average Vs Change", f"{vs_change:+.1f}%")
                    
                    with col3:
                        rhob_change = ((results_df['RHOB_modeled'] - results_df['RHOB_measured']) / results_df['RHOB_measured'] * 100).mean()
                        st.metric("Average RHOB Change", f"{rhob_change:+.1f}%")
                    
                    with col4:
                        st.metric("Depth Points Modeled", len(results_df))
                    
                    # Calculate R² values
                    r2_vp = calculate_r2(results_df['Vp_measured'], results_df['Vp_modeled'])
                    r2_vs = calculate_r2(results_df['Vs_measured'], results_df['Vs_modeled'])
                    r2_rhob = calculate_r2(results_df['RHOB_measured'], results_df['RHOB_modeled'])
                    
                    # R² metrics with color coding
                    st.subheader("Model Quality Metrics (R²)")
                    col1, col2, col3 = st.columns(3)
                    
                    # Color code based on R² value
                    vp_color = "green" if r2_vp > 0.7 else "orange" if r2_vp > 0.5 else "red"
                    vs_color = "green" if r2_vs > 0.7 else "orange" if r2_vs > 0.5 else "red"
                    rhob_color = "green" if r2_rhob > 0.7 else "orange" if r2_rhob > 0.5 else "red"
                    
                    with col1:
                        st.metric("Vp R²", f"{r2_vp:.4f}", delta_color="off")
                        st.markdown(f"<p style='color: {vp_color}; font-weight: bold;'>"
                                  f"{'Excellent' if r2_vp > 0.9 else 'Good' if r2_vp > 0.7 else 'Reasonable' if r2_vp > 0.5 else 'Poor'}"
                                  f"</p>", unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Vs R²", f"{r2_vs:.4f}", delta_color="off")
                        st.markdown(f"<p style='color: {vs_color}; font-weight: bold;'>"
                                  f"{'Excellent' if r2_vs > 0.9 else 'Good' if r2_vs > 0.7 else 'Reasonable' if r2_vs > 0.5 else 'Poor'}"
                                  f"</p>", unsafe_allow_html=True)
                    
                    with col3:
                        st.metric("RHOB R²", f"{r2_rhob:.4f}", delta_color="off")
                        st.markdown(f"<p style='color: {rhob_color}; font-weight: bold;'>"
                                  f"{'Excellent' if r2_rhob > 0.9 else 'Good' if r2_rhob > 0.7 else 'Reasonable' if r2_rhob > 0.5 else 'Poor'}"
                                  f"</p>", unsafe_allow_html=True)
                    
                    # Export results
                    st.subheader("Export Results")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        filename = st.text_input("Output filename", "fluid_substitution_results.csv")
                    
                    with col2:
                        st.markdown("### Download Results")
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=filename,
                            mime="text/csv",
                            type="primary"
                        )
                    
                    # Store results for potential further analysis
                    st.session_state.fluid_substitution_results = results_df
                    
                else:
                    st.error("No valid results generated. Please check your input parameters and data quality.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    This Rock Physics Analysis Tool provides:
    - Data quality assessment
    - Elastic properties calculation
    - Rock physics modeling (Single Point & Depth Range)
    - Fluid substitution analysis
    - Interactive visualization
    
    Upload your own CSV data or use the sample data provided.
    """)

if __name__ == "__main__":
    main()
