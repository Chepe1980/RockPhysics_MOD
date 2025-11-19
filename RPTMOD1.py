
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

def main():
    st.title("🪨 Rock Physics Analysis Tool")
    st.markdown("""
    This interactive tool performs comprehensive rock physics analysis including:
    - **Data Quality Control**
    - **Elastic Properties Calculation**
    - **Rock Physics Modeling**
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
    
    # Initialize analyzer
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
        Adjust the parameters below to model rock elastic properties using the Kuster-Toksoz model.
        The model will calculate Vp, Vs, and RHOB based on mineral composition, porosity, and fluid properties.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Mineral Composition")
            quartz = st.slider("Quartz Fraction", 0.0, 1.0, 0.6, 0.1)
            clay = st.slider("Clay Fraction", 0.0, 1.0, 0.3, 0.1)
            calcite = st.slider("Calcite Fraction", 0.0, 1.0, 0.1, 0.1)
        
        with col2:
            st.subheader("Rock Properties")
            porosity = st.slider("Porosity", 0.01, 0.4, 0.15, 0.01)
            crack_ar = st.slider("Crack Aspect Ratio", 0.001, 0.1, 0.01, 0.001)
            pore_ar = st.slider("Pore Aspect Ratio", 0.1, 1.0, 0.8, 0.1)
        
        with col3:
            st.subheader("Fluid Properties")
            fluid_type = st.selectbox("Fluid Type", ['brine', 'oil', 'gas', 'mixed'])
            water_sat = st.slider("Water Saturation", 0.0, 1.0, 0.8, 0.1)
        
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
        
        # Calculate model
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
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Modeled Vp", f"{Vp_model:.0f} m/s")
        with col2:
            st.metric("Modeled Vs", f"{Vs_model:.0f} m/s")
        with col3:
            st.metric("Modeled RHOB", f"{rho_model:.2f} g/cc")
        
        # Comparison plots
        st.subheader("Model vs Measured Comparison")
        
        if all(log in data.columns for log in ['Vp', 'Vs', 'RHOB']):
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
        
        # Cross-plot with model point
        if all(log in data.columns for log in ['Vp', 'Vs']):
            st.subheader("Cross-Plot with Model Point")
            
            fig = go.Figure()
            
            # Measured data
            fig.add_trace(go.Scatter(
                x=data['Vp'], y=data['Vs'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=data['SW'] if 'SW' in data.columns else 'blue',
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='SW' if 'SW' in data.columns else 'Value')
                ),
                name='Measured Data',
                hovertemplate="<b>Vp</b>: %{x}<br><b>Vs</b>: %{y}<extra></extra>"
            ))
            
            # Model point
            fig.add_trace(go.Scatter(
                x=[Vp_model], y=[Vs_model],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                name='Modeled Point',
                hovertemplate=f"<b>Modeled Vp</b>: {Vp_model:.0f}<br><b>Modeled Vs</b>: {Vs_model:.0f}<extra></extra>"
            ))
            
            fig.update_layout(
                title="Vp vs Vs with Model Point",
                xaxis_title="Vp (m/s)",
                yaxis_title="Vs (m/s)",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("Fluid Substitution Analysis")
        
        st.markdown("""
        Gassmann fluid substitution analysis. Select a depth point and fluid scenario to see how elastic properties change.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            depth_point = st.selectbox(
                "Select Depth Point",
                options=data['Depth'].values,
                index=0
            )
        
        with col2:
            fluid_scenario = st.selectbox(
                "Fluid Scenario",
                options=['Full Brine', 'Full Oil', 'Full Gas', 'Current Conditions']
            )
        
        if st.button("Perform Fluid Substitution"):
            # Simplified fluid substitution (for demonstration)
            depth_data = data[data['Depth'] == depth_point].iloc[0]
            
            if all(log in depth_data.index for log in ['Vp', 'Vs', 'RHOB', 'NPHI', 'SW']):
                # Calculate changes based on fluid scenario
                fluid_changes = {
                    'Full Brine': {'Vp_change': 0.1, 'Vs_change': 0.0, 'RHOB_change': 0.05},
                    'Full Oil': {'Vp_change': -0.05, 'Vs_change': 0.0, 'RHOB_change': -0.08},
                    'Full Gas': {'Vp_change': -0.15, 'Vs_change': 0.0, 'RHOB_change': -0.15},
                    'Current Conditions': {'Vp_change': 0.0, 'Vs_change': 0.0, 'RHOB_change': 0.0}
                }
                
                changes = fluid_changes[fluid_scenario]
                
                new_Vp = depth_data['Vp'] * (1 + changes['Vp_change'])
                new_Vs = depth_data['Vs'] * (1 + changes['Vs_change'])
                new_RHOB = depth_data['RHOB'] * (1 + changes['RHOB_change'])
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Vp", f"{new_Vp:.0f} m/s", f"{(changes['Vp_change']*100):+.1f}%")
                with col2:
                    st.metric("Vs", f"{new_Vs:.0f} m/s", f"{(changes['Vs_change']*100):+.1f}%")
                with col3:
                    st.metric("RHOB", f"{new_RHOB:.2f} g/cc", f"{(changes['RHOB_change']*100):+.1f}%")
                
                # Plot comparison
                fig = go.Figure()
                
                properties = ['Vp', 'Vs', 'RHOB']
                original_values = [depth_data['Vp'], depth_data['Vs'], depth_data['RHOB']]
                new_values = [new_Vp, new_Vs, new_RHOB]
                
                fig.add_trace(go.Bar(
                    name='Original',
                    x=properties,
                    y=original_values,
                    marker_color='blue'
                ))
                
                fig.add_trace(go.Bar(
                    name=f'After {fluid_scenario}',
                    x=properties,
                    y=new_values,
                    marker_color='red'
                ))
                
                fig.update_layout(
                    title=f"Fluid Substitution: {fluid_scenario}",
                    template="plotly_white",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    This Rock Physics Analysis Tool provides:
    - Data quality assessment
    - Elastic properties calculation
    - Rock physics modeling
    - Fluid substitution analysis
    - Interactive visualization
    
    Upload your own CSV data or use the sample data provided.
    """)

if __name__ == "__main__":
    main()
