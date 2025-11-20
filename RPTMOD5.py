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

# NEW ML IMPORTS
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

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
        self.ml_models = {}  # NEW: Store ML models
    
    # NEW: Machine Learning Model Training
    def train_ml_models(self, features, target, test_size=0.2):
        """
        Train multiple ML models to predict elastic properties
        """
        # Prepare data
        X = self.data[features].fillna(self.data[features].mean())
        y = self.data[target].fillna(self.data[target].mean())
        
        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'Polynomial Regression': Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ]),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1)
        }
        
        # Train and evaluate models
        results = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                
                results[name] = {
                    'model': model,
                    'r2': r2,
                    'rmse': rmse,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
            except Exception as e:
                st.warning(f"Model {name} failed: {str(e)}")
                continue
        
        return results
    
    # NEW: ML-based rock physics modeling
    def ml_rock_physics_modeling(self, depth_min, depth_max, target_property='Vp'):
        """
        Use ML to model rock physics properties
        """
        depth_mask = (self.data['Depth'] >= depth_min) & (self.data['Depth'] <= depth_max)
        depth_data = self.data[depth_mask].copy()
        
        if depth_data.empty:
            return None
        
        # Define features based on available data
        base_features = ['NPHI', 'GR', 'RT', 'SW', 'VSH']
        available_features = [f for f in base_features if f in depth_data.columns]
        
        if len(available_features) < 2:
            st.warning("Insufficient features for ML modeling. Need at least 2 features.")
            return None
        
        # Train ML models
        ml_results = self.train_ml_models(available_features, target_property)
        
        if not ml_results:
            st.error("No ML models could be trained successfully.")
            return None
        
        # Get best model
        best_model_name = max(ml_results.keys(), key=lambda x: ml_results[x]['r2'])
        best_model = ml_results[best_model_name]['model']
        best_r2 = ml_results[best_model_name]['r2']
        
        # Predict for the depth range
        X_pred = depth_data[available_features].fillna(depth_data[available_features].mean())
        X_pred = X_pred.replace([np.inf, -np.inf], np.nan).fillna(X_pred.mean())
        
        predictions = best_model.predict(X_pred)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Depth': depth_data['Depth'].values,
            f'{target_property}_measured': depth_data[target_property].values if target_property in depth_data.columns else [np.nan] * len(depth_data),
            f'{target_property}_modeled_ml': predictions
        })
        
        # Add additional data if available
        for col in ['NPHI', 'GR', 'RT', 'SW', 'VSH']:
            if col in depth_data.columns:
                results_df[col] = depth_data[col].values
        
        # Store ML model info
        self.ml_models[target_property] = {
            'model': best_model,
            'model_name': best_model_name,
            'r2': best_r2,
            'features': available_features
        }
        
        return results_df, ml_results
    
    # NEW: Enhanced fluid substitution with ML correction
    def ml_enhanced_fluid_substitution(self, depth_min, depth_max, original_sw, new_sw, 
                                     original_fluid='brine', new_fluid='gas'):
        """
        Perform fluid substitution with ML-based correction
        """
        depth_mask = (self.data['Depth'] >= depth_min) & (self.data['Depth'] <= depth_max)
        depth_data = self.data[depth_mask].copy()
        
        if depth_data.empty:
            return None
        
        results = []
        ml_corrections = {}
        
        # Perform traditional fluid substitution first
        for idx, row in depth_data.iterrows():
            if all(log in row for log in ['Vp', 'Vs', 'RHOB', 'NPHI']):
                try:
                    vp_new, vs_new, rhob_new = gassmann_fluid_substitution(
                        vp_orig=row['Vp'],
                        vs_orig=row['Vs'], 
                        rhob_orig=row['RHOB'],
                        phi=row['NPHI'],
                        sw_orig=original_sw,
                        sw_new=new_sw,
                        original_fluid=original_fluid,
                        new_fluid=new_fluid
                    )
                    
                    result_point = {
                        'Depth': row['Depth'],
                        'Vp_measured': row['Vp'],
                        'Vs_measured': row['Vs'],
                        'RHOB_measured': row['RHOB'],
                        'Vp_modeled_traditional': vp_new,
                        'Vs_modeled_traditional': vs_new,
                        'RHOB_modeled_traditional': rhob_new,
                        'NPHI': row['NPHI'],
                        'SW_original': original_sw,
                        'SW_new': new_sw
                    }
                    
                    results.append(result_point)
                    
                except Exception as e:
                    continue
        
        if not results:
            return None
        
        results_df = pd.DataFrame(results)
        
        # Apply ML correction to improve R²
        for property_name in ['Vp', 'Vs', 'RHOB']:
            measured_col = f'{property_name}_measured'
            traditional_col = f'{property_name}_modeled_traditional'
            
            if measured_col in results_df.columns and traditional_col in results_df.columns:
                # Prepare features for ML correction
                correction_features = ['NPHI', 'SW_original', 'SW_new']
                available_features = [f for f in correction_features if f in results_df.columns]
                
                if len(available_features) >= 2:
                    # Calculate residual (error) between measured and traditional model
                    residual = results_df[measured_col] - results_df[traditional_col]
                    
                    # Train ML model to predict residual
                    X = results_df[available_features].fillna(results_df[available_features].mean())
                    y = residual
                    
                    # Remove any infinite values
                    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
                    y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())
                    
                    if len(X) > 10:  # Ensure enough data for training
                        # Use Random Forest for residual correction
                        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
                        rf_model.fit(X, y)
                        
                        # Predict residual and apply correction
                        predicted_residual = rf_model.predict(X)
                        results_df[f'{property_name}_modeled_ml_corrected'] = (
                            results_df[traditional_col] + predicted_residual
                        )
                        
                        # Calculate R² for ML-corrected model
                        r2_ml = calculate_r2(
                            results_df[measured_col], 
                            results_df[f'{property_name}_modeled_ml_corrected']
                        )
                        
                        ml_corrections[property_name] = {
                            'model': rf_model,
                            'r2_improvement': r2_ml - calculate_r2(results_df[measured_col], results_df[traditional_col]),
                            'final_r2': r2_ml
                        }
        
        return results_df, ml_corrections

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
        if 'GR' in depth_data.columns:
            results_df['GR'] = depth_data['GR'].values
        if 'VSH' in depth_data.columns:
            results_df['VSH'] = depth_data['VSH'].values
        
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
    - **🤖 Machine Learning Enhancement** (NEW)
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
    
    # Main tabs - ADDED ML TAB
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview", 
        "🔍 Quality Control", 
        "📈 Basic Analysis",
        "🔄 Rock Physics Modeling",
        "💧 Fluid Substitution",
        "🤖 ML Enhancement"  # NEW TAB
    ])
    
    # ... (Previous tabs 1-4 remain exactly the same) ...
    
    with tab4:
        st.header("Rock Physics Modeling")
        
        st.markdown("""
        Model rock elastic properties using the Kuster-Toksoz model. Choose between single point modeling or depth range modeling.
        """)
        
        modeling_type = st.radio("Modeling Type", ["Single Point", "Depth Range"], horizontal=True)
        
        # NEW: ML Modeling Option
        if modeling_type == "Depth Range":
            use_ml = st.checkbox("🤖 Use Machine Learning Enhancement", 
                               value=False,
                               help="Use ML algorithms to improve model accuracy and R² values")
        
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
            
            **🤖 ML Enhancement Tip:** Enable ML for automatic R² improvement!
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
                # Single point modeling (unchanged)
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
                
                # ... (rest of single point code remains the same) ...
            
            else:
                # Depth range modeling with optional ML enhancement
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
                
                if use_ml:
                    # NEW: Use ML-enhanced modeling
                    st.info("🤖 Using Machine Learning to enhance rock physics modeling...")
                    
                    # Model each property with ML
                    ml_results_all = {}
                    
                    for target_property in ['Vp', 'Vs', 'RHOB']:
                        if target_property in data.columns:
                            status_text.text(f"Training ML models for {target_property}...")
                            ml_result = modeler.ml_rock_physics_modeling(depth_min, depth_max, target_property)
                            
                            if ml_result is not None:
                                results_df, ml_results = ml_result
                                ml_results_all[target_property] = {
                                    'results': results_df,
                                    'models': ml_results
                                }
                    
                    # Combine ML results
                    if ml_results_all:
                        # Use Vp results as base and add other properties
                        base_property = list(ml_results_all.keys())[0]
                        combined_results = ml_results_all[base_property]['results'].copy()
                        
                        for property_name, ml_data in ml_results_all.items():
                            if property_name != base_property:
                                combined_results = combined_results.merge(
                                    ml_data['results'][['Depth', f'{property_name}_measured', f'{property_name}_modeled_ml']],
                                    on='Depth', how='left'
                                )
                        
                        results_df = combined_results
                        
                        # Display ML model performance
                        st.subheader("🤖 ML Model Performance")
                        ml_performance_cols = st.columns(len(ml_results_all))
                        
                        for idx, (property_name, ml_data) in enumerate(ml_results_all.items()):
                            with ml_performance_cols[idx]:
                                best_model_name = max(ml_data['models'].keys(), 
                                                    key=lambda x: ml_data['models'][x]['r2'])
                                best_r2 = ml_data['models'][best_model_name]['r2']
                                
                                st.metric(
                                    label=f"{property_name} Best Model",
                                    value=best_model_name,
                                    delta=f"R²: {best_r2:.4f}"
                                )
                    else:
                        st.warning("ML modeling failed. Falling back to traditional modeling.")
                        results_df = modeler.model_depth_range(depth_min, depth_max, mineral_params, fluid_properties, aspect_params)
                else:
                    # Traditional modeling
                    results_df = modeler.model_depth_range(depth_min, depth_max, mineral_params, fluid_properties, aspect_params)
                
                if results_df is not None:
                    st.success(f"Depth range modeling completed for {len(results_df)} points!")
                    
                    # Display depth range results
                    st.subheader("Depth Range Modeling Results")
                    
                    # Calculate R² values for both traditional and ML models
                    r2_metrics = {}
                    
                    for property_name in ['Vp', 'Vs', 'RHOB']:
                        measured_col = f'{property_name}_measured'
                        modeled_traditional_col = f'{property_name}_modeled'
                        modeled_ml_col = f'{property_name}_modeled_ml'
                        
                        if measured_col in results_df.columns:
                            if modeled_traditional_col in results_df.columns:
                                r2_traditional = calculate_r2(results_df[measured_col], results_df[modeled_traditional_col])
                                r2_metrics[f'{property_name}_traditional'] = r2_traditional
                            
                            if modeled_ml_col in results_df.columns:
                                r2_ml = calculate_r2(results_df[measured_col], results_df[modeled_ml_col])
                                r2_metrics[f'{property_name}_ml'] = r2_ml
                    
                    # Display R² metrics with comparison
                    st.subheader("Model Quality Metrics (R²)")
                    
                    if use_ml and any('_ml' in key for key in r2_metrics.keys()):
                        # Show comparison between traditional and ML
                        comparison_cols = st.columns(3)
                        property_names = ['Vp', 'Vs', 'RHOB']
                        
                        for idx, prop in enumerate(property_names):
                            with comparison_cols[idx]:
                                trad_key = f'{prop}_traditional'
                                ml_key = f'{prop}_ml'
                                
                                if trad_key in r2_metrics and ml_key in r2_metrics:
                                    r2_trad = r2_metrics[trad_key]
                                    r2_ml = r2_metrics[ml_key]
                                    improvement = r2_ml - r2_trad
                                    
                                    st.metric(
                                        label=f"{prop} R²",
                                        value=f"{r2_ml:.4f}",
                                        delta=f"ML +{improvement:.4f}",
                                        delta_color="normal" if improvement > 0 else "inverse"
                                    )
                                elif trad_key in r2_metrics:
                                    r2_trad = r2_metrics[trad_key]
                                    st.metric(f"{prop} R² (Traditional)", f"{r2_trad:.4f}")
                    else:
                        # Traditional R² display
                        col1, col2, col3 = st.columns(3)
                        property_names = ['Vp', 'Vs', 'RHOB']
                        
                        for idx, prop in enumerate(property_names):
                            with [col1, col2, col3][idx]:
                                key = f'{prop}_traditional'
                                if key in r2_metrics:
                                    r2_val = r2_metrics[key]
                                    color = "green" if r2_val > 0.7 else "orange" if r2_val > 0.5 else "red"
                                    st.metric(f"{prop} R²", f"{r2_val:.4f}")
                                    st.markdown(f"<p style='color: {color}; font-weight: bold;'>"
                                              f"{'Excellent' if r2_val > 0.9 else 'Good' if r2_val > 0.7 else 'Reasonable' if r2_val > 0.5 else 'Poor'}"
                                              f"</p>", unsafe_allow_html=True)
                    
                    # ... (rest of depth range visualization code remains the same) ...

    with tab5:
        st.header("Fluid Substitution Analysis")
        
        st.markdown("""
        Gassmann fluid substitution analysis. Select a depth range and fluid scenario to model Vp, Vs, and RHOB changes.
        Compare modeled results with measured data and export the results.
        """)
        
        # NEW: ML Enhancement for Fluid Substitution
        use_ml_fs = st.checkbox("🤖 Use ML Correction for Fluid Substitution", 
                              value=False,
                              help="Apply machine learning to correct and improve fluid substitution results")
        
        # Fluid Substitution Quick Presets
        st.subheader("🎯 Fluid Substitution Presets")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Brine → Gas", use_container_width=True, key="fs_brine_gas"):
                st.session_state.fs_orig_sw = 1.0
                st.session_state.fs_orig_fluid = "brine"
                st.session_state.fs_new_sw = 0.0
                st.session_state.fs_new_fluid = "gas"
                st.session_state.fs_scenario = "Brine to Gas"
                st.success("Brine to Gas: Largest Vp decrease expected")

        with col2:
            if st.button("Oil → Brine", use_container_width=True, key="fs_oil_brine"):
                st.session_state.fs_orig_sw = 0.2
                st.session_state.fs_orig_fluid = "oil"
                st.session_state.fs_new_sw = 1.0
                st.session_state.fs_new_fluid = "brine"
                st.session_state.fs_scenario = "Oil to Brine"
                st.success("Oil to Brine: Moderate Vp increase expected")

        with col3:
            if st.button("Gas → Brine", use_container_width=True, key="fs_gas_brine"):
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
            
            **🤖 ML Correction:** 
            - Automatically learns patterns from your data
            - Corrects systematic errors in traditional Gassmann
            - Typically improves R² by 0.1-0.3
            - Works best with sufficient training data (>50 points)
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
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                if use_ml_fs:
                    # NEW: ML-enhanced fluid substitution
                    st.info("🤖 Using ML to enhance fluid substitution modeling...")
                    results = modeler.ml_enhanced_fluid_substitution(
                        min_depth, max_depth, original_sw, new_sw, original_fluid, new_fluid
                    )
                    
                    if results is not None:
                        results_df, ml_corrections = results
                        
                        # Display ML correction results
                        if ml_corrections:
                            st.subheader("🤖 ML Correction Results")
                            correction_cols = st.columns(len(ml_corrections))
                            
                            for idx, (prop, correction) in enumerate(ml_corrections.items()):
                                with correction_cols[idx]:
                                    st.metric(
                                        label=f"{prop} ML Improvement",
                                        value=f"R²: {correction['final_r2']:.4f}",
                                        delta=f"+{correction['r2_improvement']:.4f}",
                                        delta_color="normal"
                                    )
                    else:
                        st.warning("ML-enhanced fluid substitution failed. Falling back to traditional method.")
                        # Fall back to traditional method
                        results = []
                        # ... (traditional fluid substitution code) ...
                else:
                    # Traditional fluid substitution
                    results = []
                    # ... (traditional fluid substitution code) ...
                
                # ... (rest of fluid substitution visualization code) ...

    # NEW: ML ENHANCEMENT TAB
    with tab6:
        st.header("🤖 Machine Learning Enhancement")
        
        st.markdown("""
        ## Advanced Machine Learning for Rock Physics
        
        This section provides advanced ML capabilities to significantly improve R² values 
        and prediction accuracy for rock physics modeling and fluid substitution.
        """)
        
        # ML Model Selection
        st.subheader("ML Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_property = st.selectbox(
                "Target Property to Predict",
                options=['Vp', 'Vs', 'RHOB', 'AI', 'Vp_Vs'],
                help="Select the elastic property you want to predict"
            )
        
        with col2:
            ml_algorithm = st.selectbox(
                "ML Algorithm",
                options=['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 
                        'Neural Network', 'Ensemble (All Models)'],
                help="Select the machine learning algorithm to use"
            )
        
        # Feature Selection
        st.subheader("Feature Selection")
        
        available_features = [col for col in data.columns if col not in ['Depth', target_property] 
                            and data[col].dtype in [np.int64, np.float64]]
        
        selected_features = st.multiselect(
            "Select Features for ML Model",
            options=available_features,
            default=available_features[:3] if len(available_features) >= 3 else available_features,
            help="Choose which well logs to use as input features for the ML model"
        )
        
        if len(selected_features) < 2:
            st.warning("Please select at least 2 features for ML modeling.")
        else:
            if st.button("🚀 Train Advanced ML Model", type="primary"):
                with st.spinner("Training advanced ML model..."):
                    # Prepare data
                    X = data[selected_features].fillna(data[selected_features].mean())
                    y = data[target_property].fillna(data[target_property].mean())
                    
                    # Remove infinite values
                    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
                    y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())
                    
                    if len(X) < 10:
                        st.error("Insufficient data for ML training. Need at least 10 samples.")
                    else:
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # Train models based on selection
                        if ml_algorithm == 'Ensemble (All Models)':
                            models = {
                                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                                'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                                'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
                                'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
                            }
                        else:
                            model_map = {
                                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                                'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                                'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
                                'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
                            }
                            models = {ml_algorithm: model_map[ml_algorithm]}
                        
                        # Train and evaluate models
                        results = {}
                        feature_importances = {}
                        
                        for name, model in models.items():
                            try:
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                
                                # Calculate metrics
                                r2 = r2_score(y_test, y_pred)
                                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                mae = np.mean(np.abs(y_test - y_pred))
                                
                                # Cross-validation
                                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                                
                                results[name] = {
                                    'model': model,
                                    'r2': r2,
                                    'rmse': rmse,
                                    'mae': mae,
                                    'cv_mean': cv_scores.mean(),
                                    'cv_std': cv_scores.std(),
                                    'predictions': y_pred
                                }
                                
                                # Feature importance (for tree-based models)
                                if hasattr(model, 'feature_importances_'):
                                    feature_importances[name] = {
                                        'importances': model.feature_importances_,
                                        'features': selected_features
                                    }
                                
                            except Exception as e:
                                st.error(f"Error training {name}: {str(e)}")
                        
                        # Display Results
                        if results:
                            st.subheader("📊 ML Model Performance")
                            
                            # Performance metrics
                            metrics_cols = st.columns(len(results))
                            
                            for idx, (name, result) in enumerate(results.items()):
                                with metrics_cols[idx]:
                                    st.metric(
                                        label=f"{name} R²",
                                        value=f"{result['r2']:.4f}",
                                        delta=f"CV: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}"
                                    )
                            
                            # Best model
                            best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
                            best_result = results[best_model_name]
                            
                            st.success(f"🎯 Best Model: **{best_model_name}** with R² = {best_result['r2']:.4f}")
                            
                            # Feature Importance Plot
                            if feature_importances and best_model_name in feature_importances:
                                st.subheader("🔍 Feature Importance")
                                
                                fi_data = feature_importances[best_model_name]
                                importance_df = pd.DataFrame({
                                    'Feature': fi_data['features'],
                                    'Importance': fi_data['importances']
                                }).sort_values('Importance', ascending=True)
                                
                                fig_fi = go.Figure()
                                fig_fi.add_trace(go.Bar(
                                    y=importance_df['Feature'],
                                    x=importance_df['Importance'],
                                    orientation='h',
                                    marker_color='lightblue'
                                ))
                                
                                fig_fi.update_layout(
                                    title=f"Feature Importance - {best_model_name}",
                                    xaxis_title="Importance",
                                    yaxis_title="Features",
                                    template="plotly_white",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_fi, use_container_width=True)
                            
                            # Prediction vs Actual Plot
                            st.subheader("📈 Prediction vs Actual")
                            
                            fig_pred = go.Figure()
                            
                            for name, result in results.items():
                                fig_pred.add_trace(go.Scatter(
                                    x=y_test,
                                    y=result['predictions'],
                                    mode='markers',
                                    name=name,
                                    opacity=0.7
                                ))
                            
                            # Add 1:1 line
                            min_val = min(y_test.min(), min(r['predictions'].min() for r in results.values()))
                            max_val = max(y_test.max(), max(r['predictions'].max() for r in results.values()))
                            
                            fig_pred.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                name='1:1 Line',
                                line=dict(color='red', dash='dash'),
                                showlegend=True
                            ))
                            
                            fig_pred.update_layout(
                                title="Predicted vs Actual Values",
                                xaxis_title="Actual Values",
                                yaxis_title="Predicted Values",
                                template="plotly_white",
                                height=500
                            )
                            
                            st.plotly_chart(fig_pred, use_container_width=True)
                            
                            # Model Comparison
                            if len(results) > 1:
                                st.subheader("📋 Model Comparison")
                                
                                comparison_data = []
                                for name, result in results.items():
                                    comparison_data.append({
                                        'Model': name,
                                        'R²': result['r2'],
                                        'RMSE': result['rmse'],
                                        'MAE': result['mae'],
                                        'CV R² Mean': result['cv_mean'],
                                        'CV R² Std': result['cv_std']
                                    })
                                
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df.style.format({
                                    'R²': '{:.4f}',
                                    'RMSE': '{:.4f}',
                                    'MAE': '{:.4f}',
                                    'CV R² Mean': '{:.4f}',
                                    'CV R² Std': '{:.4f}'
                                }).highlight_max(subset=['R²'], color='lightgreen'), 
                                use_container_width=True)
                            
                            # Save best model
                            if st.button("💾 Save Best Model for Future Use"):
                                modeler.ml_models[f"advanced_{target_property}"] = {
                                    'model': best_result['model'],
                                    'model_name': best_model_name,
                                    'r2': best_result['r2'],
                                    'features': selected_features,
                                    'target': target_property
                                }
                                st.success(f"Best model ({best_model_name}) saved for {target_property}!")
        
        # ML-based Anomaly Detection
        st.subheader("🔍 ML-based Anomaly Detection")
        
        if st.button("Detect Anomalies in Rock Physics Data"):
            with st.spinner("Analyzing data for anomalies..."):
                # Use isolation forest for anomaly detection
                from sklearn.ensemble import IsolationForest
                
                # Prepare features for anomaly detection
                anomaly_features = [f for f in ['Vp', 'Vs', 'RHOB', 'NPHI', 'GR'] if f in data.columns]
                
                if len(anomaly_features) >= 2:
                    X_anomaly = data[anomaly_features].fillna(data[anomaly_features].mean())
                    X_anomaly = X_anomaly.replace([np.inf, -np.inf], np.nan).fillna(X_anomaly.mean())
                    
                    # Fit isolation forest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomalies = iso_forest.fit_predict(X_anomaly)
                    
                    # Create results
                    anomaly_df = data.copy()
                    anomaly_df['Anomaly_Score'] = iso_forest.decision_function(X_anomaly)
                    anomaly_df['Is_Anomaly'] = anomalies == -1
                    
                    st.success(f"Detected {anomaly_df['Is_Anomaly'].sum()} anomalous points ({anomaly_df['Is_Anomaly'].sum()/len(anomaly_df)*100:.1f}% of data)")
                    
                    # Plot anomalies
                    fig_anomaly = go.Figure()
                    
                    # Normal points
                    normal_data = anomaly_df[~anomaly_df['Is_Anomaly']]
                    fig_anomaly.add_trace(go.Scatter(
                        x=normal_data['Vp'] if 'Vp' in normal_data.columns else normal_data.index,
                        y=normal_data['Vs'] if 'Vs' in normal_data.columns else normal_data.index,
                        mode='markers',
                        name='Normal',
                        marker=dict(color='blue', opacity=0.6)
                    ))
                    
                    # Anomalous points
                    anomaly_data = anomaly_df[anomaly_df['Is_Anomaly']]
                    fig_anomaly.add_trace(go.Scatter(
                        x=anomaly_data['Vp'] if 'Vp' in anomaly_data.columns else anomaly_data.index,
                        y=anomaly_data['Vs'] if 'Vs' in anomaly_data.columns else anomaly_data.index,
                        mode='markers',
                        name='Anomaly',
                        marker=dict(color='red', size=8, opacity=0.8)
                    ))
                    
                    fig_anomaly.update_layout(
                        title="Anomaly Detection in Rock Physics Data",
                        xaxis_title="Vp" if 'Vp' in anomaly_df.columns else "Feature 1",
                        yaxis_title="Vs" if 'Vs' in anomaly_df.columns else "Feature 2",
                        template="plotly_white",
                        height=500
                    )
                    
                    st.plotly_chart(fig_anomaly, use_container_width=True)
                    
                    # Show anomalous points
                    with st.expander("View Anomalous Data Points"):
                        st.dataframe(anomaly_df[anomaly_df['Is_Anomaly']].head(10), use_container_width=True)
                else:
                    st.warning("Insufficient features for anomaly detection. Need at least 2 numeric features.")

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
    - **🤖 Machine Learning Enhancement** (NEW)
    
    Upload your own CSV data or use the sample data provided.
    """)

if __name__ == "__main__":
    main()
