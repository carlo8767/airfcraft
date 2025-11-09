import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- PLOTTING CONSTANTS (Based on our troubleshooting) ---
AIRFOIL_FILE = 'combinedAirfoilDataLabeled.csv'
AIRFOIL_NAME = '2032c'
SCALING_FACTOR = 1000000.0
THICKNESS_VISUAL_FACTOR = 100.0  # Exaggerates Z-axis for visibility


# --- 2D Airfoil Geometry Functions (The fixed logic) ---

def calculate_polynomial_y(x_points, coefficients):
    """Calculates the Y-coordinate (thickness) based on 31-coeff polynomial."""
    y_points = np.zeros_like(x_points, dtype=float)
    for i, coeff in enumerate(coefficients):
        power = 30 - i
        y_points += coeff * (x_points ** power)
    return y_points


def get_airfoil_coords(df, airfoil_name):
    """Retrieves the final normalized X and Y coordinates for a single airfoil profile."""
    airfoil_data = df[df['airfoilName'] == airfoil_name]
    if airfoil_data.empty: return None, None
    upper_coeffs = airfoil_data.filter(regex='upperSurfaceCoeff').iloc[0].values
    lower_coeffs = airfoil_data.filter(regex='lowerSurfaceCoeff').iloc[0].values

    x_calc = np.linspace(0.0, 1.0, 100)

    # Apply scaling factor only to the upper surface (The definitive fix for this data)
    y_upper_scaled = calculate_polynomial_y(x_calc, upper_coeffs) / SCALING_FACTOR
    y_lower_scaled = calculate_polynomial_y(x_calc, lower_coeffs)

    # Combine X and Y into a closed loop (Upper TE->LE, then Lower LE->TE)
    x_profile = np.concatenate((x_calc[::-1], x_calc))
    y_profile = np.concatenate((y_upper_scaled[::-1], y_lower_scaled))

    return x_profile, y_profile


def plot_3d_wing(x_profile_norm, y_profile_norm, cr, sem_span, sweep_deg, taper_ratio):
    # ... (Steps 1, 2, 3: WING GEOMETRY AND COORDINATE GENERATION - REMAIN UNCHANGED) ...
    # The coordinate generation logic remains exactly the same as the last version.

    # 1. PARAMETERS SETUP
    n_span = 20
    sweep_rad = np.deg2rad(sweep_deg)
    y_span_right = np.linspace(0, sem_span, n_span)
    chord_at_y = cr * (1 - (1 - taper_ratio) * (y_span_right / sem_span))
    x_LE_at_y = y_span_right * np.tan(sweep_rad)

    X_right, Y_right, Z_right = [], [], []
    for i in range(n_span):
        c_i = chord_at_y[i]
        x_LE_i = x_LE_at_y[i]
        x_airfoil = (x_profile_norm * c_i) + x_LE_i
        z_airfoil = y_profile_norm * c_i * THICKNESS_VISUAL_FACTOR
        y_airfoil = np.full_like(x_airfoil, y_span_right[i])
        X_right.append(x_airfoil);
        Y_right.append(y_airfoil);
        Z_right.append(z_airfoil)

    # 4. MIRRORING: Create the Left Semi-Span
    Y_left = [-y_coord for y_coord in Y_right]
    X_left = X_right
    Z_left = Z_right

    # 5. COMBINE FOR PLOTTING
    X_full = np.concatenate(X_left + X_right)
    Y_full = np.concatenate(Y_left + Y_right)
    Z_full = np.concatenate(Z_left + Z_right)
    X_plot = X_left + X_right
    Y_plot = Y_left + Y_right
    Z_plot = Z_left + Z_right

    # 6. PLOTTING
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(X_plot)):
        ax.plot(X_plot[i], Y_plot[i], Z_plot[i], color='blue', linewidth=0.5, alpha=0.7)

    # Set Labels and Title
    total_span = sem_span * 2.0
    ax.set_xlabel('X (Chord)')
    ax.set_ylabel('Y (Span)')
    ax.set_zlabel(f'Z (Thickness x{THICKNESS_VISUAL_FACTOR:.0f})')
    ax.set_title(f'3D Full Wing: $\\Lambda$={sweep_deg}° | $c_r$={cr} | $b$={total_span}')

    # --- FINAL SCALE FIX: FORCE THE AXIS LIMITS AND ASPECT RATIO ---

    # X Limits: Based on the chord + sweep
    x_min = X_full.min()
    x_max = X_full.max()
    ax.set_xlim(x_min - 0.5, x_max + 0.5)

    # Y Limits: Span from -semi_span to +semi_span
    ax.set_ylim(-sem_span - 0.5, sem_span + 0.5)

    # Z Limits: Thickness centered around zero
    z_min = Z_full.min()
    z_max = Z_full.max()
    ax.set_zlim(z_min * 1.5, z_max * 1.5)

    # Crucial Fix: Set the aspect ratio based on true dimensions (X:Y) and a fixed Z ratio
    # True X range (approx 3.0), True Y range (10.0), Z range is forced visually to be small
    ax.set_box_aspect([1, (total_span / cr), 0.2])  # <-- THE KEY FIX

    plt.show()


# =================================================================
#           MAIN EXECUTION BLOCK
# =================================================================

if __name__ == '__main__':
    try:
        # NOTE: If running outside of a Kaggle environment, ensure the file is in the script's directory.
        df_full = pd.read_csv(AIRFOIL_FILE, low_memory=False)

        # 1. Get the 2D Airfoil Coordinates
        x_norm, y_norm = get_airfoil_coords(df_full, AIRFOIL_NAME)

        if x_norm is None:
            raise ValueError(f"Could not load normalized coordinates for {AIRFOIL_NAME}. Check data loading.")

        # 2. DEFINE THE WING DIMENSIONS (User Inputs)
        WING_DIMENSIONS = {
            'root_chord': 2.0,  # meters (Cr)
            'semi_span': 5.0,  # meters (b/2)
            'sweep_angle_deg': 25,  # degrees (Lambda)
            'taper_ratio': 0.5  # Tip Chord / Root Chord (Lambda_taper)
        }

        # 3. Plot the 3D Wing
        plot_3d_wing(
            x_profile_norm=x_norm,
            y_profile_norm=y_norm,
            cr=WING_DIMENSIONS['root_chord'],
            sem_span=WING_DIMENSIONS['semi_span'],
            sweep_deg=WING_DIMENSIONS['sweep_angle_deg'],
            taper_ratio=WING_DIMENSIONS['taper_ratio']
        )

    except FileNotFoundError:
        print(f"❌ Error: The file '{AIRFOIL_FILE}' was not found. Please ensure it is in the correct directory.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")