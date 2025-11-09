import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mlcroissant as mlc
import zipfile
import os

# --- PLOTTING CONSTANTS (Based on our troubleshooting) ---
AIRFOIL_FILE = 'combinedAirfoilDataLabeled.csv'
AIRFOIL_NAME = '2032c'
SCALING_FACTOR = 1000000.0
THICKNESS_VISUAL_FACTOR = 100.0  # Exaggerates Z-axis for visibility
ZIP_FILE_NAME = 'archive.zip'


# ---------------------------------------

# --- VALIDATION FUNCTION (NEW) ---

def validate_wing_parameters(wing_dims):
    """
    Verifies that the wing dimension parameters are physically correct and positive.

    Args:
        wing_dims (dict): Dictionary containing the wing parameters.

    Raises:
        ValueError: If any parameter is invalid.
    """
    cr = wing_dims['root_chord']
    sem_span = wing_dims['semi_span']
    sweep_deg = wing_dims['sweep_angle_deg']
    taper_ratio = wing_dims['taper_ratio']

    # 1. Check for positive dimensions
    if cr <= 0:
        raise ValueError(f"❌ Validation Error: 'root_chord' must be positive (currently: {cr}).")
    if sem_span <= 0:
        raise ValueError(f"❌ Validation Error: 'semi_span' must be positive (currently: {sem_span}).")

    # 2. Check Taper Ratio (must be greater than 0)
    # Taper ratio of 1 is a rectangular wing; 0 is a hypothetical pointed tip.
    # We enforce a small positive minimum to prevent division/scaling errors near zero.
    if taper_ratio <= 0:
        raise ValueError(f"❌ Validation Error: 'taper_ratio' must be greater than 0 (currently: {taper_ratio}).")

    # 3. Check Sweep Angle bounds (typically between -90 and 90 degrees)
    if not -90 <= sweep_deg <= 90:
        raise ValueError(
            f"❌ Validation Error: 'sweep_angle_deg' must be between -90 and 90 degrees (currently: {sweep_deg}).")

    print("✅ Wing parameters validated successfully.")
    return True


# --- UNZIP FUNCTION (Unchanged) ---

def unzip_specific_file(zip_path, file_to_extract, output_path='.'):
    # ... (function body remains unchanged) ...
    if not os.path.exists(zip_path):
        print(f"❌ Error: ZIP file not found at '{zip_path}'.")
        return False
    try:
        print(f"⏳ ZIP file found. Attempting to extract {file_to_extract}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            if file_to_extract not in zf.namelist():
                print(f"❌ Error: File '{file_to_extract}' not found inside '{zip_path}'.")
                return False
            zf.extract(file_to_extract, path=output_path)
            print(f"✅ Successfully extracted **{file_to_extract}**.")
            return True
    except Exception as e:
        print(f"❌ An error occurred during unzipping: {e}")
        return False


# --- 2D Airfoil Geometry Functions (Unchanged) ---

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
    # 1. PARAMETERS SETUP
    n_span = 20
    sweep_rad = np.deg2rad(sweep_deg)
    y_span_right = np.linspace(0, sem_span, n_span)
    chord_at_y = cr * (1 - (1 - taper_ratio) * (y_span_right / sem_span))
    x_LE_at_y = y_span_right * np.tan(sweep_rad)

    # --- NEW: Calculate Key Dimensions & Ratios ---
    total_span = sem_span * 2.0
    tip_chord = cr * taper_ratio
    # Area (S_ref) for a simple tapered wing is the average chord times the span
    wing_area = ((cr + tip_chord) / 2.0) * total_span
    # Aspect Ratio (AR) is b^2 / S_ref
    aspect_ratio = (total_span ** 2) / wing_area

    print(f"\n📊 Wing Geometric Parameters:")
    print(f"   Root Chord (Cr): {cr:.2f} m")
    print(f"   Tip Chord (Ct): {tip_chord:.2f} m")
    print(f"   Total Span (b): {total_span:.2f} m")
    print(f"   Reference Area (S_ref): {wing_area:.2f} m²")
    print(f"   Aspect Ratio (AR): {aspect_ratio:.2f}")
    # -----------------------------------------------

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
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(X_plot)):
        ax.plot(X_plot[i], Y_plot[i], Z_plot[i], color='blue', linewidth=0.5, alpha=0.7)

    # --- ADD DIMENSION MARKERS (TOTAL SPAN AND ROOT CHORD) ---

    # Marker for Total Span (b) at the Trailing Edge (TE)
    X_TE_center = x_LE_at_y[0] + cr * x_profile_norm[0]
    ax.plot([X_TE_center, X_TE_center], [-sem_span, sem_span], [0, 0],
            color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Span (b={total_span:.1f}m)')

    # Marker for Root Chord (Cr) on the Y=0 plane
    x_min_root = x_LE_at_y[0]
    x_max_root = x_LE_at_y[0] + cr
    ax.plot([x_min_root, x_max_root], [0, 0], [0, 0],
            color='green', linestyle=':', linewidth=2, alpha=0.8, label=f'Chord (Cr={cr:.1f}m)')

    # -----------------------------------------------------------

    # Set Labels and Title
    ax.set_xlabel('X (Chord, m)')
    ax.set_ylabel('Y (Span, m)')
    ax.set_zlabel(f'Z (Thickness x{THICKNESS_VISUAL_FACTOR:.0f})')
    ax.set_title(
        f'3D Full Wing: $\\Lambda$={sweep_deg}° | $c_r$={cr:.1f}m | $b$={total_span:.1f}m \nAR={aspect_ratio:.1f} | $S_{{ref}}$={wing_area:.1f}m²',
        fontsize=14)

    ax.legend(loc='upper right')

    # --- FINAL SCALE FIX ---
    x_min = X_full.min()
    x_max = X_full.max()
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(-sem_span - 0.5, sem_span + 0.5)
    z_min = Z_full.min()
    z_max = Z_full.max()
    ax.set_zlim(z_min * 1.5, z_max * 1.5)
    ax.set_box_aspect([1, (total_span / cr), 0.2])

    plt.show()



# =================================================================
#           MAIN EXECUTION BLOCK
# =================================================================

if __name__ == '__main__':
    try:
        # --- DATA LOADING WITH FALLBACK ---

        # 1. Check if the CSV is already available
        if not os.path.exists(AIRFOIL_FILE):
            if not unzip_specific_file(ZIP_FILE_NAME, AIRFOIL_FILE):
                raise FileNotFoundError(
                    f"Missing required file **{AIRFOIL_FILE}** and failed to extract it from **{ZIP_FILE_NAME}**.")

        # 2. Load the data
        df_full = pd.read_csv(AIRFOIL_FILE, low_memory=False)
        print(f"✅ Dataframe loaded successfully from **{AIRFOIL_FILE}**.")

        # 3. DEFINE THE WING DIMENSIONS (User Inputs)
        WING_DIMENSIONS = {
            'root_chord': 2.0,  # meters (Cr)
            'semi_span': 5.0,  # meters (b/2)
            'sweep_angle_deg': 25,  # degrees (Lambda)
            'taper_ratio': 0.5  # Tip Chord / Root Chord (Lambda_taper)
        }

        # --- NEW: VALIDATE INPUTS BEFORE PROCEEDING ---
        validate_wing_parameters(WING_DIMENSIONS)

        # 4. Get the 2D Airfoil Coordinates
        x_norm, y_norm = get_airfoil_coords(df_full, AIRFOIL_NAME)

        if x_norm is None:
            raise ValueError(f"Could not load normalized coordinates for {AIRFOIL_NAME}. Check data loading.")

        # 5. Plot the 3D Wing
        plot_3d_wing(
            x_profile_norm=x_norm,
            y_profile_norm=y_norm,
            cr=WING_DIMENSIONS['root_chord'],
            sem_span=WING_DIMENSIONS['semi_span'],
            sweep_deg=WING_DIMENSIONS['sweep_angle_deg'],
            taper_ratio=WING_DIMENSIONS['taper_ratio']
        )

    except (FileNotFoundError, ValueError) as e:
        # Catch specific known errors (file missing or invalid geometry)
        print(f"🛑 Execution Halted due to Error: {e}")
    except Exception as e:
        # Catch any other unexpected errors
        print(f"❌ An unexpected error occurred: {e}")