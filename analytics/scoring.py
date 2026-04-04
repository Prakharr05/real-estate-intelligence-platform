import math
import sys
import os
from sqlalchemy.orm import Session

# Add root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import SessionLocal
from models import Society, BuilderFloor, Plot, InfrastructureDriver

# --- 1. COMPREHENSIVE SECTOR COORDINATES (EXPANDED) ---
SECTOR_COORDS = {
    # Old Faridabad / NIT belt
    "Sector 1":  (28.3800, 77.3000), "Sector 2":  (28.3750, 77.3050),
    "Sector 3":  (28.3680, 77.3120), "Sector 5":  (28.3650, 77.3100),
    "Sector 7":  (28.3650, 77.3250), "Sector 8":  (28.3600, 77.3200),
    "Sector 9":  (28.3650, 77.3350), "Sector 10": (28.3700, 77.3400),
    "Sector 11": (28.3780, 77.3350),

    # Central Faridabad
    "Sector 14": (28.4050, 77.3100), "Sector 15": (28.4010, 77.3150),
    "Sector 16": (28.3950, 77.3200), "Sector 17": (28.3900, 77.3250),
    "Sector 19": (28.4100, 77.3200), "Sector 21": (28.4250, 77.2950),
    "Sector 23": (28.4150, 77.3050), "Sector 28": (28.4350, 77.3150),

    # Greater Faridabad / Expressway belt
    "Sector 31": (28.4500, 77.3100), "Sector 35": (28.4400, 77.3000),
    "Sector 37": (28.4650, 77.3050), "Sector 39": (28.4600, 77.3150),
    "Sector 43": (28.4550, 77.3000), "Sector 45": (28.4600, 77.2900),
    "Sector 46": (28.4550, 77.3100), "Sector 48": (28.4200, 77.2850),
    "Sector 49": (28.4300, 77.2900), "Sector 50": (28.4250, 77.3050),
    "Sector 52": (28.3950, 77.2900),

    # Ballabgarh belt
    "Sector 59": (28.3750, 77.3150), "Sector 63": (28.3450, 77.3150),
    "Sector 64": (28.3480, 77.3250), "Sector 65": (28.3410, 77.3310),
    "Sector 68": (28.3250, 77.3300), "Sector 69": (28.3180, 77.3350),
    "Sector 70": (28.3100, 77.3400),

    # Neharpar / New Faridabad belt
    "Sector 75": (28.3850, 77.3500), "Sector 76": (28.3800, 77.3600),
    "Sector 77": (28.3750, 77.3650), "Sector 78": (28.3740, 77.3780),
    "Sector 81": (28.3720, 77.3480), "Sector 82": (28.3670, 77.3450),
    "Sector 83": (28.3760, 77.3550), "Sector 84": (28.3710, 77.3610),
    "Sector 85": (28.3600, 77.3550), "Sector 86": (28.3550, 77.3600),
    "Sector 88": (28.3650, 77.3750), "Sector 89": (28.3850, 77.3700),
    "Sector 92": (28.3500, 77.3700),

    # Far Neharpar / Jewar influence zone
    "Sector 97": (28.3611, 77.3820), "Sector 98": (28.3550, 77.3850),
    "Sector 110": (28.3350, 77.3950),

    # New additions
    # Missing sectors
    "Sector 21B": (28.4200, 77.3000), "Sector 21C": (28.4180, 77.2980),
    "Sector 21D": (28.4160, 77.2960), "Sector 27A": (28.4320, 77.3120),
    "Sector 41":  (28.4580, 77.3080), "Sector 42":  (28.4620, 77.3060),
    "Sector 72":  (28.3480, 77.3420), "Sector 73":  (28.3520, 77.3470),
    "Sector 79":  (28.3690, 77.3530), "Sector 80":  (28.3730, 77.3510),
    "Sector 87":  (28.3580, 77.3680), "Sector 104": (28.3420, 77.3880),
    "Sector 114": (28.3300, 77.3920), "Sector 143": (28.3150, 77.4050),
    "Sector 15A": (28.4020, 77.3130), "Sector 16A": (28.3960, 77.3180),
}

# --- 2. INFRASTRUCTURE MATURITY WEIGHTS ---
# Existing infrastructure = full weight (certain, already priced in but reliable)
# Upcoming infrastructure = 0.6x (higher upside risk, not yet delivered)
MATURITY_WEIGHTS = {
    "Expressway":   1.0,
    "Airport Link": 1.0,
    "Metro":        1.0,
    "Future Metro": 0.6,   # Piyali Chowk, Pali Chowk — upcoming stations
}

# --- 3. CATEGORY-SPECIFIC ALPHA THRESHOLDS ---
# Plots: land appreciates faster → easier to earn HIGH
# Floors: mid-tier investment vehicle
# Societies: more end-use than pure investment → harder to earn HIGH
ALPHA_THRESHOLDS = {
    "plots":          (7.0, 4.0),   # (HIGH cutoff, STABLE cutoff)
    "builder_floors": (7.5, 5.0),
    "societies":      (8.0, 5.5),
}

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2)**2)
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def calculate_connectivity_alpha():
    db: Session = SessionLocal()
    drivers = db.query(InfrastructureDriver).all()

    if not drivers:
        print("❌ ERROR: No Infrastructure Drivers in DB.")
        return

    # Map each table to its threshold key
    table_threshold_map = {
        Society:      "societies",
        BuilderFloor: "builder_floors",
        Plot:         "plots",
    }

    for table, threshold_key in table_threshold_map.items():
        properties = db.query(table).all()
        high_cut, stable_cut = ALPHA_THRESHOLDS[threshold_key]
        print(f"\n🧠 Recalculating {len(properties)} rows in "
              f"{table.__tablename__} "
              f"[HIGH>{high_cut} | STABLE>{stable_cut}]...")

        for prop in properties:

            # 1. Resolve sector coords
            sector_name = prop.sector
            if sector_name == "Unknown":
                t = prop.title.lower()
                if "tajupur"   in t: sector_name = "Sector 98"
                elif "bhopani" in t: sector_name = "Sector 89"
                elif "maujpur" in t: sector_name = "Sector 98"
                elif "basantpur" in t: sector_name = "Sector 37"
                elif "palwali"  in t: sector_name = "Sector 81"
                elif "nacholi"  in t: sector_name = "Sector 86"
                elif "pali"     in t: sector_name = "Sector 52"

            lat, lon = prop.latitude, prop.longitude
            if not lat or not lon:
                lat, lon = SECTOR_COORDS.get(sector_name, (None, None))

            if not lat:
                print(f"  ⚠️ No coords for: {sector_name} — skipping")
                continue

            # 2. Connectivity score with maturity-adjusted weights
            contributions = []
            best_dist = 99.0
            closest_expressway_dist = 99.0

            for driver in drivers:
                dist = haversine_distance(lat, lon, driver.latitude, driver.longitude)
                best_dist = min(best_dist, dist)

                # Track closest expressway/airport for bonus
                if driver.driver_type in ("Expressway", "Airport Link"):
                    closest_expressway_dist = min(closest_expressway_dist, dist)

                # Apply maturity weight on top of the driver's own weight_factor
                maturity = MATURITY_WEIGHTS.get(driver.driver_type, 1.0)
                adjusted_weight = driver.weight_factor * maturity
                contributions.append(adjusted_weight / (dist + 1))

            contributions.sort(reverse=True)

            # Primary driver (100%) + secondary drivers (50% each)
            raw_score = (contributions[0] +
                         sum(c * 0.5 for c in contributions[1:])) if contributions else 0

            # 3. Normalize to 0–10
            connectivity_score = round(min(raw_score * 1.5, 10.0), 2)

            # 4. Expressway proximity bonus (<2km of any expressway or airport link)
            if closest_expressway_dist < 2.0:
                connectivity_score = round(min(connectivity_score * 1.2, 10.0), 2)

            # 5. Infra maturity score — separate signal
            # Ratio of maturity-adjusted score vs raw score (how "certain" the infra is)
            raw_contributions_unweighted = [
                driver.weight_factor / (haversine_distance(lat, lon, driver.latitude, driver.longitude) + 1)
                for driver in drivers
            ]
            raw_contributions_unweighted.sort(reverse=True)
            raw_unweighted = (raw_contributions_unweighted[0] +
                              sum(c * 0.5 for c in raw_contributions_unweighted[1:])
                              ) if raw_contributions_unweighted else 0
            maturity_score = round(min(raw_unweighted * 1.5, 10.0), 2)

            # 6. Composite alpha (60% connectivity + 40% maturity)
            composite = round((connectivity_score * 0.60) + (maturity_score * 0.40), 2)
            final_score = min(composite, 10.0)

            # Corner plot bonus — adds 0.5 points for plots only
            if table == Plot:
                if hasattr(prop, 'is_corner_plot') and prop.is_corner_plot == "Yes":
                    final_score = round(min(final_score + 0.5, 10.0), 2)

            # 7. Category-specific alpha rating
            if final_score > high_cut:
                alpha = "HIGH"
            elif final_score > stable_cut:
                alpha = "STABLE"
            else:
                alpha = "VALUE"

            # 8. Save to DB
            prop.connectivity_score = final_score
            prop.infra_proximity_km = round(best_dist, 2)
            prop.alpha_rating = alpha
            prop.latitude = lat
            prop.longitude = lon

        db.commit()
        print(f"  ✅ Done — {table.__tablename__}")

    db.close()
    print("\n🚀 All-Clear: Scores recalculated with refined Alpha logic.")

if __name__ == "__main__":
    calculate_connectivity_alpha()