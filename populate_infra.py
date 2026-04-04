from database import SessionLocal
from models import InfrastructureDriver

def seed_infrastructure():
    db = SessionLocal()
    
    # Comprehensive List: Existing Metro + Expressway + Future Links
    drivers = [
        # --- Delhi-Mumbai Expressway (High Alpha) ---
        {"name": "Expway Exit - Sec 65", "type": "Expressway", "lat": 28.3410, "lon": 77.3310, "w": 1.5},
        {"name": "Expway Exit - Sec 14/17", "type": "Expressway", "lat": 28.3880, "lon": 77.3220, "w": 1.3},
        {"name": "Mohna Interchange (Jewar Link)", "type": "Airport Link", "lat": 28.3180, "lon": 77.4200, "w": 1.5},

        # --- Current Violet Line Metro (Faridabad Stretch) ---
        {"name": "Sarai Metro", "type": "Metro", "lat": 28.4636, "lon": 77.3005, "w": 1.0},
        {"name": "NHPC Chowk Metro", "type": "Metro", "lat": 28.4500, "lon": 77.3069, "w": 1.0},
        {"name": "Mewala Maharajpur Metro", "type": "Metro", "lat": 28.4414, "lon": 77.3101, "w": 1.0},
        {"name": "Sector 28 Metro", "type": "Metro", "lat": 28.4286, "lon": 77.3117, "w": 1.1},
        {"name": "Badkhal Mor Metro", "type": "Metro", "lat": 28.4116, "lon": 77.3135, "w": 1.2},
        {"name": "Old Faridabad Metro", "type": "Metro", "lat": 28.4014, "lon": 77.3153, "w": 1.2},
        {"name": "Neelam Chowk Ajronda Metro", "type": "Metro", "lat": 28.3900, "lon": 77.3170, "w": 1.2},
        {"name": "Bata Chowk Metro (Interchange)", "type": "Metro", "lat": 28.3814, "lon": 77.3188, "w": 1.4},
        {"name": "Escorts Mujesar Metro", "type": "Metro", "lat": 28.3639, "lon": 77.3208, "w": 1.1},
        {"name": "Raja Nahar Singh Metro", "type": "Metro", "lat": 28.3394, "lon": 77.3264, "w": 1.2},

        # --- Future Gurgaon-Faridabad Link Nodes (Approved) ---
        {"name": "Piyali Chowk (Future Metro)", "type": "Future Metro", "lat": 28.3850, "lon": 77.2950, "w": 1.3},
        {"name": "Pali Chowk (Future Metro)", "type": "Future Metro", "lat": 28.3800, "lon": 77.2600, "w": 1.3}
    ]

    try:
        for d in drivers:
            exists = db.query(InfrastructureDriver).filter_by(name=d["name"]).first()
            if not exists:
                db.add(InfrastructureDriver(
                    name=d["name"], driver_type=d["type"],
                    latitude=d["lat"], longitude=d["lon"], weight_factor=d["w"]
                ))
        db.commit()
        print(f"✅ Seeded {len(drivers)} Alpha Drivers into PostgreSQL.")
    except Exception as e:
        print(f"❌ Error: {e}"); db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_infrastructure()