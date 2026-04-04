from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Enum
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime
import enum

class Base(DeclarativeBase):
    pass

class PropertyCategory(enum.Enum):
    SOCIETY = "society"
    BUILDER_FLOOR = "builder_floor"
    PLOT = "plot"

class PropertyAlphaMixin:
    """Shared traits for all Faridabad listings to ensure data integrity."""
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), default="Unknown")
    title = Column(String(255), nullable=False)
    price = Column(Float, index=True)  # Numeric price for ML training
    display_price = Column(String(50)) # e.g., "1.25 Cr"
    address = Column(Text)
    sector = Column(String(100), index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    
    # --- Market Intelligence Columns ---
    connectivity_score = Column(Float, default=0.0) # 0 to 10 scale
    infra_proximity_km = Column(Float, default=0.0)
    price_per_sqft = Column(Float)
    alpha_rating = Column(String(10)) # e.g., "High", "Stable"
    last_scraped = Column(DateTime, default=datetime.utcnow)

class InfrastructureDriver(Base):
    """Stores the anchor points like Expressway Exits and Jewar Link points."""
    __tablename__ = 'infra_drivers'
    id = Column(Integer, primary_key=True)
    name = Column(String(100)) # e.g., "DND-KMP Exit Sector 65"
    driver_type = Column(String(50)) # Metro, Expressway, Airport
    latitude = Column(Float)
    longitude = Column(Float)
    weight_factor = Column(Float, default=1.0) # Importance for Score

class Society(Base, PropertyAlphaMixin):
    __tablename__ = 'societies'
    society_name = Column(String(150), index=True)
    bhk_type = Column(String(50))
    area_sqft = Column(Float)
    total_towers = Column(Integer)
    possession_status = Column(String(50)) # Ready to move / Under Const
    amenities = Column(Text) # List of amenities for RAG context

class BuilderFloor(Base, PropertyAlphaMixin):
    __tablename__ = 'builder_floors'
    bhk_type = Column(String(50))
    area_sqft = Column(Float)
    floor_no = Column(Integer)
    total_floors = Column(Integer, default=4) # Standard Faridabad 'Stilt+4'
    stilt_parking = Column(String(10)) 
    terrace_rights = Column(String(10))

class Plot(Base, PropertyAlphaMixin):
    __tablename__ = 'plots'
    plot_area_sqyd = Column(Float)
    facing = Column(String(20)) # North, East, etc.
    road_width_meter = Column(Float)
    is_corner_plot = Column(String(10))