import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from models import Base  # This pulls in your Triple-Table blueprint

# 1. Load Environment Variables
# Create a .env file with: DB_URL=postgresql://user:password@localhost:5432/faridabad_alpha
load_dotenv()

DATABASE_URL = os.getenv("DB_URL")

# --- ADD THIS CHECK ---
if DATABASE_URL is None:
    raise ValueError("❌ ERROR: DB_URL not found in .env file. Check your file name and variable keys!")
# ----------------------


# 2. Create the Engine
# 'echo=False' keeps the console clean. Set to 'True' if you want to see the raw SQL.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# 3. Create a Session Factory
# scoped_session is great for Streamlit/FastAPI as it handles multiple user requests safely.
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

def init_db():
    """
    The 'Magic' Function:
    This command looks at models.py and creates the actual tables 
    (Societies, Plots, Floors, Infra) in PostgreSQL if they don't exist.
    """
    try:
        # Base.metadata contains the 'knowledge' of your triple-table structure
        Base.metadata.create_all(bind=engine)
        print("✅ Success: Faridabad Alpha Triple-Table Architecture is Live in Postgres.")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")

def get_db():
    """Dependency for our future FastAPI/Streamlit integration."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    init_db()