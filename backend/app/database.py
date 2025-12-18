"""
Database module for storing processed model architectures.
Uses SQLite for simple, file-based persistence.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager

# Database file path - use environment variable for Docker, fallback to local
DB_PATH = Path(os.environ.get("DATABASE_PATH", Path(__file__).parent.parent / "models.db"))


def get_connection():
    """Get a database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize the database tables."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS saved_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                framework TEXT NOT NULL,
                total_parameters INTEGER DEFAULT 0,
                layer_count INTEGER DEFAULT 0,
                architecture_json TEXT NOT NULL,
                thumbnail TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_hash TEXT UNIQUE
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_name ON saved_models(name)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON saved_models(created_at)
        """)


def save_model(
    name: str,
    framework: str,
    total_parameters: int,
    layer_count: int,
    architecture: dict,
    file_hash: Optional[str] = None,
    thumbnail: Optional[str] = None
) -> int:
    """
    Save a model architecture to the database.
    Returns the saved model ID.
    """
    architecture_json = json.dumps(architecture)
    
    with get_db() as conn:
        # Check if model with same hash already exists
        if file_hash:
            existing = conn.execute(
                "SELECT id FROM saved_models WHERE file_hash = ?",
                (file_hash,)
            ).fetchone()
            
            if existing:
                # Update existing entry
                conn.execute("""
                    UPDATE saved_models 
                    SET name = ?, framework = ?, total_parameters = ?, 
                        layer_count = ?, architecture_json = ?, 
                        thumbnail = ?, created_at = CURRENT_TIMESTAMP
                    WHERE file_hash = ?
                """, (name, framework, total_parameters, layer_count, 
                      architecture_json, thumbnail, file_hash))
                return existing['id']
        
        # Insert new entry
        cursor = conn.execute("""
            INSERT INTO saved_models 
            (name, framework, total_parameters, layer_count, architecture_json, file_hash, thumbnail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, framework, total_parameters, layer_count, 
              architecture_json, file_hash, thumbnail))
        
        return cursor.lastrowid


def get_saved_models() -> List[dict]:
    """Get all saved models (metadata only, not full architecture)."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, name, framework, total_parameters, layer_count, 
                   thumbnail, created_at
            FROM saved_models
            ORDER BY created_at DESC
        """).fetchall()
        
        return [dict(row) for row in rows]


def get_model_by_id(model_id: int) -> Optional[dict]:
    """Get a specific model with full architecture."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT id, name, framework, total_parameters, layer_count,
                   architecture_json, thumbnail, created_at
            FROM saved_models
            WHERE id = ?
        """, (model_id,)).fetchone()
        
        if row:
            result = dict(row)
            result['architecture'] = json.loads(result['architecture_json'])
            del result['architecture_json']
            return result
        
        return None


def delete_model(model_id: int) -> bool:
    """Delete a model by ID. Returns True if deleted."""
    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM saved_models WHERE id = ?",
            (model_id,)
        )
        return cursor.rowcount > 0


def model_exists_by_hash(file_hash: str) -> Optional[int]:
    """Check if a model with the given hash exists. Returns ID if exists."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT id FROM saved_models WHERE file_hash = ?",
            (file_hash,)
        ).fetchone()
        
        return row['id'] if row else None


# Initialize database on module load
init_db()
