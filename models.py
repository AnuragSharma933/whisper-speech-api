from pydantic import BaseModel
from typing import Optional, List, Dict


class TranscribeResponse(BaseModel):
    transcript: str
    language: str
    summary: Optional[str]
    sentiment: Optional[Dict]
    translation: Optional[str]
    entities: Optional[List[Dict]]
    word_timestamps: Optional[List[Dict]]