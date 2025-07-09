import dateparser
from datetime import datetime
import re

def extract_timeframe_range(text, reference_time=None):
    """
    Attempts to parse any date expressions from the user query.
    Returns (start_time, end_time) if found, otherwise (None, None).
    """
    if reference_time is None:
        reference_time = datetime.utcnow()

    # Look for "from X to Y" or "between X and Y"
    match = re.search(r'(from|between)\s+(.*?)\s+(to|and)\s+(.*)', text, re.IGNORECASE)
    if match:
        start_expr = match.group(2)
        end_expr = match.group(4)
    else:
        # Try simpler phrases like "last week", "past 30 days", etc.
        start_expr = text
        end_expr = "now"

    start = dateparser.parse(start_expr, settings={'RELATIVE_BASE': reference_time})
    end = dateparser.parse(end_expr, settings={'RELATIVE_BASE': reference_time})

    if start and end:
        return start.isoformat(), end.isoformat()
    else:
        return None, None
