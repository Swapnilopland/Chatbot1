import re
from datetime import datetime
from dateparser.search import search_dates

def clean_text(text):
    """Remove ordinal suffixes (e.g., '24th' → '24')."""
    return re.sub(r'(\d+)(st|nd|rd|th)', r'\1', text)

def try_parse_date(day, month, year):
    """Try to parse a date using several formats and return a datetime object or None."""
    for fmt in ("%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(f"{day} {month} {year}", fmt)
        except ValueError:
            continue
    return None

def extract_range(text):
    """
    Attempt to extract a date range from text.
    Checks multiple patterns:
      1. Same-month ranges (e.g., "24 to 25 apr" or "24-25 april 2025").
      2. Month-first ranges with same month (e.g., "apr 24-25 2025").
      3. Cross-month ranges (e.g., "30 Apr to 1st May").
    Returns a tuple of two datetime objects if a range is found, else None.
    """
    patterns = [
        # Pattern 1: Same month with day range.
        r'(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\s+([A-Za-z]+)\s*(\d{4})?',
        # Pattern 2: Month first, same month range.
        r'([A-Za-z]+)\s+(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\s*(\d{4})?',
        # Pattern 3: Cross-month range, e.g., "30 Apr to 1 May"
        r'(\d{1,2})\s*([A-Za-z]+)\s*(?:to|-)\s*(\d{1,2})\s*([A-Za-z]+)\s*(\d{4})?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            groups = match.groups()
            # Pattern 1: day1, day2, month, [year]
            if re.match(r'\d', groups[0]) and len(groups) == 4:
                day1, day2, month, year = groups
            # Pattern 2: month, day1, day2, [year]
            elif re.match(r'[A-Za-z]', groups[0]) and len(groups) == 4:
                month, day1, day2, year = groups
            # Pattern 3: Cross-month range: day1, month1, day2, month2, [year]
            elif len(groups) == 5:
                day1, month1, day2, month2, year = groups
                # Use month1 for the first date and month2 for the second date.
                if not year:
                    year = str(datetime.now().year)
                date1 = try_parse_date(day1, month1, year)
                date2 = try_parse_date(day2, month2, year)
                if date1 and date2:
                    return date1, date2
                else:
                    continue
            else:
                continue

            # If year is missing, default to the current year.
            if not year:
                year = str(datetime.now().year)
            date1 = try_parse_date(day1, month, year)
            date2 = try_parse_date(day2, month, year)
            if date1 and date2:
                return date1, date2
    return None

def extract_single(text):
    """
    Try matching several patterns for a single date.
    Returns a datetime object if found, else None.
    """
    patterns = [
        # Day-Month: "24 apr", "24 april" (with or without space)
        r'(\d{1,2})\s*([A-Za-z]+)\s*(\d{4})?',
        # Month-Day: "apr 24", "april 24"
        r'([A-Za-z]+)\s*(\d{1,2})\s*(\d{4})?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            groups = match.groups()
            if groups[0].isdigit():
                day, month, year = groups
            else:
                month, day, year = groups
            if not year:
                year = str(datetime.now().year)
            date_obj = try_parse_date(day, month, year)
            if date_obj:
                return date_obj
    return None

def extract_dates(text):
    """
    Extract begin_date and end_date from text.
      1. Pre-process text.
      2. Attempt to extract a date range using multiple patterns.
      3. If no range is found, try to extract a single date.
      4. Finally, fall back on dateparser.search_dates.
    Returns a tuple (begin_date, end_date) as "YYYY-MM-DD" strings.
    """
    text = clean_text(text)

    # 1. Try explicit date range.
    range_result = extract_range(text)
    if range_result:
        from_date, to_date = range_result
        return from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d")
    
    # 2. Try explicit single date.
    single_date = extract_single(text)
    if single_date:
        date_str = single_date.strftime("%Y-%m-%d")
        return date_str, date_str

    # 3. Fallback: use dateparser.search_dates.
    date_matches = search_dates(text, settings={"RELATIVE_BASE": datetime.now()})
    if date_matches:
        unique_dates = []
        for _, dt in date_matches:
            if dt not in unique_dates:
                unique_dates.append(dt)
        if len(unique_dates) == 1:
            date_str = unique_dates[0].strftime("%Y-%m-%d")
            return date_str, date_str
        elif len(unique_dates) >= 2:
            return unique_dates[0].strftime("%Y-%m-%d"), unique_dates[1].strftime("%Y-%m-%d")
    
    return None, None