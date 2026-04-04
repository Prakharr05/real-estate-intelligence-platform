from seleniumbase import SB
import pandas as pd
from database import SessionLocal
from models import Society, BuilderFloor, Plot
import re
import time

# --- 1. PRICE CLEANING (AVERAGING + MIXED UNIT FIX) ---
def smart_clean_price(price_str, category="plot"):
    if not price_str: return None
    raw_str = str(price_str).lower().replace('₹', '').replace(',', '').strip()
    parts = re.split(r'\s*-\s*|\s+to\s+', raw_str)
    
    def process_part(p, full_context):
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", p)
        if not numbers: return None
        num = float(numbers[0])
        local_ctx, global_ctx = f" {p} ", f" {full_context} "

        if 'cr' in local_ctx or 'crore' in local_ctx: return int(num * 10000000)
        if 'lac' in local_ctx or 'lakh' in local_ctx or ' l ' in local_ctx: return int(num * 100000)
        
        if 'cr' in global_ctx or 'crore' in global_ctx:
            if num >= 10: return int(num * 100000) 
            return int(num * 10000000)
        if 'lac' in global_ctx or 'lakh' in global_ctx or ' l ' in global_ctx: return int(num * 100000)

        if num > 100000: return int(num)
        if num < 100:
            if category in ["society", "floor"] and num >= 15: return int(num * 100000)
            else: return int(num * 10000000)
        return int(num)

    try:
        results = [process_part(p, raw_str) for p in parts if process_part(p, raw_str)]
        return int(sum(results) / len(results)) if results else None
    except: return None

# --- 2. SECTOR EXTRACTION (RESTORED FULL LOCALITY LIST) ---
def extract_sector(text):
    if not text: return "Unknown"
    match = re.search(r'(?:Sector|Sec|Sec-)\s*(\d+[A-Za-z]*)', text, re.IGNORECASE)
    if match: return f"Sector {match.group(1)}"
    
    text_lower = text.lower()
    localities = {
        "neharpar": "Sector 80", "greenfield": "Sector 43", "green field colony": "Sector 43", 
        "charmwood": "Sector 39", "sainik colony": "Sector 49", "spring field": "Sector 31", 
        "nit": "Sector 1", "tajupur": "Sector 98", "bhopani": "Sector 89", "maujpur": "Sector 98",
        "basantpur": "Sector 37", "roshan nagar": "Sector 31", "palwali": "Sector 81",
        "kabulpur": "Sector 82", "nacholi": "Sector 86", "pali": "Sector 52", "ashoka enclave": "Sector 35",
        "dayal basti": "Sector 37", "indraprastha": "Sector 31", "shiv durga vihar": "Sector 37",
        "kheri road": "Sector 82", "dabua": "Sector 50", "jawahar colony": "Sector 52",
        "sanjay colony": "Sector 23", "ballabgarh": "Sector 2", "tigaon": "Sector 78",
        "manjhawali": "Sector 92"
    }
    for key, sector in localities.items():
        if key in text_lower: return sector
    return "Unknown"

# --- 3. AREA EXTRACTION (V2: RETURNS DUAL FORMATS) ---
def extract_area_v2(text):
    matches = re.findall(r'(\d+[\.\d]*)\s*(?:-|to)?\s*(\d+[\.\d]*)?\s*(sq\.?yd|sq\.?ft|gaj|sq\.\s*yards?)', text, re.IGNORECASE)
    if not matches: return None, None
    
    sqft_list = []
    seen = set()  # deduplicate
    for m in matches:
        for val in [m[0], m[1]]:
            if not val: continue
            v = float(val)
            if v in seen: continue
            seen.add(v)
            if 'ft' in m[2].lower(): sqft_list.append(v)
            else: sqft_list.append(v * 9.0)
            
    avg_sqft = sum(sqft_list) / len(sqft_list)
    return round(avg_sqft / 9.0, 2), round(avg_sqft, 2)

# --- 4. RATE EXTRACTION (IMAGE-BASED K/SQFT LOGIC) ---
def extract_rate_k_format(text):
    if not text: return None
    matches = re.findall(r'(\d+\.?\d*)\s*k/sq\.?ft', text.lower())
    if not matches: return None
    rates = [float(m) * 1000 for m in matches]
    return round(sum(rates) / len(rates), 2)

# ============================================================
# --- 5. SOCIETY NAME EXTRACTION ---
# ============================================================

def extract_society_name(card, h2_elem):
    # First try: T_arrangeElementsSpaceBetween (multi-BHK project listings)
    if h2_elem:
        parent = h2_elem.parent
        if parent:
            for sib in parent.find_previous_siblings():
                if sib.get('class') and 'T_arrangeElementsSpaceBetween' in sib.get('class', []):
                    raw = sib.get_text(strip=True)
                    clean = re.sub(r'[●•·]?\s*RERA\b.*$', '', raw, flags=re.IGNORECASE).strip()
                    if clean:
                        return clean

    # Second try: scan card text lines — name appears immediately after the title line
    all_lines = [t.strip() for t in card.get_text(separator="\n").splitlines() if t.strip()]
    for i, line in enumerate(all_lines):
        # Find the line that contains the BHK/title text
        if h2_elem and h2_elem.get_text(strip=True) in line:
            # Check next 1-2 lines for the society name
            for j in range(i + 1, min(i + 3, len(all_lines))):
                candidate = all_lines[j]
                # Skip RERA, prices, numbers, area, possession etc.
                if re.search(r'RERA|₹|sq\.?ft|sq\.?yd|ready|possession|resale|builtup|highlight|updated|contact|\d+d ago', 
                             candidate, re.IGNORECASE):
                    continue
                if re.match(r'^[\d/\s]+$', candidate):
                    continue
                if len(candidate) < 4:
                    continue
                clean = re.sub(r'[●•·]?\s*RERA\b.*$', '', candidate, flags=re.IGNORECASE).strip()
                if clean:
                    return clean

    # Third try: extract name from title itself
    if h2_elem:
        h2_text = h2_elem.get_text(strip=True)
        name_match = re.search(
            r'\bin\s+(.+?)\s*,\s*(?:Sector\s*\d+|Faridabad)',
            h2_text, re.IGNORECASE
        )
        if name_match:
            candidate = name_match.group(1).strip()
            if not re.match(r'^sector\s*\d+$|^faridabad$', candidate, re.IGNORECASE):
                return candidate

    return "Unknown"
# ============================================================
# --- 6. MULTI-BHK DETECTION ---
# ============================================================

def detect_multi_bhk(title):
    """
    Detects if a title contains multiple BHK types.
    Returns a list of floats like [2.0, 3.0] for "2, 3 BHK"
    or [3.0] for "3 BHK Flat" (single — existing behavior preserved).

    Handles patterns like:
      "2, 3 BHK Flats..."
      "2 & 3 BHK Flats..."
      "2 and 3 BHK Flats..."
      "2, 3, 4 BHK Flats..."
      "3.5 BHK Flat..." → single [3.5]
    """
    multi_match = re.search(
        r'((?:\d+\.?\d*\s*(?:,|&|and)\s*)+\d+\.?\d*)\s*bhk',
        title, re.IGNORECASE
    )
    if multi_match:
        raw = multi_match.group(1)
        bhk_values = re.findall(r'\d+\.?\d*', raw)
        if len(bhk_values) > 1:
            return [float(v) for v in bhk_values]

    # Single BHK fallback — existing behavior
    single_match = re.search(r'(\d+\.?\d*)\s*bhk', title, re.IGNORECASE)
    if single_match:
        return [float(single_match.group(1))]

    return []  # No BHK found (e.g. plot)

# ============================================================
# --- 7. CARD-LEVEL BHK PRICE PARSING ---
# ============================================================

def parse_card_bhk_prices(card, bhk_values, category):
    """
    Reads the listing card HTML for side-by-side BHK price blocks.
    On housing.com, multi-BHK cards show something like:
      <span>2 BHK Flat</span> <span>₹29.37 L - 31.38 L</span>
      <span>3 BHK Flat</span> <span>₹40 L - 40.05 L</span>
    Returns a dict: {2.0: price_int, 3.0: price_int}
    Falls back to None for any BHK not found.
    """
    result = {bhk: None for bhk in bhk_values}
    card_text = card.get_text(separator="\n")
    lines = [l.strip() for l in card_text.splitlines() if l.strip()]

    for i, line in enumerate(lines):
        bhk_label_match = re.match(r'^(\d+\.?\d*)\s*bhk', line, re.IGNORECASE)
        if bhk_label_match:
            bhk_num = float(bhk_label_match.group(1))
            if bhk_num not in result:
                continue
            # Price is usually on the next 1-2 lines
            for j in range(i + 1, min(i + 4, len(lines))):
                if '₹' in lines[j] or re.search(r'\d+.*(?:l\b|lac|lakh|cr)', lines[j], re.IGNORECASE):
                    price = smart_clean_price(lines[j], category=category)
                    if price:
                        result[bhk_num] = price
                        break

    return result

# ============================================================
# --- 8. INNER PAGE BHK DATA SCRAPING (FIXED: ACCURATE AREA) ---
# ============================================================

def scrape_inner_bhk_data(sb, inner_soup, bhk_values, category):
    """
    Since housing.com renders floor plan tabs via JavaScript (not visible to
    BeautifulSoup), we use a smarter approach:
    
    1. Extract avg price/sqft from page text
    2. Use each BHK's price (from tab label) + avg rate to calculate area
    3. Fall back to overall size range split proportionally if rate not found
    """
    result = {bhk: {'price': None, 'area_sqft': None, 'area_sqyd': None} for bhk in bhk_values}

    try:
        page_text = inner_soup.get_text(separator="\n")
        lines = [l.strip() for l in page_text.splitlines() if l.strip()]

        # --- Extract avg price per sqft from page (e.g. "₹6.2 K/sq.ft") ---
        avg_rate = None
        for line in lines:
            rate_match = re.search(r'(\d+\.?\d*)\s*k/sq\.?ft', line, re.IGNORECASE)
            if rate_match:
                avg_rate = float(rate_match.group(1)) * 1000
                break

        # --- Extract overall size range as fallback (e.g. "474 - 646 sq.ft") ---
        overall_min, overall_max = None, None
        for line in lines:
            range_match = re.search(
                r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*sq\.?\s*ft', line, re.IGNORECASE)
            if range_match:
                overall_min = float(range_match.group(1))
                overall_max = float(range_match.group(2))
                break

        # --- For each BHK: get price from tab label, calculate area ---
        for bhk in bhk_values:
            bhk_label = int(bhk) if bhk == int(bhk) else bhk

            # Step 1: Get price from BHK tab label text
            # Tab labels look like: "2 BHK Apartm... 29.37 - 31.38 L"
            try:
                updated_soup = sb.get_beautiful_soup()
                tab_elems = updated_soup.find_all(
                    lambda tag: tag.name in ['button', 'div', 'span'] and
                    re.search(
                        rf'^{re.escape(str(bhk_label))}\s*bhk',
                        tag.get_text(strip=True), re.IGNORECASE)
                )
                for tab_elem in tab_elems:
                    tab_full_text = tab_elem.get_text(separator=" ")
                    price_in_tab = re.search(
                        r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*(l\b|lac|lakh|cr|crore)',
                        tab_full_text, re.IGNORECASE
                    )
                    if price_in_tab:
                        lo = smart_clean_price(
                            f"{price_in_tab.group(1)} {price_in_tab.group(3)}", category=category)
                        hi = smart_clean_price(
                            f"{price_in_tab.group(2)} {price_in_tab.group(3)}", category=category)
                        if lo and hi:
                            result[bhk]['price'] = int((lo + hi) / 2)
                        elif lo:
                            result[bhk]['price'] = lo
                        break
            except Exception as e:
                print(f"    ⚠️ Price tab parse error for {bhk} BHK: {e}")

            # Step 2: Calculate area from price / avg_rate
            price = result[bhk]['price']
            if price and avg_rate:
                area_sqft = round(price / avg_rate, 2)
                result[bhk]['area_sqft'] = area_sqft
                result[bhk]['area_sqyd'] = round(area_sqft / 9.0, 2)
                print(f"    📐 {bhk} BHK: ₹{price} / ₹{avg_rate}/sqft = {area_sqft} sqft")

            # Step 3: Fallback — split overall range across BHKs by index
            # e.g. 2BHK gets the lower portion, 3BHK gets the upper portion
            elif overall_min and overall_max:
                sorted_bhks = sorted(bhk_values)
                idx = sorted_bhks.index(bhk)
                step = (overall_max - overall_min) / max(len(sorted_bhks) - 1, 1)
                area_sqft = round(overall_min + idx * step, 2)
                result[bhk]['area_sqft'] = area_sqft
                result[bhk]['area_sqyd'] = round(area_sqft / 9.0, 2)
                print(f"    📐 {bhk} BHK area (range fallback): {area_sqft} sqft")

    except Exception as e:
        print(f"    ⚠️ BHK data scraping failed: {e}")

    return result

# ============================================================
# --- 9. MAIN SCRAPE FUNCTION ---
# ============================================================

def scrape_category(url, category_type):
    db = SessionLocal()
    with SB(uc=True, headless=False) as sb:
        sb.uc_open_with_reconnect(url, 4)
        print(f"\n--- 🛠️ {category_type.upper()} PRODUCTION ENGINE ACTIVE ---")

        while True:
            cmd = input(f"[s] Scrape | [q] Quit: ").lower()
            if cmd == 'q': break
            if cmd == 's':
                soup = sb.get_beautiful_soup()
                cards = soup.find_all("article")
                print(f"👀 Found {len(cards)} listings. Starting Deep Scan...")

                for card in cards:
                    try:
                        # 1. BASE CARD EXTRACTION
                        h2_elem = card.find("h2")
                        title = h2_elem.get_text(strip=True) if h2_elem else "Unknown"


                        # Skip mixed listings like "Residential Land / Plot, 3 BHK Builder Floors in Sector 97"
                        if category_type == "floor" and re.search(r'residential land|plot', title, re.IGNORECASE):
                            print(f"  ⏭️ Skipping mixed listing: {title[:50]}")
                            continue
                        card_text = card.get_text(separator=" ")

                        sector = extract_sector(title)

                        # FIXED: Use extract_society_name for clean project name in both paths
                        project_name = extract_society_name(card, h2_elem)

                        # Safety: if name still looks like a BHK fragment, fall back to "Unknown"
                        if re.match(r'^[\d,\s&.]+$', project_name) or len(project_name) < 5:
                            project_name = "Unknown"

                        # Detect how many BHK types this listing has
                        bhk_values = detect_multi_bhk(title) if category_type in ["society", "floor"] else []
                        is_multi_bhk = len(bhk_values) > 1

                        # ── MULTI-BHK PATH ──────────────────────────────────────
                        if is_multi_bhk:
                            print(f"\n  🏢 Multi-BHK detected: {title}")
                            print(f"     BHK types: {bhk_values}")

                            # Parse card-level prices per BHK (fast, no tab click needed)
                            card_bhk_prices = parse_card_bhk_prices(card, bhk_values, category_type)

                            # Deep scan inner page for per-BHK area + refined price
                            inner_bhk_data = {bhk: {'price': None, 'area_sqft': None, 'area_sqyd': None}
                                              for bhk in bhk_values}

                            # Shared fields from inner page
                            possession, towers, floor_num, total_floors, amenities = \
                                "Not Ready", None, None, None, "Basic"

                            link_elem = card.find("a", href=True)
                            if link_elem:
                                full_url = ("https://housing.com" + link_elem['href']
                                            if link_elem['href'].startswith('/')
                                            else link_elem['href'])
                                sb.execute_script(f"window.open('{full_url}', '_blank');")
                                sb.switch_to_newest_window()
                                sb.sleep(2)
                                inner_soup = sb.get_beautiful_soup()
                                inner_text = inner_soup.get_text(separator=" | ").lower()

                                # Scrape BHK tabs on inner page for accurate area + price
                                inner_bhk_data = scrape_inner_bhk_data(
                                    sb, inner_soup, bhk_values, category_type)

                                if category_type == "society":
                                    if any(x in inner_text for x in ["ready to move", "possession status | ready"]):
                                        possession = "Ready"
                                    t_m = re.search(r'total towers \| (\d+)', inner_text)
                                    if t_m: towers = int(t_m.group(1))
                                    a_list = ["gym", "pool", "clubhouse", "security", "park",
                                              "parking", "garden", "lift"]
                                    found = [a.capitalize() for a in a_list if a in inner_text]
                                    amenities = ", ".join(found) if found else "Basic Amenities"
                                elif category_type == "floor":
                                    f_m = re.search(
                                        r'floor number \| (\d+|ground|lower|upper|first|second|third)',
                                        inner_text)
                                    if f_m: floor_num = f_m.group(1).capitalize()
                                    tf_m = re.search(r'total floors \| (\d+)', inner_text)
                                    if tf_m: total_floors = int(tf_m.group(1))

                                sb.driver.close()
                                sb.switch_to_default_window()

                            # Save one row per BHK type
                            for bhk in bhk_values:
                                # Price: prefer inner page tab data, fall back to card-level
                                price = (inner_bhk_data[bhk]['price']
                                         or card_bhk_prices.get(bhk))
                                area_sqft = inner_bhk_data[bhk]['area_sqft']
                                area_sqyd = inner_bhk_data[bhk]['area_sqyd']

                                rate_sqft = extract_rate_k_format(card_text)
                                if not rate_sqft and price and area_sqft:
                                    rate_sqft = round(price / area_sqft, 2)

                                bhk_int = int(bhk) if bhk == int(bhk) else bhk
                                bhk_title = f"{bhk_int} BHK - {title}"

                                # Dedup per BHK variant
                                if category_type == "floor":
                                    is_dup = db.query(BuilderFloor).filter_by(
                                        title=bhk_title, price=price).first()
                                else:
                                    is_dup = db.query(Society).filter_by(
                                        title=bhk_title, price=price).first()
                                if is_dup:
                                    print(f"    ⏭️ Duplicate skipped: {bhk_int} BHK")
                                    continue

                                if category_type == "floor":
                                    new_entry = BuilderFloor(
                                        name=project_name,
                                        title=bhk_title,
                                        sector=sector,
                                        price=price,
                                        display_price=str(price),
                                        total_floors=total_floors,
                                        floor_no=floor_num,
                                        bhk_type=str(bhk),
                                        price_per_sqft=rate_sqft,
                                        area_sqft=area_sqft
                                    )
                                else:
                                    new_entry = Society(
                                        name=project_name,
                                        society_name=project_name,
                                        title=bhk_title,
                                        sector=sector,
                                        price=price,
                                        display_price=str(price),
                                        possession_status=possession,
                                        total_towers=towers,
                                        bhk_type=str(bhk),
                                        price_per_sqft=rate_sqft,
                                        area_sqft=area_sqft,
                                        amenities=amenities
                                    )

                                db.add(new_entry)
                                db.commit()
                                print(f"    ✅ Saved: {bhk_int} BHK | {project_name[:25]} | "
                                      f"₹{price} | {area_sqft} sqft")

                        # ── SINGLE BHK PATH (original logic, fully preserved) ────
                        else:
                            #(plots only — take first size-price pair):
                            if category_type == "plot":
                                # Find all size-price pairs on the card e.g. "110 sq.yd Plot ₹99 L"
                                # They appear as sibling elements — grab just the first one
                                pairs = re.findall(
                                    r'(\d+\.?\d*)\s*sq\.?\s*yd[^₹]*₹\s*([\d\.\s]+(?:L|Lac|Lakh|Cr|Crore))',
                                    card_text, re.IGNORECASE
                                )
                                if pairs:
                                    first_sqyd = float(pairs[0][0])
                                    first_price_str = pairs[0][1]
                                    area_sqyd = round(first_sqyd, 2)
                                    area_sqft = round(first_sqyd * 9.0, 2)
                                    price_elem = f"₹{first_price_str.strip()}"
                                    price = smart_clean_price(price_elem, category=category_type)
                                else:
                                    # Fallback to original behavior
                                    price_elem = card.find(string=re.compile("₹"))
                                    price = smart_clean_price(price_elem, category=category_type)
                                    area_sqyd, area_sqft = extract_area_v2(card_text)
                            else:
                                price_elem = card.find(string=re.compile("₹"))
                                price = smart_clean_price(price_elem, category=category_type)
                                area_sqyd, area_sqft = extract_area_v2(card_text)
                            
                            if category_type == "plot":
                                rate_sqft = round((price / area_sqyd) / 9, 2) if price and area_sqyd else None
                            else:
                                rate_sqft = extract_rate_k_format(card_text)
                                if not rate_sqft and price and area_sqft:
                                    rate_sqft = round(price / area_sqft, 2)

                            bhk_val = None
                            if category_type in ["society", "floor"] and bhk_values:
                                bhk_val = bhk_values[0]

                            # DEDUPLICATION
                            if category_type == "plot":
                                is_dup = db.query(Plot).filter_by(title=title, price=price).first()
                            elif category_type == "floor":
                                is_dup = db.query(BuilderFloor).filter_by(title=title, price=price).first()
                            else:
                                is_dup = db.query(Society).filter_by(title=title, price=price).first()
                            if is_dup: continue

                            # DEEP SCAN
                            possession, towers, floor_num, total_floors, road_width, is_corner, amenities = \
                                "Not Ready", None, None, None, None, False, "Basic"

                            link_elem = card.find("a", href=True)
                            if link_elem:
                                full_url = ("https://housing.com" + link_elem['href']
                                            if link_elem['href'].startswith('/')
                                            else link_elem['href'])
                                sb.execute_script(f"window.open('{full_url}', '_blank');")
                                sb.switch_to_newest_window()
                                sb.sleep(2)
                                inner_soup = sb.get_beautiful_soup()
                                inner_text = inner_soup.get_text(separator=" | ").lower()


                                if not area_sqyd:
                                    if category_type == "floor":
                                        first_match = re.search(
                                            r'(\d+[\.\d]*)\s*(?:-|to|–)?\s*(\d+[\.\d]*)?\s*(sq\.?yd|sq\.?ft|gaj|sq\.\s*yards?)',
                                            inner_text, re.IGNORECASE
                                        )
                                        if first_match:
                                            v1 = float(first_match.group(1))
                                            v2 = float(first_match.group(2)) if first_match.group(2) else v1
                                            avg = (v1 + v2) / 2
                                            if 'ft' in first_match.group(3).lower():
                                                area_sqft = round(avg, 2)
                                                area_sqyd = round(avg / 9.0, 2)
                                            else:  # sq.yd or gaj
                                                area_sqyd = round(avg, 2)
                                                area_sqft = round(avg * 9.0, 2)
                                    else:
                                        area_sqyd, area_sqft = extract_area_v2(inner_text)

                                if not rate_sqft: rate_sqft = extract_rate_k_format(inner_text)

                                if category_type == "plot":
                                    w_m = re.search(
                                        r'width of facing road \| (\d+[\.\d]*)\s*(yd|m|ft)', inner_text)
                                    if w_m:
                                        val = float(w_m.group(1)); unit = w_m.group(2)
                                        val *= 0.9144 if unit == 'yd' else (0.3048 if unit == 'ft' else 1.0)
                                        road_width = round(val, 2)
                                    if "corner plot | yes" in inner_text: is_corner = True
                                elif category_type == "society":
                                    if any(x in inner_text for x in ["ready to move", "possession status | ready"]):
                                        possession = "Ready"
                                    t_m = re.search(r'total towers \| (\d+)', inner_text)
                                    if t_m: towers = int(t_m.group(1))
                                    a_list = ["gym", "pool", "clubhouse", "security", "park",
                                              "parking", "garden", "lift"]
                                    found = [a.capitalize() for a in a_list if a in inner_text]
                                    amenities = ", ".join(found) if found else "Basic Amenities"
                                elif category_type == "floor":
                                    f_m = re.search(
                                        r'floor number \| (\d+|ground|lower|upper|first|second|third)',
                                        inner_text)
                                    if f_m: floor_num = f_m.group(1).capitalize()
                                    tf_m = re.search(r'total floors \| (\d+)', inner_text)
                                    if tf_m: total_floors = int(tf_m.group(1))

                                sb.driver.close()
                                sb.switch_to_default_window()

                            if category_type == "plot":
                                rate_sqft = round((price / area_sqyd) / 9, 2) if price and area_sqyd else None
                            else:
                                rate_sqft = extract_rate_k_format(card_text)
                                if not rate_sqft and price and area_sqft:
                                    rate_sqft = round(price / area_sqft, 2)

                            # DATABASE COMMIT
                            if category_type == "plot":
                                new_entry = Plot(
                                    name=project_name,
                                    title=title,
                                    sector=sector,
                                    price=price,
                                    display_price=price_elem,
                                    plot_area_sqyd=area_sqyd,
                                    road_width_meter=road_width,
                                    is_corner_plot="Yes" if is_corner else "No",
                                    price_per_sqft=rate_sqft
                                )
                            elif category_type == "floor":
                                new_entry = BuilderFloor(
                                    name=project_name,
                                    title=title,
                                    sector=sector,
                                    price=price,
                                    display_price=price_elem,
                                    total_floors=total_floors,
                                    floor_no=floor_num,
                                    bhk_type=str(bhk_val),
                                    price_per_sqft=rate_sqft,
                                    area_sqft=area_sqft
                                )
                            else:
                                new_entry = Society(
                                    name=project_name,
                                    society_name=project_name,
                                    title=title,
                                    sector=sector,
                                    price=price,
                                    display_price=price_elem,
                                    possession_status=possession,
                                    total_towers=towers,
                                    bhk_type=str(bhk_val),
                                    price_per_sqft=rate_sqft,
                                    area_sqft=area_sqft,
                                    amenities=amenities
                                )

                            db.add(new_entry)
                            db.commit()
                            print(f"  ✅ Saved: {project_name[:25]} | Rate: ₹{rate_sqft}/sqft")

                    except Exception as e:
                        print(f"  ⚠️ Error: {e}")
                        if len(sb.driver.window_handles) > 1:
                            sb.driver.close()
                            sb.switch_to_default_window()

# --- THE MAIN SECTION ---
if __name__ == "__main__":
    print("Which category do you want to scrape?\n1. Plots | 2. Builder Floors | 3. Societies")
    choice = input("Enter 1, 2, or 3: ")
    targets = {
        "1": ("https://housing.com/in/buy/faridabad/plot-faridabad", "plot"),
        "2": ("https://housing.com/in/buy/faridabad/builderfloor-faridabad", "floor"),
        "3": ("https://housing.com/in/buy/faridabad/flat-faridabad", "society")
    }
    if choice in targets:
        url, cat = targets[choice]
        scrape_category(url, cat)
    else:
        print("Invalid choice. Exiting.")