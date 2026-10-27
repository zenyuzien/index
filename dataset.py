"""
HAI dataset generation
======================
Builds the 64-parish (Louisiana) table of the SEVEN indicators used by the Healthcare Accessibility Index.

Output of build_dataset():  one row per parish, columns
    parish, fips_code,
    primary_care_clinician_ratio_proxy      (people per primary-care physician)
    avg_drive_time_to_nearest_pcp_minutes   (minutes by road to the nearest primary-care facility)
    pct_households_no_vehicle               (% of households with no vehicle)
    non_elderly_uninsured_rate              (% of people under 65 without health insurance)
    low_income_below200fpl_pct              (% of people below 200% of the Federal Poverty Level)
    fixed_broadband_subscription_rate       (% of households WITH a broadband subscription)
    outpatient_facility_density_per_10k     (health facilities of ANY type per 10,000 residents)

--------------------------------------------------------------------------------------------------------------------
DATA SOURCES   (local path  ->  original source of that path  ->  what is taken from it)
--------------------------------------------------------------------------------------------------------------------
data/la-2025-primary_care_physicians.csv
    Source: HRSA Area Health Resources Files (AHRF) and Shortage Registries, combined with County Health Rankings
            (derived from the AMA Physician Masterfile).
    Taken:  column parish_pop_to_pcp_ratio (text such as "2,580:1") -> primary_care_clinician_ratio_proxy.

data/la_health_facilities_with_geom.csv
    Source: Louisiana Department of Health (LDH) Health Standards Section (HSS) Provider Directories, CMS Provider of
            Services (POS) data, and the NPPES NPI registry.
    Taken:  ALL rows, counted per parish (parish column) -> outpatient_facility_density_per_10k;
            latitude/longitude of the RHC / FQHC / Rural Health Clinic / Rural FQHC rows -> destinations for the drive time.

data/sdoh_parish_level_data.csv
    Source: U.S. Census Bureau American Community Survey (ACS) 5-Year Estimates.
    Taken:  no_vehicle_available and total_noof_households -> pct_households_no_vehicle.

data/uninsured_underage65.csv
    Source: U.S. Census Bureau Small Area Health Insurance Estimates (SAHIE) and ACS 5-Year Estimates.
    Taken:  PCTUI from the county rows with agecat=0, iprcat=0, racecat=0, sexcat=0 -> non_elderly_uninsured_rate.

data/louisiana_rurality_with_geom.csv
    Source: ZIP centroid coordinates (original source not documented).
    Taken:  zip_code, lat, lng -> ZIP centroids used as routing origins for avg_drive_time_to_nearest_pcp_minutes.

https://gisportal.hrsa.gov/server/rest/services/udsmapper/ZCTA/MapServer/0   (no local path; live API)
    Source: HRSA UDS Mapper, ZCTA layer (ACS-based ZIP-level data).
    Taken:  ACSG_200FPL_PCT -> low_income_below200fpl_pct;  No_HH_Broadband_PCT (+ Total_HH) ->
            fixed_broadband_subscription_rate;  TOTPOP -> ZIP population weights for drive time and low-income.

https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_county20_natl.txt   (cached in raw_cache/)
    Source: Census Bureau ZCTA-to-County Relationship File, 2020.
    Taken:  the parish holding the largest land+water area of each ZIP -> assigns every ZIP to one parish; also gives
            the 64 parish names and FIPS codes (parish, fips_code).

https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/totals/co-est2024-alldata.csv   (cached in raw_cache/)
    Source: Census Bureau Population Estimates Program, county totals, vintage 2024.
    Taken:  POPESTIMATE2024 -> the parish population in outpatient_facility_density_per_10k.

http://router.project-osrm.org   (no local path; answers cached in raw_cache/osrm_drive_times.csv)
    Source: OSRM public routing engine on OpenStreetMap road data.
    Taken:  road drive duration (seconds -> minutes) from each ZIP centroid to its nearest primary-care facility
            -> avg_drive_time_to_nearest_pcp_minutes.
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

# ====================================================================================================================
# 1. PATHS, URLS AND CONSTANTS
# ====================================================================================================================
HERE = os.path.dirname(os.path.abspath(__file__))   # the HAI folder itself; every path below is relative to it
DATA = os.path.join(HERE, "data")                   # raw input files shipped with this folder
CACHE = os.path.join(HERE, "raw_cache")             # downloaded files / OSRM answers (re-created if deleted)
os.makedirs(CACHE, exist_ok=True)

HRSA_URL = "https://gisportal.hrsa.gov/server/rest/services/udsmapper/ZCTA/MapServer/0/query"
ZCTA_COUNTY_URL = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_county20_natl.txt"
PEP_URL = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/totals/co-est2024-alldata.csv"
OSRM_URL = "http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"

# Facility types in la_health_facilities_with_geom.csv used ONLY as drive-time destinations ("nearest primary-care
# facility"). The facility-density indicator does NOT use this list: it counts every facility row in the file.
PC_TYPES = ["RHC", "FQHC", "Rural Health Clinic", "Rural Federally Qualified Health Center (FQHC)"]

# ====================================================================================================================
# 2. INDICATOR DIRECTIONS
# ====================================================================================================================
# Our own a-priori judgment (NOT derived from the data) of what a HIGH value of each indicator means:
#   +1 : higher value = worse access (a barrier)     -1 : higher value = better access
# Every index module uses this to point all seven indicators the same way before/while combining them.
SIGNS = {
    "primary_care_clinician_ratio_proxy": +1,     # more people per doctor        -> worse
    "outpatient_facility_density_per_10k": -1,    # more clinics per resident     -> better
    "avg_drive_time_to_nearest_pcp_minutes": +1,  # longer drive                  -> worse
    "pct_households_no_vehicle": +1,              # more households without a car -> worse
    "non_elderly_uninsured_rate": +1,             # more uninsured                -> worse
    "low_income_below200fpl_pct": +1,             # more low-income people        -> worse
    "fixed_broadband_subscription_rate": -1,      # more households subscribed    -> better
}
INDICATORS = list(SIGNS.keys())

# The ONE indicator that is ever imputed (see section 7). The other six have no missing or zero values.
FACILITY_COL = "outpatient_facility_density_per_10k"


# ====================================================================================================================
# 3. SMALL HELPERS
# ====================================================================================================================
def download(url, dest):
    """Download `url` to `dest` once; later calls reuse the saved file."""
    if not os.path.exists(dest):
        print(f"  downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest


def norm_zip(s):
    """Return a clean 5-digit ZIP string, or None if the value is not a usable ZIP (drops ZIP+4 and float artifacts)."""
    s = str(s).strip()
    if s in ("", "nan", "None", "-"):
        return None
    s = s.split("-")[0].split(".")[0]
    s = s.zfill(5) if s.isdigit() else s
    return s if s.isdigit() and len(s) == 5 else None


def norm_parish(p):
    """
    Lower-case parish key used to join files that spell parishes differently. It removes dots and the word "Parish",
    and merges the spellings found in the facility file: 'La Salle'/'Lasalle'/'LaSalle', 'St'/'St.'/'Saint',
    and 'DeSoto'/'De Soto'. Without this, facilities of those parishes would silently be dropped from the counts.
    """
    p = str(p).strip().lower().replace(".", "")
    p = re.sub(r"\s+parish$", "", p)                 # 'Ascension Parish' -> 'ascension'
    p = re.sub(r"^saint\b", "st", p)                 # 'Saint Charles'    -> 'st charles'
    p = p.replace("la salle", "lasalle").replace("desoto", "de soto")
    return p


def wavg(values, weights):
    """Weighted average that ignores rows where the value or the weight is missing (NaN if nothing is left)."""
    mask = values.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def haversine_miles(lat1, lon1, lat2, lon2):
    """Straight-line ("as the crow flies") distance in miles between points; works on numpy arrays."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ====================================================================================================================
# 4. ZIP-LEVEL BUILDING BLOCKS  (used by the drive-time, low-income and broadband indicators)
# ====================================================================================================================
def fetch_hrsa_zcta():
    """
    SOURCE 5: HRSA UDS Mapper API (ZCTA layer). Returns one row per Louisiana ZIP (ZCTA) with
      TOTPOP           total population of the ZIP
      ACSG_200FPL_PCT  % of people below 200% of the poverty level
      No_HH_Broadband_PCT  % of households WITHOUT a broadband subscription
      Total_HH         number of households
    Asks only for ZIPs starting with 70 or 71 (Louisiana), pages through the results, and caches them on disk.
    """
    cache = os.path.join(CACHE, "hrsa_zcta.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={"zip": str})
    rows, offset = [], 0
    while True:
        params = {
            "where": "ZCTA5CE20 LIKE '70%' OR ZCTA5CE20 LIKE '71%'",
            "outFields": "ZCTA5CE20,TOTPOP,ACSG_200FPL_PCT,No_HH_Broadband_PCT,Total_HH",
            "returnGeometry": "false",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": 1000,
        }
        with urllib.request.urlopen(HRSA_URL + "?" + urllib.parse.urlencode(params), timeout=120) as r:
            d = json.load(r)
        feats = d.get("features", [])
        rows += [f["attributes"] for f in feats]
        if not d.get("exceededTransferLimit") or not feats:   # no more pages
            break
        offset += len(feats)
    df = pd.DataFrame(rows).rename(columns={"ZCTA5CE20": "zip"})
    df["zip"] = df["zip"].astype(str)
    df.to_csv(cache, index=False)
    return df


def load_zip_table(api):
    """
    One row per Louisiana ZIP (all ZIPs the HRSA API returns) with what is needed to average ZIPs up to parishes:
      lat/lng          ZIP centroid (from louisiana_rurality_with_geom.csv)  -> routing origin; ZIPs without one get no drive time
      TOTPOP           API population of the ZIP
      ACSG_200FPL_PCT  low-income share from the API
      pop_weight       weight used when averaging ZIPs into a parish = TOTPOP; ZIPs with zero population get
                       weight NaN and are ignored.
    """
    rur = pd.read_csv(os.path.join(DATA, "louisiana_rurality_with_geom.csv"), dtype=str)
    rur["zip"] = rur["zip_code"].map(norm_zip)
    rur = rur.drop_duplicates("zip")[["zip", "lat", "lng"]]
    rur["lat"], rur["lng"] = pd.to_numeric(rur["lat"]), pd.to_numeric(rur["lng"])

    z = api[["zip", "TOTPOP", "ACSG_200FPL_PCT"]].merge(rur, on="zip", how="left")
    z["pop_weight"] = z["TOTPOP"].where(z["TOTPOP"] > 0)
    return z


def build_zip_to_parish():
    """
    SOURCE 6: Census ZCTA-to-County Relationship File (2020).
    A ZIP can straddle several parishes, but each ZIP is assigned to exactly ONE parish: the one that holds the
    largest land+water area of that ZIP. Returns a table indexed by ZIP with columns fips_code and parish.
    """
    path = download(ZCTA_COUNTY_URL, os.path.join(CACHE, "zcta_county_rel.txt"))
    rel = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8-sig")
    rel = rel[rel["GEOID_COUNTY_20"].str.startswith("22", na=False)].copy()        # 22 = Louisiana
    rel["zip"] = rel["GEOID_ZCTA5_20"].str.strip()
    rel = rel[rel["zip"].notna() & (rel["zip"] != "")]
    rel["area"] = rel["AREALAND_PART"].fillna("0").astype(int) + rel["AREAWATER_PART"].fillna("0").astype(int)
    rel = rel.sort_values("area", ascending=False, kind="stable").drop_duplicates("zip")   # keep biggest overlap
    rel["fips_code"] = rel["GEOID_COUNTY_20"].astype(int)
    rel["parish"] = rel["NAMELSAD_COUNTY_20"].str.replace(" Parish", "", regex=False)
    return rel.set_index("zip")[["fips_code", "parish"]]


def osrm_drive_minutes(zip_table):
    """
    SOURCES 2 + 8: for every ZIP centroid, find the straight-line-nearest primary-care facility (file 2), then ask the
    OSRM routing engine how many minutes it takes to DRIVE there on real roads (OpenStreetMap).
    Answers are cached; only ZIPs not yet in the cache are requested. ZIPs without a centroid get no drive time.
    Note: OSRM uses the live road network, so a fresh run can differ by ~0.1 min for a few ZIPs.
    """
    cache = os.path.join(CACHE, "osrm_drive_times.csv")

    # destination candidates: primary-care facilities that have coordinates
    fac = pd.read_csv(os.path.join(DATA, "la_health_facilities_with_geom.csv"))
    fac = fac[fac["facility_type"].isin(PC_TYPES)].dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    flat, flon = fac["latitude"].to_numpy(), fac["longitude"].to_numpy()

    # nearest facility (straight line) for each ZIP centroid
    pairs = zip_table[["zip", "lat", "lng"]].dropna().copy()
    idx = [int(np.argmin(haversine_miles(r.lat, r.lng, flat, flon))) for r in pairs.itertuples()]
    pairs["dest_lat"], pairs["dest_lon"] = flat[idx], flon[idx]

    # ask OSRM for the road route of each pair (skip pairs already cached)
    done = pd.read_csv(cache, dtype={"zip": str}) if os.path.exists(cache) else pd.DataFrame(columns=["zip", "drive_time_minutes"])
    have = dict(zip(done["zip"], done["drive_time_minutes"]))
    session = urllib.request.build_opener()
    for i, r in enumerate(pairs.itertuples()):
        if r.zip in have and pd.notna(have[r.zip]):
            continue
        url = OSRM_URL.format(lon1=r.lng, lat1=r.lat, lon2=r.dest_lon, lat2=r.dest_lat)
        try:
            with session.open(url, timeout=10) as resp:
                d = json.load(resp)
            have[r.zip] = round(d["routes"][0]["duration"] / 60, 2) if d.get("code") == "Ok" else np.nan   # seconds -> minutes
        except Exception:
            have[r.zip] = np.nan
        if i % 100 == 0:
            print(f"  OSRM {i}/{len(pairs)}")
        time.sleep(0.05)   # be polite to the public server
    out = pd.DataFrame({"zip": list(have.keys()), "drive_time_minutes": list(have.values())})
    out.to_csv(cache, index=False)
    return out


# ====================================================================================================================
# 5. THE SEVEN INDICATORS  (one function each; every function says exactly where its numbers come from)
# ====================================================================================================================
def ind_primary_care_ratio():
    """
    INDICATOR 1 - primary_care_clinician_ratio_proxy   (higher = worse)
    SOURCE 1: data/la-2025-primary_care_physicians.csv  (HRSA AHRF + Shortage Registries + County Health Rankings /
    AMA Physician Masterfile). The file stores the ratio as text such as "2,580:1" (2,580 people per physician).
    We keep the digits before the colon. This also repairs typos in the file such as "9,20:1" -> 920.
    Physician headcount only (no nurse practitioners / PAs, no FTE adjustment).
    """
    pcp = pd.read_csv(os.path.join(DATA, "la-2025-primary_care_physicians.csv"))
    pcp["key"] = pcp["parish"].map(norm_parish)
    pcp["primary_care_clinician_ratio_proxy"] = pcp["parish_pop_to_pcp_ratio"].apply(
        lambda s: int(re.sub(r"[^\d]", "", str(s).split(":")[0])) if pd.notna(s) else np.nan)
    return pcp[["key", "primary_care_clinician_ratio_proxy"]]


def ind_no_vehicle():
    """
    INDICATOR 3 - pct_households_no_vehicle   (higher = worse)
    SOURCE 3: data/sdoh_parish_level_data.csv  (U.S. Census Bureau ACS 5-Year Estimates).
    = households with no vehicle / total households x 100.
    """
    sd = pd.read_csv(os.path.join(DATA, "sdoh_parish_level_data.csv"))
    sd["key"] = sd["label_grouping"].str.replace(" Parish", "", regex=False).map(norm_parish)
    sd["pct_households_no_vehicle"] = (sd["no_vehicle_available"] / sd["total_noof_households"] * 100).round(2)
    return sd[["key", "pct_households_no_vehicle"]]


def ind_uninsured():
    """
    INDICATOR 4 - non_elderly_uninsured_rate   (higher = worse)
    SOURCE 4: data/uninsured_underage65.csv  (Census SAHIE program and ACS 5-Year Estimates; year 2023).
    The file holds many demographic slices per county, so we pick the "everyone under 65" slice:
      agecat=0 (under 65), iprcat=0 (all income levels), racecat=0 (all races), sexcat=0 (both sexes),
      countyfips>0 (county rows, not the state total). PCTUI = percent uninsured.
    """
    su = pd.read_csv(os.path.join(DATA, "uninsured_underage65.csv"))
    su = su[(su.agecat == 0) & (su.iprcat == 0) & (su.racecat == 0) & (su.sexcat == 0) & (su.countyfips > 0)].copy()
    su["fips_code"] = 22000 + su["countyfips"]     # Louisiana state code 22 + 3-digit county code
    return su[["fips_code", "PCTUI"]].rename(columns={"PCTUI": "non_elderly_uninsured_rate"})


def ind_drive_time(zt):
    """
    INDICATOR 5 - avg_drive_time_to_nearest_pcp_minutes   (higher = worse)
    SOURCES 2 + 8 (facility coordinates + OSRM). Each ZIP already has its drive time; here ZIPs are averaged into a
    parish, weighted by ZIP population, so populous ZIPs count more.
    """
    d = zt.dropna(subset=["fips_code"]).groupby("fips_code").apply(
        lambda g: round(wavg(g["drive_time_minutes"], g["pop_weight"]), 2))
    return d.rename("avg_drive_time_to_nearest_pcp_minutes").reset_index().astype({"fips_code": int})


def ind_low_income(zt):
    """
    INDICATOR 6 - low_income_below200fpl_pct   (higher = worse)
    SOURCE 5: HRSA UDS Mapper API, field ACSG_200FPL_PCT (ZIP level, ACS-based). ZIPs are averaged into a parish,
    weighted by ZIP population (pop_weight).
    """
    d = zt.dropna(subset=["fips_code"]).groupby("fips_code").apply(
        lambda g: round(wavg(g["ACSG_200FPL_PCT"], g["pop_weight"]), 2))
    return d.rename("low_income_below200fpl_pct").reset_index().astype({"fips_code": int})


def ind_broadband(api, zip_parish):
    """
    INDICATOR 7 - fixed_broadband_subscription_rate   (higher = better)
    SOURCE 5: HRSA UDS Mapper API, field No_HH_Broadband_PCT = % of households WITHOUT a broadband subscription.
    Averaged into parishes using ALL API ZIPs, weighted by number of households (Total_HH); then flipped
    (100 - value) so the published column is the share that DOES subscribe.
    """
    a = api.copy()
    a["fips_code"] = a["zip"].map(zip_parish["fips_code"])
    nobb = a.dropna(subset=["fips_code"]).groupby("fips_code").apply(
        lambda g: round(wavg(g["No_HH_Broadband_PCT"], g["Total_HH"].where(g["Total_HH"] > 0)), 2))
    bb = (100 - nobb).round(2).rename("fixed_broadband_subscription_rate")
    return bb.reset_index().astype({"fips_code": int})


def ind_facility_density(df):
    """
    INDICATOR 2 - outpatient_facility_density_per_10k   (higher = better)
    SOURCE 2: data/la_health_facilities_with_geom.csv  (LDH HSS Provider Directories, CMS POS, NPPES).
    SOURCE 7: Census PEP vintage 2024 for the parish population.
    = number of facility rows of ANY type in the parish / parish population x 10,000.
    Parishes with no rows in the file come out as exactly 0 - see impute_zero_facility_density().
    `df` is the parish table (needs columns fips_code and key).
    """
    pep = pd.read_csv(download(PEP_URL, os.path.join(CACHE, "co-est2024-alldata.csv")), encoding="latin1")
    pep = pep[(pep["STATE"] == 22) & (pep["COUNTY"] > 0)].copy()          # Louisiana county rows only
    pep["fips_code"] = 22000 + pep["COUNTY"]

    fac = pd.read_csv(os.path.join(DATA, "la_health_facilities_with_geom.csv"))
    fac["key"] = fac["parish"].map(norm_parish)
    counts = fac.groupby("key").size().rename("facility_count")           # every facility type counts

    out = df[["fips_code", "key"]].merge(pep[["fips_code", "POPESTIMATE2024"]], on="fips_code", how="left")
    out = out.merge(counts, on="key", how="left")
    out["facility_count"] = out["facility_count"].fillna(0)             # no rows in the file -> zero facilities
    out["outpatient_facility_density_per_10k"] = (out["facility_count"] / out["POPESTIMATE2024"] * 10000).round(3)
    return out[["fips_code", "outpatient_facility_density_per_10k"]]


# ====================================================================================================================
# 6. ASSEMBLE THE PARISH TABLE
# ====================================================================================================================
def build_indicators():
    """Run every step above and join the seven indicators into one 64-row parish table (sorted by parish name)."""
    # -- ZIP-level ingredients ----------------------------------------------------------------------------------
    print("Fetching HRSA API ...")
    api = fetch_hrsa_zcta()                       # ZIP-level API table
    zip_parish = build_zip_to_parish()            # ZIP -> parish assignment
    zt = load_zip_table(api).join(zip_parish, on="zip")   # ZIP table, now tagged with its parish

    print("Routing (OSRM) ...")
    zt = zt.merge(osrm_drive_minutes(zt), on="zip", how="left")   # add the drive time of every ZIP

    # -- the list of 64 parishes (one row each) ------------------------------------------------------------------
    parishes = zip_parish.reset_index(drop=True).drop_duplicates("fips_code")[["parish", "fips_code"]]
    parishes["fips_code"] = parishes["fips_code"].astype(int)
    parishes["key"] = parishes["parish"].map(norm_parish)         # spelling-proof join key

    # -- attach the seven indicators (keys: fips_code for numeric-coded sources, key for name-coded sources) ------
    df = parishes
    df = df.merge(ind_drive_time(zt), on="fips_code", how="left")                  # 5
    df = df.merge(ind_low_income(zt), on="fips_code", how="left")                  # 6
    df = df.merge(ind_broadband(api, zip_parish), on="fips_code", how="left")      # 7
    df = df.merge(ind_primary_care_ratio(), on="key", how="left")                  # 1
    df = df.merge(ind_no_vehicle(), on="key", how="left")                          # 3
    df = df.merge(ind_uninsured(), on="fips_code", how="left")                     # 4
    df = df.merge(ind_facility_density(df), on="fips_code", how="left")            # 2

    return df.sort_values("parish").reset_index(drop=True)


# ====================================================================================================================
# 7. IMPUTATION - ONE COLUMN ONLY (outpatient_facility_density_per_10k), K-NEAREST-NEIGHBOURS
# ====================================================================================================================
KNN_K = 5    # number of most-similar parishes whose density is averaged


def impute_zero_facility_density(df):
    """
    Fills the zero values of FACILITY_COL (outpatient_facility_density_per_10k) and touches NO other column.

    A parish with no rows in the facility file gets a density of exactly 0. That is a gap in the source, not proof the
    parish has no facilities, so the 0 is treated as MISSING and filled by K-nearest-neighbours (KNN):
      1. describe every parish by its OTHER six indicators, each standardized (z-score) so no indicator dominates
         just because of its units;
      2. measure the Euclidean distance between the zero parish and every parish that has a real density;
      3. take the KNN_K closest parishes ("most similar profile") and average THEIR facility densities.
    Today this only affects West Baton Rouge.
    """
    others = [c for c in INDICATORS if c != FACILITY_COL]
    X = df[others].to_numpy(float)
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)                    # step 1
    zero = (df[FACILITY_COL] == 0).to_numpy()
    donors = np.where(~zero)[0]                                          # parishes with a real density
    for i in np.where(zero)[0]:
        dist = np.sqrt(((Z[donors] - Z[i]) ** 2).sum(axis=1))           # step 2
        nearest = donors[np.argsort(dist)[:KNN_K]]                       # step 3
        df.loc[df.index[i], FACILITY_COL] = round(df.loc[df.index[nearest], FACILITY_COL].mean(), 3)
        print("  KNN neighbours of", df.loc[df.index[i], "parish"], "->", df.loc[df.index[nearest], "parish"].tolist())
    return df


# ====================================================================================================================
# 8. PUBLIC ENTRY POINT
# ====================================================================================================================
def build_dataset():
    """Parish x indicator table (64 rows): build all seven indicators, then fill the zero facility density (KNN)."""
    return impute_zero_facility_density(build_indicators())
