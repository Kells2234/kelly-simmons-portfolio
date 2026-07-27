import io
import math
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point


# ===============================
# CONFIGURATION
# ===============================

ADDRESS = "718 8th St, Columbus, GA 31906"
RADII_MILES = [1, 3, 5, 7, 9, 11]
ACS_YEAR = 2024
TIGER_YEAR = 2024
OUTPUT_FILE = "output/columbus_radius_demographics.csv"

# Set to True if you want church counts from OpenStreetMap
INCLUDE_CHURCH_COUNT = True

INVALID_MEDIAN_INCOME_VALUES = {-666666666, -999999999}

# Base demographics
ACS_BASE_VARS = {
    "total_population": "B01003_001E",
    "total_households": "B11001_001E",
    "median_household_income": "B19013_001E",
    "owner_occupied_units": "B25003_002E",
    "renter_occupied_units": "B25003_003E",
    "median_age": "B01002_001E",
    "white_alone": "B02001_002E",
    "black_alone": "B02001_003E",
    "hispanic_any_race": "B03003_003E",
    "edu_bachelors": "B15003_022E",
    "edu_masters": "B15003_023E",
    "edu_professional": "B15003_024E",
    "edu_doctorate": "B15003_025E",
}

# Overall marital status 18+ from B12002
ACS_MARITAL_18_PLUS_VARS = {
    "marital_total_18_plus": [
        "B12002_004E", "B12002_005E", "B12002_006E", "B12002_007E", "B12002_008E",
        "B12002_010E", "B12002_011E", "B12002_012E", "B12002_013E", "B12002_014E",
        "B12002_015E", "B12002_016E", "B12002_017E", "B12002_018E", "B12002_019E",
        "B12002_021E", "B12002_022E", "B12002_023E", "B12002_024E", "B12002_025E",
        "B12002_026E", "B12002_027E", "B12002_028E", "B12002_029E", "B12002_030E",
        "B12002_032E", "B12002_033E", "B12002_034E", "B12002_035E", "B12002_036E",
        "B12002_037E", "B12002_038E", "B12002_039E", "B12002_040E", "B12002_041E",
        "B12002_044E", "B12002_045E", "B12002_046E", "B12002_047E", "B12002_048E",
        "B12002_050E", "B12002_051E", "B12002_052E", "B12002_053E", "B12002_054E",
        "B12002_055E", "B12002_056E", "B12002_057E", "B12002_058E", "B12002_059E",
        "B12002_061E", "B12002_062E", "B12002_063E", "B12002_064E", "B12002_065E",
        "B12002_066E", "B12002_067E", "B12002_068E", "B12002_069E", "B12002_070E",
        "B12002_072E", "B12002_073E", "B12002_074E", "B12002_075E", "B12002_076E",
        "B12002_077E", "B12002_078E", "B12002_079E", "B12002_080E", "B12002_081E",
    ],
    "never_married_18_plus": [
        "B12002_004E", "B12002_005E", "B12002_006E", "B12002_007E", "B12002_008E",
        "B12002_009E", "B12002_010E", "B12002_011E", "B12002_012E", "B12002_013E",
        "B12002_014E", "B12002_015E", "B12002_016E", "B12002_017E",
        "B12002_044E", "B12002_045E", "B12002_046E", "B12002_047E", "B12002_048E",
        "B12002_049E", "B12002_050E", "B12002_051E", "B12002_052E", "B12002_053E",
        "B12002_054E", "B12002_055E", "B12002_056E", "B12002_057E",
    ],
    "married_18_plus": [
        "B12002_019E", "B12002_020E", "B12002_021E", "B12002_022E", "B12002_023E",
        "B12002_024E", "B12002_025E", "B12002_026E", "B12002_027E", "B12002_028E",
        "B12002_029E", "B12002_030E", "B12002_031E",
        "B12002_059E", "B12002_060E", "B12002_061E", "B12002_062E", "B12002_063E",
        "B12002_064E", "B12002_065E", "B12002_066E", "B12002_067E", "B12002_068E",
        "B12002_069E", "B12002_070E", "B12002_071E",
    ],
    "separated_18_plus": [
        "B12002_032E", "B12002_033E", "B12002_034E", "B12002_035E", "B12002_036E",
        "B12002_037E", "B12002_038E", "B12002_039E", "B12002_040E", "B12002_041E",
        "B12002_042E",
        "B12002_072E", "B12002_073E", "B12002_074E", "B12002_075E", "B12002_076E",
        "B12002_077E", "B12002_078E", "B12002_079E", "B12002_080E", "B12002_081E",
        "B12002_082E",
    ],
    "widowed_18_plus": [
        "B12002_083E", "B12002_084E", "B12002_085E", "B12002_086E", "B12002_087E",
        "B12002_088E", "B12002_089E", "B12002_090E", "B12002_091E", "B12002_092E",
        "B12002_093E", "B12002_094E", "B12002_095E",
        "B12002_099E", "B12002_100E", "B12002_101E", "B12002_102E", "B12002_103E",
        "B12002_104E", "B12002_105E", "B12002_106E", "B12002_107E", "B12002_108E",
        "B12002_109E", "B12002_110E", "B12002_111E",
    ],
    "divorced_18_plus": [
        "B12002_096E", "B12002_097E", "B12002_098E", "B12002_099E", "B12002_100E",
        "B12002_101E", "B12002_102E", "B12002_103E", "B12002_104E", "B12002_105E",
        "B12002_106E", "B12002_107E", "B12002_108E",
        "B12002_112E", "B12002_113E", "B12002_114E", "B12002_115E", "B12002_116E",
        "B12002_117E", "B12002_118E", "B12002_119E", "B12002_120E", "B12002_121E",
        "B12002_122E", "B12002_123E", "B12002_124E",
    ],
}

# Household structure with own children under 18 from B11003
ACS_PARENT_VARS = {
    "two_parent_households_with_children": ["B11003_003E", "B11003_004E", "B11003_005E"],
    "single_father_households_with_children": ["B11003_011E", "B11003_012E", "B11003_013E"],
    "single_mother_households_with_children": ["B11003_016E", "B11003_017E", "B11003_018E"],
}

# Race-specific marital status tables
RACE_MARITAL_TABLES = {
    "white_alone": "B12002A",
    "black_alone": "B12002B",
    "white_non_hispanic": "B12002H",
    "hispanic": "B12002I",
}


# ===============================
# HELPERS
# ===============================

def clean_income_values(series: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")
    cleaned = cleaned.replace(list(INVALID_MEDIAN_INCOME_VALUES), pd.NA)
    cleaned = cleaned.where(cleaned > 0, pd.NA)
    return cleaned


def geocode_address_census(address: str) -> tuple[float, float, str]:
    url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    params = {
        "address": address,
        "benchmark": "Public_AR_Current",
        "format": "json",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    matches = data["result"]["addressMatches"]
    if not matches:
        raise ValueError(f"No Census geocoder match found for: {address}")

    best = matches[0]
    return best["coordinates"]["y"], best["coordinates"]["x"], best["matchedAddress"]


def get_geographies_from_geocoder(address: str) -> tuple[str, str]:
    url = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
    params = {
        "address": address,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    matches = data["result"]["addressMatches"]
    if not matches:
        raise ValueError(f"No Census geography match found for: {address}")

    county_record = matches[0]["geographies"]["Counties"][0]
    return county_record["STATE"], county_record["COUNTY"]


def get_local_projected_crs(lat: float, lon: float) -> str:
    zone = int(math.floor((lon + 180) / 6) + 1)
    return f"EPSG:{32600 + zone}"


def weighted_average(values: pd.Series, weights: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0) & (values > 0)
    if valid.sum() == 0:
        return None
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def build_summed_columns(df: pd.DataFrame, mappings: dict[str, list[str]]) -> pd.DataFrame:
    """
    Build multiple derived sum columns at once to avoid DataFrame fragmentation.
    """
    output = {}

    for new_col, cols in mappings.items():
        available = [c for c in cols if c in df.columns]
        if not available:
            output[new_col] = pd.Series(pd.NA, index=df.index)
        else:
            output[new_col] = (
                df[available]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .sum(axis=1)
            )

    return pd.DataFrame(output, index=df.index)


# ===============================
# ACS PULLS
# ===============================

def fetch_acs(year: int, state_fips: str, county_fips: str, variables: list[str]) -> pd.DataFrame:
    """
    Fetch ACS variables in batches because the Census API limits how many
    variables can be requested at once.
    """
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    geo_cols = ["state", "county", "tract"]

    merged_df = None

    for var_chunk in chunk_list(variables, 45):
        params = {
            "get": ",".join(var_chunk),
            "for": "tract:*",
            "in": f"state:{state_fips} county:{county_fips}",
        }

        response = requests.get(base_url, params=params, timeout=90)
        response.raise_for_status()
        rows = response.json()

        df = pd.DataFrame(rows[1:], columns=rows[0])

        for col in df.columns:
            if col not in geo_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["GEOID"] = df["state"] + df["county"] + df["tract"]

        keep_cols = ["GEOID"] + [c for c in df.columns if c not in geo_cols + ["GEOID"]]
        df = df[keep_cols]

        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, on="GEOID", how="outer")

    return merged_df


def get_base_acs_data(year: int, state_fips: str, county_fips: str) -> pd.DataFrame:
    df = fetch_acs(year, state_fips, county_fips, list(ACS_BASE_VARS.values()))
    rename_map = {v: k for k, v in ACS_BASE_VARS.items()}
    df = df.rename(columns=rename_map)
    df["median_household_income"] = clean_income_values(df["median_household_income"])
    df["bachelors_or_higher"] = (
        df["edu_bachelors"].fillna(0)
        + df["edu_masters"].fillna(0)
        + df["edu_professional"].fillna(0)
        + df["edu_doctorate"].fillna(0)
    )
    return df.copy()


def get_marital_18_plus_data(year: int, state_fips: str, county_fips: str) -> pd.DataFrame:
    needed_vars = sorted({v for cols in ACS_MARITAL_18_PLUS_VARS.values() for v in cols})
    df = fetch_acs(year, state_fips, county_fips, needed_vars)

    summed_df = build_summed_columns(df, ACS_MARITAL_18_PLUS_VARS)
    df = pd.concat([df[["GEOID"]], summed_df], axis=1)

    return df.copy()


def get_parent_household_data(year: int, state_fips: str, county_fips: str) -> pd.DataFrame:
    needed_vars = sorted({v for cols in ACS_PARENT_VARS.values() for v in cols})
    df = fetch_acs(year, state_fips, county_fips, needed_vars)

    summed_df = build_summed_columns(df, ACS_PARENT_VARS)
    summed_df["single_parent_households_with_children"] = (
        summed_df["single_father_households_with_children"].fillna(0)
        + summed_df["single_mother_households_with_children"].fillna(0)
    )

    df = pd.concat([df[["GEOID"]], summed_df], axis=1)
    return df.copy()


def get_race_specific_marital_data(year: int, state_fips: str, county_fips: str) -> pd.DataFrame:
    results = None

    for race_key, table in RACE_MARITAL_TABLES.items():
        vars_needed = [
            f"{table}_001E",
            f"{table}_003E", f"{table}_004E", f"{table}_005E", f"{table}_006E", f"{table}_007E",
            f"{table}_009E", f"{table}_010E", f"{table}_011E", f"{table}_012E", f"{table}_013E",
        ]
        try:
            df = fetch_acs(year, state_fips, county_fips, vars_needed)

            race_output = pd.DataFrame(index=df.index)
            race_output[f"{race_key}_marital_total_15_plus"] = df[f"{table}_001E"]
            race_output[f"{race_key}_never_married_15_plus"] = (
                df[f"{table}_003E"].fillna(0) + df[f"{table}_009E"].fillna(0)
            )
            race_output[f"{race_key}_married_15_plus"] = (
                df[f"{table}_004E"].fillna(0) + df[f"{table}_010E"].fillna(0)
            )
            race_output[f"{race_key}_separated_15_plus"] = (
                df[f"{table}_005E"].fillna(0) + df[f"{table}_011E"].fillna(0)
            )
            race_output[f"{race_key}_widowed_15_plus"] = (
                df[f"{table}_006E"].fillna(0) + df[f"{table}_012E"].fillna(0)
            )
            race_output[f"{race_key}_divorced_15_plus"] = (
                df[f"{table}_007E"].fillna(0) + df[f"{table}_013E"].fillna(0)
            )

            out_df = pd.concat([df[["GEOID"]], race_output], axis=1).copy()

            results = out_df if results is None else results.merge(out_df, on="GEOID", how="outer")
        except requests.exceptions.RequestException as exc:
            print(f"Skipping unavailable or failed marital race table: {table} -> {exc}")

    if results is None:
        return pd.DataFrame(columns=["GEOID"])

    return results.copy()


# ===============================
# SHAPES / SPATIAL
# ===============================

def load_tract_shapes(year: int, state_fips: str, county_fips: str) -> gpd.GeoDataFrame:
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        extract_dir = Path("data") / f"tiger_{year}_{state_fips}"
        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_dir)

    shp_files = list(extract_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError("No shapefile found in downloaded TIGER zip.")

    gdf = gpd.read_file(shp_files[0])
    gdf = gdf[gdf["COUNTYFP"] == county_fips].copy()
    return gdf[["GEOID", "geometry"]]


def summarize_radii(tracts_gdf: gpd.GeoDataFrame, center_point_proj: Point, radii_miles: list[float]) -> pd.DataFrame:
    tracts = tracts_gdf.copy()
    tracts["tract_area_m2"] = tracts.geometry.area
    results = []

    for radius in radii_miles:
        radius_m = radius * 1609.344
        buffer_geom = center_point_proj.buffer(radius_m)

        intersected = tracts[tracts.intersects(buffer_geom)].copy()
        if intersected.empty:
            continue

        intersected["intersection_geom"] = intersected.geometry.intersection(buffer_geom)
        intersected["intersection_area_m2"] = intersected["intersection_geom"].area
        intersected["area_weight"] = (
            intersected["intersection_area_m2"] / intersected["tract_area_m2"]
        ).clip(lower=0, upper=1)

        summary = {"radius_miles": radius}

        sum_like_cols = [
            col for col in intersected.columns
            if col not in {
                "GEOID", "geometry", "intersection_geom", "tract_area_m2",
                "intersection_area_m2", "area_weight", "median_household_income", "median_age"
            }
        ]

        for col in sum_like_cols:
            if pd.api.types.is_numeric_dtype(intersected[col]):
                summary[col] = float((intersected[col].fillna(0) * intersected["area_weight"]).sum())

        summary["median_household_income_est"] = weighted_average(
            intersected["median_household_income"],
            intersected["area_weight"],
        )
        summary["median_age_est"] = weighted_average(
            intersected["median_age"],
            intersected["area_weight"],
        )

        pop = summary.get("total_population", 0)
        occ = summary.get("owner_occupied_units", 0) + summary.get("renter_occupied_units", 0)
        m18 = summary.get("marital_total_18_plus", 0)
        kids_families = summary.get("two_parent_households_with_children", 0) + summary.get("single_parent_households_with_children", 0)

        summary["pct_white"] = (summary.get("white_alone", 0) / pop * 100) if pop else None
        summary["pct_black"] = (summary.get("black_alone", 0) / pop * 100) if pop else None
        summary["pct_hispanic"] = (summary.get("hispanic_any_race", 0) / pop * 100) if pop else None
        summary["pct_bachelors_or_higher"] = (summary.get("bachelors_or_higher", 0) / pop * 100) if pop else None
        summary["pct_owner_occupied"] = (summary.get("owner_occupied_units", 0) / occ * 100) if occ else None

        summary["pct_never_married_18_plus"] = (summary.get("never_married_18_plus", 0) / m18 * 100) if m18 else None
        summary["pct_married_18_plus"] = (summary.get("married_18_plus", 0) / m18 * 100) if m18 else None
        summary["pct_separated_18_plus"] = (summary.get("separated_18_plus", 0) / m18 * 100) if m18 else None
        summary["pct_widowed_18_plus"] = (summary.get("widowed_18_plus", 0) / m18 * 100) if m18 else None
        summary["pct_divorced_18_plus"] = (summary.get("divorced_18_plus", 0) / m18 * 100) if m18 else None

        summary["pct_two_parent_households_with_children"] = (
            summary.get("two_parent_households_with_children", 0) / kids_families * 100
        ) if kids_families else None
        summary["pct_single_parent_households_with_children"] = (
            summary.get("single_parent_households_with_children", 0) / kids_families * 100
        ) if kids_families else None

        results.append(summary)

    return pd.DataFrame(results)


# ===============================
# CHURCH COUNT
# ===============================

def fetch_churches_osm(lat: float, lon: float, max_radius_miles: float) -> gpd.GeoDataFrame:
    radius_meters = int(max_radius_miles * 1609.344)
    overpass_query = f"""
    [out:json][timeout:20];
    (
      node["amenity"="place_of_worship"](around:{radius_meters},{lat},{lon});
    );
    out body;
    """

    empty_gdf = gpd.GeoDataFrame(
        columns=["osm_id", "name", "religion", "denomination", "geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    )

    try:
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=overpass_query,
            timeout=25,
            headers={"User-Agent": "radius-demographics-script/1.0"},
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        print(f"Church lookup skipped: {exc}")
        return empty_gdf

    records = []
    seen_ids = set()

    for element in data.get("elements", []):
        if element.get("type") != "node":
            continue

        osm_id = element.get("id")
        unique_id = f"node_{osm_id}"
        if unique_id in seen_ids:
            continue
        seen_ids.add(unique_id)

        lat_ = element.get("lat")
        lon_ = element.get("lon")
        if lat_ is None or lon_ is None:
            continue

        tags = element.get("tags", {})
        records.append({
            "osm_id": unique_id,
            "name": tags.get("name"),
            "religion": tags.get("religion"),
            "denomination": tags.get("denomination"),
            "geometry": Point(lon_, lat_),
        })

    if not records:
        return empty_gdf

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def count_churches_by_radius(churches_gdf: gpd.GeoDataFrame, center_point_proj: Point, radii_miles: list[float]) -> dict[float, int]:
    if churches_gdf.empty:
        return {radius: 0 for radius in radii_miles}

    counts = {}
    for radius in radii_miles:
        buffer_geom = center_point_proj.buffer(radius * 1609.344)
        counts[radius] = int(churches_gdf.within(buffer_geom).sum())
    return counts


# ===============================
# MAIN
# ===============================

def main() -> None:
    print(f"Geocoding address: {ADDRESS}")
    lat, lon, matched_address = geocode_address_census(ADDRESS)
    state_fips, county_fips = get_geographies_from_geocoder(ADDRESS)

    print(f"Matched address: {matched_address}")
    print(f"Coordinates: lat={lat}, lon={lon}")
    print(f"State FIPS: {state_fips}, County FIPS: {county_fips}")

    print("Pulling base ACS tract data...")
    base_df = get_base_acs_data(ACS_YEAR, state_fips, county_fips)

    print("Pulling marital status 18+ data...")
    marital_df = get_marital_18_plus_data(ACS_YEAR, state_fips, county_fips)

    print("Pulling parent household data...")
    parent_df = get_parent_household_data(ACS_YEAR, state_fips, county_fips)

    print("Pulling race-specific marital data...")
    race_marital_df = get_race_specific_marital_data(ACS_YEAR, state_fips, county_fips)

    print("Loading tract geometries...")
    tracts = load_tract_shapes(TIGER_YEAR, state_fips, county_fips)

    print("Joining datasets...")
    gdf = (
        tracts
        .merge(base_df, on="GEOID", how="left")
        .merge(marital_df, on="GEOID", how="left")
        .merge(parent_df, on="GEOID", how="left")
        .merge(race_marital_df, on="GEOID", how="left")
        .copy()
    )

    local_crs = get_local_projected_crs(lat, lon)
    gdf = gdf.to_crs(local_crs)

    center_geo = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    center_proj = center_geo.to_crs(local_crs).iloc[0]

    if INCLUDE_CHURCH_COUNT:
        print("Pulling church locations from OpenStreetMap...")
        churches = fetch_churches_osm(lat, lon, max(RADII_MILES))
        if not churches.empty:
            churches = churches.to_crs(local_crs)
        church_counts = count_churches_by_radius(churches, center_proj, RADII_MILES)
    else:
        church_counts = {radius: 0 for radius in RADII_MILES}

    print(f"Calculating summaries for radii: {RADII_MILES}")
    results = summarize_radii(gdf, center_proj, RADII_MILES)
    results["church_count"] = results["radius_miles"].map(church_counts)

    if "church_count" in results.columns:
        results["churches_per_1000_population"] = (
            results["church_count"] / results["total_population"] * 1000
        ).round(4)

    round_cols = [c for c in results.columns if c != "radius_miles"]
    for col in round_cols:
        if pd.api.types.is_numeric_dtype(results[col]):
            results[col] = results[col].round(2)

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    results.to_csv(output_path, index=False)

    print("\nDone.")
    print(results.head())
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()