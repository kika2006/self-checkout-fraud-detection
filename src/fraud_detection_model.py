import re
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1. File paths
cashier_file = "data/cashier_data.csv"
case_file = "data/supermarket.csv"

shrinkage_file = "data/shrinkage.csv"
stockage_missing_file = "data/stockage_missing.csv"


# 2. Department layout from team EDA file
DEPT_MAP = {
    1: "Bakery & Pastry",
    2: "Beer & Wine",
    3: "Books & Magazines",
    4: "Candy & Chips",
    5: "Care & Hygiene",
    6: "Cereals & Spreads",
    7: "Cheese & Tapas",
    8: "Dairy & Eggs",
    9: "Freezer",
    10: "Fruit & Vegetables",
    11: "Household & Pet",
    12: "Meat & Fish",
    13: "Pasta & Rice",
    14: "Salads & Meals",
    15: "Sauces & Spices",
    16: "Soda & Juices",
    17: "Special Diet",
    18: "Vegetarian & Vegan",
}

DEPT_ABBR = {
    1: "BP",
    2: "BW",
    3: "BM",
    4: "CC",
    5: "CH",
    6: "CS",
    7: "CT",
    8: "DE",
    9: "FR",
    10: "FV",
    11: "HP",
    12: "MF",
    13: "PR",
    14: "SM",
    15: "SS",
    16: "SJ",
    17: "SD",
    18: "VV",
}

GRID_LAYOUT = {
    "CI": (0, 0),
    "FV": (0, 1),
    "SM": (0, 2),
    "VV": (0, 3),
    "MF": (0, 4),

    "DE": (1, 0),
    "SS": (1, 1),
    "PR": (1, 2),
    "CT": (1, 3),
    "BP": (1, 4),

    "CS": (2, 0),
    "SD": (2, 1),
    "CC": (2, 2),
    "SJ": (2, 3),
    "FR": (2, 4),

    "CO": (3, 0),
    "BM": (3, 1),
    "CH": (3, 2),
    "HP": (3, 3),
    "BW": (3, 4),
}

DEPT_POS = {
    dept_id: GRID_LAYOUT[abbr]
    for dept_id, abbr in DEPT_ABBR.items()
    if abbr in GRID_LAYOUT
}

DEPT_IDS = sorted(DEPT_POS.keys())

DIST_MATRIX = {
    (i, j): abs(DEPT_POS[i][0] - DEPT_POS[j][0]) +
            abs(DEPT_POS[i][1] - DEPT_POS[j][1])
    for i in DEPT_IDS
    for j in DEPT_IDS
}



# 3. Helper functions
def safe_divide(a, b):
    return a / b if b != 0 else 0

def minmax(values):
    values = np.array(values).reshape(-1, 1)

    if np.nanmax(values) == np.nanmin(values):
        return np.zeros(len(values))

    return MinMaxScaler().fit_transform(values).ravel()


def entropy(values):
    if len(values) == 0:
        return 0

    counts = pd.Series(values).value_counts()
    probabilities = counts / counts.sum()

    return float(-(probabilities * np.log(probabilities + 1e-9)).sum())


def manhattan_distance(dept1, dept2):
    return DIST_MATRIX.get((dept1, dept2), 0)


def top_value_share(prices, fraction):
    if len(prices) == 0:
        return 0

    prices_sorted = sorted(prices, reverse=True)
    top_n = max(1, int(len(prices_sorted) * fraction))

    return safe_divide(sum(prices_sorted[:top_n]), sum(prices_sorted))


def clean_dataframe(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


# 4. Load external EDA files
def load_shrinkage_features(path):
    default = {
        dept: {
            "items_shrinkage": 0,
            "revenue_shrinkage": 0,
            "items_lost": 0,
            "revenue_lost": 0,
        }
        for dept in range(1, 19)
    }

    if not os.path.exists(path):
        print("shrinkage.csv not found. Shrinkage features set to 0.")
        return default

    try:
        shrinkage = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    except Exception:
        shrinkage = pd.read_csv(path, encoding="utf-8-sig")

    shrinkage.columns = [str(c).strip() for c in shrinkage.columns]

    needed_cols = [
        "department",
        "items_lost",
        "revenue_lost",
        "items_shrinkage",
        "revenue_shrinkage",
    ]

    for col in needed_cols:
        if col not in shrinkage.columns:
            print("shrinkage.csv has unexpected columns. Shrinkage features set to 0.")
            print("Columns found:", list(shrinkage.columns))
            return default

    shrinkage["department"] = pd.to_numeric(shrinkage["department"], errors="coerce")
    shrinkage = shrinkage.dropna(subset=["department"]).copy()

    for col in [
        "items_lost",
        "revenue_lost",
        "items_shrinkage",
        "revenue_shrinkage",
    ]:
        shrinkage[col] = pd.to_numeric(shrinkage[col], errors="coerce").fillna(0)

    result = default.copy()

    for _, row in shrinkage.iterrows():
        dept = int(row["department"])

        if dept in result:
            result[dept] = {
                "items_shrinkage": float(row["items_shrinkage"]),
                "revenue_shrinkage": float(row["revenue_shrinkage"]),
                "items_lost": float(row["items_lost"]),
                "revenue_lost": float(row["revenue_lost"]),
            }

    return result


def load_stockage_missing_features(path):
    default = {
        dept: {
            "stockage_missing_count": 0,
            "stockage_missing_avg_value": 0,
            "stockage_missing_total_value": 0,
        }
        for dept in range(1, 19)
    }

    if not os.path.exists(path):
        print("stockage_missing.csv not found. Stockage features set to 0.")
        return default

    try:
        raw = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    except Exception:
        raw = pd.read_csv(path, encoding="utf-8-sig")

    raw.columns = [str(c).strip() for c in raw.columns]

    if "department" not in raw.columns or "item_value" not in raw.columns:
        raw = raw.iloc[:, :2]
        raw.columns = ["department", "item_value"]

    raw = raw[["department", "item_value"]].copy()

    raw["department"] = pd.to_numeric(raw["department"], errors="coerce")
    raw["item_value"] = pd.to_numeric(raw["item_value"], errors="coerce")

    raw = raw.dropna(subset=["department", "item_value"]).copy()

    if len(raw) == 0:
        print("stockage_missing.csv had no usable rows. Stockage features set to 0.")
        return default

    raw["department"] = raw["department"].astype(int)
    raw["item_value"] = raw["item_value"].astype(float)

    grouped = (
        raw.groupby("department")["item_value"]
        .agg(["count", "mean", "sum"])
        .reset_index()
    )

    result = default.copy()

    for _, row in grouped.iterrows():
        dept = int(row["department"])

        if dept in result:
            result[dept] = {
                "stockage_missing_count": float(row["count"]),
                "stockage_missing_avg_value": float(row["mean"]),
                "stockage_missing_total_value": float(row["sum"]),
            }

    return result


SHRINKAGE_BY_DEPT = load_shrinkage_features(shrinkage_file)
STOCKAGE_BY_DEPT = load_stockage_missing_features(stockage_missing_file)


# 5. Parse transaction files and calculate features
def parse_transaction_file(file_path):
    transactions = []

    with open(file_path, "r", encoding="utf-8") as file:
        for row_id, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            match = re.search(r"Transaction\s+(\d+):", line)

            if match:
                transaction_id = int(match.group(1))
                scans_part = line.split(":", 1)[1]
            else:
                transaction_id = row_id
                scans_part = line

            scans = scans_part.split(",")

            departments = []
            times = []
            prices = []
            has_time = False

            for scan in scans:
                scan = scan.strip()

                if scan == "":
                    continue

                parts = scan.split()

                if len(parts) == 3:
                    try:
                        dept = int(parts[0])
                        time_gap = float(parts[1])
                        price = float(parts[2])

                        departments.append(dept)
                        times.append(time_gap)
                        prices.append(price)
                        has_time = True
                    except ValueError:
                        continue

                elif len(parts) == 2:
                    try:
                        dept = int(parts[0])
                        price = float(parts[1])

                        departments.append(dept)
                        times.append(0)
                        prices.append(price)
                    except ValueError:
                        continue

            if len(prices) == 0:
                continue

            n_items = len(prices)
            total_spend = sum(prices)

            avg_price = np.mean(prices)
            median_price = np.median(prices)
            max_price = max(prices)
            min_price = min(prices)

            q10_price = np.percentile(prices, 10)
            q25_price = np.percentile(prices, 25)
            q75_price = np.percentile(prices, 75)
            q90_price = np.percentile(prices, 90)

            iqr_price = q75_price - q25_price
            price_std = np.std(prices)
            price_range = max_price - min_price
            price_cv = safe_divide(price_std, avg_price)

            cheap_050 = sum(1 for p in prices if p < 0.50)
            cheap_100 = sum(1 for p in prices if p < 1)
            low_mid = sum(1 for p in prices if 1 <= p < 2)
            mid = sum(1 for p in prices if 2 <= p < 5)
            expensive_5 = sum(1 for p in prices if p >= 5)
            expensive_10 = sum(1 for p in prices if p >= 10)
            expensive_20 = sum(1 for p in prices if p >= 20)

            value_5 = sum(p for p in prices if p >= 5)
            value_10 = sum(p for p in prices if p >= 10)
            value_20 = sum(p for p in prices if p >= 20)

            n_departments = len(set(departments))
            dept_counts = pd.Series(departments).value_counts()

            top_dept_share = dept_counts.max() / n_items
            top2_dept_share = dept_counts.head(2).sum() / n_items
            top3_dept_share = dept_counts.head(3).sum() / n_items

            dept_revisits = n_items - n_departments
            dept_revisit_ratio = safe_divide(dept_revisits, n_items)
            dept_entropy = entropy(departments)

            dept_share_features = {}
            for dept_id in range(1, 19):
                dept_share_features[f"dept_{dept_id}_share"] = departments.count(dept_id) / n_items

            high_risk_depts = {2, 5, 11, 12, 16, 17}
            high_risk_items = 0
            high_risk_value = 0

            for dept, price in zip(departments, prices):
                if dept in high_risk_depts:
                    high_risk_items += 1
                    high_risk_value += price

            high_risk_dept_ratio = safe_divide(high_risk_items, n_items)
            high_risk_dept_value_share = safe_divide(high_risk_value, total_spend)

            distances = []

            for i in range(1, len(departments)):
                dist = manhattan_distance(departments[i - 1], departments[i])
                distances.append(dist)

            total_distance = sum(distances)
            avg_distance = np.mean(distances) if distances else 0
            max_distance = max(distances) if distances else 0
            distance_std = np.std(distances) if len(distances) > 1 else 0

            zero_moves = sum(1 for d in distances if d == 0)
            small_jumps = sum(1 for d in distances if d == 1)
            medium_jumps = sum(1 for d in distances if 2 <= d < 3)
            large_jumps = sum(1 for d in distances if d >= 3)
            very_large_jumps = sum(1 for d in distances if d >= 5)

            jump_ratio = safe_divide(large_jumps, len(distances))
            very_large_jump_ratio = safe_divide(very_large_jumps, len(distances))
            zero_move_ratio = safe_divide(zero_moves, len(distances))

            n_scans = n_items
            total_amount = total_spend
            avg_amount = safe_divide(total_amount, n_scans)

            unique_depts = n_departments
            dept_revisits_eda = n_scans - unique_depts

            total_dist_eda = total_distance
            avg_dist_eda = avg_distance
            max_dist_eda = max_distance
            std_dist_eda = distance_std

            n_large_jumps_eda = sum(1 for d in distances if d > 3)
            n_zero_moves_eda = zero_moves
            jump_ratio_eda = safe_divide(n_large_jumps_eda, len(distances))

            total_time = sum(times)
            avg_time = np.mean(times) if times else 0
            median_time = np.median(times) if times else 0
            max_time_gap = max(times) if times else 0
            time_std = np.std(times) if len(times) > 1 else 0

            fast_scan_ratio = safe_divide(sum(1 for t in times if t <= 5), len(times))
            long_gap_ratio = safe_divide(sum(1 for t in times if t >= 120), len(times))
            very_long_gap_ratio = safe_divide(sum(1 for t in times if t >= 300), len(times))

            seconds_per_step_values = []

            for i in range(1, len(departments)):
                dist = manhattan_distance(departments[i - 1], departments[i])

                if dist > 0:
                    seconds_per_step_values.append(safe_divide(times[i], dist))

            if seconds_per_step_values:
                min_seconds_per_step = min(seconds_per_step_values)
                avg_seconds_per_step = np.mean(seconds_per_step_values)
                median_seconds_per_step = np.median(seconds_per_step_values)
                max_seconds_per_step = max(seconds_per_step_values)
                std_seconds_per_step = np.std(seconds_per_step_values)

                fast_movement_count = sum(1 for s in seconds_per_step_values if s <= 23)
                very_fast_movement_count = sum(1 for s in seconds_per_step_values if s <= 10)

                fast_movement_ratio = safe_divide(fast_movement_count, len(seconds_per_step_values))
                very_fast_movement_ratio = safe_divide(very_fast_movement_count, len(seconds_per_step_values))
            else:
                min_seconds_per_step = 0
                avg_seconds_per_step = 0
                median_seconds_per_step = 0
                max_seconds_per_step = 0
                std_seconds_per_step = 0
                fast_movement_count = 0
                very_fast_movement_count = 0
                fast_movement_ratio = 0
                very_fast_movement_ratio = 0

            dist_per_sec = safe_divide(total_dist_eda, total_time)
            amt_per_sec = safe_divide(total_amount, total_time)

            fast_scans = sum(1 for t in times if t < 10)
            fast_scan_ratio_eda = safe_divide(fast_scans, n_scans)

            long_dwells = sum(1 for t in times if t > 600)

            low_amount_scans = sum(1 for p in prices if p < 0.50)
            low_amount_ratio = safe_divide(low_amount_scans, n_scans)

            zero_amounts = sum(1 for p in prices if p == 0)
            high_value_scans = sum(1 for p in prices if p > 10)

            fast_high_value = sum(
                1 for t, p in zip(times, prices)
                if t < 15 and p > 8
            )

            spend_per_department = safe_divide(total_spend, n_departments)
            items_per_department = safe_divide(n_items, n_departments)
            avg_price_per_department = safe_divide(avg_price, n_departments)
            spend_per_item = safe_divide(total_spend, n_items)

            value_concentration_10 = top_value_share(prices, 0.10)
            value_concentration_20 = top_value_share(prices, 0.20)
            value_concentration_50 = top_value_share(prices, 0.50)

            movement_without_spend = safe_divide(
                total_distance + large_jumps + max_distance,
                total_spend + 1
            )

            time_without_spend = safe_divide(
                total_time + max_time_gap,
                total_spend + 1
            )

            movement_time_without_spend = safe_divide(
                total_distance + max_time_gap + large_jumps,
                total_spend + 1
            )

            items_vs_movement_risk = safe_divide(
                total_distance + large_jumps,
                n_items + 1
            )

            suspicious_low_spend_movement = safe_divide(
                total_distance + max_time_gap + large_jumps,
                total_spend + n_items + 1
            )

            suspicious_fast_distance = (
                fast_movement_ratio * safe_divide(total_distance, n_items + 1)
            )

            shrinkage_items_weighted = 0
            shrinkage_revenue_weighted = 0
            shrinkage_lost_items_weighted = 0
            shrinkage_lost_revenue_weighted = 0

            stockage_missing_count_weighted = 0
            stockage_missing_avg_value_weighted = 0
            stockage_missing_total_value_weighted = 0

            max_dept_revenue_shrinkage = 0
            max_dept_items_shrinkage = 0
            max_stockage_missing_avg_value = 0

            for dept in departments:
                shrink = SHRINKAGE_BY_DEPT.get(dept, {})
                stock = STOCKAGE_BY_DEPT.get(dept, {})

                shrinkage_items_weighted += shrink.get("items_shrinkage", 0)
                shrinkage_revenue_weighted += shrink.get("revenue_shrinkage", 0)
                shrinkage_lost_items_weighted += shrink.get("items_lost", 0)
                shrinkage_lost_revenue_weighted += shrink.get("revenue_lost", 0)

                stockage_missing_count_weighted += stock.get("stockage_missing_count", 0)
                stockage_missing_avg_value_weighted += stock.get("stockage_missing_avg_value", 0)
                stockage_missing_total_value_weighted += stock.get("stockage_missing_total_value", 0)

                max_dept_revenue_shrinkage = max(
                    max_dept_revenue_shrinkage,
                    shrink.get("revenue_shrinkage", 0)
                )

                max_dept_items_shrinkage = max(
                    max_dept_items_shrinkage,
                    shrink.get("items_shrinkage", 0)
                )

                max_stockage_missing_avg_value = max(
                    max_stockage_missing_avg_value,
                    stock.get("stockage_missing_avg_value", 0)
                )

            shrinkage_items_weighted = safe_divide(shrinkage_items_weighted, n_items)
            shrinkage_revenue_weighted = safe_divide(shrinkage_revenue_weighted, n_items)
            shrinkage_lost_items_weighted = safe_divide(shrinkage_lost_items_weighted, n_items)
            shrinkage_lost_revenue_weighted = safe_divide(shrinkage_lost_revenue_weighted, n_items)

            stockage_missing_count_weighted = safe_divide(stockage_missing_count_weighted, n_items)
            stockage_missing_avg_value_weighted = safe_divide(stockage_missing_avg_value_weighted, n_items)
            stockage_missing_total_value_weighted = safe_divide(stockage_missing_total_value_weighted, n_items)

            shrinkage_value_interaction = shrinkage_revenue_weighted * total_spend
            stockage_value_interaction = stockage_missing_avg_value_weighted * total_spend

            row = {
                "transaction_id": transaction_id,

                "n_items": n_items,
                "total_spend": total_spend,
                "avg_price": avg_price,
                "median_price": median_price,
                "max_price": max_price,
                "min_price": min_price,
                "q10_price": q10_price,
                "q25_price": q25_price,
                "q75_price": q75_price,
                "q90_price": q90_price,
                "iqr_price": iqr_price,
                "price_std": price_std,
                "price_range": price_range,
                "price_cv": price_cv,

                "cheap_050_ratio": cheap_050 / n_items,
                "cheap_100_ratio": cheap_100 / n_items,
                "low_mid_ratio": low_mid / n_items,
                "mid_ratio": mid / n_items,
                "expensive_5_ratio": expensive_5 / n_items,
                "expensive_10_ratio": expensive_10 / n_items,
                "expensive_20_ratio": expensive_20 / n_items,

                "expensive_5_value_share": safe_divide(value_5, total_spend),
                "expensive_10_value_share": safe_divide(value_10, total_spend),
                "expensive_20_value_share": safe_divide(value_20, total_spend),

                "n_departments": n_departments,
                "dept_revisits": dept_revisits,
                "dept_revisit_ratio": dept_revisit_ratio,
                "top_dept_share": top_dept_share,
                "top2_dept_share": top2_dept_share,
                "top3_dept_share": top3_dept_share,
                "dept_entropy": dept_entropy,

                "high_risk_dept_ratio": high_risk_dept_ratio,
                "high_risk_dept_value_share": high_risk_dept_value_share,

                "spend_per_department": spend_per_department,
                "items_per_department": items_per_department,
                "avg_price_per_department": avg_price_per_department,
                "spend_per_item": spend_per_item,

                "value_concentration_10": value_concentration_10,
                "value_concentration_20": value_concentration_20,
                "value_concentration_50": value_concentration_50,

                "log_total_spend": np.log1p(total_spend),
                "log_n_items": np.log1p(n_items),
                "log_max_price": np.log1p(max_price),
                "log_price_std": np.log1p(price_std),

                "has_time": int(has_time),
                "total_time": total_time,
                "avg_time": avg_time,
                "median_time": median_time,
                "max_time_gap": max_time_gap,
                "time_std": time_std,
                "fast_scan_ratio": fast_scan_ratio,
                "long_gap_ratio": long_gap_ratio,
                "very_long_gap_ratio": very_long_gap_ratio,

                "total_distance": total_distance,
                "avg_distance": avg_distance,
                "max_distance": max_distance,
                "distance_std": distance_std,
                "zero_moves": zero_moves,
                "small_jumps": small_jumps,
                "medium_jumps": medium_jumps,
                "large_jumps": large_jumps,
                "very_large_jumps": very_large_jumps,
                "jump_ratio": jump_ratio,
                "very_large_jump_ratio": very_large_jump_ratio,
                "zero_move_ratio": zero_move_ratio,

                "min_seconds_per_step": min_seconds_per_step,
                "avg_seconds_per_step": avg_seconds_per_step,
                "median_seconds_per_step": median_seconds_per_step,
                "max_seconds_per_step": max_seconds_per_step,
                "std_seconds_per_step": std_seconds_per_step,
                "fast_movement_count": fast_movement_count,
                "very_fast_movement_count": very_fast_movement_count,
                "fast_movement_ratio": fast_movement_ratio,
                "very_fast_movement_ratio": very_fast_movement_ratio,

                "movement_without_spend": movement_without_spend,
                "time_without_spend": time_without_spend,
                "movement_time_without_spend": movement_time_without_spend,
                "items_vs_movement_risk": items_vs_movement_risk,
                "suspicious_low_spend_movement": suspicious_low_spend_movement,
                "suspicious_fast_distance": suspicious_fast_distance,

                "n_scans": n_scans,
                "total_amount": total_amount,
                "avg_amount": avg_amount,
                "unique_depts": unique_depts,
                "dept_revisits_eda": dept_revisits_eda,
                "total_dist_eda": total_dist_eda,
                "avg_dist_eda": avg_dist_eda,
                "max_dist_eda": max_dist_eda,
                "std_dist_eda": std_dist_eda,
                "n_large_jumps_eda": n_large_jumps_eda,
                "n_zero_moves_eda": n_zero_moves_eda,
                "jump_ratio_eda": jump_ratio_eda,
                "dist_per_sec": dist_per_sec,
                "amt_per_sec": amt_per_sec,
                "fast_scans": fast_scans,
                "fast_scan_ratio_eda": fast_scan_ratio_eda,
                "long_dwells": long_dwells,
                "low_amount_ratio": low_amount_ratio,
                "zero_amounts": zero_amounts,
                "high_value_scans": high_value_scans,
                "fast_high_value": fast_high_value,

                "shrinkage_items_weighted": shrinkage_items_weighted,
                "shrinkage_revenue_weighted": shrinkage_revenue_weighted,
                "shrinkage_lost_items_weighted": shrinkage_lost_items_weighted,
                "shrinkage_lost_revenue_weighted": shrinkage_lost_revenue_weighted,
                "stockage_missing_count_weighted": stockage_missing_count_weighted,
                "stockage_missing_avg_value_weighted": stockage_missing_avg_value_weighted,
                "stockage_missing_total_value_weighted": stockage_missing_total_value_weighted,
                "max_dept_revenue_shrinkage": max_dept_revenue_shrinkage,
                "max_dept_items_shrinkage": max_dept_items_shrinkage,
                "max_stockage_missing_avg_value": max_stockage_missing_avg_value,
                "shrinkage_value_interaction": shrinkage_value_interaction,
                "stockage_value_interaction": stockage_value_interaction,
            }

            row.update(dept_share_features)
            transactions.append(row)

    return clean_dataframe(pd.DataFrame(transactions))


# 6. Distance anomaly score
def add_distance_anomaly_score(df):
    df = df.copy()

    distance_features = [
        "total_dist_eda",
        "avg_dist_eda",
        "max_dist_eda",
        "std_dist_eda",
        "dept_revisits_eda",
        "jump_ratio_eda",
    ]

    z_parts = []

    for col in distance_features:
        mean = df[col].mean()
        std = df[col].std()

        if std == 0 or pd.isna(std):
            z_parts.append(pd.Series(0, index=df.index))
        else:
            z_parts.append((df[col] - mean) / std)

    z_df = pd.concat(z_parts, axis=1)

    df["distance_anomaly_score"] = z_df.mean(axis=1)
    df["distance_anomaly_score"] = minmax(df["distance_anomaly_score"])

    threshold_95 = df["distance_anomaly_score"].quantile(0.95)
    df["distance_anomaly_flag"] = (
        df["distance_anomaly_score"] >= threshold_95
    ).astype(int)

    return df


# 7. Load data
cashier_df = parse_transaction_file(cashier_file)
case_df = parse_transaction_file(case_file)

cashier_df = add_distance_anomaly_score(cashier_df)
case_df = add_distance_anomaly_score(case_df)

print("Cashier transactions loaded:", len(cashier_df))
print("Case transactions loaded:", len(case_df))


# 8. Feature sets
shared_features = [
    "n_items",
    "total_spend",
    "avg_price",
    "median_price",
    "max_price",
    "min_price",
    "q10_price",
    "q25_price",
    "q75_price",
    "q90_price",
    "iqr_price",
    "price_std",
    "price_range",
    "price_cv",

    "cheap_050_ratio",
    "cheap_100_ratio",
    "low_mid_ratio",
    "mid_ratio",
    "expensive_5_ratio",
    "expensive_10_ratio",
    "expensive_20_ratio",

    "expensive_5_value_share",
    "expensive_10_value_share",
    "expensive_20_value_share",

    "n_departments",
    "dept_revisits",
    "dept_revisit_ratio",
    "top_dept_share",
    "top2_dept_share",
    "top3_dept_share",
    "dept_entropy",

    "high_risk_dept_ratio",
    "high_risk_dept_value_share",

    "spend_per_department",
    "items_per_department",
    "avg_price_per_department",
    "spend_per_item",

    "value_concentration_10",
    "value_concentration_20",
    "value_concentration_50",

    "log_total_spend",
    "log_n_items",
    "log_max_price",
    "log_price_std",

    "n_scans",
    "total_amount",
    "avg_amount",
    "unique_depts",
    "dept_revisits_eda",
    "low_amount_ratio",
    "zero_amounts",
    "high_value_scans",

    "distance_anomaly_score",
    "distance_anomaly_flag",

    "shrinkage_items_weighted",
    "shrinkage_revenue_weighted",
    "shrinkage_lost_items_weighted",
    "shrinkage_lost_revenue_weighted",
    "stockage_missing_count_weighted",
    "stockage_missing_avg_value_weighted",
    "stockage_missing_total_value_weighted",
    "max_dept_revenue_shrinkage",
    "max_dept_items_shrinkage",
    "max_stockage_missing_avg_value",
    "shrinkage_value_interaction",
    "stockage_value_interaction",
]

shared_features += [f"dept_{dept_id}_share" for dept_id in range(1, 19)]

case_full_features = shared_features + [
    "total_time",
    "avg_time",
    "median_time",
    "max_time_gap",
    "time_std",
    "fast_scan_ratio",
    "long_gap_ratio",
    "very_long_gap_ratio",

    "total_distance",
    "avg_distance",
    "max_distance",
    "distance_std",
    "zero_moves",
    "small_jumps",
    "medium_jumps",
    "large_jumps",
    "very_large_jumps",
    "jump_ratio",
    "very_large_jump_ratio",
    "zero_move_ratio",

    "min_seconds_per_step",
    "avg_seconds_per_step",
    "median_seconds_per_step",
    "max_seconds_per_step",
    "std_seconds_per_step",
    "fast_movement_count",
    "very_fast_movement_count",
    "fast_movement_ratio",
    "very_fast_movement_ratio",

    "movement_without_spend",
    "time_without_spend",
    "movement_time_without_spend",
    "items_vs_movement_risk",
    "suspicious_low_spend_movement",
    "suspicious_fast_distance",

    "total_dist_eda",
    "avg_dist_eda",
    "max_dist_eda",
    "std_dist_eda",
    "n_large_jumps_eda",
    "n_zero_moves_eda",
    "jump_ratio_eda",
    "dist_per_sec",
    "amt_per_sec",
    "fast_scans",
    "fast_scan_ratio_eda",
    "long_dwells",
    "fast_high_value",
]


# 9. Scale data
cashier_shared = cashier_df[shared_features]
case_shared = case_df[shared_features]
case_full = case_df[case_full_features]

shared_scaler = StandardScaler()
cashier_shared_scaled = shared_scaler.fit_transform(cashier_shared)
case_shared_scaled = shared_scaler.transform(case_shared)

case_scaler = StandardScaler()
case_full_scaled = case_scaler.fit_transform(case_full)


# 10. Cashier-trained Isolation Forest ensemble
if_settings = []

for max_features in [0.50, 0.60, 0.75]:
    for contamination in ["auto", 0.10, 0.15]:
        for random_state in [42, 7, 21]:
            name = (
                f"cashier_if_mf{str(max_features).replace('.', '')}"
                f"_c{str(contamination).replace('.', '').replace('auto', 'auto')}"
                f"_rs{random_state}"
            )

            if_settings.append({
                "name": name,
                "max_features": max_features,
                "contamination": contamination,
                "random_state": random_state,
            })

for setting in if_settings:
    model = IsolationForest(
        n_estimators=300,
        contamination=setting["contamination"],
        max_features=setting["max_features"],
        random_state=setting["random_state"],
        n_jobs=-1,
    )

    model.fit(cashier_shared_scaled)

    raw_score = -model.decision_function(case_shared_scaled)
    case_df[setting["name"]] = minmax(raw_score)

cashier_if_cols = [s["name"] for s in if_settings]

case_df["cashier_if_score"] = case_df[cashier_if_cols].mean(axis=1)
case_df["cashier_if_max_score"] = case_df[cashier_if_cols].max(axis=1)
case_df["cashier_if_median_score"] = case_df[cashier_if_cols].median(axis=1)



# 11. Case-specific Isolation Forest ensemble

case_if_settings = []

for max_features in [0.50, 0.60, 0.75]:
    for contamination in ["auto", 0.10, 0.15]:
        for random_state in [42, 7, 21]:
            name = (
                f"case_if_mf{str(max_features).replace('.', '')}"
                f"_c{str(contamination).replace('.', '').replace('auto', 'auto')}"
                f"_rs{random_state}"
            )

            case_if_settings.append({
                "name": name,
                "max_features": max_features,
                "contamination": contamination,
                "random_state": random_state,
            })

for setting in case_if_settings:
    model = IsolationForest(
        n_estimators=300,
        contamination=setting["contamination"],
        max_features=setting["max_features"],
        random_state=setting["random_state"],
        n_jobs=-1,
    )

    model.fit(case_full_scaled)

    raw_score = -model.decision_function(case_full_scaled)
    case_df[setting["name"]] = minmax(raw_score)

case_if_cols = [s["name"] for s in case_if_settings]

case_df["case_if_score"] = case_df[case_if_cols].mean(axis=1)
case_df["case_if_max_score"] = case_df[case_if_cols].max(axis=1)
case_df["case_if_median_score"] = case_df[case_if_cols].median(axis=1)


# 12. Supporting models: LOF, PCA, K-Means
lof_cashier = LocalOutlierFactor(
    n_neighbors=20,
    contamination="auto",
    novelty=True,
)

lof_cashier.fit(cashier_shared_scaled)
lof_cashier_raw = -lof_cashier.decision_function(case_shared_scaled)
case_df["cashier_lof_score"] = minmax(lof_cashier_raw)


lof_case = LocalOutlierFactor(
    n_neighbors=20,
    contamination="auto",
)

lof_case.fit(case_full_scaled)
case_df["case_lof_score"] = minmax(-lof_case.negative_outlier_factor_)


pca_cashier = PCA(n_components=0.90)
pca_cashier.fit(cashier_shared_scaled)

case_pca_cashier = pca_cashier.transform(case_shared_scaled)
case_recon_cashier = pca_cashier.inverse_transform(case_pca_cashier)

pca_cashier_error = np.mean((case_shared_scaled - case_recon_cashier) ** 2, axis=1)
case_df["cashier_pca_score"] = minmax(pca_cashier_error)


pca_case = PCA(n_components=0.90)
case_pca = pca_case.fit_transform(case_full_scaled)
case_recon = pca_case.inverse_transform(case_pca)

pca_case_error = np.mean((case_full_scaled - case_recon) ** 2, axis=1)
case_df["case_pca_score"] = minmax(pca_case_error)


# 13. K-Means clustering 
best_k = None
best_score = -1
best_kmeans = None

for k in range(3, 11):
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        random_state=42,
    )

    labels = kmeans.fit_predict(cashier_shared_scaled)
    score = silhouette_score(cashier_shared_scaled, labels)

    if score > best_score:
        best_score = score
        best_k = k
        best_kmeans = kmeans

cashier_kmeans = best_kmeans
cashier_clusters = cashier_kmeans.predict(case_shared_scaled)

cashier_centers = cashier_kmeans.cluster_centers_
assigned_cashier_centers = cashier_centers[cashier_clusters]

cashier_cluster_distances = np.linalg.norm(
    case_shared_scaled - assigned_cashier_centers,
    axis=1,
)

case_df["cashier_cluster"] = cashier_clusters
case_df["cashier_cluster_score"] = minmax(cashier_cluster_distances)


best_case_k = None
best_case_score = -1
best_case_kmeans = None

for k in range(3, 11):
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        random_state=42,
    )

    labels = kmeans.fit_predict(case_full_scaled)
    score = silhouette_score(case_full_scaled, labels)

    if score > best_case_score:
        best_case_score = score
        best_case_k = k
        best_case_kmeans = kmeans

case_kmeans = best_case_kmeans
case_clusters = case_kmeans.predict(case_full_scaled)

case_centers = case_kmeans.cluster_centers_
assigned_case_centers = case_centers[case_clusters]

case_cluster_distances = np.linalg.norm(
    case_full_scaled - assigned_case_centers,
    axis=1,
)

case_df["case_cluster"] = case_clusters
case_df["case_cluster_score"] = minmax(case_cluster_distances)


# 14. Base risk scores before cluster profiling
case_df["value_score"] = minmax(case_df["total_spend"])

case_df["if_combined_score"] = (
    0.70 * case_df["cashier_if_score"] +
    0.20 * case_df["case_if_score"] +
    0.05 * case_df["cashier_if_median_score"] +
    0.05 * case_df["case_if_median_score"]
)

case_df["supporting_model_score"] = (
    0.20 * case_df["cashier_lof_score"] +
    0.15 * case_df["case_lof_score"] +
    0.20 * case_df["cashier_pca_score"] +
    0.15 * case_df["case_pca_score"] +
    0.15 * case_df["cashier_cluster_score"] +
    0.15 * case_df["case_cluster_score"]
)

case_df["value_risk_score"] = minmax(
    case_df["if_combined_score"] * np.sqrt(case_df["value_score"] + 1e-9)
)

case_df["expensive_basket_score"] = minmax(
    case_df["max_price"] +
    case_df["price_std"] +
    case_df["price_range"] +
    case_df["expensive_5_value_share"] +
    case_df["expensive_10_value_share"] +
    case_df["expensive_20_value_share"] +
    case_df["value_concentration_10"] +
    case_df["value_concentration_20"]
)

case_df["high_risk_department_score"] = minmax(
    case_df["high_risk_dept_ratio"] +
    case_df["high_risk_dept_value_share"]
)

case_df["eda_distance_risk_score"] = minmax(
    case_df["distance_anomaly_score"] +
    case_df["jump_ratio_eda"] +
    case_df["dist_per_sec"] +
    case_df["n_large_jumps_eda"]
)

case_df["scanner_movement_score"] = minmax(
    case_df["total_distance"] +
    case_df["large_jumps"] +
    case_df["very_large_jumps"] +
    case_df["max_distance"] +
    case_df["jump_ratio"] +
    case_df["very_large_jump_ratio"] +
    case_df["eda_distance_risk_score"]
)

case_df["scanner_time_score"] = minmax(
    case_df["max_time_gap"] +
    case_df["time_std"] +
    case_df["long_gap_ratio"] +
    case_df["very_long_gap_ratio"] +
    case_df["long_dwells"]
)

case_df["missing_items_score"] = minmax(
    case_df["movement_without_spend"] +
    case_df["time_without_spend"] +
    case_df["movement_time_without_spend"] +
    case_df["items_vs_movement_risk"] +
    case_df["suspicious_low_spend_movement"] +
    case_df["low_amount_ratio"]
)

case_df["shrinkage_stockage_score"] = minmax(
    case_df["shrinkage_revenue_weighted"] +
    case_df["stockage_missing_avg_value_weighted"] +
    case_df["max_dept_revenue_shrinkage"] +
    case_df["max_stockage_missing_avg_value"] +
    case_df["shrinkage_value_interaction"] +
    case_df["stockage_value_interaction"]
)



# 15. cluster profiling
def add_cluster_profile_scores(df):
    df = df.copy()

    profile_cols = [
        "total_spend",
        "n_items",
        "value_risk_score",
        "missing_items_score",
        "scanner_movement_score",
        "scanner_time_score",
        "eda_distance_risk_score",
        "shrinkage_stockage_score",
        "high_risk_department_score",
    ]

    case_profiles = (
        df.groupby("case_cluster")[profile_cols]
        .mean()
        .add_prefix("case_cluster_avg_")
    )

    case_sizes = df.groupby("case_cluster").size().rename("case_cluster_size")
    case_profiles = case_profiles.join(case_sizes)

    df = df.merge(
        case_profiles,
        left_on="case_cluster",
        right_index=True,
        how="left"
    )
    cashier_profiles = (
        df.groupby("cashier_cluster")[profile_cols]
        .mean()
        .add_prefix("cashier_cluster_avg_")
    )

    cashier_sizes = df.groupby("cashier_cluster").size().rename("cashier_cluster_size")
    cashier_profiles = cashier_profiles.join(cashier_sizes)

    df = df.merge(
        cashier_profiles,
        left_on="cashier_cluster",
        right_index=True,
        how="left"
    )

    df["case_small_cluster_score"] = minmax(1 / (df["case_cluster_size"] + 1))
    df["cashier_small_cluster_score"] = minmax(1 / (df["cashier_cluster_size"] + 1))

    df["case_cluster_profile_risk"] = minmax(
        df["case_cluster_avg_value_risk_score"] +
        df["case_cluster_avg_missing_items_score"] +
        df["case_cluster_avg_scanner_movement_score"] +
        df["case_cluster_avg_eda_distance_risk_score"] +
        df["case_cluster_avg_shrinkage_stockage_score"] +
        df["case_small_cluster_score"]
    )

    df["cashier_cluster_profile_risk"] = minmax(
        df["cashier_cluster_avg_value_risk_score"] +
        df["cashier_cluster_avg_missing_items_score"] +
        df["cashier_cluster_avg_scanner_movement_score"] +
        df["cashier_cluster_avg_eda_distance_risk_score"] +
        df["cashier_cluster_avg_shrinkage_stockage_score"] +
        df["cashier_small_cluster_score"]
    )

    df["cluster_profile_risk_score"] = minmax(
        0.45 * df["case_cluster_score"] +
        0.25 * df["cashier_cluster_score"] +
        0.15 * df["case_cluster_profile_risk"] +
        0.15 * df["cashier_cluster_profile_risk"]
    )

    return df


case_df = add_cluster_profile_scores(case_df)


# 16. Final score versions

case_df["score_if_strong_count"] = (
    0.70 * case_df["if_combined_score"] +
    0.09 * case_df["supporting_model_score"] +
    0.06 * case_df["missing_items_score"] +
    0.05 * case_df["value_risk_score"] +
    0.04 * case_df["eda_distance_risk_score"] +
    0.03 * case_df["shrinkage_stockage_score"] +
    0.03 * case_df["cluster_profile_risk_score"]
)

case_df["score_if_stable_eda_cluster"] = (
    0.62 * case_df["if_combined_score"] +
    0.11 * case_df["supporting_model_score"] +
    0.07 * case_df["value_risk_score"] +
    0.06 * case_df["eda_distance_risk_score"] +
    0.05 * case_df["missing_items_score"] +
    0.04 * case_df["shrinkage_stockage_score"] +
    0.05 * case_df["cluster_profile_risk_score"]
)

case_df["score_cluster_focus"] = (
    0.50 * case_df["if_combined_score"] +
    0.12 * case_df["supporting_model_score"] +
    0.13 * case_df["cluster_profile_risk_score"] +
    0.10 * case_df["eda_distance_risk_score"] +
    0.07 * case_df["missing_items_score"] +
    0.05 * case_df["value_risk_score"] +
    0.03 * case_df["shrinkage_stockage_score"]
)

case_df["score_eda_distance_focus"] = (
    0.53 * case_df["if_combined_score"] +
    0.11 * case_df["supporting_model_score"] +
    0.12 * case_df["eda_distance_risk_score"] +
    0.08 * case_df["scanner_movement_score"] +
    0.07 * case_df["missing_items_score"] +
    0.05 * case_df["value_risk_score"] +
    0.04 * case_df["cluster_profile_risk_score"]
)

case_df["score_shrinkage_value_focus"] = (
    0.48 * case_df["if_combined_score"] +
    0.11 * case_df["supporting_model_score"] +
    0.12 * case_df["shrinkage_stockage_score"] +
    0.10 * case_df["value_risk_score"] +
    0.09 * case_df["expensive_basket_score"] +
    0.05 * case_df["high_risk_department_score"] +
    0.05 * case_df["cluster_profile_risk_score"]
)

case_df["score_money_eda_cluster"] = (
    0.48 * case_df["if_combined_score"] +
    0.10 * case_df["supporting_model_score"] +
    0.14 * case_df["value_risk_score"] +
    0.10 * case_df["expensive_basket_score"] +
    0.07 * case_df["shrinkage_stockage_score"] +
    0.06 * case_df["eda_distance_risk_score"] +
    0.05 * case_df["cluster_profile_risk_score"]
)

case_df["score_scanner_missing_focus"] = (
    0.48 * case_df["if_combined_score"] +
    0.10 * case_df["supporting_model_score"] +
    0.13 * case_df["missing_items_score"] +
    0.09 * case_df["scanner_movement_score"] +
    0.08 * case_df["scanner_time_score"] +
    0.05 * case_df["eda_distance_risk_score"] +
    0.04 * case_df["cluster_profile_risk_score"] +
    0.03 * case_df["value_risk_score"]
)


# 17. Auto score versions
base_cols = [
    "if_combined_score",
    "supporting_model_score",
    "value_risk_score",
    "expensive_basket_score",
    "missing_items_score",
    "eda_distance_risk_score",
    "shrinkage_stockage_score",
    "cluster_profile_risk_score",
    "scanner_movement_score",
    "scanner_time_score",
    "high_risk_department_score",
]

weight_sets = [
    (0.72, 0.10, 0.05, 0.00, 0.04, 0.03, 0.02, 0.04, 0.00, 0.00, 0.00),
    (0.68, 0.10, 0.06, 0.00, 0.05, 0.04, 0.02, 0.05, 0.00, 0.00, 0.00),
    (0.64, 0.11, 0.07, 0.02, 0.05, 0.04, 0.03, 0.04, 0.00, 0.00, 0.00),
    (0.60, 0.11, 0.08, 0.04, 0.05, 0.04, 0.03, 0.05, 0.00, 0.00, 0.00),
    (0.55, 0.11, 0.09, 0.05, 0.05, 0.05, 0.04, 0.06, 0.00, 0.00, 0.00),
    (0.50, 0.10, 0.10, 0.07, 0.05, 0.05, 0.05, 0.06, 0.00, 0.00, 0.02),
    (0.48, 0.10, 0.08, 0.07, 0.08, 0.06, 0.04, 0.07, 0.02, 0.00, 0.00),
    (0.45, 0.10, 0.08, 0.06, 0.09, 0.07, 0.04, 0.08, 0.03, 0.00, 0.00),
    (0.42, 0.10, 0.10, 0.08, 0.08, 0.06, 0.06, 0.07, 0.01, 0.00, 0.02),
    (0.38, 0.10, 0.10, 0.10, 0.08, 0.06, 0.07, 0.07, 0.02, 0.00, 0.02),
]

auto_score_cols = []

for i, weights in enumerate(weight_sets, start=1):
    score_col = f"auto_score_{i:02d}"
    case_df[score_col] = 0

    for col, weight in zip(base_cols, weights):
        case_df[score_col] += weight * case_df[col]

    auto_score_cols.append(score_col)


# 18. Choose main score
score_to_use = "score_if_strong_count"

case_df["final_score"] = case_df[score_to_use]

case_sorted = case_df.sort_values("final_score", ascending=False).copy()
case_sorted["rank"] = range(1, len(case_sorted) + 1)


# 19. Save outputs
os.makedirs("output", exist_ok=True)
case_name = "output/" + os.path.basename(case_file).replace(".csv", "").replace(" ", "_").replace("(", "").replace(")", "")

case_sorted.to_csv(
    f"{case_name}_eda_cluster_connected_all_ranked.csv",
    index=False
)

case_sorted.head(150)["transaction_id"].to_csv(
    f"{case_name}_eda_cluster_connected_top150_main.txt",
    index=False,
    header=False
)

score_versions = [
    "score_if_strong_count",
    "score_if_stable_eda_cluster",
    "score_cluster_focus",
    "score_eda_distance_focus",
    "score_shrinkage_value_focus",
    "score_money_eda_cluster",
    "score_scanner_missing_focus",
] + auto_score_cols

for score_col in score_versions:
    temp = case_df.sort_values(score_col, ascending=False).copy()

    temp.head(150)["transaction_id"].to_csv(
        f"{case_name}_top150_{score_col}.txt",
        index=False,
        header=False
    )
