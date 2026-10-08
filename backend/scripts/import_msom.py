"""Idempotent, source-verified loader for the seven project-local MSOM CSVs.

Requires psql and a reachable PostgreSQL database. The source CSVs are read-only.
Existing imported tables are never overwritten silently.
"""
from __future__ import annotations

import csv
import hashlib
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "MSOM_Data_Driven_Challenge_2020"
SCHEMA = Path(__file__).resolve().parents[1] / "sql" / "001_msom.sql"
RELATIONSHIPS = Path(__file__).resolve().parents[1] / "sql" / "003_relationships.sql"
DATABASE = os.getenv("SERVEMIND_PG_DATABASE", "servemind")

TABLES = {
    "orders": ("JD_order_data.csv", "order_ID,user_ID,sku_ID,order_date,order_time,quantity,type,promise,original_unit_price,final_unit_price,direct_discount_per_unit,quantity_discount_per_unit,bundle_discount_per_unit,coupon_discount_per_unit,gift_item,dc_ori,dc_des"),
    "deliveries": ("JD_delivery_data.csv", "package_ID,order_ID,type,ship_out_time,arr_station_time,arr_time"),
    "skus": ("JD_sku_data.csv", "sku_ID,type,brand_ID,attribute1,attribute2,activate_date,deactivate_date"),
    "users": ("JD_user_data.csv", "user_ID,user_level,first_order_month,plus,gender,age,marital_status,education,city_level,purchase_power"),
    "inventory": ("JD_inventory_data.csv", "dc_ID,sku_ID,date"),
    "network": ("JD_network_data.csv", "region_ID,dc_ID"),
    "clicks": ("JD_click_data.csv", "sku_ID,user_ID,request_time,channel"),
}


def sql(query: str) -> str:
    result = subprocess.run(
        ["psql", "-X", "-v", "ON_ERROR_STOP=1", "-At", "--dbname", DATABASE, "-c", query],
        check=True, text=True, capture_output=True,
    )
    return result.stdout.strip()


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_one(table: str, filename: str, header: str) -> None:
    path = DATA / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as source:
        actual_header = next(csv.reader(source))
    if actual_header != header.split(","):
        raise ValueError(f"header mismatch: {filename}")
    digest = fingerprint(path)
    existing = sql(f"SELECT source_sha256 FROM msom.import_batches WHERE table_name = '{table}'")
    if existing:
        if existing != digest:
            raise RuntimeError(f"{table}: imported source differs; manual migration required")
        print(f"{table}: unchanged, skipped", flush=True)
        return
    prior = int(sql(f"SELECT count(*) FROM msom.{table}"))
    if prior:
        raise RuntimeError(f"{table}: non-empty without import record; refusing duplicate import")
    with path.open("rb") as source:
        subprocess.run(
            ["psql", "-X", "-v", "ON_ERROR_STOP=1", "--dbname", DATABASE,
             "-c", f"\\copy msom.{table} FROM STDIN WITH (FORMAT csv, HEADER true)"],
            stdin=source, check=True, stdout=subprocess.DEVNULL,
        )
    count = int(sql(f"SELECT count(*) FROM msom.{table}"))
    sql(f"INSERT INTO msom.import_batches(table_name, source_name, source_sha256, row_count) "
        f"VALUES ('{table}', '{filename}', '{digest}', {count})")
    print(f"{table}: imported {count:,} rows", flush=True)


def main() -> None:
    subprocess.run(["psql", "-X", "-v", "ON_ERROR_STOP=1", "--dbname", DATABASE,
                    "-f", str(SCHEMA)], check=True, stdout=subprocess.DEVNULL)
    for table, (filename, header) in TABLES.items():
        import_one(table, filename, header)
    subprocess.run(["psql", "-X", "-v", "ON_ERROR_STOP=1", "--dbname", DATABASE,
                    "-f", str(RELATIONSHIPS)], check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
