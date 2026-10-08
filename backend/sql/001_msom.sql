CREATE SCHEMA IF NOT EXISTS msom;

CREATE TABLE IF NOT EXISTS msom.import_batches (
    table_name text PRIMARY KEY,
    source_name text NOT NULL,
    source_sha256 text NOT NULL,
    row_count bigint NOT NULL,
    imported_at timestamptz NOT NULL DEFAULT now()
);

-- Raw text staging preserves exact CSV values. Typed projections can be added
-- after source-level validation without mutating the archived input files.
CREATE TABLE IF NOT EXISTS msom.orders (
    order_id text, user_id text, sku_id text, order_date text, order_time text,
    quantity text, type text, promise text, original_unit_price text,
    final_unit_price text, direct_discount_per_unit text,
    quantity_discount_per_unit text, bundle_discount_per_unit text,
    coupon_discount_per_unit text, gift_item text, dc_ori text, dc_des text
);
CREATE INDEX IF NOT EXISTS idx_msom_orders_id ON msom.orders (order_id);
CREATE INDEX IF NOT EXISTS idx_msom_orders_user ON msom.orders (user_id);
CREATE INDEX IF NOT EXISTS idx_msom_orders_sku ON msom.orders (sku_id);

CREATE TABLE IF NOT EXISTS msom.deliveries (
    package_id text, order_id text, type text, ship_out_time text,
    arr_station_time text, arr_time text
);
CREATE INDEX IF NOT EXISTS idx_msom_deliveries_order ON msom.deliveries (order_id);

CREATE TABLE IF NOT EXISTS msom.skus (
    sku_id text, type text, brand_id text, attribute1 text, attribute2 text,
    activate_date text, deactivate_date text
);
CREATE INDEX IF NOT EXISTS idx_msom_skus_id ON msom.skus (sku_id);

CREATE TABLE IF NOT EXISTS msom.users (
    user_id text, user_level text, first_order_month text, plus text, gender text,
    age text, marital_status text, education text, city_level text,
    purchase_power text
);
CREATE INDEX IF NOT EXISTS idx_msom_users_id ON msom.users (user_id);

CREATE TABLE IF NOT EXISTS msom.inventory (
    dc_id text, sku_id text, date text
);
CREATE INDEX IF NOT EXISTS idx_msom_inventory_sku_date ON msom.inventory (sku_id, date);

CREATE TABLE IF NOT EXISTS msom.network (
    region_id text, dc_id text
);

CREATE TABLE IF NOT EXISTS msom.clicks (
    sku_id text, user_id text, request_time text, channel text
);
CREATE INDEX IF NOT EXISTS idx_msom_clicks_user_time ON msom.clicks (user_id, request_time DESC);
