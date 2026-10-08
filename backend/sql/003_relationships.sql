-- Preserve raw CSV rows. Normalized identity tables allow foreign keys even
-- when order lines repeat and some referenced SKUs are absent from JD_sku_data.
CREATE TABLE IF NOT EXISTS msom.user_keys (
    user_id text PRIMARY KEY,
    has_user_record boolean NOT NULL DEFAULT false
);
CREATE TABLE IF NOT EXISTS msom.sku_keys (
    sku_id text PRIMARY KEY,
    has_sku_record boolean NOT NULL DEFAULT false
);
CREATE TABLE IF NOT EXISTS msom.order_keys (
    order_id text PRIMARY KEY,
    has_order_record boolean NOT NULL DEFAULT false
);

INSERT INTO msom.user_keys (user_id)
SELECT user_id FROM msom.users UNION SELECT user_id FROM msom.orders
ON CONFLICT DO NOTHING;
UPDATE msom.user_keys k SET has_user_record = true
WHERE EXISTS (SELECT 1 FROM msom.users u WHERE u.user_id = k.user_id)
  AND NOT k.has_user_record;

INSERT INTO msom.sku_keys (sku_id)
SELECT sku_id FROM msom.skus UNION SELECT sku_id FROM msom.orders
UNION SELECT sku_id FROM msom.inventory
ON CONFLICT DO NOTHING;
UPDATE msom.sku_keys k SET has_sku_record = true
WHERE EXISTS (SELECT 1 FROM msom.skus s WHERE s.sku_id = k.sku_id)
  AND NOT k.has_sku_record;

INSERT INTO msom.order_keys (order_id)
SELECT order_id FROM msom.orders UNION SELECT order_id FROM msom.deliveries
ON CONFLICT DO NOTHING;
UPDATE msom.order_keys k SET has_order_record = true
WHERE EXISTS (SELECT 1 FROM msom.orders o WHERE o.order_id = k.order_id)
  AND NOT k.has_order_record;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_msom_orders_user_key') THEN
    ALTER TABLE msom.orders ADD CONSTRAINT fk_msom_orders_user_key
      FOREIGN KEY (user_id) REFERENCES msom.user_keys(user_id);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_msom_orders_sku_key') THEN
    ALTER TABLE msom.orders ADD CONSTRAINT fk_msom_orders_sku_key
      FOREIGN KEY (sku_id) REFERENCES msom.sku_keys(sku_id);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_msom_orders_order_key') THEN
    ALTER TABLE msom.orders ADD CONSTRAINT fk_msom_orders_order_key
      FOREIGN KEY (order_id) REFERENCES msom.order_keys(order_id);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_msom_deliveries_order_key') THEN
    ALTER TABLE msom.deliveries ADD CONSTRAINT fk_msom_deliveries_order_key
      FOREIGN KEY (order_id) REFERENCES msom.order_keys(order_id);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_msom_inventory_sku_key') THEN
    ALTER TABLE msom.inventory ADD CONSTRAINT fk_msom_inventory_sku_key
      FOREIGN KEY (sku_id) REFERENCES msom.sku_keys(sku_id);
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_msom_order_keys_source ON msom.order_keys (has_order_record);
CREATE INDEX IF NOT EXISTS idx_msom_sku_keys_source ON msom.sku_keys (has_sku_record);
