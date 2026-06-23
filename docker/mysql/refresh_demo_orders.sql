USE ecs;
SET NAMES utf8mb4 COLLATE utf8mb4_general_ci;

SET @demo_user_id := '1001';
SET @now := NOW();

DROP TEMPORARY TABLE IF EXISTS demo_postsale_orders;
CREATE TEMPORARY TABLE demo_postsale_orders AS
SELECT
    order_id,
    ROW_NUMBER() OVER (ORDER BY delivered_time DESC, create_time DESC) AS rn
FROM order_info
WHERE user_id = @demo_user_id
  AND order_status IN ('已签收', '已完成', '售后中')
ORDER BY delivered_time DESC, create_time DESC
LIMIT 2;

DELETE pl
FROM postsale_logistics pl
JOIN postsale p ON p.postsale_id = pl.postsale_id
JOIN order_detail od ON od.order_detail_id = p.order_detail_id
JOIN demo_postsale_orders d ON d.order_id = od.order_id;

DELETE p
FROM postsale p
JOIN order_detail od ON od.order_detail_id = p.order_detail_id
JOIN demo_postsale_orders d ON d.order_id = od.order_id;

UPDATE order_info oi
JOIN demo_postsale_orders d ON d.order_id = oi.order_id
SET
    oi.order_status = CASE WHEN d.rn = 1 THEN '已签收' ELSE '已完成' END,
    oi.create_time = DATE_SUB(@now, INTERVAL (d.rn + 2) DAY),
    oi.payment_time = DATE_ADD(DATE_SUB(@now, INTERVAL (d.rn + 2) DAY), INTERVAL 8 MINUTE),
    oi.delivered_time = DATE_SUB(@now, INTERVAL (d.rn * 2) DAY),
    oi.complete_time = CASE
        WHEN d.rn = 1 THEN NULL
        ELSE DATE_ADD(DATE_SUB(@now, INTERVAL (d.rn * 2) DAY), INTERVAL 2 HOUR)
    END;

SELECT
    oi.order_id,
    oi.user_id,
    oi.order_status,
    oi.create_time,
    oi.delivered_time,
    DATEDIFF(@now, oi.delivered_time) AS days_since_delivered
FROM order_info oi
JOIN demo_postsale_orders d ON d.order_id = oi.order_id
ORDER BY d.rn;

SELECT
    oi.order_id,
    oi.user_id,
    oi.order_status,
    oi.create_time,
    oi.delivered_time,
    DATEDIFF(@now, oi.delivered_time) AS days_since_delivered
FROM order_info oi
WHERE oi.user_id = @demo_user_id
  AND oi.order_status IN ('已签收', '已完成')
ORDER BY oi.delivered_time DESC
LIMIT 8;
