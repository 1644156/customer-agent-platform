from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import md5
from decimal import Decimal

import pymysql


DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "123321",
    "database": "ecs",
    "charset": "utf8mb4",
    "autocommit": False,
}


@dataclass(frozen=True)
class OrderSpec:
    order_id: str
    order_status: str
    receive_id: str
    sku_id: str
    sku_name: str
    sku_price: Decimal
    sku_count: int
    create_time: datetime
    payment_time: datetime | None
    delivered_time: datetime | None
    complete_time: datetime | None
    logistics_id: str | None
    logistics_tracking: str | None


def gid(prefix: str, label: str) -> str:
    return prefix + md5(label.encode("utf-8")).hexdigest()[:16]


def fmt(dt: datetime | None) -> str | None:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None


def main() -> int:
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT receive_id FROM receive_info WHERE user_id=%s ORDER BY receive_id LIMIT 2",
                ("1001",),
            )
            receive_ids = [row[0] for row in cur.fetchall()]
            if not receive_ids:
                raise RuntimeError("未找到 1001 的收货信息，无法生成订单。")

            cur.execute(
                "SELECT sku_id, sku_name, sku_price FROM sku_info WHERE sku_id IN (%s,%s,%s,%s,%s,%s) ORDER BY sku_id",
                ("sku123458", "sku123459", "sku123461", "sku123465", "sku123472", "sku123473"),
            )
            sku_rows = {row[0]: row for row in cur.fetchall()}
            needed = ["sku123458", "sku123459", "sku123461", "sku123465", "sku123472", "sku123473"]
            missing = [sku for sku in needed if sku not in sku_rows]
            if missing:
                raise RuntimeError(f"缺少 SKU: {missing}")

            now = datetime.now()
            orders: list[OrderSpec] = [
                OrderSpec(
                    order_id=gid("ord", "1001-signed-demo-1"),
                    order_status="已签收",
                    receive_id=receive_ids[0],
                    sku_id="sku123473",
                    sku_name=sku_rows["sku123473"][1],
                    sku_price=Decimal(str(sku_rows["sku123473"][2])),
                    sku_count=1,
                    create_time=now - timedelta(days=2),
                    payment_time=now - timedelta(days=2, minutes=-8),
                    delivered_time=now - timedelta(days=1),
                    complete_time=None,
                    logistics_id=gid("lgt", "1001-signed-demo-1"),
                    logistics_tracking=f"{fmt(now - timedelta(days=2, hours=1))} 已揽收\n{fmt(now - timedelta(days=1, hours=3))} 已签收",
                ),
                OrderSpec(
                    order_id=gid("ord", "1001-completed-demo-1"),
                    order_status="已完成",
                    receive_id=receive_ids[1 if len(receive_ids) > 1 else 0],
                    sku_id="sku123472",
                    sku_name=sku_rows["sku123472"][1],
                    sku_price=Decimal(str(sku_rows["sku123472"][2])),
                    sku_count=1,
                    create_time=now - timedelta(days=5),
                    payment_time=now - timedelta(days=5, minutes=-12),
                    delivered_time=now - timedelta(days=3),
                    complete_time=now - timedelta(days=2, hours=18),
                    logistics_id=gid("lgt", "1001-completed-demo-1"),
                    logistics_tracking=f"{fmt(now - timedelta(days=5, hours=1))} 已揽收\n{fmt(now - timedelta(days=3, hours=4))} 已签收",
                ),
                OrderSpec(
                    order_id=gid("ord", "1001-waitpay-demo-1"),
                    order_status="待支付",
                    receive_id=receive_ids[0],
                    sku_id="sku123458",
                    sku_name=sku_rows["sku123458"][1],
                    sku_price=Decimal(str(sku_rows["sku123458"][2])),
                    sku_count=1,
                    create_time=now - timedelta(hours=2),
                    payment_time=None,
                    delivered_time=None,
                    complete_time=None,
                    logistics_id=None,
                    logistics_tracking=None,
                ),
                OrderSpec(
                    order_id=gid("ord", "1001-waitpay-demo-2"),
                    order_status="待支付",
                    receive_id=receive_ids[1 if len(receive_ids) > 1 else 0],
                    sku_id="sku123459",
                    sku_name=sku_rows["sku123459"][1],
                    sku_price=Decimal(str(sku_rows["sku123459"][2])),
                    sku_count=2,
                    create_time=now - timedelta(days=1, hours=1),
                    payment_time=None,
                    delivered_time=None,
                    complete_time=None,
                    logistics_id=None,
                    logistics_tracking=None,
                ),
                OrderSpec(
                    order_id=gid("ord", "1001-waitship-demo-1"),
                    order_status="待发货",
                    receive_id=receive_ids[0],
                    sku_id="sku123465",
                    sku_name=sku_rows["sku123465"][1],
                    sku_price=Decimal(str(sku_rows["sku123465"][2])),
                    sku_count=1,
                    create_time=now - timedelta(days=1, hours=2),
                    payment_time=now - timedelta(days=1, hours=1, minutes=50),
                    delivered_time=None,
                    complete_time=None,
                    logistics_id=None,
                    logistics_tracking=None,
                ),
                OrderSpec(
                    order_id=gid("ord", "1001-waitship-demo-2"),
                    order_status="待发货",
                    receive_id=receive_ids[1 if len(receive_ids) > 1 else 0],
                    sku_id="sku123461",
                    sku_name=sku_rows["sku123461"][1],
                    sku_price=Decimal(str(sku_rows["sku123461"][2])),
                    sku_count=3,
                    create_time=now - timedelta(days=2, hours=2),
                    payment_time=now - timedelta(days=2, hours=1, minutes=45),
                    delivered_time=None,
                    complete_time=None,
                    logistics_id=None,
                    logistics_tracking=None,
                ),
            ]

            order_ids = [o.order_id for o in orders]
            logistics_ids = [o.logistics_id for o in orders if o.logistics_id]

            placeholders = ",".join(["%s"] * len(order_ids))

            cur.execute(
                f"""
                DELETE pl
                FROM postsale_logistics pl
                JOIN postsale p ON p.postsale_id = pl.postsale_id
                JOIN order_detail od ON od.order_detail_id = p.order_detail_id
                WHERE od.order_id IN ({placeholders})
                """,
                order_ids,
            )
            cur.execute(
                f"""
                DELETE p
                FROM postsale p
                JOIN order_detail od ON od.order_detail_id = p.order_detail_id
                WHERE od.order_id IN ({placeholders})
                """,
                order_ids,
            )
            cur.execute(
                f"DELETE FROM order_logistics WHERE order_id IN ({placeholders})",
                order_ids,
            )
            if logistics_ids:
                log_placeholders = ",".join(["%s"] * len(logistics_ids))
                cur.execute(
                    f"DELETE FROM logistics WHERE logistics_id IN ({log_placeholders})",
                    logistics_ids,
                )
            cur.execute(
                f"DELETE FROM order_detail WHERE order_id IN ({placeholders})",
                order_ids,
            )
            cur.execute(
                f"DELETE FROM order_info WHERE order_id IN ({placeholders})",
                order_ids,
            )

            for order in orders:
                cur.execute(
                    """
                    INSERT INTO order_info (
                        order_id, create_time, payment_time, delivered_time, complete_time,
                        user_id, receive_id, order_status
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        order.order_id,
                        fmt(order.create_time),
                        fmt(order.payment_time),
                        fmt(order.delivered_time),
                        fmt(order.complete_time),
                        "1001",
                        order.receive_id,
                        order.order_status,
                    ),
                )

                order_detail_id = gid("ordd", order.order_id)
                total_amount = order.sku_price * order.sku_count
                cur.execute(
                    """
                    INSERT INTO order_detail (
                        order_detail_id, order_id, sku_id, sku_name, sku_count,
                        total_amount, discount_amount, final_amount
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        order_detail_id,
                        order.order_id,
                        order.sku_id,
                        order.sku_name,
                        order.sku_count,
                        str(total_amount),
                        "0.00",
                        str(total_amount),
                    ),
                )

                if order.logistics_id:
                    cur.execute(
                        """
                        INSERT INTO logistics (
                            logistics_id, create_time, delivered_time, logistics_tracking, logistics_category
                        ) VALUES (%s,%s,%s,%s,%s)
                        """,
                        (
                            order.logistics_id,
                            fmt(order.create_time + timedelta(hours=1)),
                            fmt(order.delivered_time),
                            order.logistics_tracking,
                            None,
                        ),
                    )
                    cur.execute(
                        """
                        INSERT INTO order_logistics (order_id, logistics_id)
                        VALUES (%s, %s)
                        """,
                        (order.order_id, order.logistics_id),
                    )

            conn.commit()

            cur.execute(
                """
                SELECT order_id, order_status, create_time, delivered_time, complete_time
                FROM order_info
                WHERE user_id='1001'
                  AND order_id IN (%s,%s,%s,%s,%s,%s)
                ORDER BY create_time DESC
                """,
                order_ids,
            )
            rows = cur.fetchall()
            for row in rows:
                print("\t".join(str(col) for col in row))
        return 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
