from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType

# uid is the entity (join key), so it is not listed as a feature below.
uid = Entity(name="uid", value_type=ValueType.STRING, join_keys=["uid"])

source = FileSource(
    path="fraud_transaction.parquet",
    timestamp_field="event_timestamp",
)

schema = (
    # amount and z-scores
    [Field(name="TransactionAmt", dtype=Float32)]
    + [Field(name=c, dtype=Float32) for c in ["zscore", "zscore_uid", "zscore_productcd"]]
    # rolling velocity
    + [Field(name=c, dtype=Int64) for c in ["txn_count_1h", "txn_count_24h"]]
    + [Field(name=c, dtype=Float32) for c in ["txn_amt_sum_1h", "txn_amt_sum_24h"]]
    # date parts
    + [Field(name=c, dtype=Int64) for c in ["year", "month", "day"]]
    # device / proxy
    + [Field(name=c, dtype=String) for c in ["DeviceType", "device_proxy"]]
    + [Field(name=c, dtype=Int64) for c in ["proxy_uid_count", "uid_proxy_count", "proxy_txn_count"]]
    # account
    + [Field(name="account_start_day", dtype=Float32)]
)

fraud_features = FeatureView(
    name="fraud_features",
    entities=[uid],
    ttl=timedelta(0),
    schema=schema,
    source=source,
    online=True,
)