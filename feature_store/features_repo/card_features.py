from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType

uid = Entity(name="uid", value_type=ValueType.STRING, join_keys=["uid"])

source = FileSource(
    path="data/features.parquet",
    timestamp_field="event_timestamp",
)

V_COLS = ("V12 V13 V14 V15 V17 V19 V20 V23 V24 V25 V26 V27 V29 V31 V35 V36 V37 V38 V39 V40 "
          "V41 V42 V44 V45 V46 V47 V48 V51 V53 V54 V55 V56 V59 V61 V62 V64 V65 V66 V67 V68 "
          "V69 V73 V75 V76 V77 V78 V79 V80 V82 V83 V85 V86 V87 V88 V90 V96 V99 V100 V124 "
          "V126 V129 V130 V131 V136 V137 V282 V283 V285 V287 V290 V291 V294 V309 V310 "
          "V312 V313 V314 V315 V320 V321").split()

C_D_COLS = "C1 C5 C9 C13 D1 D4 D10 D15 card1 card2 card3 card5 addr1 addr2".split()

schema = (
    [Field(name="TransactionAmt", dtype=Float32)]
    + [Field(name=c, dtype=Float32) for c in C_D_COLS]
    + [Field(name=c, dtype=Float32) for c in V_COLS]
    + [Field(name=c, dtype=String) for c in
       ["ProductCD", "card4", "card6", "P_emaildomain", "M6", "DeviceType", "device_proxy"]]
    + [Field(name="account_start_day", dtype=Float32)]
    + [Field(name=c, dtype=Float32) for c in ["zscore", "zscore_uid", "zscore_productcd"]]
    + [Field(name=c, dtype=Int64) for c in
       ["proxy_uid_count", "uid_proxy_count", "proxy_txn_count", "txn_count_1h", "txn_count_24h"]]
    + [Field(name=c, dtype=Float32) for c in ["txn_amt_sum_1h", "txn_amt_sum_24h"]]
)

fraud_features = FeatureView(
    name="fraud_features",
    entities=[uid],
    ttl=timedelta(0),
    schema=schema,
    source=source,
    online=True,
)