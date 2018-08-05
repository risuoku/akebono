from google.cloud import bigquery


class Client:
    def __init__(self):
        self._client = bigquery.Client()
        self._client.use_legacy_sql = False
        self._client.use_query_cache = False

    @property
    def client(self):
        return self._client

    def query_sync(self, q):
        qr = self.client.query(q)
        qrresult = qr.result(timeout=3600.0)
        rows = list(qrresult)
        snames = [a.name for a in qrresult.schema]
        r = [dict(zip(snames, [row[sn] for sn in snames])) for row in rows]
        return r
