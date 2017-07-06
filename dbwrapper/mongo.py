# -*- coding: utf-8 -*-

import pymongo

class MongoClass(object):
    def __init__(self, host="localhost", port=27017, db_name=None, coll_name=None):
        self._client = pymongo.MongoClient(host=host, port=port)
        self._db = None
        self._coll = None
        if db_name is not None:
            self._db = self._client.get_database(db_name)
        if db_name is not None and coll_name is not None:
            self._coll = self._db.get_collection(coll_name)

    def get_db(self, db_name):
        return self.client.get_database(name=db_name)

    def get_collection(self, coll_name, db_name=None):
        if db_name is None:
            return self.db.get_collection(coll_name)
        return self.client.get_database(db_name).get_collection(coll_name)

    def set_db(self, db_name):
        self._db = self.client.get_database(db_name)

    def set_collection(self, coll_name):
        self._coll = self.db.get_collection(coll_name)

    @property
    def client(self):
        return self._client

    @property
    def db(self):
        return self._db

    @property
    def collection(self):
        return self._coll

    @property
    def stock_db(self):
        return self.client.get_database("cubeopen")

    @property
    def market_coll(self):
        return self.client.get_database("cubeopen").get_collection("market_daily")