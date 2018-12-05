import numpy as np
import pandas as pd
import datetime
import queue
from threading import Thread

from root.nested import get_logger

from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.common import *
from ibapi.contract import Contract

class TestApp(EWrapper, EClient):

    def __init__(self):

        self.logger = get_logger()
        EClient.__init__(self, self)

    def error(self, reqId:TickerId, errorCode:int, errorString:str):
        
        self.logger.error("Error in requestId %s, error code is %s, error msg is %s", str(reqId), str(errorCode), errorString )

    # callback function for EClient.reqContractDetails
    def contractDetails(self, reqId, contractDetails):
        
        self.logger.info("contractDetails for requestId %s, contractDetails = %s", reqId, contractDetails )
        return super().contractDetails(reqId, contractDetails)

    def historicalData(self, reqId, bar):
        print ("HistoricalData. ReqId:", reqId, "BarData.", bar)
        return super().historicalData(reqId, bar)

    def historicalDataEnd(self, reqId, start, end):
        super().historicalDataEnd(reqId, start, end)
        print ("HistoricalDataEnd. ReqId:", reqId, "from", start, "to", end)


def main():

    app = TestApp()
    app.connect(host = "127.0.0.1", port = 7496, clientId = 0) # 7496 is port for TWS, 4001 for IBGateway

    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    contract.primaryExchange = "NASDAQ"

    # reqContractDetails is part of the ibapi.EClient imported code
    app.reqContractDetails(10, contract)
    # if we make a call to reqContractDetails, we must implement its callback
    # for reqContractDetails, callback is EWrapper.contractDetails
    end_datetime = datetime.datetime.today().strftime("%Y%m%d %H:%H:%S %Z")
    tickerid = 11
    duration = "1 Y"
    barSize = "1 day"
    whatToShow = "ADJUSTED_LAST"
    if whatToShow == "ADJUSTED_LAST":
        end_datetime = ""
    app.reqHistoricalData(tickerid, contract, end_datetime, duration, barSize, whatToShow, 1, 1, False, [])

    app.run()

if __name__ == "__main__":

    main()

    
