import random
import time
import dotenv
from pocketoptionapi.stable_api import PocketOption
import logging
import os
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')

dotenv.DotEnv()

ssid = (r'42["auth",{"session":"vtftn12e6f5f5008moitsd6skl","isDemo":1,"uid":27658142,"platform":1}]') #os.getenv("SSID")
print(ssid)
api = PocketOption(ssid)

if __name__ == "__main__":
    api.connect()
    time.sleep(5)

    print(api.check_connect(), "check connect")

    print(api.get_balance())
