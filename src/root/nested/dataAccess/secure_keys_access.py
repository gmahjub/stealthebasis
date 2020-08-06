import gspread
from oauth2client.service_account import ServiceAccountCredentials
from root.nested import get_logger
from root.nested.SysOs.os_mux import OSMuxImpl
import os
from datetime import datetime, timedelta
import pandas as pd

""" Reference Site : 
https://www.twilio.com/blog/2017/02/an-easy-way-to-read-and-write-to-a-google-spreadsheet-in-python.html
"""


class SecureKeysAccess:
    SCOPE = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive',
             'https://www.googleapis.com/auth/drive.file',
             'https://www.googleapis.com/auth/drive.appdata',
             'https://www.googleapis.com/auth/drive.apps.readonly']
    CACHED_INFO_DIR = "/workspace/data/cachedinfo/"
    MYSQL_CACHED_INFO_FILE = "mysql_server.csv"
    VENDOR_KEYS_FILE = "vendor_api_keys.csv"
    CACHED_INFO_SWITCHER = {
        "mysql_server_ip": OSMuxImpl.get_proper_path(CACHED_INFO_DIR) + MYSQL_CACHED_INFO_FILE,
        "mysql_server_user": OSMuxImpl.get_proper_path(CACHED_INFO_DIR) + MYSQL_CACHED_INFO_FILE,
        "mysql_server_secret": OSMuxImpl.get_proper_path(CACHED_INFO_DIR) + MYSQL_CACHED_INFO_FILE,
        "vendor_api_keys": OSMuxImpl.get_proper_path(CACHED_INFO_DIR) + VENDOR_KEYS_FILE
    }
    LOGGER = get_logger()

    def __init__(self):

        self.logger = get_logger()
        # init access to local files
        google_api_filesdir = '/workspace/data/googleapi/'
        local_data_file_pwd = OSMuxImpl.get_proper_path(google_api_filesdir)
        google_api_creds_file = local_data_file_pwd + "google_driveAccess_client_secrets.json"
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive',
                 'https://www.googleapis.com/auth/drive.file',
                 'https://www.googleapis.com/auth/drive.appdata',
                 'https://www.googleapis.com/auth/drive.apps.readonly']
        self.google_api_creds = ServiceAccountCredentials.from_json_keyfile_name(google_api_creds_file, scope)
        self.logger.info("SecreKeysAccess.__init__: self.google_api_creds file %s", google_api_creds_file)

    def authorize_client(self):

        return gspread.authorize(self.google_api_creds)

    def open_google_sheet(self,
                          authorized_client,
                          sheetFileName):

        self.logger.info("SecureKeysAccess.open_google_sheet: sheetFileName is %s", sheetFileName)
        sheet = authorized_client.open(sheetFileName).sheet1
        return sheet

    def get_sheet_records(self,
                          sheet):

        list_of_hashes = sheet.get_all_records()
        return list_of_hashes

    def get_access_key(self,
                       sheet,
                       vendor_name):

        list_of_hashes = sheet.get_all_records()
        for hasher in list_of_hashes:
            if hasher['Vendor'] == vendor_name:
                url = hasher['Url']
                username = hasher['Username']
                secret = hasher['Secret']
                return url, username, secret
        self.logger.error("SecureKeysAccess.get_access_key(): input vendor_name %s not found in access_keys",
                          vendor_name)

    def get_vendor_api_key(self,
                           sheet,
                           vendor_name):

        list_of_hashes = sheet.get_all_records()
        for hasher in list_of_hashes:
            if hasher['Vendor'] == vendor_name:
                return hasher['API_KEY']
        self.logger.error("SecureKeysAccess.get_vendor_api_key(): input vendor_name %s not found in api_keys!",
                          vendor_name)

    @staticmethod
    def get_ticker_simid_static(ticker):

        google_api_filesdir = '/workspace/data/googleapi'
        local_data_file_pwd = OSMuxImpl.get_proper_path(google_api_filesdir)
        google_api_creds_file = local_data_file_pwd + "google_driveAccess_client_secrets.json"
        google_api_creds = ServiceAccountCredentials.from_json_keyfile_name(google_api_creds_file,
                                                                            SecureKeysAccess.SCOPE)
        client = gspread.authorize(google_api_creds)
        sheet_filename = "simfin_simid.csv"
        simfin_simid_sheet = client.open(sheet_filename).sheet1
        list_of_hashes = simfin_simid_sheet.get_all_records()
        for hash in list_of_hashes:
            if hash['Symbol'] == ticker:
                return hash['SIM_ID']
        return ""

    @staticmethod
    def get_vendor_api_key_static(vendor):
        cached_vendor_api_key = SecureKeysAccess.check_cached_info(info_type="vendor_api_keys", info_field=vendor)
        if cached_vendor_api_key is None:
            google_api_filesdir = '/workspace/data/googleapi/'
            local_data_file_pwd = OSMuxImpl.get_proper_path(google_api_filesdir)
            google_api_creds_file = local_data_file_pwd + "google_driveAccess_client_secrets.json"
            google_api_creds = ServiceAccountCredentials.from_json_keyfile_name(google_api_creds_file,
                                                                                SecureKeysAccess.SCOPE)
            client = gspread.authorize(google_api_creds)
            sheet_filename = "ghazy_mahjub_api_keys.csv"
            api_key_sheet = client.open(sheet_filename).sheet1
            list_of_hashes = api_key_sheet.get_all_records()
            for hasher in list_of_hashes:
                if hasher['Vendor'] == vendor:
                    return hasher['API_KEY']
            return ""
        return cached_vendor_api_key

    @staticmethod
    def __get_mysql_info_object():
        google_api_filesdir = '/workspace/data/googleapi/'
        local_data_file_pwd = OSMuxImpl.get_proper_path(google_api_filesdir)
        google_api_creds_file = local_data_file_pwd + "google_driveAccess_client_secrets.json"
        google_api_creds = ServiceAccountCredentials.from_json_keyfile_name(google_api_creds_file,
                                                                            SecureKeysAccess.SCOPE)
        client = gspread.authorize(google_api_creds)
        sheet_filename = "mysql_server.csv"
        return client, sheet_filename

    @staticmethod
    def get_mysql_server_hostname():

        client, sheet_filename = SecureKeysAccess.__get_mysql_info_object()
        mysql_info_sheet = client.open(sheet_filename).sheet1
        list_of_hashes = mysql_info_sheet.get_all_values()
        for hasher in list_of_hashes:
            if hasher[0] == 'Hostname':
                return hasher[1]

    @staticmethod
    def check_cached_info(info_type, info_field):
        info_file = SecureKeysAccess.CACHED_INFO_SWITCHER[info_type]
        last_modified_time = datetime.fromtimestamp(os.path.getmtime(info_file))
        return_info = None
        if last_modified_time - datetime.now() < timedelta(days=30):
            df = pd.read_csv(filepath_or_buffer=info_file, index_col=0, header=None)
            df.index.name = 'Name'
            df.columns = [['Value']]
            print (df.loc[info_field]['Value'])
            return_info = df.loc[info_field]['Value'][0]
        return return_info

    @staticmethod
    def get_mysql_server_ip():
        cached_server_ip = SecureKeysAccess.check_cached_info("mysql_server_ip", 'LAN IP')
        if cached_server_ip is None:
            client, sheet_filename = SecureKeysAccess.__get_mysql_info_object()
            mysql_info_sheet = client.open(sheet_filename).sheet1
            list_of_hashes = mysql_info_sheet.get_all_values()
            for hasher in list_of_hashes:
                if hasher[0] == 'LAN IP':
                    return hasher[1]
        return cached_server_ip

    @staticmethod
    def get_mysql_server_user():
        cached_server_user = SecureKeysAccess.check_cached_info("mysql_server_user", "sftp_username_python")
        if cached_server_user is None:
            client, sheet_filename = SecureKeysAccess.__get_mysql_info_object()
            mysql_info_sheet = client.open(sheet_filename).sheet1
            list_of_hashes = mysql_info_sheet.get_all_values()
            for hasher in list_of_hashes:
                if hasher[0] == 'sftp_username_python':
                    return hasher[1]
        return cached_server_user

    @staticmethod
    def get_mysql_server_secret():
        cached_server_secret = SecureKeysAccess.check_cached_info("mysql_server_secret", "sftp_secret_python")
        if cached_server_secret is None:
            client, sheet_filename = SecureKeysAccess.__get_mysql_info_object()
            mysql_info_sheet = client.open(sheet_filename).sheet1
            list_of_hashes = mysql_info_sheet.get_all_values()
            for hasher in list_of_hashes:
                if hasher[0] == 'sftp_secret_python':
                    return hasher[1]
        return cached_server_secret

    @staticmethod
    def get_mysql_win_uploads_dir():

        client, sheet_filename = SecureKeysAccess.__get_mysql_info_object()
        mysql_info_sheet = client.open(sheet_filename).sheet1
        list_of_hashes = mysql_info_sheet.get_all_values()
        for hasher in list_of_hashes:
            if hasher[0] == 'mysql_win_uploads_dir':
                return hasher[1]

    @staticmethod
    def get_mysql_win_uploads_dir_str():

        client, sheet_filename = SecureKeysAccess.__get_mysql_info_object()
        mysql_info_sheet = client.open(sheet_filename).sheet1
        list_of_hashes = mysql_info_sheet.get_all_values()
        for hasher in list_of_hashes:
            if hasher[0] == 'mysql_win_uploads_dir_str':
                return hasher[1]

    @staticmethod
    def insert_simid(ticker, simid):

        google_api_filesdir = '/workspace/data/googleapi/'
        local_data_file_pwd = OSMuxImpl.get_proper_path(google_api_filesdir)
        google_api_creds_file = local_data_file_pwd + "google_driveAccess_client_secrets.json"
        google_api_creds = ServiceAccountCredentials.from_json_keyfile_name(
            google_api_creds_file, SecureKeysAccess.SCOPE)
        client = gspread.authorize(google_api_creds)
        sheet_filename = "simfin_simid.csv"
        simfin_simid_sheet = client.open(sheet_filename).sheet1
        the_recs = simfin_simid_sheet.get_all_records()
        num_recs = len(the_recs)
        SecureKeysAccess.LOGGER.info("there are %s entries in simfin_simid_file %s",
                                     num_recs, sheet_filename)
        for rec in the_recs:
            if rec['Symbol'] == ticker:
                SecureKeysAccess.LOGGER.warn(
                    "attempting to insert a record where the ticker %s already exists, use update instead!",
                    ticker)
                return
        row_to_insert = [ticker, simid]
        simfin_simid_sheet.insert_row(row_to_insert, num_recs + 2)
        SecureKeysAccess.LOGGER.info("inserted row value %s at index %s",
                                     str(row_to_insert), str(num_recs + 1))

    @staticmethod
    def insert_api_key(vendor, api_key):

        google_api_filesdir = '/workspace/data/googleapi/'
        local_data_file_pwd = OSMuxImpl.get_proper_path(google_api_filesdir)
        google_api_creds_file = local_data_file_pwd + "google_driveAccess_client_secrets.json"
        google_api_creds = ServiceAccountCredentials.from_json_keyfile_name(
            google_api_creds_file, SecureKeysAccess.SCOPE)
        client = gspread.authorize(google_api_creds)
        sheet_filename = "ghazy_mahjub_api_keys.csv"
        api_key_sheet = client.open(sheet_filename).sheet1
        the_recs = api_key_sheet.get_all_records()
        num_recs = len(the_recs)
        SecureKeysAccess.LOGGER.info("there are %s entries in api_key_file %s",
                                     num_recs, sheet_filename)
        # we don't want to insert duplicate VENDOR - check if VENDOR already exists
        # throw warning/error if the vendor already exists.
        # the_recs is a list of dictionaries
        for rec in the_recs:
            if rec['Vendor'] == vendor:
                SecureKeysAccess.LOGGER.warn(
                    "attempting to insert a record where the vendor %s already exists, use update instead!",
                    vendor)
                return
        row_to_insert = [vendor, api_key]
        api_key_sheet.insert_row(row_to_insert, num_recs + 2)
        SecureKeysAccess.LOGGER.info("inserted row value %s at index %s",
                                     str(row_to_insert), str(num_recs + 1))

    @staticmethod
    def update_api_key(vendor, api_key):

        google_api_filesdir = '/workspace/data/googleapi/'
        local_data_file_pwd = OSMuxImpl.get_proper_path(google_api_filesdir)
        google_api_creds_file = local_data_file_pwd + "google_driveAccess_client_secrets.json"
        google_api_creds = ServiceAccountCredentials.from_json_keyfile_name(
            google_api_creds_file, SecureKeysAccess.SCOPE)
        client = gspread.authorize(google_api_creds)
        sheet_filename = "ghazy_mahjub_api_keys.csv"
        api_key_sheet = client.open(sheet_filename).sheet1
        cell = None
        try:
            cell = api_key_sheet.find(vendor)
        except gspread.CellNotFound:
            SecureKeysAccess.LOGGER.error("no vendor %s found in sheet %s ",
                                          vendor, sheet_filename)
        finally:
            if cell is not None:
                if cell.col != 1:
                    SecureKeysAccess.LOGGER.error("expected vendor name %s to be in 1st column, "
                                                  "found it in column number %s, aborting!",
                                                  vendor, str(cell.col))
                else:
                    api_key_sheet.update_cell(cell.row, cell.col + 1, api_key)
                    SecureKeysAccess.LOGGER.info('updated vendor %s with new api_key %s',
                                                 vendor, api_key)
            else:
                return

    @staticmethod
    def delete_vendor(vendor):

        google_api_filesdir = '/workspace/data/googleapi/'
        local_data_file_pwd = OSMuxImpl.get_proper_path(google_api_filesdir)
        google_api_creds_file = local_data_file_pwd + "google_driveAccess_client_secrets.json"
        google_api_creds = ServiceAccountCredentials.from_json_keyfile_name(
            google_api_creds_file, SecureKeysAccess.SCOPE)
        client = gspread.authorize(google_api_creds)
        sheet_filename = "ghazy_mahjub_api_keys.csv"
        api_key_sheet = client.open(sheet_filename).sheet1
        cell = None
        try:
            cell = api_key_sheet.find(vendor)
        except gspread.CellNotFound:
            SecureKeysAccess.LOGGER.error("Vendor %s does not exist in sheet %s ",
                                          vendor, sheet_filename)
        finally:
            if cell is not None:
                api_key_sheet.delete_row(cell.row)
                SecureKeysAccess.LOGGER.warning("deleted vendor %s and corresponding "
                                                "api key!", vendor)
            else:
                return


if __name__ == '__main__':
    ska = SecureKeysAccess()
    hostname = SecureKeysAccess.get_mysql_server_hostname()
    uploads_dir = SecureKeysAccess.get_mysql_win_uploads_dir()
    ska.logger.info("Testing SecureKeysAccess: uploads dir is %s", uploads_dir)
