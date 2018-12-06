import gspread
from oauth2client.service_account import ServiceAccountCredentials
from root.nested.SysOs.os_mux import OSMuxImpl
from root.nested import get_logger


class SecureKeysAccess:

    SCOPE = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive',
             'https://www.googleapis.com/auth/drive.file',
             'https://www.googleapis.com/auth/drive.appdata',
             'https://www.googleapis.com/auth/drive.apps.readonly']
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
        return (sheet)

    def get_sheet_records(self,
                          sheet):

        list_of_hashes = sheet.get_all_records()
        print (list_of_hashes[0])
        print (list_of_hashes)

    def get_vendor_api_key(self,
                           sheet,
                           vendor_name):

        list_of_hashes = sheet.get_all_records()
        for hash in list_of_hashes:
            if hash['Vendor'] == vendor_name:
                return hash['API_KEY']
        self.logger.error("SecureKeysAccess.get_vendor_api_key(): input vendor_name %s not found in api_keys!", vendor_name)

    @staticmethod
    def get_vendor_api_key_static(vendor):

        google_api_filesdir = '/workspace/data/googleapi/'
        local_data_file_pwd = OSMuxImpl.get_proper_path(google_api_filesdir)
        google_api_creds_file = local_data_file_pwd + "google_driveAccess_client_secrets.json"
        google_api_creds = ServiceAccountCredentials.from_json_keyfile_name(google_api_creds_file,
                                                                            SecureKeysAccess.SCOPE)
        client = gspread.authorize(google_api_creds)
        sheet_filename = "ghazy_mahjub_api_keys.csv"
        api_key_sheet = client.open(sheet_filename).sheet1
        list_of_hashes = api_key_sheet.get_all_records()
        for hash in list_of_hashes:
            if hash['Vendor'] == vendor:
                return hash['API_KEY']
        return ""

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
        SecureKeysAccess.LOGGER.info("there are %s entires in api_key_file %s",
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
                    api_key_sheet.update_cell(cell.row, cell.col+1, api_key)
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

SecureKeysAccess.insert_api_key('IBKR', "fuck off")
SecureKeysAccess.update_api_key("IBKR", "dgh78_98!")
SecureKeysAccess.delete_vendor('IBKR')