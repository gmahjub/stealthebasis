import gspread
from oauth2client.service_account import ServiceAccountCredentials
from root.nested.SysOs.os_mux import OSMuxImpl
from root.nested import get_logger

class SecureKeysAccess:

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
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive',
                 'https://www.googleapis.com/auth/drive.file',
                 'https://www.googleapis.com/auth/drive.appdata',
                 'https://www.googleapis.com/auth/drive.apps.readonly']
        google_api_creds = ServiceAccountCredentials.from_json_keyfile_name(google_api_creds_file, scope)
        client = gspread.authorize(google_api_creds)
        sheet_filename = "ghazy_mahjub_api_keys.csv"
        api_key_sheet = client.open(sheet_filename).sheet1
        list_of_hashes = api_key_sheet.get_all_records()
        for hash in list_of_hashes:
            if hash['Vendor'] == vendor:
                return hash['API_KEY']
        return ""


ska = SecureKeysAccess()
authorized_client = ska.authorize_client()
ghazy_mahjub_api_keys_sheet = ska.open_google_sheet(authorized_client = authorized_client,
                                                    sheetFileName="ghazy_mahjub_api_keys.csv")
api_key = ska.get_vendor_api_key(ghazy_mahjub_api_keys_sheet, "QUANDL")